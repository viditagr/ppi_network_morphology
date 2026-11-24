"""Extract single-cell embeddings for the OpenCell PPI benchmark with the models in benchmarks/models.py.

Run from the repo root, one model at a time (multi-GPU via accelerate, as for HPAv23):

    accelerate launch benchmarks/OpenCell/extract_features.py --model MorphEm  --output_folder opencell_features/morphem
    accelerate launch benchmarks/OpenCell/extract_features.py --model dinov3   --output_folder opencell_features/dinov3
    accelerate launch benchmarks/OpenCell/extract_features.py --model subcell  --output_folder opencell_features/subcell \
        --config_path /mnt/cephfs/mir/jcaicedo/morphem/dataset/models/subcell_models/DNA-Protein_ViT-ProtS-Pool.yaml

Quick smoke test on two proteins and 20 crops each (single GPU):

    python benchmarks/OpenCell/extract_features.py --model dinov3 --output_folder /tmp/oc_test \
        --proteins PSMD3,PSMD6 --max_crops_per_protein 20

Per model, the input is the 2-channel crop [nucleus, protein]:
    MorphEm  each channel embedded separately at 224 px, CLS tokens concatenated -> 768-d
    dinov3   one composite RGB image (R = nucleus, G = B = protein) at 224 px -> 384-d
    subcell  2-channel DNA-Protein model at --image_size (default 448, its training size) -> 1536-d

Outputs in --output_folder:
    crop_features.npy      [n_crops, d] float16, rows aligned with crop_metadata.parquet
    crop_metadata.parquet  crop_idx, hgnc_id, hgnc_symbol, name, FOV_id, file_id, row
    fov_features.parquet   mean embedding per (protein, FOV)
    protein_features.parquet  mean embedding per protein (+ n_crops, n_fovs); joins nodes.csv on hgnc_id
    run_config.json        arguments used
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torchvision.transforms import v2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarks/
sys.path.insert(0, str(Path(__file__).resolve().parent))         # benchmarks/OpenCell/
from models import get_model  # noqa: E402
from opencell_dataloader import OpenCellCropDataset, load_crop_index  # noqa: E402

accelerator = Accelerator()


def to_numpy(features):
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    return features.reshape(features.shape[0], -1)  # SubCell returns [B, 1, D]


def build_model(args):
    model_instance = get_model(
        model_name=args.model,
        device=accelerator.device,
        model_path=args.model_path,
        model_size=args.model_size,
        config_path=args.config_path,
        dinov3_mode="nuc_rgb",
    )
    model, transform = model_instance.get_model()
    model.to(accelerator.device)
    if transform is None:  # SubCell has no transform in models.py; resize to its input size
        transform = v2.Resize(size=(args.image_size, args.image_size), antialias=True)
    return model_instance, transform


def extract(dataloader, model_instance, tmp_dir):
    feats, idxs = [], []
    for images, crop_idx in tqdm(dataloader, desc=f"GPU {accelerator.process_index}",
                                 disable=not accelerator.is_local_main_process):
        feats.append(to_numpy(model_instance(images)))
        idxs.append(crop_idx.cpu().numpy())
    np.savez(tmp_dir / f"process_{accelerator.process_index}.npz",
             features=np.concatenate(feats) if feats else np.zeros((0, 0), np.float32),
             crop_idx=np.concatenate(idxs) if idxs else np.zeros(0, np.int64))


def combine(crops, out, tmp_dir):
    parts = [np.load(tmp_dir / f"process_{i}.npz") for i in range(accelerator.num_processes)]
    feats = np.concatenate([p["features"] for p in parts if len(p["crop_idx"])])
    idx = np.concatenate([p["crop_idx"] for p in parts])
    # accelerate pads the last batches with repeated samples: keep one copy of each crop, in order
    idx, first = np.unique(idx, return_index=True)
    feats = feats[first]
    assert len(idx) == len(crops) and (idx == crops.crop_idx.to_numpy()).all(), "missing crops"
    for i in range(accelerator.num_processes):
        os.remove(tmp_dir / f"process_{i}.npz")

    np.save(out / "crop_features.npy", feats.astype(np.float16))
    crops.to_parquet(out / "crop_metadata.parquet", index=False)

    cols = [f"f{j}" for j in range(feats.shape[1])]
    df = pd.concat([crops[["hgnc_id", "hgnc_symbol", "FOV_id"]], pd.DataFrame(feats, columns=cols)], axis=1)
    fov = df.groupby(["hgnc_id", "hgnc_symbol", "FOV_id"])[cols].mean()
    fov.insert(0, "n_crops", df.groupby(["hgnc_id", "hgnc_symbol", "FOV_id"]).size())
    fov.reset_index().to_parquet(out / "fov_features.parquet", index=False)
    prot = df.groupby(["hgnc_id", "hgnc_symbol"])[cols].mean()
    prot.insert(0, "n_fovs", df.groupby(["hgnc_id", "hgnc_symbol"]).FOV_id.nunique())
    prot.insert(0, "n_crops", df.groupby(["hgnc_id", "hgnc_symbol"]).size())
    prot.reset_index().to_parquet(out / "protein_features.parquet", index=False)
    print(f"saved {len(feats)} crops x {feats.shape[1]} dims; {len(fov)} FOVs; {len(prot)} proteins -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["MorphEm", "dinov3", "subcell"])
    ap.add_argument("--output_folder", required=True, type=Path)
    ap.add_argument("--cytoself_dir", type=Path, default=Path("opencell_data/cytoself"))
    ap.add_argument("--nodes_csv", type=Path, default=Path("opencell_data/benchmark/nodes.csv"))
    ap.add_argument("--config_path", default=None, help="SubCell yaml (.pth beside it)")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--model_size", default="small")
    ap.add_argument("--image_size", type=int, default=448, help="SubCell input size (others use their own transform)")
    ap.add_argument("--all_imaged", action="store_true", help="also embed the 139 imaged non-bait proteins")
    ap.add_argument("--max_crops_per_protein", type=int, default=None, help="random subsample per protein (seeded)")
    ap.add_argument("--proteins", default=None, help="comma-separated gene symbols, for testing")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    out = args.output_folder
    tmp_dir = out / "tmp"
    if accelerator.is_main_process:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    accelerator.wait_for_everyone()

    crops = load_crop_index(args.cytoself_dir, args.nodes_csv, baits_only=not args.all_imaged,
                            max_crops_per_protein=args.max_crops_per_protein,
                            proteins=args.proteins.split(",") if args.proteins else None)
    if accelerator.is_main_process:
        print(f"{len(crops)} crops from {crops.hgnc_id.nunique()} proteins; model {args.model} on "
              f"{accelerator.num_processes} process(es)")

    model_instance, transform = build_model(args)
    dataset = OpenCellCropDataset(crops, args.cytoself_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    loader = accelerator.prepare(loader)

    extract(loader, model_instance, tmp_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        combine(crops, out, tmp_dir)
        tmp_dir.rmdir()


if __name__ == "__main__":
    main()
