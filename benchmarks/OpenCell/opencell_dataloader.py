"""Dataset over the cytoself single-cell crops of OpenCell.

Source files (opencell_data/cytoself/):
    Image_data0{0..9}.npy   float32 [N, 100, 100, 4], channels = (protein, nucleus,
                            nuclear distance map, nuclear segmentation labels)
    Label_data0{0..9}.csv   one row per crop, row-aligned with the arrays:
                            ensg, name, loc_grade1-3, protein_id, FOV_id

Each item is a 2-channel crop ordered [nucleus, protein] (the order SubCell's DNA-Protein model
and the DINOv3 nuc_rgb mode expect; MorphEm embeds channels independently, so order only fixes
the order of its concatenated output). Only proteins listed in nodes.csv are kept, keyed by
hgnc_id so features join directly onto nodes.csv / pairs.parquet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROTEIN, NUCLEUS = 0, 1  # channel indices in the cytoself arrays


def load_crop_index(cytoself_dir, nodes_csv, baits_only=True, max_crops_per_protein=None,
                    proteins=None, seed=0):
    """Table of the crops to embed: one row per crop with its file, row and protein ids."""
    cytoself_dir = Path(cytoself_dir)
    frames = []
    for f in sorted(cytoself_dir.glob("Label_data0*.csv")):
        lab = pd.read_csv(f, usecols=["name", "FOV_id"])
        lab["file_id"] = int(f.stem[-2:])
        lab["row"] = np.arange(len(lab))
        frames.append(lab)
    crops = pd.concat(frames, ignore_index=True)

    nodes = pd.read_csv(nodes_csv)
    if baits_only:
        nodes = nodes[nodes.is_bait]
    if proteins:
        wanted = {p.upper() for p in proteins}
        nodes = nodes[nodes.hgnc_symbol.str.upper().isin(wanted) | nodes.opencell_name.str.upper().isin(wanted)]
    name_to_node = nodes.set_index("opencell_name")[["hgnc_id", "hgnc_symbol"]]
    crops = crops.join(name_to_node, on="name", how="inner")

    if max_crops_per_protein:
        crops = (crops.groupby("hgnc_id", group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), max_crops_per_protein), random_state=seed)))
    crops = crops.sort_values(["file_id", "row"]).reset_index(drop=True)  # sequential reads
    crops.insert(0, "crop_idx", np.arange(len(crops)))
    return crops


class OpenCellCropDataset(Dataset):
    """Returns (image [2, H, W] float32 = [nucleus, protein], crop_idx).

    intensity_scale: crops are stored in [0, 1]; the repo's MorphEm transform follows the
    CHAMMI 0-255 convention, so they are multiplied by 255 by default. DINOv3 and SubCell
    min-max scale inside their wrappers, so the factor does not affect them.
    """

    def __init__(self, crops, cytoself_dir, transform=None, intensity_scale=255.0):
        self.crops = crops
        self.file_id = crops.file_id.to_numpy()
        self.row = crops.row.to_numpy()
        self.paths = {i: Path(cytoself_dir) / f"Image_data{i:02d}.npy" for i in np.unique(self.file_id)}
        self.transform = transform
        self.intensity_scale = intensity_scale
        self._arrays = {}  # opened lazily, once per worker process

    def __len__(self):
        return len(self.crops)

    def _array(self, file_id):
        if file_id not in self._arrays:
            self._arrays[file_id] = np.load(self.paths[file_id], mmap_mode="r")
        return self._arrays[file_id]

    def __getitem__(self, i):
        crop = self._array(self.file_id[i])[self.row[i]]            # (100, 100, 4)
        image = np.stack([crop[..., NUCLEUS], crop[..., PROTEIN]])   # (2, 100, 100)
        image = torch.from_numpy(image.astype(np.float32) * self.intensity_scale)
        if self.transform is not None:
            image = self.transform(image)
        return image, i
