"""Fetch OpenCell's MCL interaction clusters (communities and cores) from the OpenCell web API.

    python benchmarks/OpenCell/fetch_mcl_clusters.py --out opencell_data/clusters

The paper (Cho et al., Science 2022) clustered the IP-MS network with Markov Clustering (MCL) at two
levels: "communities" (cluster_id, coarse functional modules) and, inside them, "core complexes"
(subcluster_id, stable high-stoichiometry assemblies). These labels are not in the downloadable CSVs;
the website's network viewer gets them per pulldown from /api/pulldowns/<id>/network. This script
downloads that response for every pulldown and merges the node labels into one table.

Outputs (under --out)
    lines.json              the API's cell-line listing (gives each line's pulldown id)
    raw/<pulldown_id>.json  untouched API responses (cached; rerunning skips existing files)
    mcl_clusters.csv        one row per protein seen in any network: gene names, Ensembl ids,
                            community (cluster_id), core (subcluster_id), and how many networks agreed
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

API = "https://opencell.czbiohub.org/api"
# The clustering the website displays (taken from the site's JavaScript bundle).
CLUSTERING = "primary:mcl_i3.0_haircut:keepcore_subcluster:mcl_hybrid_stoichs_7.0_1113"
SUBCLUSTER = "core-complexes"


def get_json(url, dest, retries=5):
    if dest.exists():
        return json.loads(dest.read_text())
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            data = json.loads(body)
            dest.write_bytes(body)
            return data
        except urllib.error.HTTPError as e:
            err = e
            if e.code == 404:  # the server has no network for this pulldown; retrying won't help
                break
            time.sleep(2 ** attempt)
        except Exception as e:  # network hiccup or transient server error
            err = e
            time.sleep(2 ** attempt)
    print(f"FAILED {url}: {err!r}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("opencell_data/clusters"))
    ap.add_argument("--delay", type=float, default=0.1, help="seconds between requests (be polite)")
    args = ap.parse_args()
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    lines = get_json(f"{API}/lines?publication_ready=false", args.out / "lines.json")
    pulldowns = sorted({l["best_pulldown"]["id"] for l in lines if (l.get("best_pulldown") or {}).get("id")})
    query = urllib.parse.urlencode({"subcluster_type": SUBCLUSTER, "clustering_analysis_type": CLUSTERING})
    print(f"{len(lines)} cell lines, {len(pulldowns)} pulldowns")

    failed = []
    for i, pid in enumerate(pulldowns, 1):
        dest = raw / f"{pid}.json"
        if not dest.exists():
            if get_json(f"{API}/pulldowns/{pid}/network?{query}", dest) is None:
                failed.append(pid)
            time.sleep(args.delay)
        if i % 100 == 0 or i == len(pulldowns):
            print(f"  {i}/{len(pulldowns)} fetched, {len(failed)} failed", flush=True)

    # Merge: a protein appears in many networks; record every (community, core) it was given.
    seen = defaultdict(Counter)
    info = {}
    for f in sorted(raw.glob("*.json")):
        for node in json.loads(f.read_text())["nodes"]:
            d = node["data"]
            key = ";".join(sorted(d.get("uniprot_gene_names") or [])) or d["id"]
            seen[key][(d.get("cluster_id"), d.get("subcluster_id"))] += 1
            info[key] = d
    # A protein is often unlabeled in networks where it is a peripheral hit (the viewer only draws
    # cluster boxes for part of the neighbourhood), so a blank label is "not shown", not "unclustered".
    # Take the most common non-blank label; record real conflicts (two different non-blank labels).
    rows = []
    for key, labels in seen.items():
        labeled = Counter({k: v for k, v in labels.items() if k[0] is not None})
        (community, core), n_top = labeled.most_common(1)[0] if labeled else ((None, None), 0)
        d = info[key]
        rows.append({
            "gene_names": key,
            "opencell_target_names": ";".join(d.get("opencell_target_names") or []),
            "ensg_ids": ";".join(d.get("ensg_ids") or []),
            "community_id": community,
            "core_id": core,
            "n_networks": sum(labels.values()),
            "n_networks_labeled": sum(labeled.values()),
            "n_label_variants": len(labeled),
            "label_agreement": n_top / sum(labeled.values()) if labeled else None,
        })
    df = pd.DataFrame(rows).sort_values("gene_names")
    for c in ["community_id", "core_id"]:
        df[c] = df[c].astype("Int64")
    df.to_csv(args.out / "mcl_clusters.csv", index=False)

    print(f"proteins seen: {len(df)}; in a community: {df.community_id.notna().sum()} "
          f"({df.community_id.nunique()} communities); in a core: {df.core_id.notna().sum()} "
          f"({df.core_id.nunique()} cores); conflicting labels: {(df.n_label_variants > 1).sum()}")
    if failed:
        sys.exit(f"{len(failed)} pulldowns failed; rerun to retry")


if __name__ == "__main__":
    main()
