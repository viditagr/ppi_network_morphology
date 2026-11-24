"""Build the node table and ground-truth pair table for the OpenCell PPI-vs-morphology benchmark.

    python benchmarks/OpenCell/build_ground_truth.py --data opencell_data --out opencell_data/benchmark

Inputs (under --data)
    opencell-protein-interactions.csv, opencell-library-metadata.csv,
    opencell-localization-annotations.csv, opencell-protein-abundance.csv
    cytoself/Label_data0*.csv           (cytoself single-cell crop labels)
    s3_proj/<GENE>_<ENSG>/              (one folder per imaged target)
    external/hgnc_complete_set.txt      (HGNC symbols, for ID reconciliation)
    clusters/mcl_clusters.csv           (optional; MCL communities/cores from fetch_mcl_clusters.py)

Identifier policy
    Every protein is keyed by its HGNC ID. Names are resolved *before* Ensembl IDs because the
    sources disagree on Ensembl IDs (different releases) and some are simply wrong: the S3 folder
    for LIG1 carries LRIG1's Ensembl ID and NAPG carries RAB11FIP5's. Resolution order for a name:
    current HGNC symbol > unique previous symbol > unique alias; then the Ensembl ID as fallback.
    Prey names that are protein groups ("SUMO3;SUMO2;SUMO4") resolve to the first member that
    resolves; all members are kept in `interactor_group_hgnc`.

Outputs (under --out)
    nodes.csv              one row per imaged protein (1,311), with bait status, crop/FOV counts,
                           localization labels and HEK293T abundance
    interactions_resolved.csv  every IP-MS edge with HGNC IDs on both ends (incl. non-imaged preys)
    pairs.parquet          every unordered pair of imaged baits, the benchmark ground truth
    summary.json           counts for sanity checking
"""

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

GRADES = ["annotations_grade_3", "annotations_grade_2", "annotations_grade_1"]

# Bait names in the interaction table that HGNC cannot resolve.
MANUAL_BAIT_SYMBOL = {
    # The TOMM20A pulldown uses TOMM20 as its stoichiometry reference (interaction_stoichiometry == 1.0),
    # and TOMM20 is an imaged target with no other pulldown, so this is the TOMM20 line.
    "TOMM20A": "TOMM20",
}


# --------------------------------------------------------------------------- HGNC resolution
def expand_slash(token):
    """Expand OpenCell's paralog shorthand: 'AKT1/2/3' -> AKT1, AKT2, AKT3;
    'PABPC1/1L/3/5' -> PABPC1, PABPC1L, PABPC3, PABPC5; 'ARHGAP11A/11B' -> ARHGAP11A, ARHGAP11B."""
    first, *rest = token.strip().split("/")
    m = re.search(r"[0-9]+[A-Za-z]*$", first)
    stem = first[:m.start()] if m else first
    return [first] + [stem + s for s in rest if s]


class HGNCResolver:
    def __init__(self, path):
        h = pd.read_csv(path, sep="\t", dtype=str,
                        usecols=["hgnc_id", "symbol", "prev_symbol", "alias_symbol", "ensembl_gene_id"])
        self.symbol_of = dict(zip(h.hgnc_id, h.symbol))
        self.by_symbol = {s.upper(): i for s, i in zip(h.symbol, h.hgnc_id)}
        self.by_ensg = {e: i for e, i in zip(h.ensembl_gene_id, h.hgnc_id) if isinstance(e, str)}
        self.by_prev, self.by_alias = defaultdict(set), defaultdict(set)
        for table, col in [(self.by_prev, "prev_symbol"), (self.by_alias, "alias_symbol")]:
            for hid, val in zip(h.hgnc_id, h[col]):
                if isinstance(val, str):
                    for s in val.split("|"):
                        table[s.strip().upper()].add(hid)

    def name(self, name):
        """Resolve one gene symbol -> (hgnc_id, method) or (None, None)."""
        u = str(name).strip().upper()
        if u in self.by_symbol:
            return self.by_symbol[u], "symbol"
        if len(self.by_prev.get(u, ())) == 1:
            return next(iter(self.by_prev[u])), "prev_symbol"
        if len(self.by_alias.get(u, ())) == 1:
            return next(iter(self.by_alias[u])), "alias"
        return None, None

    def resolve(self, name, ensg=None):
        """Resolve a possibly ';'-joined name with optional Ensembl fallback.
        Returns (hgnc_id, method, [all member hgnc ids])."""
        members = []
        if isinstance(name, str):
            for tok in [t for part in name.split(";") for t in expand_slash(part)]:
                hid, how = self.name(tok)
                if hid and hid not in [m[0] for m in members]:
                    members.append((hid, how))
        if members:
            return members[0][0], members[0][1], [m[0] for m in members]
        if isinstance(ensg, str) and ensg in self.by_ensg:
            return self.by_ensg[ensg], "ensembl", [self.by_ensg[ensg]]
        return None, "unresolved", []

    def frame(self, df, name_col, ensg_col=None, prefix=""):
        """Add <prefix>hgnc_id / <prefix>hgnc_symbol / <prefix>resolved_by columns."""
        keys = df[[name_col] + ([ensg_col] if ensg_col else [])].drop_duplicates()
        out = {}
        for row in keys.itertuples(index=False):
            out[tuple(row)] = self.resolve(row[0], row[1] if ensg_col else None)
        res = [out[tuple(r)] for r in df[[name_col] + ([ensg_col] if ensg_col else [])].itertuples(index=False)]
        df = df.copy()
        df[prefix + "hgnc_id"] = [r[0] for r in res]
        df[prefix + "hgnc_symbol"] = [self.symbol_of.get(r[0]) for r in res]
        df[prefix + "resolved_by"] = [r[1] for r in res]
        df[prefix + "group_hgnc"] = [";".join(r[2]) for r in res]
        return df


def labels(s):
    return set() if not isinstance(s, str) else {x.strip() for x in s.split(";") if x.strip()}


# --------------------------------------------------------------------------- nodes
def build_nodes(data, hg):
    # Imaged proteins: S3 folders and cytoself labels use identical gene names.
    s3 = pd.DataFrame([d.name.rsplit("_", 1) for d in (data / "s3_proj").iterdir() if d.is_dir()],
                      columns=["opencell_name", "s3_ensg"])
    s3["s3_folder"] = s3.opencell_name + "_" + s3.s3_ensg
    s3["n_fovs_s3"] = [len(list((data / "s3_proj" / f).glob("*_proj.tif"))) for f in s3.s3_folder]

    lab = pd.concat([pd.read_csv(f, usecols=["ensg", "name", "protein_id", "FOV_id"])
                     for f in sorted((data / "cytoself").glob("Label_data0*.csv"))])
    cs = (lab.groupby("name")
             .agg(cytoself_protein_id=("protein_id", "first"), n_crops=("protein_id", "size"),
                  n_fovs_cytoself=("FOV_id", "nunique"))
             .reset_index().rename(columns={"name": "opencell_name"}))
    nodes = s3.merge(cs, on="opencell_name", how="outer", validate="1:1", indicator=True)
    assert (nodes._merge == "both").all(), "S3 folders and cytoself labels disagree on proteins"
    nodes = hg.frame(nodes.drop(columns="_merge"), "opencell_name", "s3_ensg")
    assert nodes.hgnc_id.notna().all() and not nodes.hgnc_id.duplicated().any()

    lib = hg.frame(pd.read_csv(data / "opencell-library-metadata.csv"), "gene_name", "ensg_id")
    lib = lib[["hgnc_id", "ensg_id", "uniprot_id", "tagged_terminus"]].rename(columns={"ensg_id": "library_ensg"})
    loc = hg.frame(pd.read_csv(data / "opencell-localization-annotations.csv"), "target_name", "ensg_id")
    loc = loc[["hgnc_id"] + GRADES].rename(columns=lambda c: c.replace("annotations_", "loc_"))
    ab = hg.frame(pd.read_csv(data / "opencell-protein-abundance.csv"), "gene_name", "ensg_id")
    ab = (ab.dropna(subset=["hgnc_id"]).sort_values("hek_protein_copy_number", ascending=False)
            .drop_duplicates("hgnc_id")[["hgnc_id", "hek_rna_tpm", "hek_protein_conc_nm", "hek_protein_copy_number"]])
    for t in (lib, loc):
        assert not t.hgnc_id.dropna().duplicated().any()
    nodes = (nodes.merge(lib, on="hgnc_id", how="left")
                  .merge(loc, on="hgnc_id", how="left")
                  .merge(ab, on="hgnc_id", how="left"))

    # MCL communities / cores from the OpenCell API (benchmarks/OpenCell/fetch_mcl_clusters.py)
    mcl_path = data / "clusters" / "mcl_clusters.csv"
    if mcl_path.exists():
        mcl_all = hg.frame(pd.read_csv(mcl_path), "gene_names").sort_values("n_networks_labeled", ascending=False)
        mcl = (mcl_all.dropna(subset=["hgnc_id"]).drop_duplicates("hgnc_id")
                  [["hgnc_id", "community_id", "core_id", "n_networks_labeled", "n_label_variants"]]
                  .rename(columns={"n_networks_labeled": "mcl_n_networks_labeled",
                                   "n_label_variants": "mcl_n_label_variants"}))
        nodes = nodes.merge(mcl, on="hgnc_id", how="left")
        nodes["mcl_label_source"] = np.where(nodes.community_id.notna(), "direct", None)
        # MS protein groups ("ACTN2/4", "PRPH;VIM") are labeled under their first member only. An imaged
        # protein with no label of its own inherits the label of a labeled group it belongs to; it is
        # usually the HEK293T-expressed member (VIM, not neuronal PRPH).
        inherited = {}
        for members, com, core in zip(mcl_all.group_hgnc, mcl_all.community_id, mcl_all.core_id):
            if isinstance(members, str) and pd.notna(com):
                for h in members.split(";")[1:]:
                    inherited.setdefault(h, (com, core))  # rows are sorted by evidence
        fill = nodes.community_id.isna() & nodes.hgnc_id.isin(inherited)
        nodes.loc[fill, "community_id"] = [inherited[h][0] for h in nodes.loc[fill, "hgnc_id"]]
        nodes.loc[fill, "core_id"] = [inherited[h][1] for h in nodes.loc[fill, "hgnc_id"]]
        nodes.loc[fill, "mcl_label_source"] = "protein_group"
        for c in ["community_id", "core_id"]:
            nodes[c] = pd.to_numeric(nodes[c]).astype("Int64")
    else:
        print(f"note: {mcl_path} not found; skipping MCL community/core labels")
    return nodes


# --------------------------------------------------------------------------- pairs
def stoich_weight(interaction_stoich, abundance_stoich):
    """OpenCell's MCL edge weight (10 / 4 / 0.2). Reverse-engineered from the authors' clustering input
    (2021-opencell-figures/data/ppi_analysis/clustering/oc_interactions_stoich_weighted.csv): this rule
    reproduces all 25,130 matched edge weights exactly. Weight > 2 is what the paper calls a "core"
    (stoichiometric) interaction; an undirected pair takes the max over its two directions."""
    window = (abundance_stoich >= 0.1) & (abundance_stoich <= 10)
    return np.where(window & (interaction_stoich >= 0.2), 10.0,
                    np.where(window & (interaction_stoich >= 0.05), 4.0, 0.2))


def build_pairs(nodes, ppi):
    baits = nodes[nodes.is_bait].sort_values("hgnc_id").reset_index(drop=True)
    idx = {h: i for i, h in enumerate(baits.hgnc_id)}
    n = len(baits)

    # directed bait->prey evidence restricted to bait-bait edges; aggregate duplicates (protein groups)
    bb = ppi[ppi.target_hgnc_id.isin(idx) & ppi.interactor_hgnc_id.isin(idx)
             & (ppi.target_hgnc_id != ppi.interactor_hgnc_id)].copy()
    bb["istoich"] = bb.interaction_stoichiometry.replace([np.inf, -np.inf], np.nan)
    bb["weight"] = stoich_weight(bb.interaction_stoichiometry, bb.abundance_stoichiometry)
    bb = (bb.groupby(["target_hgnc_id", "interactor_hgnc_id"])
            .agg(pval=("pval", "max"), enrichment=("enrichment", "max"),
                 istoich=("istoich", "max"), astoich=("abundance_stoichiometry", "max"),
                 istoich_inf=("interaction_stoichiometry", lambda s: bool(np.isinf(s).any())),
                 only_group=("interactor_is_group", "all"), weight=("weight", "max"))
            .reset_index())

    # all unordered pairs i<j
    I, J = np.triu_indices(n, k=1)
    pairs = pd.DataFrame({"a": baits.hgnc_id.values[I], "b": baits.hgnc_id.values[J]})
    pairs["symbol_a"] = baits.hgnc_symbol.values[I]
    pairs["symbol_b"] = baits.hgnc_symbol.values[J]

    def direction(src, dst, tag):
        d = bb.rename(columns={"target_hgnc_id": src, "interactor_hgnc_id": dst})
        d = d.rename(columns={c: f"{c}_{tag}" for c in
                              ["pval", "enrichment", "istoich", "astoich", "istoich_inf", "only_group", "weight"]})
        return d

    pairs = pairs.merge(direction("a", "b", "ab"), on=["a", "b"], how="left")  # a pulled down b
    pairs = pairs.merge(direction("b", "a", "ba"), on=["a", "b"], how="left")  # b pulled down a
    pairs["found_ab"] = pairs.pval_ab.notna()
    pairs["found_ba"] = pairs.pval_ba.notna()
    pairs["n_directions"] = pairs.found_ab.astype(int) + pairs.found_ba.astype(int)
    pairs["is_edge"] = pairs.n_directions > 0
    pairs["reciprocal"] = pairs.n_directions == 2
    pairs["interaction_stoich_max"] = pairs[["istoich_ab", "istoich_ba"]].max(axis=1)
    flag = lambda col: pairs[col].astype("boolean").fillna(False).astype(bool)
    pairs["interaction_stoich_inf"] = flag("istoich_inf_ab") | flag("istoich_inf_ba")
    # edge supported only via ambiguous MS protein groups (e.g. "CALM2;CALM3;CALM1") in every direction found
    pairs["edge_via_group_only"] = pairs.is_edge & ~((pairs.found_ab & ~flag("only_group_ab"))
                                                      | (pairs.found_ba & ~flag("only_group_ba")))
    pairs["abundance_stoich_max"] = pairs[["astoich_ab", "astoich_ba"]].max(axis=1)
    pairs["pval_max"] = pairs[["pval_ab", "pval_ba"]].max(axis=1)
    pairs["enrichment_max"] = pairs[["enrichment_ab", "enrichment_ba"]].max(axis=1)
    pairs["mcl_weight"] = pairs[["weight_ab", "weight_ba"]].max(axis=1)
    pairs["is_core_edge"] = pairs.mcl_weight > 2
    pairs["edge_tier"] = np.select([pairs.mcl_weight == 10, pairs.mcl_weight == 4, pairs.is_edge],
                                   ["core_strong", "core", "non_core"], default="none")
    pairs = pairs.drop(columns=[c for c in pairs.columns if c.endswith(("_ab", "_ba")) and not c.startswith("found")])

    # shared interaction partners in the full network (any protein, imaged or not)
    allp = pd.Index(sorted(set(ppi.target_hgnc_id.dropna()) | set(ppi.interactor_hgnc_id.dropna())))
    e = ppi.dropna(subset=["target_hgnc_id", "interactor_hgnc_id"])
    e = e[e.target_hgnc_id != e.interactor_hgnc_id]
    r, c = allp.get_indexer(e.target_hgnc_id), allp.get_indexer(e.interactor_hgnc_id)
    A = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(len(allp),) * 2).tocsr()
    A = ((A + A.T) > 0).astype(np.int32)
    sel = allp.get_indexer(baits.hgnc_id)
    sel_ok = sel >= 0
    common = np.zeros((n, n), dtype=np.int32)
    sub = A[sel[sel_ok]]
    common[np.ix_(sel_ok, sel_ok)] = (sub @ sub.T).toarray()
    deg = np.zeros(n, dtype=np.int32)
    deg[sel_ok] = np.asarray(sub.sum(axis=1)).ravel()
    pairs["n_shared_partners"] = common[I, J]
    pairs["jaccard_partners"] = pairs.n_shared_partners / np.maximum(deg[I] + deg[J] - pairs.n_shared_partners, 1)
    pairs["second_neighbor"] = ~pairs.is_edge & (pairs.n_shared_partners > 0)

    # localization overlap (manual annotations)
    g3 = [labels(x) for x in baits.loc_grade_3]
    g23 = [labels(x) | labels(y) for x, y in zip(baits.loc_grade_3, baits.loc_grade_2)]
    inter = np.array([len(g23[i] & g23[j]) for i, j in zip(I, J)])
    union = np.array([len(g23[i] | g23[j]) for i, j in zip(I, J)])
    pairs["loc_jaccard_g23"] = inter / np.maximum(union, 1)
    pairs["share_loc_g23"] = inter > 0
    pairs["share_loc_g3"] = np.array([bool(g3[i] & g3[j]) for i, j in zip(I, J)])

    # MCL cluster co-membership (False when either protein is unlabeled)
    if "community_id" in baits:
        for col, name in [("community_id", "same_community"), ("core_id", "same_core")]:
            v = baits[col].to_numpy(dtype="float64", na_value=np.nan)
            pairs[name] = (v[I] == v[J]) & ~np.isnan(v[I])
    return pairs


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("opencell_data"))
    ap.add_argument("--out", type=Path, default=Path("opencell_data/benchmark"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hg = HGNCResolver(args.data / "external" / "hgnc_complete_set.txt")
    nodes = build_nodes(args.data, hg)

    ppi = pd.read_csv(args.data / "opencell-protein-interactions.csv")
    ppi = hg.frame(ppi, "target_gene_name", "target_ensg_id", prefix="target_")
    ppi = hg.frame(ppi, "interactor_gene_name", "interactor_ensg_id", prefix="interactor_")
    ppi = ppi.drop(columns="target_group_hgnc")

    # Manual bait overrides for names HGNC cannot resolve (evidence noted per entry).
    for bait_name, symbol in MANUAL_BAIT_SYMBOL.items():
        hid, _ = hg.name(symbol)
        m = ppi.target_gene_name == bait_name
        ppi.loc[m, ["target_hgnc_id", "target_hgnc_symbol", "target_resolved_by"]] = [hid, symbol, "manual"]

    # A bait was tested even if all it recovered was itself, so fix the bait set before filtering.
    baits = set(ppi.target_hgnc_id.dropna())

    # Drop self-detections: prey protein groups that contain the bait itself (e.g. "ARF1;ARF3" in the
    # ARF3 pulldown). With a first-member rule these would otherwise become false bait-paralog edges.
    in_group = [isinstance(t, str) and t in g.split(";")
                for t, g in zip(ppi.target_hgnc_id, ppi.interactor_group_hgnc)]
    n_self = int(np.sum(in_group))
    ppi = ppi[~np.array(in_group)].copy()
    ppi["interactor_is_group"] = ppi.interactor_group_hgnc.str.contains(";")

    imaged = set(nodes.hgnc_id)
    nodes["is_bait"] = nodes.hgnc_id.isin(baits)
    nodes["n_preys"] = nodes.hgnc_id.map(ppi.groupby("target_hgnc_id").interactor_hgnc_id.nunique()).fillna(0).astype(int)
    ppi["target_imaged"] = ppi.target_hgnc_id.isin(imaged)
    ppi["interactor_imaged"] = ppi.interactor_hgnc_id.isin(imaged)

    pairs = build_pairs(nodes, ppi)
    deg = pd.concat([pairs.loc[pairs.is_edge, "a"], pairs.loc[pairs.is_edge, "b"]]).value_counts()
    nodes["degree_bait_bait"] = nodes.hgnc_id.map(deg).fillna(0).astype(int)
    nodes.loc[~nodes.is_bait, "degree_bait_bait"] = np.nan

    first = ["hgnc_id", "hgnc_symbol", "opencell_name", "is_bait", "n_preys", "degree_bait_bait"] + \
            [c for c in ["community_id", "core_id"] if c in nodes] + [
             "n_crops", "n_fovs_cytoself", "n_fovs_s3", "loc_grade_3", "loc_grade_2", "loc_grade_1"]
    nodes = nodes[first + [c for c in nodes.columns if c not in first]].sort_values("hgnc_symbol")
    nodes.to_csv(args.out / "nodes.csv", index=False)
    ppi.to_csv(args.out / "interactions_resolved.csv", index=False)
    pairs.to_parquet(args.out / "pairs.parquet", index=False)

    unresolved_baits = sorted(ppi.loc[ppi.target_hgnc_id.isna(), "target_gene_name"].unique())
    not_imaged_baits = sorted(ppi.loc[ppi.target_hgnc_id.notna() & ~ppi.target_imaged, "target_hgnc_symbol"].unique())
    summary = {
        "imaged_proteins": len(nodes),
        "ppi_baits_distinct_names": int(ppi.target_gene_name.nunique()),
        "ppi_baits_resolved": len(baits),
        "ppi_baits_unresolved": unresolved_baits,
        "imaged_baits": int(nodes.is_bait.sum()),
        "baits_not_imaged": not_imaged_baits,
        "imaged_non_baits": int((~nodes.is_bait).sum()),
        "pairs": len(pairs),
        "edges": int(pairs.is_edge.sum()),
        "edges_reciprocal": int(pairs.reciprocal.sum()),
        "edges_via_protein_group_only": int(pairs.edge_via_group_only.sum()),
        "interaction_rows_dropped_self_group": n_self,
        "edge_tiers": pairs.edge_tier.value_counts().to_dict(),
        "core_edges": int(pairs.is_core_edge.sum()),
        "positive_rate": float(pairs.is_edge.mean()),
        "second_neighbor_pairs": int(pairs.second_neighbor.sum()),
        "pairs_share_loc_g23": int(pairs.share_loc_g23.sum()),
        "edges_share_loc_g23": int((pairs.is_edge & pairs.share_loc_g23).sum()),
        **({
            "imaged_baits_in_community": int((nodes.is_bait & nodes.community_id.notna()).sum()),
            "imaged_baits_in_core": int((nodes.is_bait & nodes.core_id.notna()).sum()),
            "communities_among_imaged_baits": int(nodes.loc[nodes.is_bait, "community_id"].nunique()),
            "cores_among_imaged_baits": int(nodes.loc[nodes.is_bait, "core_id"].nunique()),
            "pairs_same_community": int(pairs.same_community.sum()),
            "pairs_same_core": int(pairs.same_core.sum()),
            "edges_same_core": int((pairs.is_edge & pairs.same_core).sum()),
        } if "community_id" in nodes else {}),
        "prey_resolution": ppi.drop_duplicates("interactor_gene_name").interactor_resolved_by.value_counts().to_dict(),
        "interaction_rows_prey_unresolved": int(ppi.interactor_hgnc_id.isna().sum()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
