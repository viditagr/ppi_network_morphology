"""Static figures of the OpenCell bait-bait interaction network.

    python benchmarks/OpenCell/plot_network.py --data opencell_data/benchmark --out benchmarks/OpenCell/figures

Produces
    network_communities.png/.pdf   nodes coloured by MCL community (12 largest), with complex names
                                   and the best-connected genes of each community labelled
    network_degree.png/.pdf        nodes coloured and sized by number of interaction partners,
                                   with a degree histogram inset and the main hubs labelled

The layout (Kamada-Kawai on the main component, refined by a short spring pass) takes ~30 s and is
cached in <data>/layout.csv; delete that file to recompute it.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Names for the largest communities. These are NOT OpenCell annotations (OpenCell gives only numeric
# IDs); they were assigned by reading each community's member genes, and some simplify mixed
# communities: 2 also holds stress-granule proteins (G3BP2, CAPRIN1), 22 also holds PP2A, 17 also
# holds RAB5A/B and TIA1.
COMMUNITY_NAMES = {
    2: "Ribosome / SRP",
    11: "Proteasome",
    26: "Mediator",
    4: "snRNP / hnRNP",
    17: "OST complex",
    25: "RNA polymerases",
    27: "SF3a/SF3b (U2 snRNP)",
    56: "V-ATPase",
    49: "SWI/SNF (BAF)",
    47: "APC/C",
    13: "Clathrin",
    22: "CCT chaperonin",
}
TAB20 = plt.get_cmap("tab20").colors
COMMUNITY_COLORS = [TAB20[i] for i in (0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 5, 9)]


def load(data):
    nodes = pd.read_csv(data / "nodes.csv")
    pairs = pd.read_parquet(data / "pairs.parquet")
    baits = nodes[nodes.is_bait].set_index("hgnc_id")
    edges = pairs.loc[pairs.is_edge, ["a", "b", "edge_tier"]]
    return baits, edges


def layout(baits, edges, cache):
    if cache.exists():
        pos = pd.read_csv(cache, index_col=0)
        if set(pos.index) == set(baits.index):
            return pos
    G = nx.Graph()
    G.add_nodes_from(baits.index)
    for a, b, tier in edges.itertuples(index=False):
        G.add_edge(a, b, weight=3.0 if tier.startswith("core") else 1.0, dist=0.6 if tier.startswith("core") else 1.0)
    giant = max(nx.connected_components(G), key=len)
    H = G.subgraph(giant)
    p = nx.kamada_kawai_layout(H, weight="dist")
    p = nx.spring_layout(H, pos=p, weight="weight", k=1.6 / np.sqrt(len(giant)), iterations=60, seed=7)
    c0 = np.median(np.array(list(p.values())), axis=0)
    p = {h: c0 + (v - c0) * np.linalg.norm(v - c0) ** -0.35 for h, v in p.items()}  # compress the long tails
    xy = np.array(list(p.values()))
    lo, hi = xy.min(0), xy.max(0)
    p = {h: (v - lo) / (hi - lo) for h, v in p.items()}
    # everything outside the main component goes in rows underneath
    others = sorted((c for c in nx.connected_components(G) if c is not giant and c != giant), key=len, reverse=True)
    x0 = 0.0
    for comp in (c for c in others if len(c) > 1):
        for i, h in enumerate(sorted(comp)):
            p[h] = np.array([x0 + 0.02 * i, -0.08])
        x0 += 0.02 * len(comp) + 0.04
    singles = sorted(next(iter(c)) for c in others if len(c) == 1)
    for i, h in enumerate(singles):
        p[h] = np.array([(i % 36) / 35, -0.14 - 0.035 * (i // 36)])
    pos = pd.DataFrame.from_dict(p, orient="index", columns=["x", "y"])
    pos["main_component"] = pos.index.isin(giant)
    pos.to_csv(cache)
    return pos


def draw_edges(ax, pos, edges):
    seg = lambda e: np.stack([pos.loc[e.a, ["x", "y"]].to_numpy(), pos.loc[e.b, ["x", "y"]].to_numpy()], axis=1)
    core = edges.edge_tier.str.startswith("core")
    lo, hi = edges[~core], edges[core]
    ax.add_collection(LineCollection(seg(lo), colors="0.6", linewidths=0.25, alpha=0.25, zorder=1))
    ax.add_collection(LineCollection(seg(hi), colors="0.35", linewidths=0.45, alpha=0.45, zorder=1))


def label_nodes(ax, pos, genes, fontsize=7):
    """Place labels with a small offset, skipping any that would overlap an earlier one."""
    placed = []
    for g, (x, y) in genes:
        w, h = 0.011 * len(g) * fontsize / 7, 0.016 * fontsize / 7
        tx, ty = x + 0.008, y + 0.008
        if any(abs(tx + w / 2 - px) < (w + pw) / 2 + 0.004 and abs(ty - py) < (h + ph) / 2 for px, py, pw, ph in placed):
            continue
        placed.append((tx + w / 2, ty, w, h))
        ax.annotate(g, (x, y), xytext=(tx, ty), fontsize=fontsize, zorder=5,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3", shrinkA=0, shrinkB=2),
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))


def finish(ax, pos, title):
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(pos.y.min() - 0.04, 1.04)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    ymain = pos.loc[~pos.main_component, "y"]
    if len(ymain):
        ax.text(0, -0.05, "Small fragments", fontsize=7, color="0.35")
        ax.text(0, -0.11, "No partner among imaged baits", fontsize=7, color="0.35")


def plot_communities(baits, edges, pos, out):
    fig, ax = plt.subplots(figsize=(12, 12.6))
    draw_edges(ax, pos, edges)
    b = baits.join(pos)
    size = 6 + 4 * np.sqrt(b.degree_bait_bait)
    top = list(COMMUNITY_NAMES)
    other = b[~b.community_id.isin(top)]
    ax.scatter(other.x, other.y, s=size[other.index], c=np.where(other.community_id.isna(), "0.88", "0.72"),
               edgecolors="white", linewidths=0.3, zorder=2)
    handles = []
    for cid, color in zip(top, COMMUNITY_COLORS):
        m = b[b.community_id == cid]
        ax.scatter(m.x, m.y, s=size[m.index], color=color, edgecolors="white", linewidths=0.3, zorder=3)
        handles.append(Line2D([], [], marker="o", ls="", color=color, label=f"{cid}: {COMMUNITY_NAMES[cid]} ({len(m)})"))
    handles += [Line2D([], [], marker="o", ls="", color="0.72", label=f"Other communities ({other.community_id.notna().sum()})"),
                Line2D([], [], marker="o", ls="", color="0.88", label=f"No MCL community ({b.community_id.isna().sum()})")]
    # label the 3 best-connected genes of each named community
    genes = []
    for cid in top:
        m = b[b.community_id == cid].sort_values("degree_bait_bait", ascending=False).head(3)
        genes += [(r.hgnc_symbol, (r.x, r.y)) for r in m.itertuples()]
    label_nodes(ax, pos, genes)
    ax.legend(handles=handles, title="MCL community ID: name* (imaged baits)", loc="upper left", bbox_to_anchor=(1.0, 1.0),
              fontsize=8, title_fontsize=9, frameon=False)
    ax.text(1.005, 0.705, "*names assigned by reading member genes;\n community membership is OpenCell MCL", transform=ax.transAxes, fontsize=7.5, color="0.3", va="top")
    finish(ax, pos, f"OpenCell bait network coloured by MCL community "
                    f"({len(baits)} baits, {len(edges)} interactions, {b.community_id.nunique()} communities)")
    save(fig, out / "network_communities")


def plot_degree(baits, edges, pos, out):
    bins = [(0, 0, "0"), (1, 2, "1–2"), (3, 5, "3–5"), (6, 10, "6–10"), (11, 20, "11–20"), (21, 10**6, "21+")]
    cmap = plt.get_cmap("viridis", len(bins))
    b = baits.join(pos)
    d = b.degree_bait_bait.astype(int)
    k = np.select([(d >= lo) & (d <= hi) for lo, hi, _ in bins], range(len(bins)))
    size = 6 + 4 * np.sqrt(d)

    fig, ax = plt.subplots(figsize=(12, 12.6))
    draw_edges(ax, pos, edges)
    order = np.argsort(d.to_numpy())
    ax.scatter(b.x.to_numpy()[order], b.y.to_numpy()[order], s=size.to_numpy()[order], c=k[order], cmap=cmap,
               vmin=-0.5, vmax=len(bins) - 0.5, edgecolors="white", linewidths=0.3, zorder=3)
    hubs = b.sort_values("degree_bait_bait", ascending=False).head(20)
    label_nodes(ax, pos, [(f"{r.hgnc_symbol} ({int(r.degree_bait_bait)})", (r.x, r.y)) for r in hubs.itertuples()])

    color_handles = [Line2D([], [], marker="o", ls="", color=cmap(i), markersize=7,
                            label=f"{lab}  (n={(k == i).sum()})") for i, (_, _, lab) in enumerate(bins)]
    leg1 = ax.legend(handles=color_handles, title="Interaction partners", loc="upper left",
                     bbox_to_anchor=(1.0, 1.0), fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    size_handles = [plt.scatter([], [], s=6 + 4 * np.sqrt(v), color="0.5", edgecolors="white", label=str(v))
                    for v in (1, 5, 20, 50)]
    ax.legend(handles=size_handles, title="Node size = partners", loc="upper left", bbox_to_anchor=(1.0, 0.74),
              fontsize=8, title_fontsize=9, frameon=False, labelspacing=1.2)

    ins = fig.add_axes([0.83, 0.40, 0.14, 0.12])
    ins.hist(d, bins=np.arange(0, d.max() + 2) - 0.5, color=cmap(3), edgecolor="none")
    ins.set_yscale("log")
    ins.set_xlabel("partners per bait", fontsize=8)
    ins.set_ylabel("baits", fontsize=8)
    ins.tick_params(labelsize=7)
    ins.set_title(f"median {int(d.median())}, max {d.max()}", fontsize=8)
    finish(ax, pos, f"OpenCell bait network coloured by number of interaction partners "
                    f"({len(baits)} baits, {len(edges)} interactions)")
    save(fig, out / "network_degree")


def save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", stem.with_suffix(".png"), "and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("opencell_data/benchmark"))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/OpenCell/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    baits, edges = load(args.data)
    pos = layout(baits, edges, args.data / "layout.csv")
    plot_communities(baits, edges, pos, args.out)
    plot_degree(baits, edges, pos, args.out)


if __name__ == "__main__":
    main()
