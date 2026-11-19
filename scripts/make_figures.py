"""Summary figures for the README (figures/*.png), built from results/.

Every number is read from files written by the evaluation scripts:
  results/eval_*.json                 accuracy        (scripts/eval_checkpoints.py)
  results/certificate_acceptance*.json                (scripts/certificate_acceptance.py)
  results/oracle_*.json               explanations    (scripts/oracle_eval.py)
  results/rule_tests.md, rules_*.md   rule tests      (scripts/rule_tests.py)
  results/error_analysis.md           reach and size  (scripts/error_analysis.py)
The WL-family figure recomputes model outputs from the saved checkpoints.

Colors: categorical slots 1-3 of the reference palette (validated all-pairs in
light mode), neutral gray for baselines, one-hue blue ramp for the heatmap.

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import model_from_config  # noqa: E402
from src.wl_family import hinged, pair_records, wheel  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "figures"
SURFACE, INK, INK2, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})


def mean(v):
    v = [x for x in v if x == x]
    return st.mean(v) if v else float("nan")


def load_json(*names):
    out = {}
    for n in names:
        p = RES / n
        if p.exists():
            out.update(json.loads(p.read_text()))
    return out


def md_rows(path: Path) -> dict[str, list[float]]:
    """Parse '| run | 1.2 ± 0.3 | ... |' rows of a markdown table into run -> means."""
    rows = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0] or cells[0] in ("run", "model") or set(cells[0]) <= set("-:"):
            continue
        vals = []
        for c in cells[1:]:
            m = re.match(r"\*{0,2}(-?\d+(?:\.\d+)?)", c)
            vals.append(float(m.group(1)) if m else float("nan"))
        rows.setdefault(cells[0], vals)
    return rows


def finish(fig, name, note):
    fig.text(0.01, -0.06, note, fontsize=7.5, color=INK2, ha="left", va="top", wrap=True)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def bars(ax, groups, series, values, colors, ylabel, ylim=(0, 1.05), baseline=None):
    """Grouped bars: values[s][g]; 2px surface gap between neighbouring bars."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (s, c) in enumerate(zip(series, colors)):
        ax.bar(x + (i - (n - 1) / 2) * width, values[i], width=width, color=c, label=s,
               edgecolor=SURFACE, linewidth=1.2)
    if baseline is not None:
        for j, b in enumerate(baseline):
            if b == b:
                ax.hlines(b, x[j] - 0.42, x[j] + 0.42, color=BASE, linewidth=2, linestyles=(0, (2, 1.5)),
                          label="random" if j == 0 else None)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


# --------------------------------------------------------------------- 1
def fig_accuracy():
    E = load_json("eval_features_edge_medium.json", "eval_mixed_features.json", "eval_maskaug_features.json",
                  "eval_ood.json", "eval_truss.json", "eval_maskaug_mpnn_pair.json", "eval_pair_quick.json")
    sets = [("edge_medium", "medium\ntest"), ("hard_eval:decoy", "decoy"), ("hard_eval:multi", "multi"),
            ("hard_eval:symmetric", "symmetric"), ("chain_matched_eval", "chains"),
            ("ood_suite:ood_chain", "ood\nchain"), ("ood_suite:ood_large", "ood\nlarge"),
            ("ood_suite:lattice", "lattice"), ("truss_eval", "truss")]
    rows, group_ends = [], []
    for m in ("gcn", "mpnn", "gin", "dsgin"):
        for reg, run in (("original", f"ladder_{m}_medium"), ("mixed", f"mixed_{m}"),
                         ("masked", f"maskaug_{m}"), ("masked + pair", f"maskaug_{m}_pair")):
            vals = [mean([x["acc"] for x in E.get(f"{run}/{s}", [])]) for s, _ in sets]
            if not all(v != v for v in vals):
                rows.append((f"{m.upper().replace('DSGIN', 'DS-GIN')} · {reg}", vals))
        group_ends.append(len(rows) - 1)
    M = np.array([r[1] for r in rows])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.6))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(sets)), [s[1] for s in sets])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.8 else INK)
    for y in group_ends[:-1]:
        ax.axhline(y + 0.5, color=SURFACE, linewidth=4)
    ax.axvline(3.5, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(3.55, -0.75, "never trained on →", fontsize=8, color=INK2, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("test accuracy (0.5 = chance)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Accuracy: mixed and masked training fix familiar graphs, not large graphs or lattices", pad=18)
    finish(fig, "fig1_accuracy.png", "Mean over seeds (original: 3, others: 5). Hard families (decoy, multi, "
           "symmetric) share a generator with the mixed training data. Source: results/eval_*.json")


# --------------------------------------------------------------------- 2
def fig_certificates():
    C = load_json("certificate_acceptance.json", "certificate_acceptance_pair.json")
    groups = [("hard_eval:decoy", "decoy"), ("hard_eval:multi", "multi"), ("hard_eval:symmetric", "sym-\nmetric"),
              ("chain_matched_eval", "chains"), ("ood_suite:ood_large", "ood\nlarge"),
              ("ood_suite:ood_chain", "ood\nchain"), ("ood_suite:lattice", "lattice"), ("truss_eval:truss", "truss")]
    regimes = [("mixed", "mixed_{m}"), ("random masks only", "randmask_{m}"), ("masked", "maskaug_{m}")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, m, title in zip(axes, ("gin", "dsgin"), ("GIN", "DS-GIN")):
        vals = [[mean(C.get(f"{pat.format(m=m)}|{g}", [])) for g, _ in groups] for _, pat in regimes]
        bars(ax, [g[1] for g in groups], [r[0] for r in regimes], vals, [S1, S2, S3],
             "% accepting the true certificate", ylim=(0, 105))
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8.5)
    axes[0].legend(loc="lower left", ncols=3, fontsize=8.5, bbox_to_anchor=(0, 1.08))
    fig.suptitle("Accurate is not the same as relying on the proof: masked training fixes it", x=0.01, ha="left",
                 fontweight="bold", y=1.08)
    finish(fig, "fig2_certificate_acceptance.png", "Truly redundant examples the model predicts redundant: % still "
           "predicted redundant when shown only a minimal certificate. Source: results/certificate_acceptance.json")


# --------------------------------------------------------------------- 3
def fig_explainers():
    oracle = load_json("oracle_maskaug_features.json")
    groups = [("_chain_matched_eval", "chains"), ("_hard_eval:decoy", "decoy"),
              ("_hard_eval:multi", "multi"), ("_hard_eval:symmetric", "symmetric")]
    methods = [("EdgeGrad", "EdgeGrad"), ("PGExplainer", "PGExplainer"), ("GNNExplainer", "GNNExplainer")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, m, title in zip(axes, ("gin", "mpnn"), ("GIN (masked training)", "MPNN (masked training)")):
        vals = [[mean([x["slack"] for x in oracle.get(f"{d}|maskaug_{m}|{meth}|raw", [])]) for d, _ in groups]
                for meth, _ in methods]
        rand = [mean([x["slack"] for x in oracle.get(f"{d}|maskaug_{m}|random|raw", [])]) for d, _ in groups]
        bars(ax, [g[1] for g in groups], [x[1] for x in methods], vals, [S1, S2, S3],
             "slack score", baseline=rand)
        ax.set_title(title)
    axes[0].legend(loc="lower left", ncols=4, fontsize=8.5, bbox_to_anchor=(0, 1.08))
    fig.suptitle("A gradient on the unmasked graph finds certificates; soft-mask explainers do not", x=0.01,
                 ha="left", fontweight="bold", y=1.1)
    finish(fig, "fig3_explainers.png", "Slack score: 1 = the ranking reaches a minimal valid certificate first, "
           "0 = it needs every bar. Truly redundant examples. Source: results/oracle_maskaug_features.json")


# --------------------------------------------------------------------- 4
def fig_rules():
    R = {}
    for name in ("rule_tests.md", "rules_randmask_gcn.md", "rules_maskaug_mpnn_pair.md"):
        if (RES / name).exists():
            R.update(md_rows(RES / name))
    laws = ["add\n2 bars", "add\n6 bars", "remove\n2 bars", "remove\n6 bars", "attach\n8 joints", "attach\n20 joints"]
    runs = [("MPNN", "maskaug_mpnn"), ("GIN", "maskaug_gin"), ("DS-GIN", "maskaug_dsgin")]
    vals = [R[r][:6] for _, r in runs]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bars(ax, laws, [r[0] for r in runs], vals, [S1, S2, S3], "violations (%)", ylim=(0, 30))
    ax.axvspan(3.5, 5.5, color=GRID, alpha=0.45, zorder=0, linewidth=0)
    ax.text(0.0, 27, "monotonicity: learned", fontsize=9, color=INK2)
    ax.text(3.6, 27, "1-sum law: not learned", fontsize=9, color=INK2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88), fontsize=8.5)
    ax.set_title("Rule tests: laws the true rule always obeys (masked training)")
    finish(fig, "fig4_rule_tests.png", "Every perturbed graph is verified by the exact oracle; 0% = the model "
           "always obeys the law. Source: results/rule_tests.md")


# --------------------------------------------------------------------- 5
def fig_reach_size():
    text = (RES / "error_analysis.md").read_text().split("## Accuracy by ")
    reach = md_rows_from(text, "reach (redundant + matched independent)")
    size = md_rows_from(text, "graph size")
    runs = [("MPNN (sum readout)", "maskaug_mpnn", S1), ("GIN (sum readout)", "maskaug_gin", S2),
            ("MPNN (pair readout)", "maskaug_mpnn_pair", S3)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for label, run, c in runs:
        axes[0].plot([0, 1, 2], reach[run][:3], marker="o", markersize=6, linewidth=2, color=c, label=label)
        axes[1].plot([0, 1, 2, 3], size[run][:4], marker="o", markersize=6, linewidth=2, color=c, label=label)
    axes[0].set_xticks([0, 1, 2], ["≤2 hops", "3–4 hops", "5–6 hops"])
    axes[0].axvspan(1.5, 2.3, color=GRID, alpha=0.45, linewidth=0, zorder=0)
    axes[0].text(1.55, 0.53, "beyond 4 rounds", fontsize=8.5, color=INK2)
    axes[0].set_title("By how far the certificate reaches")
    axes[0].set_ylabel("accuracy")
    axes[1].set_xticks([0, 1, 2, 3], ["≤20", "21–40", "41–70", "71+"])
    axes[1].set_xlabel("joints")
    axes[1].set_title("By graph size")
    for ax in axes:
        ax.set_ylim(0.5, 1.02)
        ax.axhline(0.5, color=BASE, linewidth=1)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Accuracy already falls inside the receptive field; the whole-graph sum hurts large graphs",
                 x=0.01, ha="left", fontweight="bold", y=1.03)
    finish(fig, "fig5_reach_and_size.png", "Masked training, mean over 5 seeds. Reach: matched pairs on the stress "
           "sets. Size: all stress sets. Source: results/error_analysis.md")


def md_rows_from(sections, title_prefix):
    for sec in sections:
        if sec.startswith(title_prefix):
            tmp = RES / ".tmp_section.md"
            tmp.write_text(sec)
            rows = md_rows(tmp)
            tmp.unlink()
            return rows
    raise KeyError(title_prefix)


# --------------------------------------------------------------------- 6
def fig_wl_family():
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.6], wspace=0.55)
    for col, (fn, title) in enumerate(((wheel, "Wheel (rigid)\nproposed bar is REDUNDANT"),
                                       (hinged, "Hinged at the hub\nproposed bar is INDEPENDENT"))):
        ax = fig.add_subplot(gs[0, col])
        n, edges, cand = fn(4)
        g = nx.Graph(edges)
        hub = n - 1
        if fn is wheel:
            pos = {i: (np.cos(2 * np.pi * i / 8), np.sin(2 * np.pi * i / 8)) for i in range(8)}
        else:
            pos = {i: (-1.1 + 0.6 * np.cos(2 * np.pi * i / 4), 0.6 * np.sin(2 * np.pi * i / 4)) for i in range(4)}
            pos.update({4 + i: (1.1 + 0.6 * np.cos(np.pi + 2 * np.pi * i / 4), 0.6 * np.sin(np.pi + 2 * np.pi * i / 4))
                        for i in range(4)})
        pos[hub] = (0, 0)
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=INK2, width=1.4)
        nx.draw_networkx_edges(nx.Graph([cand]), pos, ax=ax, edge_color=S2, width=2.2, style="dashed")
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=SURFACE, edgecolors=INK, node_size=110, linewidths=1.4)
        nx.draw_networkx_nodes(g, pos, nodelist=[hub], ax=ax, node_color=INK, node_size=140)
        nx.draw_networkx_nodes(g, pos, nodelist=list(cand), ax=ax, node_color=S2, node_size=130)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        ax.set_aspect("equal")
    ax = fig.add_subplot(gs[0, 2])
    recs = [r for k in range(3, 16) for r in pair_records(k)]
    datas = [record_to_data(r) for r in recs]
    ks = list(range(3, 16))
    rows = [("GCN, MPNN, GIN", ["mixed_gcn", "mixed_mpnn", "mixed_gin", "maskaug_gin"]),
            ("DS-GIN", ["mixed_dsgin", "maskaug_dsgin"])]
    for yi, (_, runs) in enumerate(rows):
        distinct = {k: [] for k in ks}
        for run in runs:
            for ck in sorted((RES / run).glob("seed*/model.pt")):
                c = torch.load(ck, weights_only=False)
                model = model_from_config(c["config"], NUM_NODE_FEATURES)
                model.load_state_dict(c["state_dict"])
                model.eval()
                with torch.no_grad():
                    out = [model(d.x, d.edge_index, None, d.candidate)[0] for d in datas]
                for i in range(0, len(recs), 2):
                    distinct[recs[i]["k"]].append(not torch.allclose(out[i], out[i + 1], atol=1e-4))
        for k in ks:
            sep = all(distinct[k])
            ax.scatter(k, 1 - yi, s=90, marker="o", color=S1 if sep else SURFACE, edgecolors=S1 if sep else INK2,
                       linewidths=1.5, zorder=3)
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.set_xticks(ks)
    ax.set_xlabel("k (graph has 2k + 1 joints)")
    ax.set_title("Can the model tell the pair apart?", fontsize=10)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=70, color=S1, label="outputs differ (can separate)")
    ax.scatter([], [], s=70, color=SURFACE, edgecolors=INK2, linewidths=1.5, label="identical outputs (provably blind)")
    ax.legend(loc="upper center", ncols=2, fontsize=8.5)
    fig.suptitle("A provable limitation: 1-WL GNNs cannot tell these apart at any size; DS-GIN only up to k = 6",
                 x=0.01, ha="left", fontweight="bold", y=1.02)
    finish(fig, "fig6_wl_family.png", "Left: k = 4 (orange = proposed bar, black = hub). Right: whether each model's "
           "two outputs differ on the pair, for every seed. Every model predicts 'redundant' for both. "
           "Source: src/wl_family.py + checkpoints")


if __name__ == "__main__":
    fig_accuracy()
    fig_certificates()
    fig_explainers()
    fig_rules()
    fig_reach_size()
    fig_wl_family()
