"""Aggregate all saved results into markdown tables (results/tables.md).

Reads results/<run>/summary.json, seed*/metrics.json, seed*/explain_prelim.json
and the dataset JSONs. Every number is mean ± std over checkpoint seeds.

Usage: python scripts/make_tables.py
"""

from __future__ import annotations

import collections
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LADDER = [("features", "features_edge_medium", "none (no message passing)"),
          ("gcn", "ladder_gcn_medium", "< 1-WL"),
          ("mpnn", "ladder_mpnn_medium", "≤ 1-WL"),
          ("gin", "ladder_gin_medium", "= 1-WL"),
          ("dsgin", "ladder_dsgin_medium", "> 1-WL")]


def ms(vals, digits=3):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{st.mean(vals):.{digits}f} ± {st.pstdev(vals):.{digits}f}"


def seeds(run):
    return sorted((RES / run).glob("seed*/"))


def dataset_table(out):
    out.append("## 1. Datasets\n")
    out.append("| dataset | examples | base graphs | train / val / test | nodes | edges (mean) | circuit sizes (incl. candidate) | circuit share of base edges | 1-WL ceiling |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for name in ("edge_small", "edge_medium"):
        path = ROOT / "data/generated" / f"{name}.json"
        if not path.exists():
            continue
        R = json.loads(path.read_text())["records"]
        sp = collections.Counter(r["split"] for r in R)
        n = [r["num_nodes"] for r in R]
        E = [len(r["edges"]) for r in R]
        C = collections.Counter(len(r["circuit_edges"]) for r in R if r["label"] == 1)
        share = st.mean((len(r["circuit_edges"]) - 1) / len(r["edges"]) for r in R if r["label"] == 1)
        wl = subprocess.run([sys.executable, str(ROOT / "scripts/wl_diagnostic.py"), "--data", str(path)],
                            capture_output=True, text=True).stdout.strip().splitlines()[-1].split(":")[-1].strip()
        circ = ", ".join(f"{k}:{v}" for k, v in sorted(C.items()))
        out.append(f"| {name} | {len(R)} | {len({r['base_id'] for r in R})} | {sp['train']} / {sp['val']} / {sp['test']} "
                   f"| {min(n)}–{max(n)} | {st.mean(E):.1f} ({min(E)}–{max(E)}) | {circ} | {share:.2f} | {wl} |")
    out.append("\nLabels are balanced 50/50 within every base graph and split; graph size and edge count alone give 50% accuracy.\n")


def classification_table(out):
    out.append("## 2. Classification (independent vs. redundant proposed edge)\n")
    out.append("| model | expressivity | dataset | seeds | train acc | val acc | test acc | test F1 (redundant) | best epoch | sec / seed |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = LADDER + [("features", "features_edge_small", "none"), ("gin", "gin_edge_small", "= 1-WL"),
                     ("gin (first run)", "gin_edge_medium", "= 1-WL")]
    for model, run, expr in rows:
        ms_ = [json.loads((s / "metrics.json").read_text()) for s in seeds(run) if (s / "metrics.json").exists()]
        if not ms_:
            continue
        ds = "small" if "small" in run else "medium"
        out.append(f"| {model} | {expr} | {ds} | {len(ms_)} | {ms([m['train']['acc'] for m in ms_])} "
                   f"| {ms([m['val']['acc'] for m in ms_])} | **{ms([m['test']['acc'] for m in ms_])}** "
                   f"| {ms([m['test']['f1_redundant'] for m in ms_])} | {ms([m['best_epoch'] for m in ms_], 0)} "
                   f"| {ms([m['seconds'] for m in ms_], 0)} |")
    out.append("\nFeature baseline uses 7 hand-made features (n, |E|, 2n−3−|E|, endpoint degrees, common neighbours, endpoint distance). "
               "Single-feature lookup on 'common neighbours' alone reaches 0.84 on edge_small.\n")


def load_explain(run):
    return [json.loads((s / "explain_prelim.json").read_text()) for s in seeds(run) if (s / "explain_prelim.json").exists()]


def explainer_tables(out):
    graph_models = [x for x in LADDER if x[0] != "features"]
    graph_models.insert(3, ("gin (run A)", "gin_edge_medium", "= 1-WL"))
    graph_models = [(m if m != "gin" else "gin (run B)", r, e) for m, r, e in graph_models]
    have = {m: load_explain(run) for m, run, _ in graph_models}
    have = {m: v for m, v in have.items() if v}
    if not have:
        return

    out.append("## 3. Explanation quality per model (all 300 test examples; circuit metrics on the 150 redundant ones)\n")
    out.append("Logits are temperature-scaled per checkpoint to a median validation margin of 4 before explaining (T ≈ 2–3 for all models). "
               "Caveat: soft-mask explainers start with every edge weighted ≈0.5. At that point ~50% of predictions flip for every model, "
               "and the GCN's median logit margin jumps from 9.5 to ~249 (its explainer gradient vanishes), so GCN explanations stay near chance. "
               "gin run A and run B are two identical-config training runs (3 seeds each).\n")
    out.append("k = |circuit| − 1 for redundant examples, k = 4 for independent. circuit precision = fraction of selected edges in the circuit; "
               "AUROC = ranking of circuit vs. non-circuit edges; fid− = drop in predicted-class probability when keeping only the selected edges (lower = more sufficient); "
               "fid+ = mean drop when deleting top-1..k edges (higher = more necessary).\n")
    out.append("| model | temperature | explainer | circuit AUROC ↑ | per-seed AUROC | circuit precision ↑ | fid− ↓ | fid+ ↑ |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m, exps in have.items():
        temp = ms([e.get("temperature", 1.0) for e in exps], 1)
        for ex in ("oracle", "PGExplainer", "GNNExplainer", "random"):
            key = f"all/{ex}/raw"
            t = [e["table"][key] for e in exps]
            per_seed = ", ".join(f"{x['circuit_auroc']:.2f}" for x in t)
            out.append(f"| {m} | {temp} | {ex} | {ms([x['circuit_auroc'] for x in t])} | {per_seed} | {ms([x['circuit_prec'] for x in t])} "
                       f"| {ms([x['fid_minus'] for x in t])} | {ms([x['fid_plus'] for x in t])} |")
    out.append("")

    out.append("## 4. Symmetry corrections: raw vs. orbit averaging vs. whole-orbit selection\n")
    out.append("Restricted to test examples with a nontrivial candidate-stabilizer orbit. orbit split = % of selections that include part (not all) of an orbit; "
               "overshoot = extra edges whole-orbit selection adds beyond k.\n")
    out.append("| model | explainer | variant | circuit AUROC | circuit precision | fid− | fid+ | orbit split % ↓ | overshoot |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for m, exps in have.items():
        for ex in ("PGExplainer", "GNNExplainer", "random"):
            for var in ("raw", "orbit_avg", "orbit_sel"):
                t = [e["table"][f"symmetric/{ex}/{var}"] for e in exps]
                out.append(f"| {m} | {ex} | {var} | {ms([x['circuit_auroc'] for x in t])} | {ms([x['circuit_prec'] for x in t])} "
                           f"| {ms([x['fid_minus'] for x in t])} | {ms([x['fid_plus'] for x in t])} "
                           f"| {ms([100 * x['orbit_split'] for x in t], 1)} | {ms([x['overshoot'] for x in t], 2)} |")
    out.append("")

    out.append("## 5. Symmetry structure and GNNExplainer stability\n")
    out.append("| model | examples with nontrivial orbit | circuit = union of orbits | GNNExpl. Spearman: same graph, new seed | GNNExpl. Spearman: relabeled graph | mean max |Δscore| after relabeling |")
    out.append("|---|---|---|---|---|---|")
    for m, exps in have.items():
        c = [e["gnnexplainer_consistency"] for e in exps]
        out.append(f"| {m} | {ms([100 * e['frac_with_nontrivial_orbits'] for e in exps], 1)}% "
                   f"| {ms([100 * e['circuit_is_union_of_orbits'] for e in exps], 1)}% "
                   f"| {ms([x['same_graph_new_seed_spearman'] for x in c])} | {ms([x['relabeled_same_seed_spearman'] for x in c])} "
                   f"| {ms([x['relabeled_same_seed_max_abs_diff'] for x in c])} |")
    out.append("")


def main():
    out = ["# Results: symmetry-consistent GNN explanations for 2D rigidity (preliminary)\n",
           "All runs on CPU. Numbers are mean ± std over 3 checkpoint seeds unless only one exists. "
           "Regenerate with `python scripts/make_tables.py`.\n"]
    dataset_table(out)
    classification_table(out)
    explainer_tables(out)
    text = "\n".join(out) + "\n"
    (RES / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
