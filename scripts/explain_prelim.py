"""Preliminary explainer run (CPU): GNNExplainer, PGExplainer, random baseline.

For each test example (balanced subset) and each method, compare three variants:
  raw         plain top-k on raw edge scores
  orbit_avg   top-k on orbit-averaged scores
  orbit_sel   whole-orbit selection on raw scores (may exceed k)
with k = |circuit| - 1 for redundant examples (circuit minus candidate) and
k = --k-indep for independent examples.

Metrics:
  circuit_prec   |selected ∩ circuit| / |selected|           (redundant only)
  circuit_auroc  AUROC of scores vs circuit membership       (redundant only)
  fid_plus       deletion curve: mean_j [p(pred) - p(pred | top-j removed)], j=1..k
                 (removing *all* k saturates: any one circuit edge breaks redundancy)
  fid_minus      p(pred) - p(pred | only selected edges kept) (lower = more sufficient)
  orbit_split    selection contains part (not all) of some orbit
  overshoot      |selected| - k

Usage:
  python scripts/explain_prelim.py --run results/gin_edge_medium --seed-ckpt 0 --n-per-class 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, directed_to_undirected, record_to_data  # noqa: E402
from src.models import EdgeGIN  # noqa: E402
from src.orbits import (candidate_stabilizer_orbits, orbit_average, orbit_select,  # noqa: E402
                        splits_an_orbit, topk)

from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer  # noqa: E402

MODEL_CONFIG = dict(mode="multiclass_classification", task_level="graph", return_type="raw")
MODEL: EdgeGIN | None = None
GNNEXP_EPOCHS = 200


def load_model(run_dir: Path, seed: int) -> EdgeGIN:
    ckpt = torch.load(run_dir / f"seed{seed}" / "model.pt", weights_only=False)
    cfg = ckpt["config"]
    model = EdgeGIN(NUM_NODE_FEATURES, cfg["hidden"], cfg["layers"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def prob_of(model, d, pred, keep_undirected: np.ndarray | None = None) -> float:
    ei = d.edge_index
    if keep_undirected is not None:
        keep = torch.from_numpy(np.repeat(keep_undirected, 2))
        ei = ei[:, keep]
    out = model(d.x, ei, None, d.candidate)
    return torch.softmax(out, -1)[0, pred].item()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def gnnexplainer_scores(model, d, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    explainer = Explainer(model, GNNExplainer(epochs=GNNEXP_EPOCHS), explanation_type="model",
                          edge_mask_type="object", node_mask_type=None, model_config=MODEL_CONFIG)
    exp = explainer(d.x, d.edge_index, batch=None, candidate=d.candidate)
    return directed_to_undirected(exp.edge_mask.detach()).numpy()


def _gnnexp_worker(args):
    rec, seed, perm = args
    torch.set_num_threads(1)
    if perm is None:
        return gnnexplainer_scores(MODEL, record_to_data(rec), seed)
    # Relabel nodes, explain, map scores back to original edge order.
    n = rec["num_nodes"]
    p_edges = sorted(tuple(sorted((perm[u], perm[v]))) for u, v in rec["edges"])
    prec = dict(rec, edges=[list(e) for e in p_edges],
                candidate_edge=sorted(perm[c] for c in rec["candidate_edge"]), circuit_edges=None)
    s_perm = gnnexplainer_scores(MODEL, record_to_data(prec), seed)
    idx = {e: i for i, e in enumerate(p_edges)}
    return np.array([s_perm[idx[tuple(sorted((perm[u], perm[v])))]] for u, v in rec["edges"]])


def train_pgexplainer(model, train_recs, epochs: int, seed: int) -> Explainer:
    torch.manual_seed(seed)
    alg = PGExplainer(epochs=epochs, lr=0.003)
    explainer = Explainer(model, alg, explanation_type="phenomenon", edge_mask_type="object",
                          model_config=MODEL_CONFIG)
    data = [record_to_data(r) for r in train_recs]
    for epoch in range(epochs):
        losses = []
        for d in data:
            losses.append(alg.train(epoch, model, d.x, d.edge_index, target=d.y,
                                    batch=None, candidate=d.candidate))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  PGExplainer epoch {epoch:3d} loss {statistics.mean(losses):.4f}")
    return explainer


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    global MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=ROOT / "results/gin_edge_medium")
    p.add_argument("--data", type=Path, default=ROOT / "data/generated/edge_medium.json")
    p.add_argument("--seed-ckpt", type=int, default=0)
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--k-indep", type=int, default=4)
    p.add_argument("--pg-epochs", type=int, default=30)
    p.add_argument("--pg-train", type=int, default=600)
    p.add_argument("--consistency-n", type=int, default=60)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.set_num_threads(8)
    MODEL = load_model(args.run, args.seed_ckpt)
    records = json.loads(args.data.read_text())["records"]
    test = [r for r in records if r["split"] == "test"]
    sample = []
    for label in (0, 1):
        pool = [r for r in test if r["label"] == label]
        sample += rng.sample(pool, min(args.n_per_class, len(pool)))
    datas = [record_to_data(r) for r in sample]
    print(f"explaining {len(sample)} test examples")

    # ---- explanations ----
    t = time.time()
    with Pool(args.workers) as pool:
        gnn = pool.map(_gnnexp_worker, [(r, args.seed, None) for r in sample])
    print(f"GNNExplainer: {time.time() - t:.1f}s")

    t = time.time()
    train = [r for r in records if r["split"] == "train"]
    pg = train_pgexplainer(MODEL, rng.sample(train, min(args.pg_train, len(train))), args.pg_epochs, args.seed)
    pg_scores = []
    for d in datas:
        exp = pg(d.x, d.edge_index, target=d.y, batch=None, candidate=d.candidate)
        pg_scores.append(directed_to_undirected(exp.edge_mask.detach()).numpy())
    print(f"PGExplainer (train + explain): {time.time() - t:.1f}s")

    nprng = np.random.default_rng(args.seed)
    rand = [nprng.random(len(r["edges"])) for r in sample]
    # Oracle: circuit edges score 1 (redundant only; random for independent).
    oracle = [d.circuit_mask.numpy().astype(float) + 0.01 * nprng.random(len(r["edges"])) if r["label"] == 1
              else nprng.random(len(r["edges"])) for r, d in zip(sample, datas)]
    methods = {"GNNExplainer": gnn, "PGExplainer": pg_scores, "random": rand, "oracle": oracle}

    # ---- orbits ----
    orbits = [candidate_stabilizer_orbits(r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"]))
              for r in sample]
    nontrivial = [any(len(o) > 1 for o in orbs) for orbs in orbits]
    circuit_is_union = []
    for r, d, orbs in zip(sample, datas, orbits):
        if r["label"] == 1:
            cm = d.circuit_mask.numpy()
            circuit_is_union.append(all(cm[o].all() or (~cm[o]).all() for o in orbs))

    # ---- metrics ----
    rows = []
    for mname, all_scores in methods.items():
        for j, (r, d, orbs, s) in enumerate(zip(sample, datas, orbits, all_scores)):
            cm = d.circuit_mask.numpy()
            E = len(s)
            k = int(cm.sum()) if r["label"] == 1 else min(args.k_indep, E)
            pred = MODEL(d.x, d.edge_index, None, d.candidate).argmax(-1).item()
            p0 = prob_of(MODEL, d, pred)
            avg = orbit_average(s, orbs)
            variants = {"raw": (s, topk(s, k)), "orbit_avg": (avg, topk(avg, k)), "orbit_sel": (avg, orbit_select(s, orbs, k))}
            for vname, (scores, sel) in variants.items():
                mask = np.zeros(E, dtype=bool)
                mask[sel] = True
                row = {
                    "method": mname, "variant": vname, "label": r["label"],
                    "symmetric": nontrivial[j],
                    "fid_plus": float(np.mean([p0 - prob_of(MODEL, d, pred, ~np.isin(np.arange(E), sel[:jj]))
                                               for jj in range(1, max(k, 1) + 1)])),
                    "fid_minus": p0 - prob_of(MODEL, d, pred, mask),
                    "orbit_split": splits_an_orbit(sel, orbs),
                    "overshoot": len(sel) - k,
                }
                if r["label"] == 1:
                    row["circuit_prec"] = cm[sel].mean() if len(sel) else 0.0
                    row["circuit_auroc"] = auroc(scores, cm)
                rows.append(row)
    # ---- relabeling consistency (GNNExplainer) ----
    cons_idx = rng.sample(range(len(sample)), min(args.consistency_n, len(sample)))
    jobs_rerun, jobs_perm = [], []
    for j in cons_idx:
        n = sample[j]["num_nodes"]
        perm = list(range(n))
        rng.shuffle(perm)
        jobs_rerun.append((sample[j], args.seed + 1, None))
        jobs_perm.append((sample[j], args.seed, perm))
    with Pool(args.workers) as pool:
        rerun = pool.map(_gnnexp_worker, jobs_rerun)
        permd = pool.map(_gnnexp_worker, jobs_perm)
    cons = {
        "same_graph_new_seed_spearman": float(np.nanmean([spearman(gnn[j], s) for j, s in zip(cons_idx, rerun)])),
        "relabeled_same_seed_spearman": float(np.nanmean([spearman(gnn[j], s) for j, s in zip(cons_idx, permd)])),
        "relabeled_same_seed_max_abs_diff": float(np.mean([np.abs(gnn[j] - s).max() for j, s in zip(cons_idx, permd)])),
    }

    # ---- report ----
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {"n_examples": len(sample),
               "frac_with_nontrivial_orbits": float(np.mean(nontrivial)),
               "circuit_is_union_of_orbits": float(np.mean(circuit_is_union)),
               "gnnexplainer_consistency": cons, "table": {}}
    print(f"\nexamples with a nontrivial candidate-stabilizer orbit: {np.mean(nontrivial):.1%}")
    print(f"redundant examples whose circuit is a union of orbits: {np.mean(circuit_is_union):.1%}")
    print("GNNExplainer consistency:", {k: round(v, 3) for k, v in cons.items()})

    header = f"{'method':13s} {'variant':10s} {'subset':9s} {'circ_prec':>9s} {'circ_auc':>8s} {'fid+':>6s} {'fid-':>6s} {'split%':>6s} {'over':>5s}"
    for subset_name, filt in [("all", lambda r: True), ("symmetric", lambda r: r["symmetric"])]:
        print("\n" + header)
        for mname in methods:
            for vname in ("raw", "orbit_avg", "orbit_sel"):
                rs = [r for r in rows if r["method"] == mname and r["variant"] == vname and filt(r)]
                red = [r for r in rs if r["label"] == 1]
                line = {
                    "n": len(rs),
                    "circuit_prec": mean([r["circuit_prec"] for r in red]),
                    "circuit_auroc": mean([r["circuit_auroc"] for r in red]),
                    "fid_plus": mean([r["fid_plus"] for r in rs]),
                    "fid_minus": mean([r["fid_minus"] for r in rs]),
                    "orbit_split": mean([float(r["orbit_split"]) for r in rs]),
                    "overshoot": mean([r["overshoot"] for r in rs]),
                }
                summary["table"][f"{subset_name}/{mname}/{vname}"] = line
                print(f"{mname:13s} {vname:10s} {subset_name:9s} {line['circuit_prec']:9.3f} {line['circuit_auroc']:8.3f} "
                      f"{line['fid_plus']:6.3f} {line['fid_minus']:6.3f} {100 * line['orbit_split']:6.1f} {line['overshoot']:5.2f}")

    out = args.run / f"seed{args.seed_ckpt}" / "explain_prelim.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
