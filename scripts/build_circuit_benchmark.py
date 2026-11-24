"""Experiment 0: build and gate the composed-circuit benchmark (Family B).

For each number of components k, builds chain/tree twins (same components and
decoys, different composition depth), each a feature-matched (independent,
redundant) pair. Gates from notes/preregistration.md, checked on every record:
  G1 oracle label == intended label == PyRigi pebble-game label
  G2 stored certificate is valid and minimal
  G3 the seven shortcut features are identical within every pair
  G4 (added after the v1 audit) a degree-profile classifier, with no rigidity
     reasoning, is at chance (<= 0.55 cross-validated); v2 additionally
     requires identical degree profiles within every pair
Also reports H1b: fraction of pairs whose members have different 1-WL
signatures (candidate marked). Nothing is saved if a gate fails.

Usage: python scripts/build_circuit_benchmark.py --twins-per-k 30 --out data_new/generated/circuit_bench.json
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_hard_dataset import signature  # noqa: E402
from scripts.wl_diagnostic import wl_signature  # noqa: E402
import torch  # noqa: E402

from scripts.probe_by_depth import cv_probe  # noqa: E402
from src.circuit_generator import degree_profile, make_twin_pairs  # noqa: E402
from src.laman_diagnostics import candidate_features  # noqa: E402
from src.oracle import is_redundant  # noqa: E402
from src.rigidity_labels import make_graph  # noqa: E402


def _task(args):
    k, idx, seed, version = args
    return make_twin_pairs(random.Random(seed), f"k{k}_{idx}", k, version)


def pyrigi_redundant(n, edges, cand):
    def rank(es):
        basis = []
        for e in es:
            if make_graph(n, basis + [e]).is_Rd_independent(dim=2):
                basis.append(e)
        return len(basis)
    return rank(edges + [cand]) == rank(edges)


def gate_record(r):
    n, cand = r["num_nodes"], tuple(r["candidate_edge"])
    edges = [tuple(e) for e in r["edges"]]
    g1 = r["label"] == r["intended_label"] == int(pyrigi_redundant(n, edges, cand))
    g2 = True
    if r["label"] == 1:
        body = [tuple(e) for e in r["circuit_edges"] if tuple(e) != cand]
        g2 = set(body) <= set(edges) and is_redundant(n, body, cand) and all(
            not is_redundant(n, [e for e in body if e != d], cand) for d in body)
    return g1, g2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6, 8, 10, 12])
    p.add_argument("--twins-per-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--version", type=int, default=2, choices=[1, 2],
                   help="1: Laman decoys, 7-feature matching (confounded: degree-profile baseline 0.69); "
                        "2: circuit-component decoys + full degree-profile matching")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", type=Path, default=ROOT / "data_new/generated/circuit_bench.json")
    args = p.parse_args()

    tasks, i = [], 0
    for k in args.ks:
        for _ in range(2 * args.twins_per_k):  # oversample: some tasks find no matched pair
            tasks.append((k, i, args.seed * 1_000_003 + i, args.version))
            i += 1
    with Pool(args.workers) as pool:
        groups = pool.map(_task, tasks, chunksize=2)

    seen, kept, per_k = set(), [], collections.Counter()
    for (k, _, _, _), grp in zip(tasks, groups):
        if not grp or per_k[k] >= args.twins_per_k:
            continue
        sigs = [signature(r) for r in grp]
        if any(s in seen for s in sigs):
            continue
        seen.update(sigs)
        per_k[k] += 1
        kept += grp

    with Pool(args.workers) as pool:
        gates = pool.map(gate_record, kept, chunksize=4)
    g1 = sum(a for a, _ in gates)
    g2 = sum(b for _, b in gates)
    pairs = collections.defaultdict(list)
    for r in kept:
        pairs[r["pair_id"]].append(r)
    g3 = wl_diff = 0
    for rs in pairs.values():
        f = [candidate_features(r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"])) for r in rs]
        g3 += f[0] == f[1]
        w = [wl_signature(r["num_nodes"], r["edges"], r["candidate_edge"]) for r in rs]
        wl_diff += w[0] != w[1]
        for r in rs:
            r["wl_signature_differs_from_partner"] = w[0] != w[1]

    # G4: a rigidity-free degree-profile classifier must be at chance (v2 matches profiles exactly)
    prof = torch.tensor([degree_profile(r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"]))
                         for r in kept], dtype=torch.float)
    labels = torch.tensor([float(r["label"]) for r in kept])
    g4_acc = cv_probe(prof, labels, [r["pair_id"] for r in kept])
    g4_same = sum(degree_profile(rs[0]["num_nodes"], [tuple(e) for e in rs[0]["edges"]], tuple(rs[0]["candidate_edge"]))
                  == degree_profile(rs[1]["num_nodes"], [tuple(e) for e in rs[1]["edges"]], tuple(rs[1]["candidate_edge"]))
                  for rs in pairs.values())
    print(f"records {len(kept)}  pairs {len(pairs)}  twins per k {dict(per_k)}")
    print(f"G1 labels (oracle = intended = PyRigi): {g1}/{len(kept)}")
    print(f"G2 certificates valid and minimal:      {g2}/{len(kept)}")
    print(f"G3 shortcut features identical in pair: {g3}/{len(pairs)}")
    print(f"G4 degree profiles identical in pair:   {g4_same}/{len(pairs)}; degree-profile classifier (CV) {g4_acc:.3f}")
    print(f"H1b pairs with different 1-WL signatures: {wl_diff}/{len(pairs)} ({100 * wl_diff / len(pairs):.1f}%)")
    depth = collections.defaultdict(list)
    for r in kept:
        depth[(r["family"], r["n_components"])].append(r["composition_depth"])
    print("depth by (shape, components):", {f"{s}/{k}": sorted(set(v)) for (s, k), v in sorted(depth.items())})
    g4_ok = g4_acc <= 0.55 and (args.version == 1 or g4_same == len(pairs))
    if g1 != len(kept) or g2 != len(kept) or g3 != len(pairs) or not g4_ok:
        sys.exit("GATE FAILED: benchmark not saved")
    meta = {"purpose": "Experiment 0 composed-circuit benchmark (evaluation only)",
            "gates": {"G1": f"{g1}/{len(kept)}", "G2": f"{g2}/{len(kept)}", "G3": f"{g3}/{len(pairs)}",
                      "G4_profiles_identical": f"{g4_same}/{len(pairs)}", "G4_degree_profile_cv_acc": round(g4_acc, 3)},
            "version": args.version,
            "H1b_wl_distinguishable_pairs": f"{wl_diff}/{len(pairs)}",
            "args": {k: str(v) for k, v in vars(args).items()}}
    args.out.write_text(json.dumps({"meta": meta, "records": kept}) + "\n", encoding="utf-8")
    print(f"gates passed; saved {len(kept)} records to {args.out}")


if __name__ == "__main__":
    main()
