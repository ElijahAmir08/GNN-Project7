"""Build the harder evaluation set (decoy / multi / symmetric families).

Usage:
    python scripts/build_hard_dataset.py --per-class 60 --out data_new/generated/hard_eval.json

All records get split="test" (evaluation-only). Every family is built
as feature-matched (independent, redundant) pairs on the same graph, so labels
are balanced and the seven shortcut features carry no signal; duplicate
examples (isomorphic graph with the candidate marked) are dropped.
"""

from __future__ import annotations

import os

# One BLAS thread per worker: many worker processes each spawning one thread
# per core oversubscribes the machine and makes the run far slower.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import json
import random
import statistics as st
import sys
from multiprocessing import Pool
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hard_generator import MAKERS  # noqa: E402
from src.orbits import candidate_stabilizer_orbits  # noqa: E402

FAMILIES = list(MAKERS)


def _task(args) -> list[dict]:
    """One task -> a feature-matched (independent, redundant) pair, kept or
    dropped together. Every family is built as matched pairs, so no
    classifier on the seven shortcut features can beat 50%."""
    family, idx, seed, _ = args
    rng = random.Random(seed)
    group = MAKERS[family](rng, f"{family}_{idx}") or []
    for rec in group:
        edges = [tuple(e) for e in rec["edges"]]
        orbits = candidate_stabilizer_orbits(rec["num_nodes"], edges, tuple(rec["candidate_edge"]))
        rec["has_nontrivial_orbit"] = any(len(o) > 1 for o in orbits)
    return group


def signature(rec) -> str:
    g = nx.Graph()
    g.add_nodes_from(range(rec["num_nodes"]))
    g.add_edges_from(map(tuple, rec["edges"]))
    for w in g.nodes:
        g.nodes[w]["c"] = str(int(w in rec["candidate_edge"]))
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="c")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=60, help="examples per label per family")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", type=Path, default=ROOT / "data_new/generated/hard_eval.json")
    args = p.parse_args()

    tasks, i = [], 0
    for family in FAMILIES:
        for want in (0, 1):  # each task yields one 0/1 pair; `want` only spreads seeds
            for _ in range(3 * args.per_class):  # oversample: some tasks find no matched pair
                tasks.append((family, i, args.seed * 1_000_003 + i, want))
                i += 1
    with Pool(args.workers) as pool:
        groups = pool.map(_task, tasks, chunksize=2)

    seen, kept = set(), collections.defaultdict(list)
    for group in groups:
        if not group:
            continue
        sigs = [signature(r) for r in group]
        keys = [(r["family"], r["label"]) for r in group]
        if any(sg in seen for sg in sigs) or len(set(sigs)) < len(sigs) \
                or any(len(kept[k]) >= args.per_class for k in keys):
            continue
        for rec, sg, k in zip(group, sigs, keys):
            seen.add(sg)
            rec["split"] = "test"
            kept[k].append(rec)

    records = [r for fam in FAMILIES for lab in (0, 1) for r in kept[(fam, lab)]]
    meta = {"purpose": "evaluation_only", "families": FAMILIES,
            "labels": "exact oracle (src/oracle.py)", "args": {k: str(v) for k, v in vars(args).items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "records": records}) + "\n", encoding="utf-8")

    print(f"{'family':10s} {'label':>5s} {'n':>4s} {'nodes':>9s} {'edges':>9s} {'k=|C|-1':>8s} "
          f"{'circ share':>10s} {'#certs':>7s} {'dep base':>8s} {'sym':>5s}")
    for fam in FAMILIES:
        for lab in (0, 1):
            rs = kept[(fam, lab)]
            if not rs:
                print(f"{fam:10s} {lab:5d}    0")
                continue
            nodes = [r["num_nodes"] for r in rs]
            E = [len(r["edges"]) for r in rs]
            red = [r for r in rs if r["label"] == 1]
            k = [len(r["circuit_edges"]) - 1 for r in red]
            share = [(len(r["circuit_edges"]) - 1) / len(r["edges"]) for r in red]
            nc = [len(r["certificates"]) for r in red]
            print(f"{fam:10s} {lab:5d} {len(rs):4d} {min(nodes):3d}–{max(nodes):<3d}   {st.mean(E):5.1f}    "
                  f"{(st.mean(k) if k else 0):6.1f}   {(st.mean(share) if share else 0):8.2f}   "
                  f"{(st.mean(nc) if nc else 0):5.1f}   {100 * st.mean(r['base_dependent'] for r in rs):5.0f}%  "
                  f"{100 * st.mean(r['has_nontrivial_orbit'] for r in rs):3.0f}%")
    print(f"saved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
