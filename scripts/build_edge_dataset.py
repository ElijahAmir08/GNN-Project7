"""Build the edge-level (independent vs. redundant proposed edge) dataset.

Usage:
    python scripts/build_edge_dataset.py --bases-per-config 40 --out data/generated/edge_small.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.edge_dataset import _base_task, assign_splits  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    p.add_argument("--deletes", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--bases-per-config", type=int, default=40)
    p.add_argument("--per-class", type=int, default=3, help="max candidates per label per base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, default=ROOT / "data/generated/edge_small.json")
    args = p.parse_args()

    tasks, i = [], 0
    for n in args.nodes:
        for k in args.deletes:
            for _ in range(args.bases_per_config):
                tasks.append((i, n, k, args.seed * 1_000_003 + i, args.per_class))
                i += 1

    with Pool(args.workers) as pool:
        results = pool.map(_base_task, tasks, chunksize=4)

    bases = [(bid, n, edges) for bid, n, edges, _ in results]
    split_of = assign_splits(bases, (0.7, 0.15, 0.15), random.Random(args.seed))
    records = []
    for bid, _, _, exs in results:
        for rec in exs:
            rec["split"] = split_of[bid]
            records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "task": "edge-level generic 2D rigidity: label 0 = independent, 1 = redundant",
        "edge_order": "edges sorted lexicographically as (u, v) with u < v; explanation score i <-> edges[i]",
        "circuit_edges": "fundamental circuit including candidate_edge (label 1 only)",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    args.out.write_text(json.dumps({"meta": meta, "records": records}) + "\n", encoding="utf-8")

    counts = collections.Counter((r["split"], r["label"]) for r in records)
    skipped = sum(1 for _, _, _, exs in results if not exs)
    print(f"bases: {len(bases)} ({skipped} had only one class and were skipped)")
    print(f"examples: {len(records)}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}  independent={counts[(s, 0)]:5d}  redundant={counts[(s, 1)]:5d}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
