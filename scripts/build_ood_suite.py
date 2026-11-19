"""Build the out-of-distribution stress suite (evaluation only, split="test").

Families (see src/ood_generator.py): ood_large (24-48 vertices), ood_chain
(3-4 glued Laman modules, circuits spanning the chain), lattice (bond-diluted
triangular lattices, 36-100 sites). ood_large and ood_chain are
feature-matched pairs; lattice is not (check the feature baseline).

Usage:
    python scripts/build_ood_suite.py --pairs 60 --lattices 40 --out data_new/generated/ood_suite.json
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import statistics as st  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_hard_dataset import signature  # noqa: E402
from src.ood_generator import make_chain_pair, make_large_pair, make_lattice_examples  # noqa: E402

MAKERS = {"ood_large": make_large_pair, "ood_chain": make_chain_pair, "lattice": make_lattice_examples}


def _task(args):
    family, idx, seed = args
    return MAKERS[family](random.Random(seed), f"{family}_{idx}") or []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=60, help="matched pairs for ood_large and ood_chain")
    p.add_argument("--lattices", type=int, default=40, help="lattice graphs (up to 2 candidates per label each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", type=Path, default=ROOT / "data_new/generated/ood_suite.json")
    args = p.parse_args()

    want = {"ood_large": args.pairs, "ood_chain": args.pairs, "lattice": args.lattices}
    tasks, i = [], 0
    for fam, n in want.items():
        for _ in range(2 * n):  # oversample: some tasks yield nothing
            tasks.append((fam, i, args.seed * 1_000_003 + i))
            i += 1
    with Pool(args.workers) as pool:
        groups = pool.map(_task, tasks, chunksize=1)

    seen, kept = set(), collections.defaultdict(list)
    for (fam, _, _), group in zip(tasks, groups):
        if not group or len(kept[fam]) >= want[fam]:
            continue
        sigs = [signature(r) for r in group]
        if any(sg in seen for sg in sigs):
            continue
        seen.update(sigs)
        for r in group:
            r["split"] = "test"
        kept[fam].append(group)

    records = [r for fam in MAKERS for g in kept[fam] for r in g]
    meta = {"purpose": "evaluation_only (out-of-distribution rule tests)", "args": {k: str(v) for k, v in vars(args).items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "records": records}) + "\n", encoding="utf-8")
    for fam in MAKERS:
        rs = [r for g in kept[fam] for r in g]
        red = [r for r in rs if r["label"] == 1]
        print(f"{fam:10s} examples {len(rs):4d} (labels {sum(r['label'] == 0 for r in rs)}/{len(red)})  "
              f"nodes {min(r['num_nodes'] for r in rs)}-{max(r['num_nodes'] for r in rs)}  "
              f"mean |E| {st.mean(len(r['edges']) for r in rs):.0f}  "
              f"mean k=|C|-1 {st.mean(len(r['circuit_edges']) - 1 for r in red):.1f}  "
              f"dependent bases {100 * st.mean(r['base_dependent'] for r in rs):.0f}%")
    print(f"saved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
