"""Add small wheel/hinge pairs (src/wl_family.py) to a training set.

Train/val use k in --train-k only (default 3..6, i.e. 7-13 joints); every
copy gets a random node relabelling so models cannot rely on vertex order.
Larger k (7-15) stay held out in data_new/generated/wl_family.json.

Usage:
  python scripts/build_family_train.py --data data_new/generated/mixed_masked_train.json \
      --out data_new/generated/mixed_masked_family_train.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wl_family import pair_records  # noqa: E402


def relabel(rec, rng, tag):
    n = rec["num_nodes"]
    perm = list(range(n))
    rng.shuffle(perm)
    edges = sorted(sorted((perm[a], perm[b])) for a, b in rec["edges"])
    u, v = rec["candidate_edge"]
    return dict(rec, graph_id=f"{rec['graph_id']}_{tag}", edges=edges, candidate_edge=sorted([perm[u], perm[v]]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data_new/generated/mixed_masked_train.json")
    p.add_argument("--train-k", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument("--copies", type=int, default=30, help="relabelled copies per pair for train")
    p.add_argument("--val-copies", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "data_new/generated/mixed_masked_family_train.json")
    args = p.parse_args()
    rng = random.Random(args.seed)

    blob = json.loads(args.data.read_text())
    added = []
    for k in args.train_k:
        for split, copies in (("train", args.copies), ("val", args.val_copies)):
            for c in range(copies):
                for rec in pair_records(k):
                    added.append(dict(relabel(rec, rng, f"{split}{c}"), split=split))
    blob["records"] += added
    blob.setdefault("meta", {})["wl_family_training"] = {k: str(v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps(blob) + "\n", encoding="utf-8")
    c = collections.Counter((r["split"], r["label"]) for r in added)
    print(f"added {len(added)} wheel/hinge records for k={args.train_k}: {dict(c)}; saved to {args.out}")


if __name__ == "__main__":
    main()
