"""Build a size- and family-diverse training set (train/val/test splits).

  train/val: edge_medium train/val
           + Henneberg-base examples on 11-16 nodes (own splits)
           + hard families (decoy / multi / symmetric) from a DIFFERENT seed than
             the held-out hard_eval set, split 85/15 by matched pair
  test:      edge_medium test only (comparable with earlier runs)

Hard-family training pairs with any record isomorphic (graph + marked
candidate) to a held-out hard_eval record are dropped as whole pairs. Chains and other stress sets
remain fully unseen.

Usage:
  python scripts/build_mixed_dataset.py --out data_new/generated/mixed_train.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_hard_dataset import signature  # noqa: E402


def run(cmd):
    subprocess.run([sys.executable, "-W", "ignore", *cmd], check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--medium", type=Path, default=ROOT / "data/generated/edge_medium.json")
    p.add_argument("--hard-eval", type=Path, default=ROOT / "data_new/generated/hard_eval.json")
    p.add_argument("--hard-per-class", type=int, default=300)
    p.add_argument("--large-bases-per-config", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", type=Path, default=ROOT / "data_new/generated/mixed_train.json")
    args = p.parse_args()
    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        large_path, hard_path = Path(tmp) / "large.json", Path(tmp) / "hard_train.json"
        run(["scripts/build_edge_dataset.py", "--nodes", "11", "12", "13", "14", "15", "16",
             "--bases-per-config", str(args.large_bases_per_config), "--seed", str(100 + args.seed),
             "--workers", str(args.workers), "--out", str(large_path)])
        run(["scripts/build_hard_dataset.py", "--per-class", str(args.hard_per_class),
             "--seed", str(100 + args.seed), "--workers", str(args.workers), "--out", str(hard_path)])
        large = json.loads(large_path.read_text())["records"]
        hard = json.loads(hard_path.read_text())["records"]

    medium = json.loads(args.medium.read_text())["records"]
    for r in medium:
        r.setdefault("family", "henneberg_small")
    large = [r for r in large if r["split"] != "test"]  # keep its test portion out entirely
    for r in large:
        r["family"] = "henneberg_large"

    held_out = {signature(r) for r in json.loads(args.hard_eval.read_text())["records"]}
    by_pair = collections.defaultdict(list)
    for r in hard:
        by_pair[r["pair_id"]].append(r)
    # Drop whole pairs (not single records) so every family stays label-balanced.
    leaky = {pid for pid, rs in by_pair.items() if any(signature(r) in held_out for r in rs)}
    dropped = sum(len(by_pair.pop(pid)) for pid in leaky)
    pair_ids = sorted(by_pair)
    rng.shuffle(pair_ids)
    n_val = round(0.15 * len(pair_ids))
    for i, pid in enumerate(pair_ids):
        for r in by_pair[pid]:
            r["split"] = "val" if i < n_val else "train"
    hard = [r for pid in pair_ids for r in by_pair[pid]]

    records = medium + large + hard
    counts = collections.Counter((r["family"], r["split"]) for r in records)
    labels = collections.Counter((r["family"], r["label"]) for r in records if r["split"] == "train")
    meta = {"purpose": "mixed training set", "args": {k: str(v) for k, v in vars(args).items()},
            "dropped_isomorphic_to_hard_eval": dropped}
    args.out.write_text(json.dumps({"meta": meta, "records": records}) + "\n", encoding="utf-8")
    for fam in ("henneberg_small", "henneberg_large", "decoy", "multi", "symmetric"):
        print(f"{fam:16s} train {counts[(fam, 'train')]:5d} (labels {labels[(fam, 0)]}/{labels[(fam, 1)]})  "
              f"val {counts[(fam, 'val')]:4d}  test {counts[(fam, 'test')]:4d}")
    print(f"dropped {dropped} hard records (whole pairs) overlapping hard_eval; saved {len(records)} to {args.out}")


if __name__ == "__main__":
    main()
