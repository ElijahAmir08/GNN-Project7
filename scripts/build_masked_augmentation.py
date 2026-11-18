"""Masked-subgraph augmentation with exact oracle labels (model-side fix).

Explainers and fidelity metrics query the model on masked graphs (subsets of
the edges). Models trained only on full graphs can be wrong there, e.g. MPNN
keeps predicting 'redundant' on subgraphs that contain no certificate. With
the exact rigidity oracle we can label any masked graph, so we add masked
copies of training examples to the training data:

  random     keep a random 30-90% of the edges (negatives downsampled to
             the number of positives, per split, to keep labels balanced)
  near_cert  (redundant parents) a certificate minus one edge, plus random
             off-certificate edges -> usually independent: the hard negatives
             that 'faithful but invalid' explanations exploit
  cert_plus  (redundant parents) a certificate plus random off-certificate
             edges (same extra-edge distribution) -> redundant

Every masked record is labeled by src/oracle.py, inherits its parent's split
(no leakage into val/test), and gets family "masked_<kind>". Masked graphs keep
all vertices (isolated vertices allowed). Recomputed certificates are not
stored (circuit_edges=None): these records are for training only.

Usage:
  python scripts/build_masked_augmentation.py --data data_new/generated/mixed_train.json \
      --out data_new/generated/mixed_masked_train.json
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
from src.oracle import truth_label  # noqa: E402


def masked_copies(args) -> list[dict]:
    rec, seed, n_random = args
    rng = random.Random(seed)
    n, cand = rec["num_nodes"], tuple(rec["candidate_edge"])
    edges = [tuple(e) for e in rec["edges"]]
    subsets = []
    for _ in range(n_random):
        keep = rng.uniform(0.3, 0.9)
        subsets.append(("random", [e for e in edges if rng.random() < keep]))
    if rec["label"] == 1 and rec.get("circuit_edges"):
        cert = [tuple(e) for e in rec["circuit_edges"] if tuple(e) != cand]
        drop = rng.choice(cert)
        rest = [e for e in edges if e not in cert]
        # Same distribution of extra off-certificate edges for both kinds, so
        # edge count does not reveal the label (the oracle still decides it:
        # extras can complete another certificate in multi-certificate bases).
        extra = rng.sample(rest, rng.randint(0, len(rest)))
        subsets.append(("near_cert", [e for e in cert if e != drop] + extra))
        extra = rng.sample(rest, rng.randint(0, len(rest)))
        subsets.append(("cert_plus", cert + extra))
    out = []
    for i, (kind, sub) in enumerate(subsets):
        sub = sorted(set(sub))
        out.append({
            "graph_id": f"{rec['graph_id']}_m{i}",
            "base_id": rec.get("base_id", rec["graph_id"]),
            "num_nodes": n,
            "edges": [list(e) for e in sub],
            "candidate_edge": list(cand),
            "label": truth_label(n, sub, cand),
            "circuit_edges": None,
            "split": rec["split"],
            "family": f"masked_{kind}",
            "parent_id": rec["graph_id"],
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data_new/generated/mixed_train.json")
    p.add_argument("--n-random", type=int, default=1, help="random masks per parent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", type=Path, default=ROOT / "data_new/generated/mixed_masked_train.json")
    args = p.parse_args()

    blob = json.loads(args.data.read_text())
    records = blob["records"]
    parents = [r for r in records if r["split"] in ("train", "val")]  # never augment test
    tasks = [(r, args.seed * 1_000_003 + i, args.n_random) for i, r in enumerate(parents)]
    with Pool(args.workers) as pool:
        masked = [m for group in pool.map(masked_copies, tasks, chunksize=16) for m in group]

    # Random masks are almost always independent; keep the augmented data
    # label-balanced by downsampling random-mask negatives to the positives,
    # per split (near_cert / cert_plus are close to balanced by construction).
    rng = random.Random(args.seed)
    balanced = [m for m in masked if m["family"] != "masked_random"]
    for split in ("train", "val"):
        rand = [m for m in masked if m["family"] == "masked_random" and m["split"] == split]
        pos = [m for m in rand if m["label"] == 1]
        neg = [m for m in rand if m["label"] == 0]
        balanced += pos + rng.sample(neg, min(len(neg), len(pos)))
    masked = balanced

    out = records + masked
    meta = dict(blob.get("meta", {}), masked_augmentation={k: str(v) for k, v in vars(args).items()})
    args.out.write_text(json.dumps({"meta": meta, "records": out}) + "\n", encoding="utf-8")
    c = collections.Counter((m["family"], m["split"], m["label"]) for m in masked)
    for fam in ("masked_random", "masked_near_cert", "masked_cert_plus"):
        print(f"{fam:17s} train 0/1 = {c[(fam, 'train', 0)]}/{c[(fam, 'train', 1)]}   "
              f"val 0/1 = {c[(fam, 'val', 0)]}/{c[(fam, 'val', 1)]}")
    print(f"added {len(masked)} masked records; saved {len(out)} to {args.out}")


if __name__ == "__main__":
    main()
