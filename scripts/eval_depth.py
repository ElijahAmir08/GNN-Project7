"""Evaluate the expressivity x depth experiment, varying rounds at test time.

For each run (and, for recurrent models, each round count in --rounds):
  family_train   pair accuracy on wheel/hinge pairs with k in 3..6 (seen sizes)
  family_heldout pair accuracy on k in 7..15 (never trained on)
  family_k       pair accuracy per k
  and accuracy on the medium test set, chains, ood_chain, ood_large, lattice.
Pair accuracy = both members of a pair classified correctly (chance = 25%,
always-'redundant' = 0%).

Usage: python scripts/eval_depth.py --runs fam_gin fam_dsgin fam_recmpnn fam_recdsgin
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import model_from_config  # noqa: E402
from src.wl_family import pair_records  # noqa: E402

SETS = {
    "medium": ("data/generated/edge_medium.json", None),
    "chains": ("data_new/generated/chain_matched_eval.json", None),
    "ood_chain": ("data_new/generated/ood_suite.json", "ood_chain"),
    "ood_large": ("data_new/generated/ood_suite.json", "ood_large"),
    "lattice": ("data_new/generated/ood_suite.json", "lattice"),
}


@torch.no_grad()
def predict(model, datas):
    out = []
    for b in DataLoader(datas, batch_size=128):
        out += model(b.x, b.edge_index, b.batch, b.candidate).argmax(-1).tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--rounds", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16, 24])
    p.add_argument("--out", type=Path, default=ROOT / "results/eval_depth.md")
    args = p.parse_args()
    torch.set_num_threads(16)

    fam = [r for k in range(3, 16) for r in pair_records(k)]
    fam_d = [record_to_data(r) for r in fam]
    sets = {}
    for name, (path, family) in SETS.items():
        recs = [r for r in json.loads((ROOT / path).read_text())["records"]
                if r.get("split") == "test" and (family is None or r.get("family") == family)]
        sets[name] = (recs, [record_to_data(r) for r in recs])

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            settings = args.rounds if hasattr(model, "rounds") else [None]
            for rounds in settings:
                if rounds is not None:
                    model.rounds = rounds
                key = f"{run}" + (f"@{rounds}" if rounds is not None else "")
                pf = predict(model, fam_d)
                per_k = collections.defaultdict(list)
                for i in range(0, len(fam), 2):
                    per_k[fam[i]["k"]].append(pf[i] == fam[i]["label"] and pf[i + 1] == fam[i + 1]["label"])
                res[key]["family_train"].append(mean([x for k in range(3, 7) for x in per_k[k]]))
                res[key]["family_heldout"].append(mean([x for k in range(7, 16) for x in per_k[k]]))
                for k, v in per_k.items():
                    res[key][f"k{k}"].append(mean(v))
                for name, (recs, datas) in sets.items():
                    pr = predict(model, datas)
                    res[key][name].append(mean([y == r["label"] for y, r in zip(pr, recs)]))

    cols = ["family_train", "family_heldout", "medium", "chains", "ood_chain", "ood_large", "lattice"]
    lines = ["# Expressivity x depth (rounds varied at test time for recurrent models)\n",
             "family_* = pair accuracy on wheel/hinge pairs (both members right); train k = 3..6, held-out k = 7..15.\n",
             "| run@rounds | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for key, d in res.items():
        lines.append(f"| {key} | " + " | ".join(f"{st.mean(d[c]):.2f}" for c in cols) + " |")
    lines += ["\n## Wheel/hinge pair accuracy by k\n",
              "| run@rounds | " + " | ".join(f"k={k}" for k in range(3, 16)) + " |", "|---|" + "---|" * 13]
    for key, d in res.items():
        lines.append(f"| {key} | " + " | ".join(f"{st.mean(d[f'k{k}']):.2f}" for k in range(3, 16)) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({k: dict(v) for k, v in res.items()}, indent=1))
    print(text)


def mean(v):
    v = [float(x) for x in v]
    return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    main()
