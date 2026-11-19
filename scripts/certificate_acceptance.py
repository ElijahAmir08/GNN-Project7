"""Does the model accept the TRUE certificate on its own?

For every truly redundant example the model also predicts redundant, show the
model only a minimal valid certificate (record["circuit_edges"] minus the
candidate) and check whether it still predicts redundant. 100% = a correct
proof is always sufficient for the model. This is the oracle-explanation
F∧V of scripts/oracle_eval.py, computed directly (no explainers needed).

Usage:
  python scripts/certificate_acceptance.py --runs mixed_gin maskaug_gin randmask_gin ...
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

DEFAULT_DATA = ["data_new/generated/hard_eval.json", "data_new/generated/chain_matched_eval.json",
                "data_new/generated/ood_suite.json", "data_new/generated/truss_eval.json"]


@torch.no_grad()
def predict(model, recs):
    out = []
    for b in DataLoader([record_to_data(r) for r in recs], batch_size=256):
        out += model(b.x, b.edge_index, b.batch, b.candidate).argmax(-1).tolist()
    return out


def cert_only(r):
    cand = tuple(r["candidate_edge"])
    cert = sorted(tuple(e) for e in r["circuit_edges"] if tuple(e) != cand)
    return dict(r, edges=[list(e) for e in cert], circuit_edges=None, certificates=None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=ROOT / "results/certificate_acceptance.json")
    args = p.parse_args()
    torch.set_num_threads(8)

    groups = collections.defaultdict(list)
    for path in args.data:
        for r in json.loads((ROOT / path).read_text())["records"]:
            if r.get("split") == "test" and r["label"] == 1 and r.get("circuit_edges"):
                groups[f"{Path(path).stem}:{r.get('family', '')}".rstrip(":")].append(r)

    res = {}
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            for g, recs in groups.items():
                full = predict(model, recs)
                only = predict(model, [cert_only(r) for r in recs])
                kept = [o for f, o in zip(full, only) if f == 1]
                res.setdefault(f"{run}|{g}", []).append(100 * sum(kept) / len(kept) if kept else float("nan"))

    args.out.write_text(json.dumps(res, indent=1))
    names = list(groups)
    print("| run | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for run in args.runs:
        cells = []
        for g in names:
            v = [x for x in res.get(f"{run}|{g}", []) if x == x]
            cells.append(f"{st.mean(v):.0f} ± {st.pstdev(v):.0f}" if v else "–")
        print(f"| {run} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
