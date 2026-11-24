"""Days 3-4: existing checkpoints on the composed-circuit benchmark, by depth.

Reports, per run (mean over seeds) and per (shape, composition depth):
accuracy, accuracy on each class, and pair accuracy (both members of a matched
pair right; chance 25%). Chain and tree twins with the same number of
components have the same size, so chain-vs-tree at equal k isolates depth
from size (H2b). Recurrent models use their saved evaluation rounds (10,
chosen on validation). Never pooled across depths.

Usage: python scripts/eval_circuit_depth.py --runs mixed_gin maskaug_gin ...
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


@torch.no_grad()
def predict(model, datas):
    out = []
    for b in DataLoader(datas, batch_size=128):
        out += model(b.x, b.edge_index, b.batch, b.candidate).argmax(-1).tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data_new/generated/circuit_bench.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/eval_circuit_depth.md")
    args = p.parse_args()
    torch.set_num_threads(16)
    recs = json.loads(args.data.read_text())["records"]
    datas = [record_to_data(r) for r in recs]
    keys = sorted({(r["n_components"], r["family"].split("_")[1]) for r in recs})

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            pred = predict(model, datas)
            groups = collections.defaultdict(list)
            for r, y in zip(recs, pred):
                groups[(r["n_components"], r["family"].split("_")[1])].append((r, y))
            for key, items in groups.items():
                ok = [y == r["label"] for r, y in items]
                by_pair = collections.defaultdict(list)
                for (r, _), c in zip(items, ok):
                    by_pair[r["pair_id"]].append(c)
                res[run][key + ("acc",)].append(sum(ok) / len(ok))
                res[run][key + ("pair",)].append(sum(all(v) for v in by_pair.values()) / len(by_pair))
                res[run][key + ("indep",)].append(st.mean(c for (r, _), c in zip(items, ok) if r["label"] == 0))

    depth_of = {(r["n_components"], r["family"].split("_")[1]): r["composition_depth"] for r in recs}
    lines = ["# Composed-circuit benchmark: existing checkpoints by composition depth\n",
             "Pair accuracy (both members of a matched pair right; chance 0.25). Chain and tree twins with "
             "the same number of components k have identical size; depth = chain k−1, tree ⌊log2 k⌋.\n"]
    for shape in ("chain", "tree"):
        ks = [k for k, s in keys if s == shape]
        lines.append(f"\n## {shape}: pair accuracy by components k (depth in brackets)\n")
        lines.append("| run | " + " | ".join(f"k={k} [d={depth_of[(k, shape)]}]" for k in ks) + " |")
        lines.append("|---|" + "---|" * len(ks))
        for run in args.runs:
            lines.append(f"| {run} | " + " | ".join(f"{st.mean(res[run][(k, shape, 'pair')]):.2f}" for k in ks) + " |")
    lines.append("\n## Independent-class accuracy (chain), to expose a 'redundant' bias\n")
    ks = [k for k, s in keys if s == "chain"]
    lines.append("| run | " + " | ".join(f"k={k}" for k in ks) + " |")
    lines.append("|---|" + "---|" * len(ks))
    for run in args.runs:
        lines.append(f"| {run} | " + " | ".join(f"{st.mean(res[run][(k, 'chain', 'indep')]):.2f}" for k in ks) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({r: {"|".join(map(str, k)): v for k, v in d.items()}
                                                          for r, d in res.items()}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
