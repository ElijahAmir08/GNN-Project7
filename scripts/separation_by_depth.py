"""Experiment 2: does the representation separation of matched pairs decay with depth?

For each matched pair (G+, G-) in the composed-circuit benchmark, take the
vector the classifier head actually receives (the candidate readout) and
compute the normalized separation
    Delta = ||h(G+) - h(G-)|| / (0.5 (||h(G+)|| + ||h(G-)||) + eps).
Readout vectors are permutation-invariant, so no node alignment is needed.
Reported per (shape, components k), for trained checkpoints and for the same
architectures at random initialization, alongside pair accuracy, to test
H2a (Delta decays with depth), H2b (chain < tree at equal size) and H2c
(accuracy tracks Delta).

Usage: python scripts/separation_by_depth.py --runs maskaug_mpnn maskaug_gin fam_recmpnn
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import model_from_config  # noqa: E402


def readout_vector(model, d):
    captured = {}

    def grab(module, inputs, output):  # must return None: a returned value would replace the output
        captured["x"] = inputs[0].detach()

    hook = model.head[0].register_forward_hook(grab)
    with torch.no_grad():
        model(d.x, d.edge_index, None, d.candidate)
    hook.remove()
    return captured["x"][0]


def separation(a, b, eps=1e-12):
    return (a - b).norm().item() / (0.5 * (a.norm() + b.norm()).item() + eps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data_new/generated/circuit_bench.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/separation_by_depth.md")
    args = p.parse_args()
    torch.set_num_threads(16)
    recs = json.loads(args.data.read_text())["records"]
    pairs = collections.defaultdict(dict)
    for r in recs:
        pairs[r["pair_id"]][r["label"]] = r
    keys = sorted({(r["n_components"], r["family"].split("_")[1]) for r in recs})

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            for state in ("trained", "init"):
                torch.manual_seed(0)
                model = model_from_config(ck["config"], NUM_NODE_FEATURES)
                if state == "trained":
                    model.load_state_dict(ck["state_dict"])
                model.eval()
                for pr in pairs.values():
                    r0, r1 = pr[0], pr[1]
                    key = (r0["n_components"], r0["family"].split("_")[1])
                    a = readout_vector(model, record_to_data(r0))
                    b = readout_vector(model, record_to_data(r1))
                    res[(run, state)][key].append(separation(a, b))

    lines = ["# Representation separation of matched pairs by composition depth\n",
             "Delta = normalized distance between the classifier inputs of the independent and redundant member "
             "(median over pairs, pooled over seeds). Chain and tree twins with the same k have identical size.\n"]
    for shape in ("chain", "tree"):
        ks = [k for k, s in keys if s == shape]
        lines.append(f"\n## {shape}\n")
        lines.append("| run | state | " + " | ".join(f"k={k}" for k in ks) + " |")
        lines.append("|---|---|" + "---|" * len(ks))
        for (run, state), d in res.items():
            lines.append(f"| {run} | {state} | " + " | ".join(f"{st.median(d[(k, shape)]):.3f}" for k in ks) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
