"""Is beyond-1-WL expressivity even needed on this dataset?

Two examples are 1-WL-indistinguishable for our readout if, after colour
refinement with candidate endpoints marked, they have the same multiset of
node colours and the same (unordered) pair of endpoint colours. Any such pair
with different labels caps the accuracy of every 1-WL model (gcn, mpnn, gin).

The 1-WL accuracy ceiling predicts each indistinguishable group's majority
label; dsgin (> 1-WL) is not bound by it.

Usage: python scripts/wl_diagnostic.py --data data/generated/edge_medium.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def wl_signature(n: int, edges: list[list[int]], cand: list[int]) -> tuple:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Integer hashes are deterministic within a process, so colours are
    # comparable across graphs. n rounds is enough for refinement to stabilize.
    colors = [hash(("cand",)) if i in cand else hash(("node",)) for i in range(n)]
    for _ in range(n):
        colors = [hash((colors[i], tuple(sorted(colors[j] for j in adj[i])))) for i in range(n)]
    u, v = cand
    return (n, tuple(sorted(colors)), tuple(sorted((colors[u], colors[v]))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/generated/edge_medium.json"))
    args = p.parse_args()
    records = json.loads(args.data.read_text())["records"]

    groups: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        groups[wl_signature(r["num_nodes"], r["edges"], r["candidate_edge"])][r["label"]] += 1

    conflicting = [c for c in groups.values() if len(c) == 2]
    n_conf = sum(sum(c.values()) for c in conflicting)
    ceiling = sum(max(c.values()) for c in groups.values()) / len(records)
    print(f"examples: {len(records)}   1-WL classes: {len(groups)}")
    print(f"classes containing both labels: {len(conflicting)} ({n_conf} examples)")
    print(f"1-WL accuracy ceiling: {ceiling:.4f}")


if __name__ == "__main__":
    main()
