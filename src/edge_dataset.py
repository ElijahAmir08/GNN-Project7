"""Edge-level rigidity dataset: is a proposed edge independent or redundant?

Recipe (see notes/edge_task_design.md):
  1. Start from a Laman graph (rigid + independent, 2n - 3 edges).
  2. Delete k >= 1 random edges -> an independent but flexible base graph.
     (Subsets of independent sets are independent, so the fundamental-circuit
     preconditions of `label_proposed_edge` hold.)
  3. Every non-edge of the base is a candidate; PyRigi labels it
     0 = independent or 1 = redundant, with a unique circuit when redundant.
  4. Sample the same number of candidates from each class per base graph,
     so the label is balanced *within* each base (no base-level shortcut).

Splits are assigned per base graph, and isomorphic base graphs are grouped
into the same split so no structure leaks across train/val/test.
"""

from __future__ import annotations

import random
from itertools import combinations

import networkx as nx

from src.laman import random_laman_graph
from src.rigidity_labels import label_proposed_edge


def make_base_graph(num_nodes: int, num_delete: int, rng: random.Random) -> list[tuple[int, int]]:
    laman = random_laman_graph(num_nodes, rng)
    drop = set(rng.sample(range(len(laman)), num_delete))
    return [e for i, e in enumerate(laman) if i not in drop]


def examples_for_base(
    base_id: str,
    num_nodes: int,
    base_edges: list[tuple[int, int]],
    rng: random.Random,
    per_class: int,
) -> list[dict]:
    """Label all non-edges, then keep up to `per_class` of each label (balanced)."""
    existing = set(base_edges)
    labeled = {0: [], 1: []}
    for cand in combinations(range(num_nodes), 2):
        if cand in existing:
            continue
        rec = label_proposed_edge(num_nodes, base_edges, cand)
        labeled[rec["label"]].append(rec)

    k = min(per_class, len(labeled[0]), len(labeled[1]))
    out = []
    for label in (0, 1):
        for j, rec in enumerate(rng.sample(labeled[label], k)):
            u, v = rec["candidate_edge"]
            out.append({
                "graph_id": f"{base_id}_c{u}-{v}",
                "base_id": base_id,
                "num_nodes": num_nodes,
                "edges": rec["existing_edges"],
                "candidate_edge": rec["candidate_edge"],
                "label": rec["label"],
                "circuit_edges": rec["circuit_edges"],
            })
    return out


def _base_task(args):
    idx, num_nodes, num_delete, seed, per_class = args
    rng = random.Random(seed)
    base_id = f"n{num_nodes}_k{num_delete}_b{idx}"
    edges = make_base_graph(num_nodes, num_delete, rng)
    return base_id, num_nodes, edges, examples_for_base(base_id, num_nodes, edges, rng, per_class)


def assign_splits(
    bases: list[tuple[str, int, list[tuple[int, int]]]],
    fractions: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, str]:
    """Group isomorphic bases, then assign whole groups to train/val/test."""
    groups: list[list[tuple[str, nx.Graph]]] = []
    by_hash: dict[str, list[int]] = {}
    for base_id, n, edges in bases:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        h = nx.weisfeiler_lehman_graph_hash(g)
        for gi in by_hash.get(h, []):
            if nx.is_isomorphic(groups[gi][0][1], g):
                groups[gi].append((base_id, g))
                break
        else:
            by_hash.setdefault(h, []).append(len(groups))
            groups.append([(base_id, g)])

    order = list(range(len(groups)))
    rng.shuffle(order)
    total = len(bases)
    cut_train, cut_val = fractions[0] * total, (fractions[0] + fractions[1]) * total
    split_of, seen = {}, 0
    for gi in order:
        name = "train" if seen < cut_train else "val" if seen < cut_val else "test"
        for base_id, _ in groups[gi]:
            split_of[base_id] = name
        seen += len(groups[gi])
    return split_of
