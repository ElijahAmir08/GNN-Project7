"""Exact generic 2D rigidity oracle for arbitrary edge subsets.

rigidity_rank(n, edges): rank of the 2D rigidity matrix at a random
realization. A random realization is generic with probability 1, so this is
the generic rank (the rank of the edge set in the 2D rigidity matroid). We
take the max over a few realizations as a guard against numerical bad luck.

is_redundant(n, S, c): candidate c is dependent on edge set S, i.e.
rank(S + c) == rank(S). Unlike circuit-membership shortcuts, this is valid for
ANY S, including dependent base graphs with several circuits through c.

Explanation-level ground truth built on it:
  certificate_valid(S)   the explanation S alone proves c is redundant
  truth_label(S)         true label of c on the masked graph S (what a perfect
                         model should predict when only S is kept)
"""

from __future__ import annotations

import numpy as np

_REALIZATIONS = 2


def _rank(num_nodes: int, edges, rng: np.random.Generator) -> int:
    if len(edges) == 0:
        return 0
    p = rng.random((num_nodes, 2))
    m = np.zeros((len(edges), 2 * num_nodes))
    for row, (u, v) in enumerate(edges):
        d = p[u] - p[v]
        m[row, 2 * u:2 * u + 2] = d
        m[row, 2 * v:2 * v + 2] = -d
    return int(np.linalg.matrix_rank(m))


def rigidity_rank(num_nodes: int, edges, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    return max(_rank(num_nodes, list(edges), rng) for _ in range(_REALIZATIONS))


def is_redundant(num_nodes: int, edges, candidate, seed: int = 0) -> bool:
    edges = [tuple(e) for e in edges]
    return rigidity_rank(num_nodes, edges + [tuple(candidate)], seed) == rigidity_rank(num_nodes, edges, seed)


def truth_label(num_nodes: int, edges, candidate) -> int:
    """1 = redundant, 0 = independent, for the candidate on edge set `edges`."""
    return int(is_redundant(num_nodes, edges, candidate))


def certificate_valid(num_nodes: int, base_edges, candidate, selected_idx) -> bool:
    """Does the selected subset of base edges on its own prove redundancy?"""
    sel = [tuple(base_edges[i]) for i in selected_idx]
    return is_redundant(num_nodes, sel, candidate)
