"""Generate exact generic 2D rigidity labels and minimal dependency circuits.

Initial scope: independent base graphs with a candidate NON-edge.
For an independent base, any dependent added edge has a unique fundamental
circuit containing that candidate. This is our Day 1 toy-data interface.
"""

from __future__ import annotations

from pyrigi import Graph


def normalize_edge(edge: tuple[int, int]) -> tuple[int, int]:
    u, v = edge
    if u == v:
        raise ValueError("Self-loops are not allowed in our simple graphs")
    return (min(u, v), max(u, v))


def normalize_edges(num_nodes: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized = [normalize_edge(edge) for edge in edges]
    if any(u < 0 or v >= num_nodes for u, v in normalized):
        raise ValueError("An edge endpoint is outside 0..num_nodes-1")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate undirected edges in the graph")
    return sorted(normalized)


def make_graph(num_nodes: int, edges: list[tuple[int, int]]) -> Graph:
    if num_nodes < 2:
        raise ValueError("At least two vertices are required")
    g = Graph()
    g.add_vertices(list(range(num_nodes)))  # Preserve isolated vertices.
    g.add_edges(normalize_edges(num_nodes, edges))
    return g


def fundamental_circuit(
    num_nodes: int,
    base_edges: list[tuple[int, int]],
    candidate_edge: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Return one minimal dependent set including candidate, or None.

    Preconditions: base edges are independent and candidate is not in base.
    For such a base, a dependent candidate belongs to a unique circuit.
    We remove unnecessary base edges and verify the remaining graph is a circuit.
    """
    current = normalize_edges(num_nodes, base_edges + [candidate_edge])
    candidate = normalize_edge(candidate_edge)
    if make_graph(num_nodes, current).is_Rd_independent(dim=2):
        return None

    for edge in list(current):
        if edge == candidate:
            continue
        trial = [item for item in current if item != edge]
        if not make_graph(num_nodes, trial).is_Rd_independent(dim=2):
            current = trial

    if not make_graph(num_nodes, current).is_Rd_circuit(dim=2):
        raise AssertionError("Circuit verification failed")
    return sorted(current)


def label_proposed_edge(
    num_nodes: int,
    base_edges: list[tuple[int, int]],
    candidate_edge: tuple[int, int],
) -> dict:
    """0 = independent; 1 = redundant. Certificate includes the candidate."""
    base = normalize_edges(num_nodes, base_edges)
    candidate = normalize_edge(candidate_edge)
    if candidate in base:
        raise ValueError("Candidate already exists; it must be a proposed non-edge")
    if not make_graph(num_nodes, base).is_Rd_independent(dim=2):
        raise ValueError("Day 1 generator requires an independent base graph")

    augmented = make_graph(num_nodes, base + [candidate])
    redundant = not augmented.is_Rd_independent(dim=2)
    circuit = fundamental_circuit(num_nodes, base, candidate) if redundant else None

    return {
        "num_nodes": num_nodes,
        "existing_edges": [list(edge) for edge in base],
        "candidate_edge": list(candidate),
        "label": 1 if redundant else 0,
        "label_name": "redundant" if redundant else "independent",
        "circuit_edges": None if circuit is None else [list(edge) for edge in circuit],
    }
