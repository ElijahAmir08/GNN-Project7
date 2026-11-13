"""Compute Edge Orbits under automorphisms"""

import networkx as nx

def compute_edge_orbits (
        num_nodes: int,
        edges: list[tuple[int, int]],
) -> list[list[int]]:
    """
    Returns groups of edge indices that are symmetry-equivalent    
    """
    # Normalize each edge to have each edge begin with smaller endpoint
    normalized = [tuple(sorted(edge)) for edge in edges]

    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate edges found")
    
    # Validate the endpoints of each edge
    if any(u == v or u < 0 or v >= num_nodes for u, v in normalized):
        raise ValueError("Invalid edge endpoints")


    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(normalized)

    # Map each edge to its respective index
    edge_index = {
        edge: i for i, edge in enumerate(normalized)
    }

    # Enumerate graph automorphisms 
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph, graph
    )

    # Store every valid automorphism as a list
    automorphisms = list(matcher.isomorphisms_iter())

    remaining = set(range(len(normalized)))
    orbits = []

    while remaining:
        # Select edge that is not in an orbit
        i = min(remaining)
        u, v = normalized[i]

        # Find every edge that (u,v) can map to with automorphism
        orbit = sorted({
            edge_index[tuple(sorted((mapping[u], mapping[v])))]
            for mapping in automorphisms
        })
        orbits.append(orbit)
        remaining.difference_update(orbit)
    return orbits