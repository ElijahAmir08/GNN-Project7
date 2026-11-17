"""Join verified Laman graphs along one shared edge."""

from src.rigidity_labels import make_graph, normalize_edge, normalize_edges


def glue_laman_graphs(left_n, left_edges, right_n, right_edges, left_edge, right_edge):
    """Identify the chosen edge endpoints, keeping one copy of the shared edge.

    The union is Laman with n = left_n + right_n - 2 and m = 2n - 3.
    Return the edges and module mappings for checking circuit coverage.
    """
    left_edges = normalize_edges(left_n, left_edges)
    right_edges = normalize_edges(right_n, right_edges)
    left_edge, right_edge = normalize_edge(left_edge), normalize_edge(right_edge)
    if left_edge not in left_edges or right_edge not in right_edges:
        raise ValueError("The shared edge must exist in each module")
    if not make_graph(left_n, left_edges).is_min_rigid(dim=2):
        raise ValueError("Left module must be Laman")
    if not make_graph(right_n, right_edges).is_min_rigid(dim=2):
        raise ValueError("Right module must be Laman")

    mapping = {right_edge[0]: left_edge[0], right_edge[1]: left_edge[1]}
    next_vertex = left_n
    for vertex in range(right_n):
        if vertex not in mapping:
            mapping[vertex] = next_vertex
            next_vertex += 1
    mapped_right = sorted(normalize_edge((mapping[u], mapping[v])) for u, v in right_edges)
    edges = sorted(set(left_edges) | set(mapped_right))
    if len(edges) != 2 * next_vertex - 3 or not make_graph(next_vertex, edges).is_min_rigid(dim=2):
        raise AssertionError("The shared-edge union should be Laman")
    return {
        "num_nodes": next_vertex, "edges": edges, "shared_edge": left_edge,
        "module_vertices": [list(range(left_n)), sorted(mapping.values())],
        "module_edges": [left_edges, mapped_right],
        "right_vertex_map": [mapping[v] for v in range(right_n)],
    }


def cross_module_candidates(chain):
    """Nonedges with one exclusive vertex from each module."""
    shared = set(chain["shared_edge"])
    left = sorted(set(chain["module_vertices"][0]) - shared)
    right = sorted(set(chain["module_vertices"][1]) - shared)
    return [normalize_edge((u, v)) for u in left for v in right]


def circuit_module_support(chain, circuit_edges):
    """Count circuit edges exclusive to each module; remove the shared edge."""
    circuit = set(map(tuple, circuit_edges)) - {tuple(chain["shared_edge"])}
    return [len(circuit & set(edges)) for edges in chain["module_edges"]]
