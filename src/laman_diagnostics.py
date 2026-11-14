"""Circuit-aware candidate checks on a small Laman sample."""

from itertools import combinations

import networkx as nx

from src.orbits import candidate_stabilizer_orbits
from src.rigidity_labels import label_proposed_edge, make_graph, normalize_edge


def candidate_features(num_nodes, edges, candidate) -> tuple[int, ...]:
    """Same seven features as models.shortcut_features, without importing PyG."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    u, v = candidate
    degrees = sorted((graph.degree[u], graph.degree[v]))
    distance = nx.shortest_path_length(graph, u, v) if nx.has_path(graph, u, v) else num_nodes
    return (
        num_nodes, len(edges), 2 * num_nodes - 3 - len(edges), *degrees,
        len(list(nx.common_neighbors(graph, u, v))), distance,
    )


def candidate_sort_key(record):
    circuit_size = len(record["circuit_edges"])
    candidate_edge = record["candidate_edge"]
    return (-circuit_size, candidate_edge)


def find_matched_pair(num_nodes, parent_edges, min_circuit_size=10,
                      candidate_edges=None, independent_candidates=None, min_distance=0):
    """Find one opposite-label pair with equal features on the same flexible base.

    Remove an edge outside a candidate's circuit. Restoring that edge is
    independent; the original candidate remains redundant. Optionally search
    other independent candidates with the same features instead of restoring
    the removed edge. Keep an off-circuit edge and prefer larger circuits.
    """
    if not make_graph(num_nodes, parent_edges).is_min_rigid(dim=2):
        raise ValueError("Parent must be a Laman graph")
    existing = set(parent_edges)
    if candidate_edges is None:
        candidate_edges = list(combinations(range(num_nodes), 2))
    candidate_edges = sorted({normalize_edge(edge) for edge in candidate_edges} - existing)
    if independent_candidates is not None:
        independent_candidates = sorted({normalize_edge(edge) for edge in independent_candidates} - existing)
    search = set(candidate_edges) | set(independent_candidates or [])
    parent_records = {}
    for candidate in sorted(search):
        if candidate not in existing:
            rec = label_proposed_edge(num_nodes, parent_edges, candidate)
            parent_records[candidate] = rec
    candidates = [parent_records[edge] for edge in candidate_edges
                  if len(parent_records[edge]["circuit_edges"]) >= min_circuit_size]
    candidates.sort(key=candidate_sort_key)

    for rec in candidates:
        circuit = set(map(tuple, rec["circuit_edges"]))
        outside = sorted(existing - circuit)
        if len(outside) < 2:
            continue
        candidate = tuple(rec["candidate_edge"])
        for removed in outside:
            base = sorted(existing - {removed})
            features = candidate_features(num_nodes, base, candidate)
            if features[-1] < min_distance:
                continue
            alternatives = [removed] if independent_candidates is None else independent_candidates
            for alternative in alternatives:
                if independent_candidates is not None:
                    # Removing an edge breaks exactly the parent circuits containing it.
                    other_circuit = set(map(tuple, parent_records[alternative]["circuit_edges"]))
                    if removed not in other_circuit:
                        continue
                if features != candidate_features(num_nodes, base, alternative):
                    continue
                redundant = label_proposed_edge(num_nodes, base, candidate)
                independent = label_proposed_edge(num_nodes, base, alternative)
                assert redundant["label"] == 1 and independent["label"] == 0
                assert redundant["circuit_edges"] == rec["circuit_edges"]
                return removed, independent, redundant
    return None


def dataset_record(base_id, labeled, parent_code, family):
    """Keep Salaar's edge-record fields and add source provenance."""
    u, v = labeled["candidate_edge"]
    return {
        "graph_id": f"{base_id}_c{u}-{v}", "base_id": base_id,
        "num_nodes": labeled["num_nodes"], "edges": labeled["existing_edges"],
        "candidate_edge": labeled["candidate_edge"], "label": labeled["label"],
        "circuit_edges": labeled["circuit_edges"],
        "parent_code": parent_code, "construction_family": family,
    }


def isomorphic_base_groups(records):
    """Group base IDs by topology, including bases from different parents."""
    groups, seen = [], set()
    for record in records:
        base_id = record["base_id"]
        if base_id in seen:
            continue
        seen.add(base_id)
        graph = nx.Graph()
        graph.add_nodes_from(range(record["num_nodes"]))
        graph.add_edges_from(record["edges"])
        for representative, ids in groups:
            if nx.is_isomorphic(graph, representative):
                ids.append(base_id)
                break
        else:
            groups.append((graph, [base_id]))
    return [ids for _, ids in groups]


def diagnose_record(record):
    edges = list(map(tuple, record["edges"]))
    n = record["num_nodes"]
    if not make_graph(n, edges).is_Rd_independent(dim=2):
        raise ValueError("Diagnostics require an independent base")
    features = candidate_features(n, edges, record["candidate_edge"])
    orbits = candidate_stabilizer_orbits(n, edges, tuple(record["candidate_edge"]))
    circuit = set(map(tuple, record["circuit_edges"] or []))
    mask = [edge in circuit for edge in edges]
    return {
        "graph_id": record["graph_id"], "parent_code": record["parent_code"],
        "construction_family": record["construction_family"],
        "features": features, "num_nodes": n, "num_edges": len(edges),
        "rigidity_rank": len(edges),  # Independence was checked above.
        "label": record["label"], "degree_min": features[3], "degree_max": features[4],
        "common_neighbors": features[5], "candidate_distance": features[6],
        "circuit_size": len(circuit) if circuit else None,
        "circuit_base_fraction": sum(mask) / len(edges) if circuit else None,
        "off_circuit_edges": len(edges) - sum(mask) if circuit else None,
        "orbit_sizes": sorted(map(len, orbits)),
        "has_nontrivial_edge_orbit": any(len(orbit) > 1 for orbit in orbits),
        "circuit_is_union_of_orbits": all(
            all(mask[i] for i in orbit) or not any(mask[i] for i in orbit)
            for orbit in orbits
        ) if circuit else None,
        "common_neighbor_prediction": int(features[5] >= 2),
    }
