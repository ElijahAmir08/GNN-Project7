""" Build a small labeled dataset for rigidity classification"""

import json
from pathlib import Path

import networkx as nx
from src.laman_loader import load_laman_graphs
from src.dataset_generator import find_nonrigid_graphs
from src.graph_rigidity import is_graph_rigid

DATASET_DIR = Path("data/raw/laman")
OUTPUT_PATH = Path("data/processed/rigidity_seed.json")
TARGET_PAIRS = 12

def to_networkx(num_nodes, edges):
    # Convert edge list into a NetworkX graph
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    return graph

def is_duplicate(num_nodes, edges, existing_graphs):
    # Check if the graph is a duplicate of any existing graph
    candidate_graph = to_networkx(num_nodes, edges)
    for graph in existing_graphs:
        if graph["num_nodes"] != num_nodes:
            continue
        previous_graph = to_networkx(graph["num_nodes"], graph["edges"])
        if nx.is_isomorphic(candidate_graph, previous_graph):
            return True
    return False

def make_record(graph_id, num_nodes, edges, label, source):
    normalized_edges = sorted(
        tuple(sorted(edge)) for edge in edges
    )
    return {
        "graph_id": graph_id,
        "num_nodes": num_nodes,
        "edges": [list(edge) for edge in normalized_edges],
        "label": label,
        "source": source
    }

def main():
    records = []
    pair_count = 0

    for num_nodes in [5, 6, 7, 8]:
        graphs = load_laman_graphs(num_nodes, DATASET_DIR)
        
        print(f"Loaded {len(graphs)} Laman graphs with {num_nodes} nodes.")

        for rigid_edges in graphs:
            if pair_count >= TARGET_PAIRS:
                break
            # Attempt to make a nonrigid graph
            nonrigid_edges = find_nonrigid_graphs(num_nodes, rigid_edges)
            if nonrigid_edges is None:
                continue
            if is_duplicate(num_nodes, nonrigid_edges, records):
                continue
            if is_duplicate(num_nodes, rigid_edges, records):
                continue
            # Verify the labels
            assert is_graph_rigid(num_nodes, rigid_edges), "Rigid graph should be rigid."
            assert not is_graph_rigid(num_nodes, nonrigid_edges), "Nonrigid graph should be nonrigid."
            # Ensuure both graphs have the same number of edges
            assert len(rigid_edges) == len(nonrigid_edges), "Both graphs should have the same number of edges."
            assert len(rigid_edges) == 2 * num_nodes - 3, "Laman graph should have 2n-3 edges."

            pair_count += 1
            pair_id = f"{num_nodes}_{pair_count}"
            records.append(make_record(f"{pair_id}_rigid", num_nodes, rigid_edges, 1, "laman"))
            records.append(make_record(f"{pair_id}_nonrigid", num_nodes, nonrigid_edges, 0, "laman_nonrigid_conversion"))

            if pair_count < TARGET_PAIRS:
                print(f"Found {pair_count} pairs of rigid/nonrigid graphs so far.")

    # Create the output directory if it doesn't exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write the records to a JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    print(f"Saved {len(records)} graphs to {OUTPUT_PATH}")
    print(f"Rigid: {sum(record['label'] == 1 for record in records)}, Nonrigid: {sum(record['label'] == 0 for record in records)}")

if __name__ == "__main__":
    main()