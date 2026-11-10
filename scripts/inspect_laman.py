"""Uses a real Laman graph to test the Laman graph loader."""

from pathlib import Path
from src.laman_loader import load_laman_graphs
from src.graph_rigidity import is_graph_rigid

DATASET_DIR = Path("data/raw/laman")

def main():
    for num_nodes in [4, 5, 6]:
        graphs = load_laman_graphs(num_nodes, DATASET_DIR)
        print(f"Loaded {len(graphs)} Laman graphs with {num_nodes} nodes.")
        for edges in graphs[:10]:
            is_rigid = is_graph_rigid(num_nodes, edges)
            print(f"Graph with edges {edges} is {'rigid' if is_rigid else 'not rigid'}.")
            assert len(edges) == 2 * num_nodes - 3, f"Graph with {num_nodes} nodes should have {2 * num_nodes - 3} edges."
            assert is_rigid, f"Graph with edges {edges} should be rigid."

if __name__ == "__main__":
    main()