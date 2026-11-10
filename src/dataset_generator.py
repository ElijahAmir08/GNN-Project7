"""This file serves as a way to generate nonrigid graphs using the Laman graphs dataset.
    As of now the laman set has only graphs that are minimally rigid, and the GNN can learn this to always predict 1.
    To generate nonrigid graphs, we can take a laman graph and find all edges that can exist but do not, and swap this possible edge with an existing edge.
    Until a swap that makes the graph nonrigid is found, we can keep swapping edges. This makes it harder for the GNN to learn shortcuts, and forces it to learn the actual rigidity of the graph."""

from itertools import combinations
from src.graph_rigidity import is_graph_rigid

def find_nonrigid_graphs(
        num_nodes: int,
        edges: list[tuple[int, int]],
    ) -> list[tuple[int,int]] | None:
        # Store existing edges in a set
        # Sort endpoints of edges to ensure (a, b) and (b, a) are treated the same
        existing_edges = {tuple(sorted(edge)) for edge in edges}
        # Find all possible edges that could exist
        all_possible_edges = set(combinations(range(num_nodes), 2))
        # Find edges that could exist but do not
        missing_edges = sorted(all_possible_edges - existing_edges)
        # Try swapping each missing edge with an existing edge
        for removed_edge in existing_edges:
                remaining_edges = existing_edges - {removed_edge}
                # Replace with each missing edge and check rigidity
                for added_edge in missing_edges:
                    modified_edges = sorted(remaining_edges | {added_edge})
                    if not is_graph_rigid(num_nodes, modified_edges):
                        return modified_edges
        return None