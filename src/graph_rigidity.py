"""Generic 2D graph rigid checks"""
from pyrigi import Graph

def is_graph_rigid(num_nodes: int, 
                    edges: list[tuple[int, int]],
                    ) -> bool:
    """ Checks if a graph is rigid in 2D
        num_nodes = num of verticies from 0 to n-1
        edges = undirected edges already in the graph
    """
    # Create an empty Pyrigi graph
    graph = Graph()
    #Add edges and nodes to the graph
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    # Check if the graph is rigid in 2D
    return graph.is_rigid(dim=2)