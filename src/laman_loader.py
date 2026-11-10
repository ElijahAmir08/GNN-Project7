from pathlib import Path
"""Decode graph data from Laman graphs dataset."""


def decode_laman_integer(encoded: int, num_nodes: int) -> list[tuple[int, int]]:
    """Decode a Laman graph integer into an undirected edge list.

    Example: encoded = 31 (011111), num_nodes = 4
      returns [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    """

    # Determine number of upper-triangle entries
    # that exist for num_nodes.
    num_edges = num_nodes * (num_nodes - 1) // 2
    # Convert encoded integer into a binary string padding with leading zeros to match the width
    binary_string = format(encoded, f"0{num_edges}b")
    # Iterate through pairs of nodes (i, j) where i < j, in upper-triangle row order
    # Add edge (i, j) to the edge list if the corresponding bit is 1
    edges = []
    index = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if binary_string[index] == "1":
                edges.append((i, j))
            index += 1

    # Return the list of edges
    return edges


def load_laman_graphs(
        num_nodes: int,
        dataset_dir: Path ) -> list[list[tuple[int, int]]]:
        """Load laman graphs with num_nodes vertices
           
           Reads LamanGraphsN.txt and decodes each graph into the undirected edge list
        """
        # Create the path to the Laman graphs text file for the specified number of nodes
        text_file = dataset_dir / f"LamanGraphs{num_nodes}.txt"
        # Read the contents
        contents = text_file.read_text(encoding="utf-8").strip()
        # Check file format
        if not (contents.startswith("{") and contents.endswith("}")):
            raise ValueError(f"Invalid format in {text_file}. Expected contents to be enclosed in braces.")
        #Remove the outer braces
        contents = contents[1:-1].strip()

        if not contents:
            return []
        # Convert comma seperated ints into edge lists
        graphs = []
        for value in contents.split(","):
             encoded = int(value.strip())
             edges = decode_laman_integer(encoded, num_nodes)
             graphs.append(edges)
        return graphs

