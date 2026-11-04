"""Load graph data from Laman graphs dataset."""

def decode_laman_integer(encoded: int, num_nodes: int) -> list[tuple[int, int]]:
    """Decode a Laman graph integer into an undirected edge list.

       Example: encoded = 31 (011111), num_nodes = 4
         returns [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    """ 

    # Determine number of upper-triangle entries
    # that exist for num_nodes.
    num_edges = num_nodes * (num_nodes - 1) // 2
    #Convert encoded integer into a binary string padding with leading zeros to match the width
    binary_string = format(encoded, f'0{num_edges}b')
    #Iterate through pairs of nodes (i, j) where i < j, in upper-triangle row order
    # Add edge (i, j) to the edge list if the corresponding bit is 1
    edges = []
    index = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if binary_string[index] == '1':
                edges.append((i, j))
            index += 1

    #Return the list of edges
    return edges