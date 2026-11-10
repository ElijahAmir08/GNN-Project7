import unittest
from src.dataset_generator import find_nonrigid_graphs
from src.graph_rigidity import is_graph_rigid

class DatasetGeneratorTests(unittest.TestCase):
    def test_find_nonrigid_graphs(self):
        rigid_five = [
            (0, 1), (1, 2), (2, 3),
            (0, 3), (0, 2),
            (0, 4), (2, 4)
        ]
        #Verify the starting graph is rigid
        self.assertTrue(is_graph_rigid(5, rigid_five))
        result = find_nonrigid_graphs(5, rigid_five)

        # A edge swap should be found that makes the graph nonrigid
        self.assertIsNotNone(result)

        # Both graphs should have the same number of edges
        self.assertEqual(len(result), len(rigid_five))
        # The resulting graph should be nonrigid
        self.assertFalse(is_graph_rigid(5, result))

if __name__ == "__main__":
    unittest.main()