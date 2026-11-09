import unittest
from src.graph_rigidity import is_graph_rigid

class TestGraphRigidity(unittest.TestCase):
    def test_four_node_examples(self):
        # Four-node Examples
        square = [
            (0,1), (1,2), (2,3), (0,3)
        ]
        self.assertFalse(is_graph_rigid(4, square))
    def test_four_node_braced_square(self):
        braced_square = [
            (0,1), (1,2), (2,3), (0,3), (0,2)
        ]
        self.assertTrue(is_graph_rigid(4, braced_square))
    def test_four_node_crossed_square(self):
        crossed_square = [
            (0,1), (1,2), (2,3), (0,3), (0,2), (1,3)
        ]
        self.assertTrue(is_graph_rigid(4, crossed_square))

    # Five-node rigid example
    def test_five_node_examples(self):

        rigid_five = [
        (0,1), (0,2), (1,2), (0,3), (1,3), (0, 4), (2, 4)
        ]
        self.assertTrue(is_graph_rigid(5, rigid_five))
    # Five-node non-rigid example
    def test_five_node_non_rigid(self):
        non_rigid_five = [
            (0,1), (0,2), (0, 3), (1,2), (1,3), (2,3), (0,4)
        ]
        self.assertFalse(is_graph_rigid(5, non_rigid_five))

if __name__ == '__main__':
    unittest.main()