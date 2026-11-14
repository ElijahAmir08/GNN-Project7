"""Verify shared-edge gluing and matched candidates across both modules."""

import unittest

from pyrigi import Graph

from src.chain_graphs import glue_laman_graphs, cross_module_candidates, circuit_module_support
from src.laman_diagnostics import candidate_features, find_matched_pair
from src.laman_loader import decode_laman_integer
from src.rigidity_labels import label_proposed_edge, make_graph


class ChainGraphTests(unittest.TestCase):
    def test_two_triangles_form_four_vertex_laman_graph(self):
        triangle = [(0, 1), (0, 2), (1, 2)]
        graph = glue_laman_graphs(3, triangle, 3, triangle, (0, 1), (0, 1))
        self.assertEqual(graph["num_nodes"], 4)
        self.assertEqual(len(graph["edges"]), 5)
        self.assertEqual(cross_module_candidates(graph), [(2, 3)])
        rec = label_proposed_edge(4, graph["edges"], (2, 3))
        self.assertEqual(len(rec["circuit_edges"]), 6)

    def test_dataset_modules_overlap_only_on_chosen_edge(self):
        left = decode_laman_integer(6672166, 8)
        right = decode_laman_integer(20486935, 8)
        graph = glue_laman_graphs(8, left, 8, right, (2, 7), (3, 5))
        self.assertEqual(graph["num_nodes"], 14)
        self.assertEqual(len(graph["edges"]), 25)
        self.assertEqual(graph["edges"], sorted(set(graph["edges"])))
        self.assertEqual(set(graph["module_vertices"][0]) & set(graph["module_vertices"][1]), {2, 7})
        self.assertEqual(set(graph["module_edges"][0]) & set(graph["module_edges"][1]), {(2, 7)})
        self.assertTrue(make_graph(14, graph["edges"]).is_Rd_independent(dim=2))

    def test_reject_missing_attachment_or_non_laman_module(self):
        triangle = [(0, 1), (0, 2), (1, 2)]
        with self.assertRaises(ValueError):
            glue_laman_graphs(3, triangle, 3, triangle, (0, 3), (0, 1))
        # K4 plus a pendant vertex has the right count but is not Laman.
        bad = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)]
        self.assertEqual(len(bad), 2 * 5 - 3)
        with self.assertRaises(ValueError):
            glue_laman_graphs(5, bad, 3, triangle, (0, 1), (0, 1))

    def test_cross_candidates_depend_on_shared_edge(self):
        module = decode_laman_integer(31, 4)
        graph = glue_laman_graphs(4, module, 4, module, (2, 3), (2, 3))
        base_without_seam = [edge for edge in graph["edges"] if edge != graph["shared_edge"]]
        for candidate in cross_module_candidates(graph):
            with self.subTest(candidate=candidate):
                before = label_proposed_edge(6, graph["edges"], candidate)
                self.assertIn(list(graph["shared_edge"]), before["circuit_edges"])
                after = label_proposed_edge(6, base_without_seam, candidate)
                self.assertEqual(after["label"], 0)

    def test_identical_modules_have_copy_exchange_before_deletion(self):
        module = decode_laman_integer(31, 4)
        graph = glue_laman_graphs(4, module, 4, module, (2, 3), (2, 3))
        swap = [4, 5, 2, 3, 0, 1]
        transformed = {tuple(sorted((swap[u], swap[v]))) for u, v in graph["edges"]}
        self.assertEqual(transformed, set(graph["edges"]))
        self.assertEqual({swap[0], swap[4]}, {0, 4})

    def test_both_labels_cross_modules_with_long_matched_dependency(self):
        left = decode_laman_integer(12896013502, 9)
        right = decode_laman_integer(6578871661, 9)
        graph = glue_laman_graphs(9, left, 9, right, (5, 6), (0, 4))
        candidates = cross_module_candidates(graph)
        # Candidate pools also accept reversed pairs and ignore existing edges.
        pool = [tuple(reversed(edge)) for edge in candidates + [graph["edges"][0]]]
        pair = find_matched_pair(16, graph["edges"], min_circuit_size=16,
                                 candidate_edges=pool, independent_candidates=pool,
                                 min_distance=5)
        self.assertIsNotNone(pair)
        removed, independent, redundant = pair
        self.assertEqual((independent["label"], redundant["label"]), (0, 1))
        self.assertIsNone(independent["circuit_edges"])
        base = list(map(tuple, redundant["existing_edges"]))
        vectors = []
        for rec in (independent, redundant):
            self.assertIn(tuple(rec["candidate_edge"]), candidates)
            self.assertEqual(rec["existing_edges"], redundant["existing_edges"])
            vectors.append(candidate_features(16, base, rec["candidate_edge"]))
        self.assertEqual(vectors[0], vectors[1])
        self.assertEqual(vectors[0][-1], 5)
        circuit = set(map(tuple, redundant["circuit_edges"]))
        self.assertGreaterEqual(len(circuit), 16)
        self.assertNotIn(removed, circuit)
        self.assertGreater(len(set(base) - circuit), 0)
        self.assertTrue(all(count > 0 for count in circuit_module_support(graph, circuit)))
        self.assertTrue(Graph(list(circuit)).is_Rd_circuit(dim=2))
        for module_edges in graph["module_edges"]:
            on_circuit = sorted((set(module_edges) & circuit) - {graph["shared_edge"]})[0]
            reduced = [edge for edge in base if edge != on_circuit]
            self.assertEqual(label_proposed_edge(16, reduced, tuple(redundant["candidate_edge"]))["label"], 0)


if __name__ == "__main__":
    unittest.main()
