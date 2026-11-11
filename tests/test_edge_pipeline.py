import random

import torch
from pyrigi import Graph
from torch_geometric.loader import DataLoader

from src.data import NUM_NODE_FEATURES, directed_to_undirected, record_to_data
from src.edge_dataset import _base_task, assign_splits
from src.laman import random_laman_graph
from src.models import EdgeGIN
from src.rigidity_labels import label_proposed_edge


def test_henneberg_graphs_are_laman():
    rng = random.Random(0)
    for n in range(3, 11):
        for _ in range(5):
            edges = random_laman_graph(n, rng)
            g = Graph()
            g.add_vertices(list(range(n)))
            g.add_edges(edges)
            assert len(edges) == 2 * n - 3
            assert g.is_min_rigid(dim=2)


def test_circuit_found_when_circuit_does_not_span_all_vertices():
    # K4 on {0,1,2,3} minus one edge, plus a pendant-ish vertex 4 hanging off.
    base = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)]
    rec = label_proposed_edge(5, base, (2, 3))
    assert rec["label"] == 1
    assert sorted(map(tuple, rec["circuit_edges"])) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_base_task_balanced_and_circuits_contain_candidate():
    _, _, _, exs = _base_task((0, 8, 2, 123, 3))
    labels = [r["label"] for r in exs]
    assert labels.count(0) == labels.count(1)
    for r in exs:
        if r["label"] == 1:
            assert r["candidate_edge"] in r["circuit_edges"]
        else:
            assert r["circuit_edges"] is None


def test_isomorphic_bases_share_split():
    tri_a = [(0, 1), (0, 2), (1, 2)]
    tri_b = [(0, 1), (1, 2), (0, 2)]
    path = [(0, 1), (1, 2)]
    split = assign_splits([("a", 3, tri_a), ("b", 3, tri_b), ("c", 3, path)], (0.34, 0.33, 0.33), random.Random(0))
    assert split["a"] == split["b"]


def _record(edges, cand, n=5):
    return {"graph_id": "t", "num_nodes": n, "edges": edges, "candidate_edge": cand,
            "label": 1, "circuit_edges": [[0, 1], [1, 2]]}


def test_edge_alignment_contract():
    edges = [[0, 1], [0, 3], [1, 2], [2, 4]]
    d = record_to_data(_record(edges, [3, 4]))
    assert d.edge_index.size(1) == 2 * len(edges)
    for i, (u, v) in enumerate(edges):
        assert d.edge_index[:, 2 * i].tolist() == [u, v]
        assert d.edge_index[:, 2 * i + 1].tolist() == [v, u]
    assert d.circuit_mask.tolist() == [True, False, True, False]
    m = torch.arange(8, dtype=torch.float)
    assert directed_to_undirected(m).tolist() == [0.5, 2.5, 4.5, 6.5]
    assert d.x[:, 1].tolist() == [0, 0, 0, 1, 1]


def test_candidate_offsets_when_batched():
    d1 = record_to_data(_record([[0, 1]], [0, 2], n=3))
    d2 = record_to_data(_record([[0, 1]], [1, 2], n=4))
    batch = next(iter(DataLoader([d1, d2], batch_size=2)))
    assert batch.candidate.tolist() == [[0, 2], [4, 5]]


def test_gin_invariant_to_relabeling_and_endpoint_swap():
    torch.manual_seed(0)
    model = EdgeGIN(NUM_NODE_FEATURES, hidden=16, num_layers=3).eval()
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    cand = (0, 3)
    d = record_to_data(_record([list(e) for e in edges], list(cand)))
    out = model(d.x, d.edge_index, None, d.candidate)

    perm = [3, 0, 4, 1, 2]
    p_edges = sorted(tuple(sorted((perm[u], perm[v]))) for u, v in edges)
    p_cand = [perm[cand[1]], perm[cand[0]]]  # also swap endpoint order
    pd = record_to_data(_record([list(e) for e in p_edges], sorted(p_cand)))
    pd.candidate = torch.tensor([p_cand])
    out_p = model(pd.x, pd.edge_index, None, pd.candidate)
    assert torch.allclose(out, out_p, atol=1e-5)
