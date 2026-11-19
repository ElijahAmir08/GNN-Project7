import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data import NUM_NODE_FEATURES, record_to_data
from src.models import GRAPH_MODELS


def rec(edges, cand, n):
    return {"graph_id": "t", "num_nodes": n, "edges": edges, "candidate_edge": cand, "label": 0, "circuit_edges": None}


@pytest.mark.parametrize("name", list(GRAPH_MODELS))
def test_pair_readout_batches_and_relabels_consistently(name):
    torch.manual_seed(0)
    model = GRAPH_MODELS[name](NUM_NODE_FEATURES, hidden=16, num_layers=2, readout="pair").eval()
    d1 = record_to_data(rec([[0, 1], [0, 2], [1, 2], [2, 3]], [0, 3], 4))
    d2 = record_to_data(rec([[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]], [1, 3], 5))
    batch = next(iter(DataLoader([d1, d2], batch_size=2)))
    out_b = model(batch.x, batch.edge_index, batch.batch, batch.candidate)
    out_1 = torch.cat([model(d.x, d.edge_index, None, d.candidate) for d in (d1, d2)])
    assert torch.allclose(out_b, out_1, atol=1e-5)
    perm = [2, 0, 3, 1]
    p_edges = sorted(sorted((perm[u], perm[v])) for u, v in [[0, 1], [0, 2], [1, 2], [2, 3]])
    dp = record_to_data(rec(p_edges, sorted([perm[0], perm[3]]), 4))
    assert torch.allclose(model(dp.x, dp.edge_index, None, dp.candidate), out_1[:1], atol=1e-5)


def test_pair_readout_ignores_disconnected_structure():
    """With no whole-graph term, a component not connected to the candidate
    cannot change the output (sum readout does change it)."""
    torch.manual_seed(0)
    base = rec([[0, 1], [0, 2], [1, 2], [2, 3]], [0, 3], 4)
    extra = rec([[0, 1], [0, 2], [1, 2], [2, 3], [4, 5], [4, 6], [5, 6]], [0, 3], 7)
    for readout, should_match in (("pair", True), ("sum", False)):
        model = GRAPH_MODELS["gin"](NUM_NODE_FEATURES, hidden=16, num_layers=2, readout=readout).eval()
        a, b = (record_to_data(r) for r in (base, extra))
        same = torch.allclose(model(a.x, a.edge_index, None, a.candidate), model(b.x, b.edge_index, None, b.candidate), atol=1e-5)
        assert same == should_match
