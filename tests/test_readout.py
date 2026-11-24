import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data import NUM_NODE_FEATURES, record_to_data
from src.models import GRAPH_MODELS


def rec(edges, cand, n):
    return {"graph_id": "t", "num_nodes": n, "edges": edges, "candidate_edge": cand, "label": 0, "circuit_edges": None}


@pytest.mark.parametrize("readout", ["pair", "mean"])
@pytest.mark.parametrize("name", list(GRAPH_MODELS))
def test_pair_readout_batches_and_relabels_consistently(name, readout):
    torch.manual_seed(0)
    model = GRAPH_MODELS[name](NUM_NODE_FEATURES, hidden=16, num_layers=2, readout=readout).eval()
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


def test_mean_readout_global_term_does_not_grow_with_copies():
    """Attaching more and more disjoint (unmarked) triangles makes the sum
    readout's global term grow without bound, while the mean readout's global
    term stays bounded. (Mean pooling is size-normalized, not invariant: the
    added nodes still change the average.)"""
    from torch_geometric.nn import global_add_pool, global_mean_pool
    torch.manual_seed(0)
    model = GRAPH_MODELS["gin"](NUM_NODE_FEATURES, hidden=16, num_layers=2).eval()
    base = [[0, 1], [0, 2], [1, 2], [2, 3]]
    norms = {"sum": [], "mean": []}
    for copies in (1, 4, 16):
        edges = list(base)
        for c in range(copies):
            o = 4 + 3 * c
            edges += [[o, o + 1], [o, o + 2], [o + 1, o + 2]]
        d = record_to_data(rec(edges, [0, 3], 4 + 3 * copies))
        batch = torch.zeros(d.num_nodes, dtype=torch.long)
        with torch.no_grad():
            h = model.node_embeddings(d.x, d.edge_index, batch)
        norms["sum"].append(global_add_pool(h, batch).norm().item())
        norms["mean"].append(global_mean_pool(h, batch).norm().item())
    assert norms["sum"][2] > 3 * norms["sum"][0]                 # grows with size
    assert max(norms["mean"]) < 1.5 * min(norms["mean"])          # stays bounded


def test_aux_pairs_batch_with_correct_offsets_and_head_runs():
    torch.manual_seed(0)
    r1 = dict(rec([[0, 1], [1, 2]], [0, 2], 3), aux_pairs=[[0, 2]], aux_labels=[1])
    r2 = dict(rec([[0, 1], [1, 2], [2, 3]], [0, 3], 4), aux_pairs=[[0, 2], [1, 3]], aux_labels=[0, 1])
    batch = next(iter(DataLoader([record_to_data(r1), record_to_data(r2)], batch_size=2)))
    assert batch.aux_index.t().tolist() == [[0, 2], [3, 5], [4, 6]]      # second graph offset by 3 nodes
    assert batch.aux_y.tolist() == [1.0, 0.0, 1.0]
    model = GRAPH_MODELS["gin"](NUM_NODE_FEATURES, hidden=16, num_layers=2).eval()
    model.enable_aux(16)
    model(batch.x, batch.edge_index, batch.batch, batch.candidate)
    assert model.aux_logits(batch.aux_index).shape == (3,)
