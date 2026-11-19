import pytest
import torch

from scripts.wl_diagnostic import wl_signature
from src.data import NUM_NODE_FEATURES, record_to_data
from src.models import GRAPH_MODELS
from src.oracle import is_redundant
from src.rigidity_labels import make_graph
from src.wl_family import hinged, pair_records, wheel


@pytest.mark.parametrize("k", range(3, 11))
def test_family_is_1wl_identical_with_opposite_labels(k):
    (n1, e1, c1), (n2, e2, c2) = hinged(k), wheel(k)
    assert n1 == n2 == 2 * k + 1 and len(e1) == len(e2) == 4 * k
    assert wl_signature(n1, [list(e) for e in e1], list(c1)) == wl_signature(n2, [list(e) for e in e2], list(c2))
    assert not is_redundant(n1, e1, c1)          # hinge: independent
    assert is_redundant(n2, e2, c2)              # wheel is rigid: redundant
    assert make_graph(n2, e2).is_rigid(dim=2)    # PyRigi agrees the wheel is rigid
    assert not make_graph(n1, e1).is_rigid(dim=2)


@pytest.mark.parametrize("name", ["gcn", "mpnn", "gin"])
def test_1wl_models_cannot_separate_the_pair(name):
    torch.manual_seed(0)
    model = GRAPH_MODELS[name](NUM_NODE_FEATURES, hidden=16, num_layers=4).eval()
    for k in (3, 5, 8):
        a, b = (record_to_data(r) for r in pair_records(k))
        out_a = model(a.x, a.edge_index, None, a.candidate)
        out_b = model(b.x, b.edge_index, None, b.candidate)
        assert torch.allclose(out_a, out_b, atol=1e-5)
