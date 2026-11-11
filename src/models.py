"""Models for edge-level rigidity classification.

EdgeGIN: GIN encoder over the base graph; the candidate pair (u, v) is read
out symmetrically as [h_u + h_v, h_u * h_v, sum-pooled graph embedding].
The readout is invariant to swapping u and v and to node relabelings.

forward(x, edge_index, batch, candidate) keeps x / edge_index as the first two
arguments so torch_geometric.explain.Explainer can wrap the model directly.

FeatureMLP: shortcut-check baseline on hand-made features only (no message
passing). If it matches the GIN, the dataset has a shortcut.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import degree


def mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class EdgeGIN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers):
            self.convs.append(GINConv(mlp([dim, hidden, hidden]), train_eps=True))
            dim = hidden
        # Jumping-knowledge concat of all layers keeps local + global info.
        jk = in_dim + num_layers * hidden
        self.head = mlp([3 * jk, hidden, 2], dropout=dropout)

    def node_embeddings(self, x, edge_index):
        hs = [x]
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            hs.append(h)
        return torch.cat(hs, dim=-1)

    def forward(self, x, edge_index, batch=None, candidate=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embeddings(x, edge_index)
        hu, hv = h[candidate[:, 0]], h[candidate[:, 1]]
        g = global_add_pool(h, batch)
        return self.head(torch.cat([hu + hv, hu * hv, g], dim=-1))


def shortcut_features(data) -> torch.Tensor:
    """Per-example: n, |E|, 2n-3-|E|, sorted endpoint degrees, endpoint
    common-neighbour count, shortest-path distance between endpoints."""
    feats = []
    graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
    for d in graphs:
        n, m = d.num_nodes, d.edge_index.size(1) // 2
        deg = degree(d.edge_index[0], n)
        u, v = d.candidate[0].tolist()
        adj = [set() for _ in range(n)]
        for a, b in d.edge_index.t().tolist():
            adj[a].add(b)
        common = len(adj[u] & adj[v])
        dist, frontier, seen = 0, {u}, {u}
        while frontier and v not in frontier:
            frontier = {w for f in frontier for w in adj[f]} - seen
            seen |= frontier
            dist += 1
        dist = dist if v in frontier else n
        du, dv = sorted([deg[u].item(), deg[v].item()])
        feats.append([n, m, 2 * n - 3 - m, du, dv, common, dist])
    return torch.tensor(feats, dtype=torch.float)


class FeatureMLP(nn.Module):
    NUM_FEATURES = 7

    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(self.NUM_FEATURES)
        self.net = mlp([self.NUM_FEATURES, hidden, hidden, 2], dropout=dropout)

    def forward(self, feats):
        return self.net(self.norm(feats))
