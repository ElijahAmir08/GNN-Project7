"""Model hierarchy for edge-level rigidity classification.

Expressivity ladder (all permutation-equivariant, all explainer-compatible):

  rung  name       expressivity              notes
  0     features   none (no message passing) shortcut check on hand-made features
  1     gcn        < 1-WL                    normalized mean aggregation cannot count
  2     mpnn       = 1-WL (upper bound)      Gilmer-style: MLP messages + GRU update
  3     gin        = 1-WL                    simplest injective sum aggregation
  4     dsgin      > 1-WL                    node-marked subgraph GNN (DS-GNN):
                                             one GIN run per marked node, averaged

Shared candidate readout: [h_u + h_v, h_u * h_v, sum-pool(h)], invariant to
swapping u and v and to node relabelings.

Every graph model has forward(x, edge_index, batch, candidate) and reads the
graph only through MessagePassing layers, so PyG explainer edge masks reach
every place the graph structure enters the model. No random features or
Laplacian eigenvectors are used: they break automorphism equivariance, which
would make the symmetry analysis meaningless.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, MessagePassing, global_add_pool
from torch_geometric.utils import degree


def mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class CandidateReadout(nn.Module):
    """Base class: subclasses implement node_embeddings -> [N, emb_dim]."""

    def __init__(self, emb_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.head = mlp([3 * emb_dim, hidden, 2], dropout=dropout)

    def node_embeddings(self, x, edge_index, batch):
        raise NotImplementedError

    def forward(self, x, edge_index, batch=None, candidate=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embeddings(x, edge_index, batch)
        hu, hv = h[candidate[:, 0]], h[candidate[:, 1]]
        g = global_add_pool(h, batch)
        return self.head(torch.cat([hu + hv, hu * hv, g], dim=-1))


# --------------------------------------------------------------------- rung 1
class EdgeGCN(CandidateReadout):
    def __init__(self, in_dim, hidden=64, num_layers=4, dropout=0.1):
        super().__init__(in_dim + num_layers * hidden, hidden, dropout)
        dims = [in_dim] + [hidden] * num_layers
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(num_layers))

    def node_embeddings(self, x, edge_index, batch):
        hs, h = [x], x
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(h, edge_index)))
            hs.append(h)
        return torch.cat(hs, dim=-1)


# --------------------------------------------------------------------- rung 2
class MessageMLPConv(MessagePassing):
    """m_ij = MLP([h_i, h_j]); aggregate by sum."""

    def __init__(self, dim):
        super().__init__(aggr="add")
        self.msg = mlp([2 * dim, dim, dim])

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j):
        return self.msg(torch.cat([h_i, h_j], dim=-1))


class EdgeMPNN(CandidateReadout):
    """Gilmer et al. (2017)-style MPNN without edge features: MLP messages,
    sum aggregation, GRU node update (weights shared across steps)."""

    def __init__(self, in_dim, hidden=64, num_layers=4, dropout=0.1):
        super().__init__(in_dim + num_layers * hidden, hidden, dropout)
        self.num_layers = num_layers
        self.encode = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(MessageMLPConv(hidden) for _ in range(num_layers))
        self.gru = nn.GRUCell(hidden, hidden)

    def node_embeddings(self, x, edge_index, batch):
        hs, h = [x], torch.relu(self.encode(x))
        for conv in self.convs:
            h = self.gru(conv(h, edge_index), h)
            hs.append(h)
        return torch.cat(hs, dim=-1)


# --------------------------------------------------------------------- rung 3
class EdgeGIN(CandidateReadout):
    def __init__(self, in_dim, hidden=64, num_layers=4, dropout=0.1):
        super().__init__(in_dim + num_layers * hidden, hidden, dropout)
        dims = [in_dim] + [hidden] * num_layers
        self.convs = nn.ModuleList(
            GINConv(mlp([dims[i], hidden, hidden]), train_eps=True) for i in range(num_layers)
        )

    def node_embeddings(self, x, edge_index, batch=None):
        hs, h = [x], x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            hs.append(h)
        return torch.cat(hs, dim=-1)


# --------------------------------------------------------------------- rung 4
class CopyGINConv(MessagePassing):
    """GIN layer on features of shape [N, C, d] (C = subgraph copies).
    node_dim=0 so explainer masks broadcast over copies and features."""

    def __init__(self, in_dim, hidden):
        super().__init__(aggr="add", node_dim=0)
        self.eps = nn.Parameter(torch.zeros(1))
        self.net = mlp([in_dim, hidden, hidden])

    def forward(self, h, edge_index, valid):
        z = (1 + self.eps) * h + self.propagate(edge_index, h=h)
        out = torch.zeros(*z.shape[:2], self.net[-1].out_features, device=z.device)
        out[valid] = self.net(z[valid])  # BatchNorm sees only real copies
        return out

    def message(self, h_j):
        return h_j


class EdgeDSGIN(CandidateReadout):
    """Node-marked subgraph GNN (DS-GNN, Bevilacqua et al. 2022).

    Copy c of a graph marks its c-th node; a shared GIN runs on all copies in
    parallel and node embeddings are averaged over copies. Strictly more
    expressive than 1-WL, still permutation-equivariant. Copies are stored as
    a tensor dimension (not replicated edges) so explainer masks align.
    A final ordinary GIN layer gives PGExplainer an [N, d] embedding.
    """

    def __init__(self, in_dim, hidden=64, num_layers=3, dropout=0.1):
        emb = in_dim + num_layers * hidden + hidden
        super().__init__(emb, hidden, dropout)
        dims = [in_dim + 1] + [hidden] * num_layers
        self.convs = nn.ModuleList(CopyGINConv(dims[i], hidden) for i in range(num_layers))
        self.final = GINConv(mlp([in_dim + num_layers * hidden, hidden, hidden]), train_eps=True)

    def node_embeddings(self, x, edge_index, batch):
        n = x.size(0)
        ptr = torch.zeros(int(batch.max()) + 2 if n else 1, dtype=torch.long, device=x.device)
        ptr[1:] = torch.bincount(batch, minlength=ptr.numel() - 1).cumsum(0)
        local = torch.arange(n, device=x.device) - ptr[batch]
        sizes = (ptr[1:] - ptr[:-1])[batch]                    # graph size per node
        c_max = int(sizes.max()) if n else 1
        copies = torch.arange(c_max, device=x.device)
        valid = copies[None, :] < sizes[:, None]               # [N, C]
        mark = (local[:, None] == copies[None, :]).float()     # [N, C]

        h = torch.cat([x[:, None, :].expand(n, c_max, x.size(1)), mark[..., None]], dim=-1)
        h = h * valid[..., None]
        hs = [x]
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, valid))
            pooled = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            hs.append(pooled)
        hcat = torch.cat(hs, dim=-1)
        return torch.cat([hcat, torch.relu(self.final(hcat, edge_index))], dim=-1)


# --------------------------------------------------------------------- rung 0
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


GRAPH_MODELS = {"gcn": EdgeGCN, "mpnn": EdgeMPNN, "gin": EdgeGIN, "dsgin": EdgeDSGIN}
ALL_MODELS = ["features", *GRAPH_MODELS]


def build_model(name: str, in_dim: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    if name == "features":
        return FeatureMLP(hidden, dropout)
    return GRAPH_MODELS[name](in_dim, hidden, layers, dropout)
