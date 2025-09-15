from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Embedding, Linear
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GCNConv, SAGEConv
from utils import tied_topk_indices


class PowerMethod(torch.nn.Module):
    """Symmetry breaking model using the power method."""

    def __init__(self, k, out_dim):
        super().__init__()
        self.k = k
        self.out_dim = out_dim

    def forward(self, v0, adj_t):
        v = v0
        for _ in range(self.k):
            v = adj_t.matmul(v)
        return v


class SymmetryBreakingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels, hidden_channels, normalize=False
        )  # Note: This assumes the adj matrix is normalized
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.out_dim = hidden_channels

    def forward(self, v0, adj_t):
        x = self.conv1(v0, adj_t).relu()
        return self.conv2(x, adj_t)


class ProductTupleEncoder(torch.nn.Module):
    """A baseline tuple encoder that takes the element-wise product of node embeddings."""

    def __init__(self):
        super().__init__()

    def forward(self, X, adj_t, tuples_coo, **kwargs):
        return X[tuples_coo].prod(dim=0)


class Classifier(torch.nn.Module):
    """A wrapper module combining a data encoder, tuple encoder, and MLP head."""

    def __init__(
        self,
        data_encoder,
        tuple_encoder,
        in_dim,
        out_dim,
        linear_classifier,
        train_head_only=False,
    ):
        super().__init__()
        self.data_encoder = data_encoder
        self.tuple_encoder = tuple_encoder
        self.mlp = (
            MLP([in_dim, in_dim, out_dim], norm=None, dropout=0.0)
            if not linear_classifier
            else Linear(in_dim, out_dim)
        )
        self.train_head_only = train_head_only

    def forward(self, data, tuples_coo, adj_t):
        with torch.set_grad_enabled(not self.train_head_only and self.training):
            X, tuples_coo = self.data_encoder(data, tuples_coo)
            X = self.tuple_encoder(X, adj_t, tuples_coo)
        return self.mlp(X)


class Holo(torch.nn.Module):
    """The Holo-GNN tuple encoder using symmetry breaking."""

    def __init__(self, *, n_breakings: int, symmetry_breaking_model):
        super().__init__()
        self.n_breakings = n_breakings
        self.symmetry_breaking_model = symmetry_breaking_model

        self.ln = torch.nn.LayerNorm(symmetry_breaking_model.out_dim)

    def get_nodes_to_break(self, adj_t, n_breakings=8):
        node_degrees = adj_t.sum(-1)
        return tied_topk_indices(node_degrees, k=n_breakings)

    def forward(self, X, adj_t, tuples_coo, group_idx=None):
        break_node_indices = self.get_nodes_to_break(
            adj_t, n_breakings=self.n_breakings
        )

        one_hot_breakings = F.one_hot(break_node_indices, X.size(0)).unsqueeze(-1)
        holo_repr = self.symmetry_breaking_model(
            torch.cat(
                (
                    X.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),
                    one_hot_breakings,
                ),
                dim=-1,
            ),
            adj_t,
        )  # (t, n, f), where n includes both movies and users
        holo_repr = self.ln(holo_repr)

        set_of_link_repr = holo_repr[:, tuples_coo].prod(dim=1)  # (t, k, f)

        if group_idx is not None:
            link_repr = torch_scatter.scatter(
                set_of_link_repr, group_idx, 0, reduce="mean"
            )
        else:
            link_repr = set_of_link_repr.mean(0, keepdim=True)  # (l=1, k, f)

        return link_repr.transpose(0, 1).flatten(1, 2)  # (k, f*l)


# =================================================================================
# MovieLens-Specific Modules
# =================================================================================
class BipartiteSAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin1 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        user_x = self.conv1(
            (x_dict["movie"], x_dict["user"]),
            edge_index_dict[("movie", "rev_rates", "user")],
        ).relu()

        movie_x = self.conv2(
            (user_x, x_dict["movie"]),
            edge_index_dict[("user", "rates", "movie")],
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x),
            edge_index_dict[("movie", "rev_rates", "user")],
        ).relu()

        return {"user": self.lin1(user_x), "movie": self.lin2(movie_x)}


class SAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class MovieLensEncoder(torch.nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels,
        *,
        relation_schema: Sequence[Literal["user", "movie"]],
    ):
        super().__init__()
        self.user_emb = Embedding(1, hidden_channels)
        self.gnn_encoder = BipartiteSAGEEncoder(hidden_channels, out_channels)
        # self.gnn_encoder = to_hetero(
        #     SAGEEncoder(hidden_channels, out_channels), metadata
        # )
        self.relation_schema = relation_schema

    def forward(self, data, tuples_coo) -> tuple[Data, torch.Tensor]:
        assert len(self.relation_schema) == tuples_coo.size(0)
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        n_users = x_dict["user"].size(0)

        z_dict = {}
        x_dict["user"] = self.user_emb.weight.repeat((n_users, 1))
        z_dict = self.gnn_encoder(x_dict, edge_index_dict)
        X = torch.vstack((z_dict["user"], z_dict["movie"]))

        new_entity_index = []
        for entity, entity_index in zip(self.relation_schema, tuples_coo):
            offset = 0 if entity == "user" else n_users
            new_entity_index.append(entity_index + offset)

        return X, torch.vstack(new_entity_index)


# =================================================================================
# Planetoid-Specific Modules
# =================================================================================
class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        *,
        alpha=0.1,
        theta=0.5,
        shared_weights=True,
        dropout=0.0,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCNConv(
                    in_channels,
                    hidden_channels,
                    normalize=False,
                )
            )
            in_channels = hidden_channels

        self.dropout = dropout

    def forward(self, x, adj_t):
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, adj_t)
            x = x.relu()

        return F.dropout(x, self.dropout, training=self.training)


class PlanetoidEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.gnn_encoder = GCNEncoder(
            in_channels, hidden_channels, num_layers, dropout=dropout
        )

    def forward(self, data, tuples_coo) -> tuple[Data, torch.Tensor]:
        x, adj_t = data.x, data.adj_t
        X = self.gnn_encoder(x, adj_t)
        return X, tuples_coo
