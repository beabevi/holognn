import operator

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import multi_class_accuracy, roc_auc_score


def node_pred(data):
    data = T.ToSparseTensor()(data)
    data.adj_t = gcn_norm(data.adj_t, add_self_loops=True)
    return (
        (
            (
                data,
                torch.sparse_coo_tensor(
                    indices=data.train_mask.nonzero().view(1, -1),
                    values=data.y[data.train_mask],
                ).coalesce(),
            ),
            (
                data,
                torch.sparse_coo_tensor(
                    indices=data.val_mask.nonzero().view(1, -1),
                    values=data.y[data.val_mask],
                ).coalesce(),
            ),
            (
                data,
                torch.sparse_coo_tensor(
                    indices=data.test_mask.nonzero().view(1, -1),
                    values=data.y[data.test_mask],
                ).coalesce(),
            ),
        ),
        (F.cross_entropy, multi_class_accuracy, -float("inf"), operator.gt),
    )


def _to_sparse(data):
    edge_label_index = data.edge_label_index
    edge_label = data.edge_label

    data.edge_label_index = None
    data.edge_label = None

    data = T.ToSparseTensor(remove_edge_index=False)(data)

    data.edge_label_index = edge_label_index
    data.edge_label = edge_label
    return data


def link_pred(data):
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
    )(data)

    train_data = _to_sparse(train_data)
    val_data = _to_sparse(val_data)
    test_data = _to_sparse(test_data)

    train_data.adj_t = gcn_norm(train_data.adj_t, add_self_loops=True)
    val_data.adj_t = gcn_norm(val_data.adj_t, add_self_loops=True)
    test_data.adj_t = gcn_norm(test_data.adj_t, add_self_loops=True)

    return (
        (
            (
                train_data,
                torch.sparse_coo_tensor(
                    indices=train_data.edge_label_index,
                    values=train_data.edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
            (
                val_data,
                torch.sparse_coo_tensor(
                    indices=val_data.edge_label_index,
                    values=val_data.edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
            (
                test_data,
                torch.sparse_coo_tensor(
                    indices=test_data.edge_label_index,
                    values=test_data.edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
        ),
        (F.binary_cross_entropy_with_logits, roc_auc_score, -float("inf"), operator.gt),
    )


def planetoid_split_data(data, task: str):
    return {
        "node_pred": node_pred,
        "link_pred": link_pred,
    }[task](data)
