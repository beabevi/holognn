import random
from typing import Callable
import numpy as np
import torch

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score as _roc_auc_score
from torch_geometric.utils import negative_sampling


def tied_topk_indices(values, k, expansion=2):
    """
    >>> tied_topk_indices(torch.tensor([4,4,4,5,5]), 2, 2).sort()[0]
    tensor([3, 4])
    >>> tied_topk_indices(torch.tensor([4,1,4,5,5,1]), 3, 2).sort()[0]
    tensor([0, 2, 3, 4])
    """
    assert len(values) >= k * expansion

    values, indices = torch.topk(values, k * expansion)
    assert values[k - 1] != values[-1], (
        "Cannot break ties within expansion.\nTry a larger expansion value"
    )

    return indices[: k + ((values[k - 1] == values[k:]).sum())]


def init_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def rmse(pred, target):
    return F.mse_loss(pred, target).sqrt()


def binary_accuracy(pred, target):
    return ((torch.sigmoid(pred) > 0.5).float() == target).float().mean().item()


def multi_class_accuracy(pred, target):
    pred = pred.argmax(dim=-1)
    return (pred == target).float().mean().item()


def roc_auc_score(pred, target):
    return _roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())


class CooSampler:
    def __init__(self, coos, values, batch_size: int, shuffle: bool = False):
        assert coos.size(1) == values.size(0)
        self.coos = coos
        self.values = values
        self.shuffle = shuffle
        n_groups, rem = divmod(self.values.size(0), batch_size)

        self.batch_sizes = [batch_size] * n_groups
        if rem > 0:
            self.batch_sizes.append(rem)

    def __len__(self):
        return len(self.batch_sizes)

    def __iter__(self):
        size = self.values.size(0)
        perm_coos = self.coos
        perm_values = self.values
        if self.shuffle:
            perm = torch.randperm(size)
            perm_coos = self.coos[:, perm]
            perm_values = self.values[perm]

        return iter(
            [
                (coos, values)
                for (coos, values) in zip(
                    torch.split(perm_coos, self.batch_sizes, dim=1),
                    torch.split(perm_values, self.batch_sizes, dim=0),
                )
            ]
        )


def get_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sample_negatives: bool = False,
):
    def get_negatives(train_data, pos_y_coos, pos_y_values):
        neg_y_coos = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=pos_y_coos.size(-1),
            method="sparse",
        )

        y_coos = torch.cat(
            [pos_y_coos, neg_y_coos],
            dim=-1,
        )
        y_values = torch.cat([pos_y_values, torch.zeros_like(pos_y_values)], dim=0)
        return y_coos, y_values

    def fun(train_data, adj, y_dl) -> float:
        model.train()
        optimizer.zero_grad()
        tot_loss = 0
        for y_coos, y_values in y_dl:
            if sample_negatives:
                y_coos, y_values = get_negatives(train_data, y_coos, y_values)

            out = model(train_data, y_coos, adj)
            loss = loss_fn(out, y_values)
            loss.backward()
            tot_loss += float(loss)
        optimizer.step()
        return tot_loss

    return fun
