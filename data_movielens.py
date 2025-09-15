import functools
import itertools as it
import operator
import random
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from utils import binary_accuracy, rmse


def get_adj(data):
    n_users, n_movies = data.x_dict["user"].size(0), data.x_dict["movie"].size(0)
    user2movies = data.edge_index_dict[("user", "rates", "movie")].clone()
    user2movies[1] = user2movies[1] + n_users
    movies2user = data.edge_index_dict[("movie", "rev_rates", "user")].clone()
    movies2user[0] = movies2user[0] + n_users
    edge_index = torch.hstack((user2movies, movies2user))
    adj_t = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(n_users + n_movies, n_users + n_movies),
    )

    # Pre-compute GCN normalization.
    return gcn_norm(adj_t, add_self_loops=True)


def user_movie(data):
    # Perform a link-level split into training, validation, and test edges:
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[("user", "rates", "movie")],
        rev_edge_types=[("movie", "rev_rates", "user")],
    )(data)

    @functools.wraps(rmse)
    def clamped_rmse(pred, target):
        return rmse(pred.clamp(min=0, max=5), target)

    return (
        (
            (
                train_data,
                torch.sparse_coo_tensor(
                    indices=train_data["user", "movie"].edge_label_index,
                    values=train_data["user", "movie"].edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
            (
                val_data,
                torch.sparse_coo_tensor(
                    indices=val_data["user", "movie"].edge_label_index,
                    values=val_data["user", "movie"].edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
            (
                test_data,
                torch.sparse_coo_tensor(
                    indices=test_data["user", "movie"].edge_label_index,
                    values=test_data["user", "movie"].edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
        ),
        ("user", "movie"),
        (F.mse_loss, clamped_rmse, float("inf"), operator.lt),
    )


def movie_movie(data):
    # Generate the co-occurence matrix of movies<>movies:
    metapath = [("movie", "rev_rates", "user"), ("user", "rates", "movie")]
    data = T.AddMetaPaths(metapaths=[metapath])(data)

    # Apply normalization to filter the metapath:
    _, edge_weight = gcn_norm(
        data["movie", "movie"].edge_index,
        num_nodes=data["movie"].num_nodes,
        add_self_loops=False,
    )  # type: ignore
    edge_index = data["movie", "movie"].edge_index[:, edge_weight > 0.002]

    data["movie", "metapath_0", "movie"].edge_index = edge_index

    # Perform a link-level split into training, validation, and test edges:
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
        edge_types=[("movie", "metapath_0", "movie")],
    )(data)
    return (
        (
            (
                train_data,
                torch.sparse_coo_tensor(
                    indices=train_data["movie", "movie"].edge_label_index,
                    values=train_data["movie", "movie"].edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
            (
                val_data,
                torch.sparse_coo_tensor(
                    indices=val_data["movie", "movie"].edge_label_index,
                    values=val_data["movie", "movie"].edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
            (
                test_data,
                torch.sparse_coo_tensor(
                    indices=test_data["movie", "movie"].edge_label_index,
                    values=test_data["movie", "movie"].edge_label.unsqueeze(-1),
                ).coalesce(),
            ),
        ),
        ("movie", "movie"),
        (
            F.binary_cross_entropy_with_logits,
            binary_accuracy,
            -float("inf"),
            operator.gt,
        ),
    )


def split_indices(size, val_ratio, test_ratio):
    train_ratio = 1.0 - val_ratio - test_ratio
    perm = torch.randperm(size)
    sizes = [int(train_ratio * size), int(val_ratio * size)]
    sizes.append(size - sum(sizes))
    return perm, sizes


def user_movie_user(data):
    dev = data["user"].x.device
    data = data.to("cpu")

    movie2users = defaultdict(set)
    for movie, user in zip(*data["movie", "rev_rates", "user"].edge_index):
        movie2users[movie.item()].add(user.item())

    population = set(range(data["user"].x.size(0)))
    rel_coos = []
    neg_rel_coos = []
    for user, movie in zip(*data["user", "rates", "movie"].edge_index):
        if (watched_by_users := movie2users.get(movie.item())) is not None:
            rel_coos.append(
                torch.vstack(
                    (
                        torch.tensor([user]).repeat(len(watched_by_users)),
                        torch.tensor([movie]).repeat(len(watched_by_users)),
                        torch.tensor(list(watched_by_users)),
                    )
                )
            )
            remaining = sorted(population - watched_by_users)
            anti_users = random.sample(
                remaining, min(len(watched_by_users), len(remaining))
            )
            neg_rel_coos.append(
                torch.vstack(
                    (
                        torch.tensor([user]).repeat(len(anti_users)),
                        torch.tensor([movie]).repeat(len(anti_users)),
                        torch.tensor(anti_users),
                    )
                )
            )

    rel_coos = torch.hstack(rel_coos)
    rel_perm, sizes = split_indices(rel_coos.size(1), val_ratio=0.1, test_ratio=0.1)
    rel_coos = torch.split(rel_coos[:, rel_perm], sizes, dim=1)

    neg_rel_coos = torch.hstack(neg_rel_coos)
    neg_rel_perm, sizes = split_indices(
        neg_rel_coos.size(1), val_ratio=0.1, test_ratio=0.1
    )
    neg_rel_coos = torch.split(neg_rel_coos[:, neg_rel_perm], sizes, dim=1)

    results = []
    for split_rel_coos, split_neg_rel_coos in zip(rel_coos, neg_rel_coos):
        rel_ys = torch.tensor(1.0).repeat(split_rel_coos.size(1))
        neg_rel_ys = torch.tensor(0.0).repeat(split_neg_rel_coos.size(1))
        results.append(
            torch.sparse_coo_tensor(
                indices=torch.hstack((split_rel_coos, split_neg_rel_coos)),
                values=torch.hstack((rel_ys, neg_rel_ys)).unsqueeze(-1),
            )
            .coalesce()
            .to(device=dev)
        )

    return (
        tuple(zip(it.repeat(data.to(dev)), results)),
        ("user", "movie", "user"),
        (
            F.binary_cross_entropy_with_logits,
            binary_accuracy,
            -float("inf"),
            operator.gt,
        ),
    )


def user_movie_movie(data):
    dev = data["user"].x.device
    data = data.to("cpu")

    user2movie = defaultdict(set)
    for user, movie in zip(*data["user", "rates", "movie"].edge_index):
        user2movie[user.item()].add(movie.item())

    population = set(range(data["movie"].x.size(0)))
    rel_coos = []
    neg_rel_coos = []
    for user in user2movie.keys():
        if (watched := user2movie.get(user)) is not None:
            moviemovie = list(zip(*combinations(watched, 2)))
            rel_coos.append(
                torch.vstack(
                    (
                        torch.tensor([user]).repeat(len(moviemovie[0])),
                        torch.tensor([moviemovie[0]]),
                        torch.tensor(moviemovie[1]),
                    )
                )
            )

            remaining = sorted(population - watched)
            anti_movie = random.choices(remaining, k=len(moviemovie[0]))
            neg_rel_coos.append(
                torch.vstack(
                    (
                        torch.tensor([user]).repeat(len(anti_movie)),
                        torch.tensor([moviemovie[0]]),
                        torch.tensor(anti_movie),
                    )
                )
            )

    rel_coos = torch.hstack(rel_coos)
    rel_perm, sizes = split_indices(rel_coos.size(1), val_ratio=0.1, test_ratio=0.1)
    rel_coos = torch.split(rel_coos[:, rel_perm], sizes, dim=1)

    neg_rel_coos = torch.hstack(neg_rel_coos)
    neg_rel_perm, sizes = split_indices(
        neg_rel_coos.size(1), val_ratio=0.1, test_ratio=0.1
    )
    neg_rel_coos = torch.split(neg_rel_coos[:, neg_rel_perm], sizes, dim=1)

    results = []
    for idx, (split_rel_coos, split_neg_rel_coos) in enumerate(
        zip(rel_coos, neg_rel_coos)
    ):
        split_rel_coos = split_rel_coos[:, : split_rel_coos.size(1) // 4]
        split_neg_rel_coos = split_neg_rel_coos[:, : split_neg_rel_coos.size(1) // 4]

        rel_ys = torch.tensor(1.0).repeat(split_rel_coos.size(1))
        neg_rel_ys = torch.tensor(0.0).repeat(split_neg_rel_coos.size(1))
        results.append(
            torch.sparse_coo_tensor(
                indices=torch.hstack((split_rel_coos, split_neg_rel_coos)),
                values=torch.hstack((rel_ys, neg_rel_ys)).unsqueeze(-1),
            )
            .coalesce()
            .to(device=dev)
        )

    return (
        tuple(zip(it.repeat(data.to(dev)), results)),
        ("user", "movie", "movie"),
        (
            F.binary_cross_entropy_with_logits,
            binary_accuracy,
            -float("inf"),
            operator.gt,
        ),
    )


def movielens_split_data(data, task: str):
    return {
        "user_movie": user_movie,
        "movie_movie": movie_movie,
        "user_movie_user": user_movie_user,
        "user_movie_movie": user_movie_movie,
    }[task](data)
