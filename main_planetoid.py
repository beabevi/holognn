from pathlib import Path
from typing import Literal

import torch
import torch_geometric.transforms as T
from data_planetoid import planetoid_split_data
from models import (
    Classifier,
    Holo,
    PlanetoidEncoder,
    PowerMethod,
    ProductTupleEncoder,
    SymmetryBreakingGNN,
)
from tap import Tap
from torch_geometric.datasets import Planetoid
from utils import (
    CooSampler,
    get_train_step,
    init_seed,
)

import wandb

TaskType = Literal["node_pred", "link_pred"]


class Config(Tap):
    task: TaskType = "node_pred"

    pretrained_path: Path | None = None

    model: Literal["gcn", "holo"] = "holo"
    hidden_channels: int = 64
    num_layers: int = 4
    linear_classifier: bool = False
    dropout: float = 0.0
    n_breakings: int = 8
    """This corresponds to the f_t in the paper"""
    symmetry_breaking_model: Literal["power_method", "gnn"] = "power_method"
    power: int = 2

    seed: int = 42
    n_epochs: int = 1500
    lr: float = 0.01
    batch_size: int = 2**20

    dataset_name: Literal["cora", "citeseer", "pubmed"] = "cora"


def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.task in ["node_pred", "link_pred"]:
        dataset = cfg.dataset_name
        path = Path.cwd() / f"data/{dataset}"
        transform = T.Compose([T.NormalizeFeatures()])
        dataset = Planetoid(path, dataset, transform=transform)
        data = dataset[0]

        data = data.to(device)

        data_encoder = PlanetoidEncoder(
            in_channels=dataset.num_features,
            hidden_channels=cfg.hidden_channels,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

        (
            ((train_data, train_y), (val_data, val_y), (test_data, test_y)),
            (loss_fn, metric_fn, init_metric, is_better),
        ) = planetoid_split_data(data, cfg.task)
        train_adj = train_data.adj_t
        val_adj = val_data.adj_t
        test_adj = test_data.adj_t

        out_dim = 1 if cfg.task == "link_pred" else dataset.num_classes
    else:
        raise ValueError(f"Unkown task {cfg.task}")

    if cfg.model == "gcn":
        tuple_encoder = ProductTupleEncoder()
        in_dim = cfg.hidden_channels
    elif cfg.model == "holo":
        in_dim = cfg.hidden_channels + 1
        if cfg.symmetry_breaking_model == "power_method":
            symmetry_breaking_model = PowerMethod(cfg.power, in_dim)
        elif cfg.symmetry_breaking_model == "gnn":
            symmetry_breaking_model = SymmetryBreakingGNN(in_dim, in_dim)
        else:
            raise ValueError(
                f"Unkown symmetry breaking model {cfg.symmetry_breaking_model}"
            )
        tuple_encoder = Holo(
            n_breakings=cfg.n_breakings,
            symmetry_breaking_model=symmetry_breaking_model,
        )
    else:
        raise NotImplementedError()
    model = Classifier(
        data_encoder,
        tuple_encoder,
        in_dim=in_dim,
        out_dim=out_dim,
        linear_classifier=cfg.linear_classifier,
        train_head_only=cfg.pretrained_path is not None,
    )
    print(model)
    model = model.to(device)

    test_y_dl = CooSampler(
        test_y.indices(), test_y.values(), batch_size=test_y._nnz(), shuffle=False
    )
    # Trigger first run to infer tensor shapes
    _ = model(test_data, next(iter(test_y_dl))[0], test_adj)

    if cfg.pretrained_path is not None:
        my_dict = torch.load(cfg.pretrained_path, map_location=device)

        mlp_keys = [key for key in my_dict if key.startswith("mlp")]
        for key in mlp_keys:
            my_dict.pop(key)
        load_out = model.load_state_dict(my_dict, strict=False)
        assert len(load_out.unexpected_keys) == 0 and all(
            [key.startswith("mlp") for key in load_out.missing_keys]
        )

    if cfg.pretrained_path is None and cfg.task == "node_pred":
        optimizer = torch.optim.Adam(
            [
                dict(
                    params=model.data_encoder.gnn_encoder.convs.parameters(),
                    weight_decay=5e-4,
                ),
                dict(params=model.tuple_encoder.parameters(), weight_decay=5e-4),
                dict(
                    params=model.mlp.parameters(),
                ),
            ],
            lr=cfg.lr,
        )
    else:
        parameters = (
            model.parameters()
            if cfg.pretrained_path is None
            else model.mlp.parameters()
        )
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr)

    train_step = get_train_step(
        model, optimizer, loss_fn, sample_negatives=cfg.task == "link_pred"
    )

    train_y_dl = CooSampler(
        train_y.indices(), train_y.values(), batch_size=cfg.batch_size, shuffle=True
    )
    val_y_dl = CooSampler(
        val_y.indices(), val_y.values(), batch_size=val_y._nnz(), shuffle=False
    )

    @torch.inference_mode()
    def test(data, adj, y_dl):
        model.eval()
        preds = []
        ys = []
        for y_coos, y_values in y_dl:
            out = model(data, y_coos, adj)
            preds.append(out)
            ys.append(y_values)
        return float(metric_fn(torch.vstack(preds), torch.vstack(ys)))

    best_val_metric, test_at_best_val = init_metric, init_metric
    for epoch in range(cfg.n_epochs):
        train_loss = train_step(train_data, train_adj, train_y_dl)
        train_loss /= len(train_y_dl)
        # train_metric = test(train_data, train_adj, train_y_dl)
        val_metric = test(val_data, val_adj, val_y_dl)
        test_metric = test(test_data, test_adj, test_y_dl)
        if is_better(val_metric, best_val_metric):
            best_val_metric = val_metric
            test_at_best_val = test_metric
            torch.save(model.state_dict(), wandb.config["model_checkpoint_path"])
        wandb.log(
            {
                f"Loss ({loss_fn.__name__})": train_loss,
                # f"Train ({metric_fn.__name__})": train_metric,
                f"Val ({metric_fn.__name__})": val_metric,
                f"Test ({metric_fn.__name__})": test_metric,
                f"Test @ Best Val ({metric_fn.__name__})": test_at_best_val,
            }
        )
        print(
            f"Epoch: {epoch:04d}, Loss: {train_loss:.4f}, "  # Train: {train_metric:.4f}, "
            f"Val: {val_metric:.4f}, Test: {test_metric:.4f} "
            f"Test @ Best Val {test_at_best_val:.4f}"
        )


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    cfg = Config().parse_args()
    init_seed(cfg.seed)

    dict_cfg = cfg.as_dict()
    wandb.init(project="holo", config=dict_cfg)
    if cfg.pretrained_path is not None:
        wandb.config.update({"pretrained_path_name": cfg.pretrained_path.name})
    wandb.config.update(
        {
            "model_checkpoint_path": str(
                (Path(wandb.run.dir) / f"{cfg.model}-{cfg.task}.pt").absolute()
            )
        }
    )

    # now = datetime.now()
    # ts = now.strftime("%m%d-%H:%M")
    # prefix = f"mod={cfg.model}-task={cfg.task}"
    # parent = Path("runs/")
    # parent.mkdir(parents=True, exist_ok=True)
    # run_dir = parent / f"{prefix}-{config_hash(cfg)}-{ts}"
    # run_dir.mkdir(parents=True)

    for k, v in dict_cfg.items():
        print(f"{k}: {v}")

    main(cfg)
