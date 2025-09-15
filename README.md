This repository contains the official code of the paper **[Holographic Node Representations: Pre-training Task-Agnostic Node Embeddings](https://openreview.net/forum?id=tGYFikNONB) (ICLR 2025)**.

<p align="center">
<img src=./holo.png>
</p>

## Setup

1.  **Create Environment**: First, create and activate a Conda environment using the provided commands.

    ```bash
    conda create --prefix ./env python=3.10
    conda activate ./env
    ```

2.  **Install Packages**: Install all the necessary packages from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to install PyTorch and the PyG stack separately according to your specific CUDA version. Please see the official PyTorch Geometric installation instructions if you encounter issues.)*

## Reproducing Results

To reproduce the results reported in the paper, it is necessary to perform hyperparameter tuning. The details of the experimental setup and the hyperparameter grids for each dataset can be found in **Appendix E** of the paper.

## Usage

The experiments are organized into two distinct entry points:
* `main_movielens.py`: For all tasks related to the MovieLens dataset.
* `main_planetoid.py`: For all tasks related to the Planetoid datasets (Cora, CiteSeer, PubMed).

The intended workflow is to first pre-train a model on one task and then use the saved model checkpoint to adapt to other tasks within the same dataset.

### Step 1: Pre-training on a Task

Run one of the main scripts, specifying the dataset and the task you want to pre-train on. The script will use `wandb` to log results and save the best model checkpoint to a `wandb` run directory.

**Planetoid Example (pre-training on node prediction for Cora):**
```bash
python main_planetoid.py --dataset_name cora --task node_pred --model holo
```

**MovieLens Example (pre-training on the movie-movie link task):**
```bash
python main_movielens.py --task movie_movie --model holo
```

### Step 2: Running with a Checkpoint
After the pre-training run is complete, copy the path to the saved checkpoint (`.pt` file, under `model_checkpoint_path` in the wandb run) from the wandb run directory. Use this path with the `--pretrained_path` argument to adapt the model to a different task. 

The performance of these transfer experiments *after hyperparameter tuning for both the pre-training and the adaptation phase (Appendix E)* can be compared to the results in the paper.

**Planetoid Example (using the checkpoint for link prediction):**

This command evaluates the model pre-trained on node classification on the link prediction task. See Table 4 in the paper for the full Planetoid results obtained with hyperparameter tuning.

```bash
python main_planetoid.py --dataset_name cora --task link_pred --model holo --pretrained_path <path_to_your_wandb_checkpoint.pt>
```

**MovieLens Example (using the checkpoint for the user-movie-user task):**

This command evaluates the model pre-trained on movie_movie on the user_movie_user task. See Table 3 in the paper for the full MovieLens results obtained with hyperparameter tuning.

```bash
python main_movielens.py --task user_movie_user --model holo --pretrained_path <path_to_your_wandb_checkpoint.pt>
```

## Citation

If you find this work useful in your research, please cite our paper:

```
@inproceedings{bevilacqua2025holographic,
title={Holographic Node Representations: Pre-training Task-Agnostic Node Embeddings},
author={Beatrice Bevilacqua and Joshua Robinson and Jure Leskovec and Bruno Ribeiro},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
}
```
