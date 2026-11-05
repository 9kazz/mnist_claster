import csv
from collections.abc import Sequence
import torch
import matplotlib.pyplot as plt


def read_train_log(path: str) -> dict:
    metrics = {
        "Epoch":            [],
        "TrainLoss":        [],
        "ValLoss":          [],
        "TrainLabeledLoss": [],
        "ValLabeledLoss" :  [],
    }

    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            metrics["Epoch"].append(int(row["Epoch"]))
            for key in metrics:
                if key != "Epoch":
                    metrics[key].append(float(row[key]))

    return metrics


def show_train_log_plot(metrics: dict, path: str | None = None) -> None:
    nrows = 2
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 7 * nrows),
                             squeeze=False, layout="constrained")

    ax = axes[0, 0]
    ax.plot(metrics["Epoch"], metrics["TrainLoss"], label="Train (all images)")
    ax.plot(metrics["Epoch"], metrics["ValLoss"],   label="Validation")
    ax.set_title("Reconstruction loss")
    ax.set_ylabel("MSE")

    ax = axes[1, 0]
    ax.plot(metrics["Epoch"], metrics["TrainLabeledLoss"], label="Train (labeled subset)")
    ax.plot(metrics["Epoch"], metrics["ValLabeledLoss"],   label="Validation")
    ax.set_title("Reconstruction + classification loss")
    ax.set_ylabel("MSE + CrossEntropy")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if path is not None:
        fig.savefig(path, dpi=150)
    plt.show()


def show_mnist_img(
    objs: Sequence[tuple[torch.Tensor, int | str]],
    path:  str | None = None,
    title: str = "MNIST Dataset",
) -> None:
    objs_cnt = len(objs)

    if objs_cnt == 0:
        return

    MAX_IMG_IN_LINE = 4 # Keep original/reconstruction pairs on the same row
    ncols = min(objs_cnt, MAX_IMG_IN_LINE)
    nrows = (objs_cnt - 1) // ncols + 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze = False,
        # for nice output
        figsize = (2 * ncols, 2 * nrows),
        layout  = "constrained",
    )
    fig.suptitle(title)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= objs_cnt:
            continue

        img, label = objs[idx]

        ax.imshow(img.detach().cpu().reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        if isinstance(label, torch.Tensor):
            label = label.item()
        ax.set_title(label if isinstance(label, str) else f"Label: {label}")

    if path is not None:
        fig.savefig(path, dpi=150)
        
    plt.show()


def show_embeddings(
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    path: str | None = None,
) -> None:
    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()
    fig, ax = plt.subplots(figsize=(16, 16), layout="constrained")

    colors = plt.get_cmap("tab10")
    for digit in range(10):
        points = embeddings[labels == digit]
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.5,
                   color=colors(digit), label=str(digit))
    ax.legend(title="Digit", markerscale=3)
    ax.set_title("Encoder embedding space")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    if path is not None:
        fig.savefig(path, dpi=150)
    plt.show()
