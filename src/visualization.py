import csv
from pathlib import Path
from collections.abc import Sequence
import torch
import matplotlib.pyplot as plt


def read_train_log(path: str) -> dict:
    metrics = {"Epoch": [], "TrainLoss": [], "ValLoss": []}

    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        for train_key, val_key in (
            ("TrainLabeledLoss", "ValLabeledLoss"),
            ("TrainGeometryLoss", "ValGeometryLoss"),
        ):
            if train_key in columns and val_key in columns:
                metrics[train_key] = []
                metrics[val_key] = []

        for row in reader:
            metrics["Epoch"].append(int(row["Epoch"]))
            for key in metrics:
                if key != "Epoch":
                    metrics[key].append(float(row[key]))

    return metrics


def show_train_log_plot(metrics: dict, path: str | Path | None = None) -> None:
    plots = [
        ("TrainLoss", "ValLoss", "Reconstruction loss", "MSE"),
        ("TrainLabeledLoss", "ValLabeledLoss", "Reconstruction + classification loss", "MSE + CrossEntropy"),
        ("TrainGeometryLoss", "ValGeometryLoss", "Geometry loss", "Within / between variance"),
    ]
    plots = [plot for plot in plots if plot[0] in metrics and plot[1] in metrics]
    fig, axes = plt.subplots(len(plots), 1, figsize=(16, 7 * len(plots)),
                             squeeze=False, layout="constrained")

    for ax, (train_key, val_key, title, ylabel) in zip(axes.flat, plots):
        ax.plot(metrics["Epoch"], metrics[train_key], label="Train")
        ax.plot(metrics["Epoch"], metrics[val_key],   label="Validation")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if path is not None:
        fig.savefig(path, dpi=150)
    plt.show()


def show_mnist_img(
    objs: Sequence[tuple[torch.Tensor, int | str]],
    path:  str | Path | None = None,
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
    path: str | Path | None = None,
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

# def show_embeddings(
#     embeddings: torch.Tensor,
#     labels:     torch.Tensor,
#     path: str | Path | None = None,
#     centers: torch.Tensor | None = None,
# ) -> None:
#     embeddings = embeddings.detach().cpu()
#     labels = labels.detach().cpu()
#     fig, ax = plt.subplots(figsize=(16, 16), layout="constrained")

#     colors = plt.get_cmap("tab10")
#     for digit in range(10):
#         points = embeddings[labels == digit]
#         ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.5,
#                    color=colors(digit), label=str(digit))
#     if centers is not None:
#         centers = centers.detach().cpu()
#         clusters = torch.cdist(embeddings, centers).argmin(dim=1)
#         lower = embeddings.min(dim=0).values
#         upper = embeddings.max(dim=0).values
#         padding = (upper - lower).clamp_min(1.0) * 0.05
#         x = torch.linspace(lower[0] - padding[0], upper[0] + padding[0], 200,
#                            dtype=embeddings.dtype)
#         y = torch.linspace(lower[1] - padding[1], upper[1] + padding[1], 200,
#                            dtype=embeddings.dtype)
#         grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
#         grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
#         regions = torch.cdist(grid_points, centers).argmin(dim=1).reshape(grid_x.shape)

#         # Print clasters with bounds
#         for cluster in range(len(centers)):
#             mask = regions == cluster
#             if mask.any() and not mask.all():
#                 ax.contour(x, y, mask.float(), levels=[0.5], colors="black",
#                            linewidths=0.8, linestyles="--")
#         ax.plot([], [], color="black", linestyle="--", label="Cluster boundary")
#         ax.scatter(centers[:, 0], centers[:, 1], color="black", s=35,
#                    label="Cluster center", zorder=3)

#         for cluster, center in enumerate(centers):
#             members = labels[clusters == cluster]
#             if len(members) > 0:
#                 digit = torch.bincount(members, minlength=10).argmax().item()
#                 name = f"K{digit}"
#             else:
#                 name = "K (empty)"
#             ax.annotate(name, (center[0].item(), center[1].item()),
#                         xytext=(5, 5), textcoords="offset points")

#     ax.legend(title="Digit / K-means", markerscale=2)
#     ax.set_title("Encoder embedding space")
#     ax.set_xlabel("Embedding 1")
#     ax.set_ylabel("Embedding 2")
#     ax.set_aspect("equal", adjustable="datalim")
#     ax.grid(True, alpha=0.3)

#     if path is not None:
#         fig.savefig(path, dpi=150)
#     plt.show()

