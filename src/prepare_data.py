import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader, random_split

RANDOM_SEED = 42


def create_mnist_datasets(
    data_dir: str,
    val_set_size: int = 5000,
    mnist_set_size: int | None = None
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Prepare normalized datasets on CPU; training transfers individual batches."""

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
    )

    # Prepare tensors
    train_images = train_dataset.data.unsqueeze(1).float() / 255.0
    test_images  = test_dataset.data.unsqueeze(1).float()  / 255.0

    # None means use the entire MNIST training dataset
    if mnist_set_size is None:
        mnist_set_size = len(train_dataset)

    if not 0 < val_set_size < mnist_set_size <= len(train_dataset):
        raise ValueError("Invalid dataset sizes")

    # Split into train, validation and unused subsets
    train_divide, val_divide, _ = random_split(
        train_dataset,
        [
            mnist_set_size - val_set_size,
            val_set_size,
            len(train_dataset) - mnist_set_size,
        ],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_tensors = TensorDataset(
        train_images[train_divide.indices],
        train_dataset.targets[train_divide.indices],
    )

    val_tensors = TensorDataset(
        train_images[val_divide.indices],
        train_dataset.targets[val_divide.indices],
    )

    test_tensors = TensorDataset(
        test_images,
        test_dataset.targets,
    )

    return train_tensors, val_tensors, test_tensors


def make_embeddings_tensor(
        loader:  DataLoader,
        encoder: nn.Module,
        device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:

    device = torch.device(device)
    encoder.eval()

    with torch.inference_mode():
        embeddings = []
        labels     = []

        for batch_img, batch_labels in loader:
            batch_img = batch_img.to(device)

            batch_embeddings = encoder(batch_img)

            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        labels     = torch.cat(labels, dim=0)
    
    return embeddings, labels
