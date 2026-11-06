import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from pathlib import Path

import sys
from typing import TextIO


def train(
    autoencoder:      nn.Module,
    unlabeled_loader: DataLoader,
    labeled_loader:   DataLoader,
    val_loader: DataLoader,
    optimizer:  torch.optim.Optimizer,
    num_epoch:  int,
    device:     torch.device | str,
    log_dir:    TextIO=sys.stdout
) -> nn.Module:
    # Move the autoencoder before constructing the optimizer, not inside train().
    device = torch.device(device)
    for tensor in (*autoencoder.parameters(), *autoencoder.buffers()):
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise ValueError("Move the autoencoder to device before creating the optimizer")

    COEF_CLASS = 1
    COEF_GEOM  = 0.1
    reconstruction_crit = nn.MSELoss().to(device)
    classification_crit = nn.CrossEntropyLoss().to(device)

    if any(len(loader.dataset) == 0 for loader in (unlabeled_loader, labeled_loader, val_loader)):
        raise ValueError("Training and validation datasets must not be empty")

    print(
        "Epoch,TrainLoss,ValLoss,TrainLabeledLoss,ValLabeledLoss,TrainGeometryLoss,ValGeometryLoss",
        file=log_dir,
        flush=True
    )

    for epoch in tqdm(range(num_epoch)):
        train_unlabeled_loss = train_reconstruction(
            autoencoder, unlabeled_loader, optimizer, reconstruction_crit, device
        )
        train_labeled_loss = train_classification(
            autoencoder, labeled_loader, optimizer,
            reconstruction_crit, classification_crit, device, COEF_CLASS
        )
        train_geometry_loss = train_geometry(
            autoencoder, labeled_loader, optimizer, device, COEF_GEOM
        )
        val_unlabeled_loss, val_labeled_loss, val_geometry_loss = validate(
            autoencoder, val_loader, reconstruction_crit, classification_crit, device, COEF_CLASS
        )

        print(
            f"{epoch + 1},{train_unlabeled_loss:.6f},{val_unlabeled_loss:.6f},"
            f"{train_labeled_loss:.6f},{val_labeled_loss:.6f},"
            f"{train_geometry_loss:.6f},{val_geometry_loss:.6f}",
            file=log_dir,
            flush=True
        )

    return autoencoder


def train_reconstruction(
    autoencoder: nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    reconstruction_crit: nn.Module,
    device: torch.device | str,
) -> float:
    autoencoder.train()
    total_loss = 0.0

    for batch_img in loader:
        batch_img = batch_img.to(device)
        optimizer.zero_grad()

        reconstructed = autoencoder(batch_img)
        batch_loss = reconstruction_crit(reconstructed, batch_img)
        total_loss += batch_loss.item() * batch_img.size(0)

        batch_loss.backward()
        optimizer.step()

    return total_loss / len(loader.dataset)


def train_classification(
    autoencoder: nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    reconstruction_crit: nn.Module,
    classification_crit: nn.Module,
    device: torch.device | str,
    COEF_CLASS: float,
) -> float:
    autoencoder.train()
    total_loss = 0.0

    for batch_img, label in loader:
        batch_img = batch_img.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        latent        = autoencoder.encoder(batch_img)
        pred_class    = autoencoder.classifier(latent)
        reconstructed = autoencoder.decoder(latent)

        classification_loss = classification_crit(pred_class, label)
        reconstruction_loss = reconstruction_crit(reconstructed, batch_img)
        batch_loss = reconstruction_loss + COEF_CLASS * classification_loss
        total_loss += batch_loss.item() * batch_img.size(0)

        batch_loss.backward()
        optimizer.step()

    return total_loss / len(loader.dataset)


def train_geometry(
    autoencoder: nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device | str,
    coef_geom:   float,
) -> float:
    autoencoder.train()
    optimizer.zero_grad()
    all_latents = []
    all_labels  = []

    for batch_img, label in loader:
        batch_img = batch_img.to(device)
        label = label.to(device)

        all_latents.append(autoencoder.encoder(batch_img))
        all_labels.append(label)

    all_latents = torch.cat(all_latents, dim=0)
    all_labels  = torch.cat(all_labels, dim=0)
    geometry_loss = compute_geometry_loss(all_latents, all_labels)

    if all_labels.unique().numel() >= 2:
        (coef_geom * geometry_loss).backward()
        optimizer.step()

    return geometry_loss.item()


def validate(
    autoencoder: nn.Module,
    loader:      DataLoader,
    reconstruction_crit: nn.Module,
    classification_crit: nn.Module,
    device: torch.device | str,
    COEF_CLASS: float,
) -> tuple[float, float, float]:
    autoencoder.eval()
    reconstruction_total = 0.0
    labeled_total = 0.0
    all_latents = []
    all_labels  = []

    with torch.inference_mode():
        for batch_img, label in loader:
            batch_img = batch_img.to(device)
            label = label.to(device)

            latent        = autoencoder.encoder(batch_img)
            pred_class    = autoencoder.classifier(latent)
            reconstructed = autoencoder.decoder(latent)
            all_latents.append(latent)
            all_labels.append(label)

            classification_loss = classification_crit(pred_class, label)
            reconstruction_loss = reconstruction_crit(reconstructed, batch_img)

            total_loss = reconstruction_loss + COEF_CLASS * classification_loss

            reconstruction_total += reconstruction_loss.item() * batch_img.size(0)
            labeled_total += total_loss.item() * batch_img.size(0)

        geometry_loss = compute_geometry_loss(
            torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)
        ).item()

    return (
        reconstruction_total / len(loader.dataset),
        labeled_total / len(loader.dataset),
        geometry_loss,
    )


def compute_geometry_loss(latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    digits = labels.unique()
    if len(digits) < 2:
        return latents.sum() * 0.0

    # Compute class centers
    centers = torch.stack([
        latents[labels == digit].mean(dim=0)
        for digit in digits
    ])

    # Mean variance within each class
    within_variance = torch.stack([
        ((latents[labels == digit] - centers[index]) ** 2).mean(dim=0).sum()
        for index, digit in enumerate(digits)
    ]).mean()

    # Variance of class centers
    global_center = centers.mean(dim=0)
    between_variance = ((centers - global_center) ** 2).mean(dim=0).sum()

    return within_variance / (between_variance + 1e-8)
