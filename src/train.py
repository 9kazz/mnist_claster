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

    # Move the autoencoder before constructing the optimizer, not inside train()
    device = torch.device(device)

    for tensor in (*autoencoder.parameters(), *autoencoder.buffers()):
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise ValueError("Move the autoencoder to device before creating the optimizer")
    
    reconstruction_crit = torch.nn.MSELoss().to(device)
    classification_crit = torch.nn.CrossEntropyLoss().to(device)

    if len(unlabeled_loader.dataset) == 0 or len(val_loader.dataset) == 0 or len(labeled_loader.dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty")
    
    print(
        "Epoch,TrainLoss,ValLoss,TrainLabeledLoss,ValLabeledLoss",
        file=log_dir,
        flush=True
    )

    for epoch in tqdm(range(num_epoch)):
        # Train dataset pass
        autoencoder.train()
        train_unlabeled_loss = 0.0
        train_labeled_loss   = 0.0

        # Reconstruction on all training images, labels are not used here
        for batch_img in unlabeled_loader:
            batch_img = batch_img.to(device)
            
            optimizer.zero_grad()

            reconstructed = autoencoder(batch_img)

            batch_loss = reconstruction_crit(reconstructed, batch_img)
            train_unlabeled_loss += batch_loss.item() * batch_img.size(0)

            batch_loss.backward()
            optimizer.step()

        # Extra reconstruction + classification pass on the labeled subset
        for batch_img, label in labeled_loader:
            batch_img = batch_img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()

            latent        = autoencoder.encoder(batch_img)
            pred_class    = autoencoder.classifier(latent)
            reconstructed = autoencoder.decoder(latent)

            classification_loss = classification_crit(pred_class, label)
            reconstruction_loss = reconstruction_crit(reconstructed, batch_img) 
            batch_loss = classification_loss + reconstruction_loss

            train_labeled_loss += batch_loss.item() * batch_img.size(0)

            batch_loss.backward()
            optimizer.step()

        train_unlabeled_loss /= len(unlabeled_loader.dataset) # mean loss of all dataset's elements
        train_labeled_loss   /= len(labeled_loader.dataset)

        # Validation dataset pass
        autoencoder.eval()
        val_unlabeled_loss = 0.0
        val_labeled_loss   = 0.0

        with torch.inference_mode():
            for batch_img, label in val_loader:
                batch_img = batch_img.to(device)
                label = label.to(device)

                latent        = autoencoder.encoder(batch_img)
                pred_class    = autoencoder.classifier(latent)
                reconstructed = autoencoder.decoder(latent)

                classification_loss = classification_crit(pred_class, label)
                reconstruction_loss = reconstruction_crit(reconstructed, batch_img) 
                total_loss = classification_loss + reconstruction_loss

                val_unlabeled_loss += reconstruction_loss.item() * batch_img.size(0)
                val_labeled_loss   += total_loss.item() * batch_img.size(0)

        val_unlabeled_loss /= len(val_loader.dataset)
        val_labeled_loss   /= len(val_loader.dataset)

        print(
            f"{epoch + 1},{train_unlabeled_loss:.6f},{val_unlabeled_loss:.6f},"
            f"{train_labeled_loss:.6f},{val_labeled_loss:.6f}",
            file=log_dir, 
            flush=True
        )

    return autoencoder


def save_model(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)    