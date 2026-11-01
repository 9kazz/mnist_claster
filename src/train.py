import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from pathlib import Path

import sys
from typing import TextIO


def train(
    autoencoder:  nn.Module,

    train_loader: DataLoader,
    val_loader:   DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epoch: int,
    device:    torch.device | str,
    log_dir:   TextIO=sys.stdout
) -> nn.Module:

    # Move the autoencoder before constructing the optimizer, not inside train()
    device = torch.device(device)

    for tensor in (*autoencoder.parameters(), *autoencoder.buffers()):
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise ValueError("Move the autoencoder to device before creating the optimizer")
        
    criterion.to(device)

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty")
    
    print(
        "Epoch,TrainLoss,ValLoss",
        file=log_dir,
        flush=True
    )

    for epoch in tqdm(range(num_epoch)):
        # Train set pass
        autoencoder.train()
        train_loss = 0.0

        for batch_img in train_loader:
            batch_img = batch_img.to(device)
            
            optimizer.zero_grad()

            reconstructed = autoencoder(batch_img)

            batch_loss  = criterion(reconstructed, batch_img)   # mean batch loss 
            train_loss += batch_loss.item() * batch_img.size(0) # total loss of all batch elements

            batch_loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset) # mean loss of all dataset's elements

        # Validation set pass
        autoencoder.eval()
        val_loss = 0.0

        with torch.inference_mode():
            for batch_img in val_loader:
                batch_img = batch_img.to(device)

                reconstructed = autoencoder(batch_img)

                batch_loss = criterion(reconstructed, batch_img)
                val_loss += batch_loss.item() * batch_img.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"{epoch + 1},{train_loss:.6f},{val_loss:.6f}",
            file=log_dir, 
            flush=True
        )

    return autoencoder


def save_model(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)    