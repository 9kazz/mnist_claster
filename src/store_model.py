from pathlib import Path

import torch
import torch.nn as nn

from src.network import Autoencoder


def save_model(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, device: torch.device | str) -> Autoencoder:
    autoencoder = Autoencoder().to(device)

    state = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    autoencoder.load_state_dict(state)
    autoencoder.eval()

    return autoencoder