import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.prepare_data import *
from src.network import *
from src.train import *

def main(
    device:    torch.device,
    log_dir:   Path,
    model_dir: Path,
    epoch:     int
) -> None:
    train_dataset, val_dataset, _ = create_mnist_datasets("./data")

    train_loader = DataLoader(
        train_dataset.tensors[0],
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset.tensors[0],
        batch_size=64,
        shuffle=False
    )

    autoencoder = Autoencoder()
    criterion   = torch.nn.MSELoss().to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    log_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_dir.open("w", newline="") as train_log:
        train(autoencoder, train_loader, val_loader, criterion, optimizer, num_epoch=epoch, device=device, log_dir=train_log)

    save_model(autoencoder, str(model_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device",                   type=torch.device, default="cuda",            help="cpu, cuda or cuda:0")
    parser.add_argument("--log-dir",   "--log_dir",   type=Path,         default=Path("train.csv"), help="Output log path")
    parser.add_argument("--model-dir", "--model_dir", type=Path,         default=Path("model.pt"),  help="Model output path")
    parser.add_argument("--epoch",                    type=int,          default=50,                help="Number of training epoches")
    args = parser.parse_args()

    main(
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        epoch=args.epoch
    )