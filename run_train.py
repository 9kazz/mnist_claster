import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from src.prepare_data import RotatedDataset, create_mnist_datasets, RANDOM_SEED
from src.network      import Autoencoder
from src.train        import train
from src.store_model  import save_model

def main(
    device:    torch.device,
    log_dir:   Path,
    model_dir: Path,
    epoch:     int,
    rotation:  bool,
    labeled:   float,
) -> None:

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu")
        
    train_dataset, val_dataset, _ = create_mnist_datasets("./data")

    labeled_size = int(len(train_dataset) * labeled)
    if not 0 < labeled <= 1 or labeled_size == 0:
        raise ValueError("labeled must select at least one image and be at most 1")
    
    labeled_dataset, _ = random_split(
        train_dataset,
        [labeled_size, len(train_dataset) - labeled_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    if rotation:
        train_dataset   = RotatedDataset(train_dataset,   max_angle=10, image_only=True)
        labeled_dataset = RotatedDataset(labeled_dataset, max_angle=10)

    unlabeled_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    autoencoder = Autoencoder().to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    log_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_dir.open("w", newline="") as train_log:
        train(autoencoder, unlabeled_loader, labeled_loader, val_loader, optimizer, num_epoch=epoch, device=device, log_dir=train_log)

    save_model(autoencoder, str(model_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device",                   type=torch.device, default="cuda",            help="cpu, cuda or cuda:0")
    parser.add_argument("--log-dir",   "--log_dir",   type=Path,         default=Path("train.csv"), help="Output log path")
    parser.add_argument("--model-dir", "--model_dir", type=Path,         default=Path("model.pt"),  help="Model output path")
    parser.add_argument("--epoch",                    type=int,          default=50,                help="Number of training epochs")
    parser.add_argument("--labeled",                  type=float,        default=0.1,               help="Fraction of training images with labels")
    parser.add_argument("--rotation",                 action="store_trueпоче",                          help="Enable rotation augmentation")
    args = parser.parse_args()

    if not 0 < args.labeled <= 1:
        parser.error("--labeled must be in (0, 1]")

    main(
        device    = args.device,
        log_dir   = args.log_dir,
        model_dir = args.model_dir,
        epoch     = args.epoch,
        labeled   = args.labeled,
        rotation  = args.rotation,
    )