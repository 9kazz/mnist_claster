import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.prepare_data import create_mnist_datasets
from src.benchmark import load_model, run_benchmark
from src.visualization import read_train_log, show_train_log_plot


def main(
    device:    torch.device,
    train_log: Path | None,
    log_dir:   Path | None,
    model_dir: Path,
    num_pairs: int | None = None,
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu")

    _, _, test_dataset = create_mnist_datasets("./data")
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    autoencoder = load_model(str(model_dir), device=device)
    criterion   = torch.nn.MSELoss().to(device)

    run_benchmark(autoencoder, loader, criterion, device=device,
                  log_dir=log_dir, num_pairs=num_pairs)

    if train_log is not None:
        metrics = read_train_log(train_log)
        path = log_dir / "loss.png" if log_dir is not None else None
        show_train_log_plot(metrics, path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device",                   type=torch.device, default="cuda", help="cpu, cuda or cuda:0")
    parser.add_argument("--train-log", "--train_log", type=Path,         default=None,   help="Plot train/validation losses from CSV")
    parser.add_argument("--log-dir",   "--log_dir",   type=Path,         default=None,   help="Directory for plot images; omit to only display them")
    parser.add_argument("--model-dir", "--model_dir", type=Path,         default=None,   help="Model weights path")
    parser.add_argument("--num-pairs", "--num_pairs", type=int,          default=None,   help="Show the first N original/reconstructed pairs")
    args = parser.parse_args()

    if args.num_pairs is not None and args.num_pairs < 0:
        parser.error("--num-pairs must be non-negative")

    main(
        device=args.device,
        train_log=args.train_log,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        num_pairs=args.num_pairs,
    )
