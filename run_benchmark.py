import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from src.prepare_data  import create_mnist_datasets, make_embeddings_tensor
from src.benchmark     import run_benchmark
from src.store_model   import load_model
from src.visualization import read_train_log, show_train_log_plot
from src.prepare_data  import RANDOM_SEED


def main(
    device:    torch.device,
    train_log: Path | None,
    log_dir:   Path | None,
    model_dir: Path,
    knn_set:   float = 0.1, # Must match training --labeled.
    num_pairs: int  | None = None,
) -> None:

    NEIGHBORS = 10 # Count of neighbors for knn algorithm

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu")
    
    train_dataset, _, test_dataset = create_mnist_datasets("./data")
    knn_size = int(len(train_dataset) * knn_set)
    if not 0 < knn_set <= 1 or knn_size == 0:
        raise ValueError("knn_set must select at least one image and be at most 1")
    knn_dataset, _ = random_split(
        train_dataset,
        [knn_size, len(train_dataset) - knn_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=64,
        shuffle=False
    )
    knn_loader = DataLoader(
        knn_dataset,
        batch_size=64,
        shuffle=False
    )

    autoencoder = load_model(str(model_dir), device=device)
    criterion   = torch.nn.MSELoss().to(device)

    knn_embeddings, knn_labels = make_embeddings_tensor(knn_loader, autoencoder.encoder, device)
    run_benchmark(autoencoder,     test_loader,
                  knn_embeddings,  knn_labels,
                  criterion,       device=device,
                  log_dir=log_dir, num_pairs=num_pairs,
                  neighbors=NEIGHBORS
    )

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
    parser.add_argument("--knn-set",   "--knn_set",   type=float,        default=0.1,    help="Same labeled fraction as training --labeled")
    args = parser.parse_args()

    if not 0 < args.knn_set <= 1:
        parser.error("--knn-set must be in (0, 1]")

    if args.num_pairs is not None and args.num_pairs < 0:
        parser.error("--num-pairs must be non-negative")

    main(
        device    = args.device,
        train_log = args.train_log,
        log_dir   = args.log_dir,
        model_dir = args.model_dir,
        num_pairs = args.num_pairs,
        knn_set   = args.knn_set,
    )
