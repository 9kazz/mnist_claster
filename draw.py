import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from src.draw_digit   import create_gui
from src.prepare_data import create_mnist_datasets, make_embeddings_tensor, RANDOM_SEED
from src.store_model  import load_model

def main(model_path: Path, device: torch.device, knn_set: float, neighbors: int) -> None:
    model = load_model(str(model_path), device=device)
    train_dataset, _, _ = create_mnist_datasets("./data")
    knn_size = int(len(train_dataset) * knn_set)
    if not 1 <= neighbors <= knn_size:
        raise ValueError("neighbors must be between 1 and the number of labeled images")
    knn_dataset, _ = random_split(
        train_dataset,
        [knn_size, len(train_dataset) - knn_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    knn_loader = DataLoader(knn_dataset, batch_size=64, shuffle=False)
    knn_embeddings, knn_labels = make_embeddings_tensor(knn_loader, model.encoder, device)
    root = create_gui(model, device, knn_embeddings, knn_labels, neighbors)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-file", "--model-dir", type=Path,         required=True,  help="Autoencoder weights path")
    parser.add_argument("--device",                    type=torch.device, default="cpu", help="cpu, cuda or cuda:0")
    parser.add_argument("--knn-set",                   type=float,        default=0.1,   help="Same labeled fraction as training --labeled")
    parser.add_argument("--neighbors",                 type=int,          default=10,    help="Number of neighbors for kNN")
    args = parser.parse_args()
    if not 0 < args.knn_set <= 1:
        parser.error("--knn-set must be in (0, 1]")
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")

    main(model_path=args.model_file, device=args.device, knn_set=args.knn_set, neighbors=args.neighbors)
