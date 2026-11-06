import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathlib import Path

from src.clustering import kmeans, predict_knn
from src.train import compute_geometry_loss
from src.visualization import show_mnist_img, show_embeddings


def run_benchmark(
    autoencoder:    nn.Module,
    loader:         DataLoader,
    knn_embeddings: torch.Tensor,
    knn_labels:     torch.Tensor,
    criterion:      nn.Module,
    device:         torch.device | str,
    log_dir:        Path | None = None,
    num_pairs:      int  | None = None,
    neighbors:      int = 10,
) -> float:
    if num_pairs is not None and num_pairs < 0:
        raise ValueError("num_pairs must be non-negative or None")

    if len(loader.dataset) == 0:
        raise ValueError("Benchmark dataset must not be empty")

    if knn_embeddings.ndim != 2 or knn_embeddings.shape[1] != 2:
        raise ValueError("Expected knn_embeddings with shape (N, 2)")
    if knn_labels.ndim != 1 or len(knn_labels) != len(knn_embeddings):
        raise ValueError("Expected one label per kNN embedding")
    if not 1 <= neighbors <= len(knn_labels):
        raise ValueError("neighbors must not exceed the number of labeled training images")

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    autoencoder.to(device)
    criterion.to(device)
    classification_crit = nn.CrossEntropyLoss().to(device)
    autoencoder.eval()

    total_loss         = 0.0
    total_labeled_loss = 0.0
    class_correct      = 0
    knn_correct        = 0
    examples   = []
    embeddings = []
    labels     = []

    with torch.inference_mode():
        for batch_img, batch_label in loader:
            batch_img   = batch_img.to(device)
            batch_label = batch_label.to(device)
            
            embedding       = autoencoder.encoder(batch_img)
            reconstructed   = autoencoder.decoder(embedding)
            
            embeddings.append(embedding.detach().cpu())
            labels.append(batch_label.detach().cpu())

            # Classifier and kNN predict independently from the same embeddings.
            pred_classifier = autoencoder.classifier(embedding)
            pred_knn = predict_knn(
                embedding, 
                knn_embeddings, 
                knn_labels, 
                neighbors=neighbors
            )

            batch_loss = criterion(reconstructed, batch_img)
            classification_loss = classification_crit(pred_classifier, batch_label)

            total_loss  += batch_loss.item() * batch_img.size(0)
            labeled_loss = batch_loss + classification_loss
            total_labeled_loss += labeled_loss.item() * batch_img.size(0)

            class_correct += (pred_classifier.argmax(dim=1) == batch_label).sum().item()
            knn_correct   += (pred_knn.to(device) == batch_label).sum().item()

            # Save images for printing examples of reconstruction.
            if num_pairs is not None and len(examples) // 2 < num_pairs:
                count = min(num_pairs - len(examples) // 2, batch_img.size(0))
                for idx in range(count):
                    number   = batch_label[idx]
                    original = batch_img[idx].detach().cpu().clone()
                    restored = reconstructed[idx].detach().cpu().clone()
                    examples.append((original, f"Original {number}"))
                    examples.append((restored, f"Restored {number}"))

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    mean_loss = total_loss / len(loader.dataset)
    mean_labeled_loss = total_labeled_loss / len(loader.dataset)
    geometry_loss = compute_geometry_loss(embeddings, labels).item()
    class_accuracy = class_correct / len(loader.dataset)
    knn_accuracy = knn_correct / len(loader.dataset)
    print(
        f"Reconstruction loss: {mean_loss:.6f}\n"
        f"Reconstruction + classification loss: {mean_labeled_loss:.6f}\n"
        f"Geometry loss: {geometry_loss:.6f}\n"
        f"Classifier accuracy: {class_accuracy:.2%}\n"
        f"kNN accuracy: {knn_accuracy:.2%}",
        flush=True
    )

    # Print examples of reconstruction.
    if examples:
        path = log_dir / "reconstructions.png" if log_dir is not None else None
        show_mnist_img(examples, path=path, title="Original and reconstructed images")

    path = log_dir / "embeddings.png" if log_dir is not None else None
    show_embeddings(embeddings, labels, path=path)

    return mean_loss
