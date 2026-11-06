import torch

def kmeans(
    points: torch.Tensor,
    num_clusters:   int = 10,
    max_iterations: int = 100,
    seed:           int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cluster indices and centers"""
    
    if points.ndim != 2 or points.shape[1] == 0:
        raise ValueError("Expected points with shape (N, D), D > 0")
    points = points.detach().cpu()
    if not points.is_floating_point():
        points = points.float()
    if not torch.isfinite(points).all():
        raise ValueError("Point coordinates must be finite")
    if not 1 <= num_clusters <= len(points):
        raise ValueError("num_clusters must be between 1 and the number of points")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    generator = torch.Generator().manual_seed(seed)
    first     = torch.randint(len(points), (1,), generator=generator).item()
    centers   = [points[first]]
    distances = (points - centers[0]).square().sum(dim=1)

    # k-means++: prefer initial centers far from the already selected centers.
    for _ in range(1, num_clusters):
        if distances.sum() == 0:
            centers.append(points[0])
        else:
            index = torch.multinomial(distances, 1, generator=generator).item()
            centers.append(points[index])
        new_distances = (points - centers[-1]).square().sum(dim=1)
        distances = torch.minimum(distances, new_distances)
    centers = torch.stack(centers)

    for _ in range(max_iterations):
        clusters = torch.cdist(points, centers).argmin(dim=1)
        new_centers = centers.clone()
        for cluster in range(num_clusters):
            members = points[clusters == cluster]
            if len(members) > 0:
                new_centers[cluster] = members.mean(dim=0)

        converged = torch.allclose(centers, new_centers, atol=1e-4, rtol=0)
        centers = new_centers
        if converged:
            break

    clusters = torch.cdist(points, centers).argmin(dim=1)
    return clusters, centers


def predict_knn(
    pivots:     torch.Tensor,
    embeddings: torch.Tensor, # point used in train as labeled data
    labels:     torch.Tensor,
    neighbors:  int,
) -> torch.Tensor:

    if pivots.ndim != 2 or pivots.shape[1] != 2:
        raise ValueError("Expected pivots with shape (N, 2)")
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("Expected one label for each embedding")

    predictions = []

    for pivot in pivots:
        indices = k_nearest_neighbors(pivot, embeddings, neighbors)
        neighbor_labels = labels[indices.to(labels.device)]

        pred = torch.bincount(neighbor_labels).argmax().item()
        predictions.append(pred)

    return torch.tensor(predictions, dtype=torch.long, device=labels.device)


def k_nearest_neighbors(
    pivot:     torch.Tensor,
    points:    torch.Tensor,
    neighbors: int,
) -> torch.Tensor:
    """Return original point indices, ordered by Euclidean distance"""

    if pivot.shape != (2,) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected pivot with shape (2,) and points with shape (N, 2)")
    if isinstance(neighbors, bool) or not isinstance(neighbors, int):
        raise TypeError("neighbors must be an integer")
    if not 1 <= neighbors <= len(points):
        raise ValueError("neighbors must be between 1 and the number of points")

    # Search on CPU, return indices on the original points device.
    search_points = points.detach().to(device="cpu", dtype=torch.float64)
    pivot         = pivot.detach().to(device="cpu", dtype=torch.float64)

    if not torch.isfinite(search_points).all() or not torch.isfinite(pivot).all():
        raise ValueError("Point coordinates must be finite")

    x_order = search_points[:, 0].argsort()
    sorted_x = search_points[x_order, 0].contiguous()
    radius = 0.05

    while True:
        # Binary search selects only points in the vertical strip around pivot
        left  = torch.searchsorted(sorted_x, pivot[0] - radius).item()
        right = torch.searchsorted(sorted_x, pivot[0] + radius, right=True).item()
        candidates = x_order[left:right]
        # Select only point in square around pivot
        inside_y   = (search_points[candidates, 1] - pivot[1]).abs() <= radius
        candidates = candidates[inside_y]

        if len(candidates) >= neighbors:
            distances = ((search_points[candidates] - pivot) ** 2).sum(dim=1)
            nearest_distances, nearest = distances.topk(neighbors, largest=False)
            
            if nearest_distances[-1].item() <= radius ** 2:
                return candidates[nearest].to(points.device)
        
        radius *= 2