import torch

def kmeans(
    points: torch.Tensor,
    num_clusters:   int = 10,
    max_iterations: int = 100,
    seed:           int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cluster indices and centers"""
    
    points = points.detach().cpu()
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
