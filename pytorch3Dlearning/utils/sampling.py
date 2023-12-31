import torch

def knn(x, k=16):
    """
    Performs k-nearest neighbors (knn) search on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_points, num_dims).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Index tensor of shape (batch_size, num_points, k), containing the indices of the k nearest neighbors for each point.
    """
    x_t = x.transpose(2, 1)
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx

def fps(xyz, npoint):
    """
    Farthest Point Sampling (FPS) algorithm for selecting a subset of points from a point cloud.

    Args:
        xyz (torch.Tensor): Input point cloud tensor of shape (B, N, C), where B is the batch size, N is the number of points, and C is the number of dimensions.
        npoint (int): Number of points to select.

    Returns:
        torch.Tensor: Tensor of shape (B, npoint) containing the indices of the selected points.
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
