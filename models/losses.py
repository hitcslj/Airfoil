import torch
import torch.nn as nn
import torch.nn.functional as F
    

def chamfer_loss(points1, points2):
    """
    Calculates the Chamfer distance loss between two sets of points.
    
    Args:
        points1 (torch.Tensor): Tensor of shape (batch_size, num_points1, num_dimensions)
            representing the first set of points.
        points2 (torch.Tensor): Tensor of shape (batch_size, num_points2, num_dimensions)
            representing the second set of points.
    
    Returns:
        torch.Tensor: Chamfer distance loss.
    """

    # Calculate pairwise distances between points
    distances1 = torch.cdist(points1, points2, p=2)  # Shape: (batch_size, num_points1, num_points2)
    distances2 = torch.cdist(points2, points1, p=2)  # Shape: (batch_size, num_points2, num_points1)

    # Find the minimum distance for each point in points1
    min_distances1, _ = torch.min(distances1, dim=2)  # Shape: (batch_size, num_points1)

    # Find the minimum distance for each point in points2
    min_distances2, _ = torch.min(distances2, dim=2)  # Shape: (batch_size, num_points2)

    # Calculate the Chamfer distance loss
    chamfer_loss = torch.mean(min_distances1) + torch.mean(min_distances2)

    return chamfer_loss


if __name__ == '__main__':
    points1 = torch.randn(1,200,2)
    points2 = torch.randn(1,200,2)
    loss = chamfer_loss(points1, points2)
    print(loss)