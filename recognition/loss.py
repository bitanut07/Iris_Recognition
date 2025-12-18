"""
Triplet Loss Implementation
Computes triplet loss from scratch using Euclidean distance.
"""

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))
    
    where d is Euclidean distance.
    """
    
    def __init__(self, margin: float = 0.3):
        """
        Args:
            margin: Margin parameter for triplet loss (default: 0.3)
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def _euclidean_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance between two tensors.
        
        Args:
            x1: First tensor (B, D) or (B, D, H, W)
            x2: Second tensor (B, D) or (B, D, H, W)
        
        Returns:
            distances: Euclidean distances (B,)
        """
        # Flatten if needed
        if x1.dim() > 2:
            x1 = x1.view(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.view(x2.size(0), -1)
        
        # Compute squared differences
        diff = x1 - x2
        squared_diff = diff.pow(2)
        
        # Sum over feature dimension and take square root
        distances = squared_diff.sum(dim=1).sqrt()
        
        return distances
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
        
        Returns:
            loss: Scalar loss value
        """
        # Compute distances
        d_positive = self._euclidean_distance(anchor, positive)
        d_negative = self._euclidean_distance(anchor, negative)
        
        # Compute triplet loss: max(0, margin + d(anchor, positive) - d(anchor, negative))
        loss = torch.clamp(self.margin + d_positive - d_negative, min=0.0)
        
        # Return mean loss over batch
        return loss.mean()

