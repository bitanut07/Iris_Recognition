"""
ResNet18-based Embedding Model for Iris Recognition
Uses ResNet18 as backbone and outputs L2-normalized embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class IrisEmbeddingModel(nn.Module):
    """
    ResNet18-based model for generating iris embeddings.
    
    Architecture:
    - ResNet18 backbone (pretrained)
    - Remove final classification layer
    - Global Average Pooling
    - Fully connected layer (512 -> 128)
    - L2 normalization
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        """
        Args:
            embedding_dim: Dimension of output embedding vector (default: 128)
            pretrained: Whether to use pretrained ResNet18 weights (default: True)
        """
        super(IrisEmbeddingModel, self).__init__()
        
        # Load ResNet18 backbone
        resnet = resnet18(pretrained=pretrained)
        
        # Remove the final classification layer (fc)
        # Keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet18 outputs 512 features after avgpool
        # Add our own embedding layer
        self.fc = nn.Linear(512, embedding_dim)
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            embeddings: L2-normalized embeddings (B, embedding_dim)
        """
        # Extract features using ResNet18 backbone
        # Output shape: (B, 512, 1, 1) after avgpool
        features = self.backbone(x)
        
        # Flatten: (B, 512, 1, 1) -> (B, 512)
        features = features.view(features.size(0), -1)
        
        # Project to embedding dimension: (B, 512) -> (B, embedding_dim)
        embeddings = self.fc(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

