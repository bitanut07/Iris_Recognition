"""
Recognition model wrapper for biometric pipeline.

- CNN backbone (ResNet18) + embedding head (128-dim).
- Loads from a pre-trained checkpoint (trained with Triplet Loss).
- Exposes `embed(roi)` to get L2-normalized embeddings.

All inference:
- Uses GPU if available.
- Runs under `torch.no_grad()`.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class RecognitionModel(nn.Module):
    """
    ResNet18-based embedding model.

    - Backbone: ResNet18.
    - Head: FC(512 -> embedding_dim).
    - Output: L2-normalized embedding [B, embedding_dim].
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        backbone = resnet18(pretrained=pretrained)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.fc = nn.Linear(512, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        features = self.backbone(x)                # [B, 512, 1, 1]
        features = features.view(x.size(0), -1)    # [B, 512]
        emb = self.fc(features)                    # [B, D]
        emb = F.normalize(emb, p=2, dim=1)         # L2-normalize
        return emb


class RecognitionModelWrapper:
    """
    Wrapper around RecognitionModel with checkpoint loading.

    Expected:
    - Input:  ROI tensor [1, 3, H, W], already resized & normalized.
    - Output: L2-normalized embedding [1, embedding_dim].
    """

    def __init__(
        self,
        checkpoint_path: str,
        embedding_dim: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = RecognitionModel(embedding_dim=embedding_dim, pretrained=False)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Recognition checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device)
        # Accept both raw state_dict or training checkpoint dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model_state = self.model.state_dict()
        filtered_state = {k: v for k, v in state.items() if k in model_state}
        model_state.update(filtered_state)
        self.model.load_state_dict(model_state)

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(self, roi: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding for a single ROI.

        Args:
            roi: [1, 3, H, W] on CPU or GPU.

        Returns:
            embedding: [1, embedding_dim] (L2-normalized).
        """
        if roi.ndim != 4 or roi.size(0) != 1 or roi.size(1) != 3:
            raise ValueError(f"Expected ROI shape [1, 3, H, W], got {tuple(roi.shape)}")

        roi = roi.to(self.device)
        emb = self.model(roi)
        return emb.cpu()


