"""
Segmentation model wrapper for biometric pipeline.

This module:
- Loads a pre-trained segmentation model (e.g. U-Net checkpoint).
- Exposes a simple `predict_mask(frame)` API.

All inference:
- Uses GPU if available.
- Runs under `torch.no_grad()`.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from unet.unet_model import UNet  # Reuse existing UNet implementation in this repo


class SegmentationModel:
    """
    Wrapper around a segmentation network (e.g. U-Net).

    Expected:
    - Input:  Tensor [1, 3, H, W]  (RGB frame)
    - Output: Tensor [1, 1, H, W]  (binary mask in {0,1})
    """

    def __init__(
        self,
        checkpoint_path: str,
        n_channels: int = 3,
        n_classes: int = 2,
        bilinear: bool = False,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            checkpoint_path: Path to segmentation checkpoint (.pth).
            n_channels: Number of input channels (3 for RGB).
            n_classes: Number of output classes (1 for binary mask).
            bilinear: Whether UNet uses bilinear upsampling.
            device: torch.device (auto-detect if None).
            threshold: Probability threshold to binarize mask.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.threshold = threshold

        # Build model
        self.model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)

        # Load checkpoint (expects state_dict-like checkpoint)
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device)

        # Allow both raw state_dict and training checkpoints with extra keys
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Filter out non-matching keys to be more tolerant across training scripts
        model_state = self.model.state_dict()
        filtered_state = {k: v for k, v in state.items() if k in model_state}
        model_state.update(filtered_state)
        self.model.load_state_dict(model_state)

        # Move to device, eval & freeze
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def predict_mask(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Run segmentation on a single frame.

        Args:
            frame: Tensor [1, 3, H, W] on CPU or GPU.

        Returns:
            Binary mask: Tensor [1, 1, H, W] with values {0,1}.
        """
        if frame.ndim != 4 or frame.size(0) != 1 or frame.size(1) != 3:
            raise ValueError(f"Expected frame shape [1, 3, H, W], got {tuple(frame.shape)}")

        frame = frame.to(self.device)

        # Forward pass
        logits = self.model(frame)  # [1, 1, H, W] or [1, C, H, W]

        if logits.size(1) == 1:
            # Binary mask via sigmoid
            prob = torch.sigmoid(logits)
        else:
            # Multi-class; take class-1 as foreground probability
            prob = F.softmax(logits, dim=1)[:, 1:2, :, :]

        # Binarize
        mask = (prob >= self.threshold).float()
        return mask.cpu()


