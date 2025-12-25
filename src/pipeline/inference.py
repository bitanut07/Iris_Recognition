"""
End-to-end biometric inference pipeline.

Stages:
- Segmentation: frame -> binary mask -> ROI.
- Recognition: ROI -> embedding.
- Enrollment: multi-frame mean embedding -> FAISS + MongoDB.
- Recognition: frame -> embedding -> nearest neighbor with open-set threshold.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.models.segmentation import SegmentationModel
from src.models.recognition import RecognitionModelWrapper
from src.vector_db.faiss_db import FaissMongoVectorDB


@dataclass
class PipelineConfig:
    seg_checkpoint: str
    rec_checkpoint: str
    embedding_dim: int = 128
    metric: str = "cosine"  # "cosine" or "l2"
    threshold: float = 0.7  # cosine similarity threshold (or L2 distance threshold)
    image_size: int = 224
    use_crop: bool = True   # True: crop ROI by mask bbox, False: mask multiplication


class BiometricPipeline:
    """
    High-level biometric pipeline that orchestrates:
    - Segmentation
    - ROI extraction
    - Embedding computation
    - Vector DB operations (enrollment & recognition, persisted to MongoDB)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Preprocessing (ImageNet normalization)
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Models
        self.seg_model = SegmentationModel(
            checkpoint_path=config.seg_checkpoint,
            n_channels=3,
            n_classes=2,  # checkpoint được train với 2 classes (background + foreground)
            bilinear=False,
            device=self.device,
            threshold=0.5,
        )
        self.rec_model = RecognitionModelWrapper(
            checkpoint_path=config.rec_checkpoint,
            embedding_dim=config.embedding_dim,
            device=self.device,
        )

        # Vector DB backed by MongoDB
        self.db = FaissMongoVectorDB(dim=config.embedding_dim, metric=config.metric)

    @torch.no_grad()
    def _apply_mask_and_extract_roi(
        self,
        frame: torch.Tensor,  # [1, 3, H, W]
        mask: torch.Tensor,  # [1, 1, H, W]
    ) -> torch.Tensor:
        """
        Apply mask to frame and extract ROI.

        Returns:
            roi_tensor: Tensor [1, 3, H', W'] (normalized for recognition model).
        """
        # Convert to numpy for simple cropping logic
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        mask_np = mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W] {0,1}

        if self.config.use_crop:
            # Convert mask to 0/255 for finding bbox
            mask_bin = (mask_np > 0).astype(np.uint8) * 255
            coords = cv2.findNonZero(mask_bin)
            if coords is None:
                # If mask is empty, fallback to full frame
                roi_np = frame_np
            else:
                x, y, w, h = cv2.boundingRect(coords)
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame_np.shape[1] - x, w + 2 * padding)
                h = min(frame_np.shape[0] - y, h + 2 * padding)
                roi_np = frame_np[y : y + h, x : x + w, :]
        else:
            # Mask multiplication
            mask_3d = mask_np[:, :, None].astype(np.float32)
            roi_np = frame_np * mask_3d

        # Convert ROI (numpy H,W,3) to a PIL Image before transforms.
        # Handle dtype/range robustness: roi_np may be float in [0,1] (mock frames)
        # or uint8 in [0,255]. Ensure uint8 HxWx3 for Image.fromarray.
        if np.issubdtype(roi_np.dtype, np.floating):
            maxv = roi_np.max() if roi_np.size else 0.0
            if maxv <= 1.0:
                roi_uint8 = (roi_np * 255.0).round().astype(np.uint8)
            else:
                roi_uint8 = roi_np.round().astype(np.uint8)
        else:
            roi_uint8 = roi_np.astype(np.uint8)

        pil = Image.fromarray(roi_uint8)
        tensor = self.transform(pil)  # [3, H', W']
        tensor = tensor.unsqueeze(0)  # [1, 3, H', W']
        return tensor

    @torch.no_grad()
    def frame_to_embedding(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: frame -> mask -> ROI -> embedding.

        Args:
            frame: [1, 3, H, W], float32 in [0,1] or same as used in training.

        Returns:
            embedding: [1, D] L2-normalized.
        """
        if frame.ndim != 4 or frame.size(0) != 1 or frame.size(1) != 3:
            raise ValueError(f"Expected frame shape [1, 3, H, W], got {tuple(frame.shape)}")

        # 1) Segmentation
        mask = self.seg_model.predict_mask(frame)  # [1, 1, H, W]

        # 2) ROI extraction
        roi = self._apply_mask_and_extract_roi(frame, mask)  # [1, 3, H', W']

        # 3) Recognition embedding
        emb = self.rec_model.embed(roi)  # [1, D]
        return emb

    @torch.no_grad()
    def enroll_user(self, user_id: str, frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Enrollment:
        - For each frame: frame -> embedding.
        - Aggregate via mean pooling.
        - Store final embedding in FAISS + MongoDB.

        Args:
            user_id: ID of the user to enroll.
            frames: List of frame tensors [1, 3, H, W].

        Returns:
            final_embedding: [1, D] L2-normalized.
        """
        if len(frames) == 0:
            raise ValueError("frames list is empty during enrollment")

        embs = []
        for frame in frames:
            embs.append(self.frame_to_embedding(frame))

        embs_tensor = torch.cat(embs, dim=0)  # [N, D]
        mean_emb = embs_tensor.mean(dim=0, keepdim=True)  # [1, D]
        mean_emb = F.normalize(mean_emb, p=2, dim=1)  # re-normalize

        # Store in vector DB (persists to Mongo)
        self.db.add(user_id, mean_emb.squeeze(0))

        return mean_emb

    @torch.no_grad()
    def recognize(self, frame: torch.Tensor) -> Tuple[str, float]:
        """
        Recognition:
        - frame -> embedding.
        - embed -> nearest neighbor search in VectorDB.
        - threshold-based open-set decision.

        Returns:
            (user_id or "UNDEFINED", score)
        """
        emb = self.frame_to_embedding(frame).squeeze(0)  # [D]
        user_id, score = self.db.recognize(emb, threshold=self.config.threshold)
        return user_id, score


