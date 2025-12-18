"""
Triplet Dataset for Iris Recognition
Implements a custom PyTorch Dataset that:
- Loads images and their corresponding segmentation masks
- Applies masks to extract ROI (Region of Interest)
- Performs triplet sampling (anchor, positive, negative)
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TripletDataset(Dataset):
    """
    Dataset for triplet learning with masked ROI images.
    
    Each sample returns (anchor, positive, negative) where:
    - Anchor and positive share the same identity
    - Negative has a different identity
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        labels_file: str = None,
        image_size: int = 224,
        use_crop: bool = True,
    ):
        """
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing segmentation masks
            labels_file: Optional file mapping image names to identity labels.
                        If None, extracts identity from filename (assumes format: personID_*.ext)
            image_size: Target size for resizing (default: 224)
            use_crop: If True, crop by mask bounding box. If False, multiply image by mask.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.use_crop = use_crop
        
        # ImageNet normalization statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Load image and mask paths
        self.image_paths = sorted(list(self.images_dir.glob("*.bmp")) + 
                                   list(self.images_dir.glob("*.png")) +
                                   list(self.images_dir.glob("*.jpg")))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {images_dir}")
        
        # Load labels
        self.labels = self._load_labels(labels_file)
        
        # Group images by identity for triplet sampling
        self.identity_to_indices = defaultdict(list)
        for idx, img_path in enumerate(self.image_paths):
            identity = self.labels[img_path.stem]
            self.identity_to_indices[identity].append(idx)
        
        # Filter out identities with only one image (can't form triplets)
        self.identity_to_indices = {
            identity: indices 
            for identity, indices in self.identity_to_indices.items() 
            if len(indices) > 1
        }
        
        # Create list of valid indices (only those with multiple images per identity)
        self.valid_indices = []
        for indices in self.identity_to_indices.values():
            self.valid_indices.extend(indices)
        
        if not self.valid_indices:
            raise ValueError("No identities with multiple images found. Cannot form triplets.")
        
        print(f"Loaded {len(self.valid_indices)} images from {len(self.identity_to_indices)} identities")
    
    def _load_labels(self, labels_file: str = None) -> dict:
        """
        Load identity labels for each image.
        
        If labels_file is provided, reads from file (format: image_name,identity_id).
        Otherwise, extracts identity from filename (assumes format: personID_*.ext or personID.ext).
        """
        labels = {}
        
        if labels_file and Path(labels_file).exists():
            # Load from file
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        labels[parts[0]] = parts[1]
        else:
            # Extract from filename
            # Assumes format: personID_*.ext or personID.ext
            for img_path in self.image_paths:
                name = img_path.stem
                # Try to extract person ID (first part before underscore or first digits)
                if '_' in name:
                    identity = name.split('_')[0]
                else:
                    # Extract leading digits
                    identity = ''.join(c for c in name if c.isdigit())[:10] or name[:5]
                labels[name] = identity
        
        return labels
    
    def _get_mask_path(self, image_path: Path) -> Path:
        """Get corresponding mask path for an image."""
        # Try different extensions
        for ext in ['.png', '.bmp', '.jpg']:
            mask_path = self.masks_dir / f"{image_path.stem}{ext}"
            if mask_path.exists():
                return mask_path
        
        raise FileNotFoundError(f"Mask not found for {image_path}")
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to image to extract ROI.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            mask: Binary mask (H, W) with values 0 or 255
        
        Returns:
            Masked image
        """
        # Ensure mask is binary (0 or 255)
        if mask.max() > 1:
            mask_binary = (mask > 127).astype(np.uint8) * 255
        else:
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
        
        if self.use_crop:
            # Crop by mask bounding box
            coords = cv2.findNonZero(mask_binary)
            if coords is None:
                # If mask is empty, return original image
                masked = image
            else:
                x, y, w, h = cv2.boundingRect(coords)
                # Add small padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                masked = image[y:y+h, x:x+w]
        else:
            # Multiply image by mask
            if len(image.shape) == 2:
                masked = image * (mask_binary / 255.0)
            else:
                mask_3d = mask_binary[:, :, np.newaxis] if len(image.shape) == 3 else mask_binary
                masked = image * (mask_3d / 255.0)
        
        return masked
    
    def _load_and_preprocess(self, image_path: Path) -> torch.Tensor:
        """
        Load image, apply mask, resize, normalize.
        
        Returns:
            Preprocessed image tensor (C, H, W)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self._get_mask_path(image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Ensure mask and image have same size
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Apply mask to extract ROI
        masked_image = self._apply_mask(image, mask)
        
        # Convert to PIL Image for resizing
        if len(masked_image.shape) == 2:
            pil_image = Image.fromarray(masked_image, mode='L').convert('RGB')
        else:
            pil_image = Image.fromarray(masked_image)
        
        # Resize to target size
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # Convert to tensor and normalize
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(pil_image)
        tensor = self.normalize(tensor)
        
        return tensor
    
    def _get_identity(self, idx: int) -> str:
        """Get identity label for a given index."""
        img_path = self.image_paths[idx]
        return self.labels[img_path.stem]
    
    def __len__(self) -> int:
        """Return number of valid samples (can form triplets)."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet (anchor, positive, negative).
        
        Returns:
            anchor: Anchor image tensor
            positive: Positive image tensor (same identity as anchor)
            negative: Negative image tensor (different identity from anchor)
        """
        # Get anchor index
        anchor_idx = self.valid_indices[idx]
        anchor_identity = self._get_identity(anchor_idx)
        
        # Get positive (same identity, different image)
        positive_candidates = [
            i for i in self.identity_to_indices[anchor_identity] 
            if i != anchor_idx
        ]
        if not positive_candidates:
            # Fallback: use same image (shouldn't happen if validation worked)
            positive_idx = anchor_idx
        else:
            positive_idx = random.choice(positive_candidates)
        
        # Get negative (different identity)
        negative_identities = [
            identity for identity in self.identity_to_indices.keys() 
            if identity != anchor_identity
        ]
        negative_identity = random.choice(negative_identities)
        negative_idx = random.choice(self.identity_to_indices[negative_identity])
        
        # Load and preprocess images
        anchor_path = self.image_paths[anchor_idx]
        positive_path = self.image_paths[positive_idx]
        negative_path = self.image_paths[negative_idx]
        
        anchor = self._load_and_preprocess(anchor_path)
        positive = self._load_and_preprocess(positive_path)
        negative = self._load_and_preprocess(negative_path)
        
        return anchor, positive, negative

