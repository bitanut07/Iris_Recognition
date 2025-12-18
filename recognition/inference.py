"""
Inference utilities for Iris Recognition
Provides functions for:
- Generating embeddings from images
- Computing similarity scores
- Verification (matching)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from .model import IrisEmbeddingModel


class IrisRecognizer:
    """
    Wrapper class for iris recognition inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        embedding_dim: int = 128,
        image_size: int = 224,
        device: str = None,
    ):
        """
        Initialize the recognizer.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            embedding_dim: Dimension of embedding vector
            image_size: Target image size
            device: Device to run inference on (None for auto-detect)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.image_size = image_size
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Load model
        self.model = IrisEmbeddingModel(embedding_dim=embedding_dim, pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _preprocess_image(self, image: Union[str, Path, np.ndarray], mask: Union[str, Path, np.ndarray] = None, use_crop: bool = True) -> torch.Tensor:
        """
        Preprocess a single image (with optional mask).
        
        Args:
            image: Image path or numpy array
            mask: Optional mask path or numpy array
            use_crop: Whether to crop by mask or multiply
        
        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, (str, Path)):
                mask_array = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            else:
                mask_array = mask
            
            if mask_array.shape[:2] != img.shape[:2]:
                mask_array = cv2.resize(mask_array, (img.shape[1], img.shape[0]))
            
            # Apply mask
            if use_crop:
                mask_binary = (mask_array > 127).astype(np.uint8) * 255
                coords = cv2.findNonZero(mask_binary)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img.shape[1] - x, w + 2 * padding)
                    h = min(img.shape[0] - y, h + 2 * padding)
                    img = img[y:y+h, x:x+w]
            else:
                mask_3d = mask_array[:, :, np.newaxis] if len(img.shape) == 3 else mask_array
                img = img * (mask_3d / 255.0)
        
        # Convert to PIL and resize
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # Convert to tensor and normalize
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(pil_img)
        tensor = self.normalize(tensor)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def get_embeddings(
        self,
        images: Union[List[Union[str, Path]], Union[str, Path]],
        masks: Union[List[Union[str, Path]], Union[str, Path], None] = None,
        use_crop: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Generate embeddings for a batch of images.
        
        Args:
            images: Single image path or list of image paths
            masks: Optional single mask path or list of mask paths
            use_crop: Whether to crop by mask or multiply
            batch_size: Batch size for processing
        
        Returns:
            embeddings: Normalized embeddings (N, embedding_dim)
        """
        # Handle single image
        if isinstance(images, (str, Path)):
            images = [images]
            if masks is not None:
                masks = [masks] if not isinstance(masks, list) else masks
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_masks = masks[i:i+batch_size] if masks is not None else [None] * len(batch_images)
            
            # Preprocess batch
            batch_tensors = []
            for img, mask in zip(batch_images, batch_masks):
                tensor = self._preprocess_image(img, mask, use_crop)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                embeddings = self.model(batch_tensor)
            
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        return all_embeddings
    
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embeddings (N, D) or (D,)
            emb2: Second embeddings (M, D) or (D,)
        
        Returns:
            similarities: Cosine similarities
                - If both are 1D: scalar
                - If one is 1D and one is 2D: (M,) or (N,)
                - If both are 2D: (N, M)
        """
        # Ensure embeddings are 2D
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        
        # Compute cosine similarity
        # Since embeddings are L2-normalized, cosine similarity = dot product
        similarities = torch.mm(emb1, emb2.t())
        
        # Squeeze if needed
        if similarities.numel() == 1:
            similarities = similarities.item()
        elif similarities.size(0) == 1:
            similarities = similarities.squeeze(0)
        elif similarities.size(1) == 1:
            similarities = similarities.squeeze(1)
        
        return similarities
    
    def verify(
        self,
        image1: Union[str, Path],
        image2: Union[str, Path],
        mask1: Union[str, Path, None] = None,
        mask2: Union[str, Path, None] = None,
        threshold: float = 0.7,
        use_crop: bool = True,
    ) -> Tuple[bool, float]:
        """
        Verify if two images belong to the same identity.
        
        Args:
            image1: First image path
            image2: Second image path
            mask1: Optional mask for first image
            mask2: Optional mask for second image
            threshold: Similarity threshold for verification
            use_crop: Whether to crop by mask or multiply
        
        Returns:
            is_match: True if similarity >= threshold
            similarity: Cosine similarity score
        """
        # Get embeddings
        emb1 = self.get_embeddings(image1, mask1, use_crop)
        emb2 = self.get_embeddings(image2, mask2, use_crop)
        
        # Compute similarity
        similarity = self.cosine_similarity(emb1, emb2)
        
        # Convert to float if tensor
        if isinstance(similarity, torch.Tensor):
            similarity = similarity.item()
        
        # Verify
        is_match = similarity >= threshold
        
        return is_match, similarity


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Standalone function to compute cosine similarity between embeddings.
    
    Since embeddings are L2-normalized, cosine similarity = dot product.
    
    Args:
        emb1: First embeddings (N, D) or (D,)
        emb2: Second embeddings (M, D) or (D,)
    
    Returns:
        similarities: Cosine similarities
    """
    # Ensure embeddings are 2D
    if emb1.dim() == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 1:
        emb2 = emb2.unsqueeze(0)
    
    # Compute cosine similarity (dot product for L2-normalized vectors)
    similarities = torch.mm(emb1, emb2.t())
    
    return similarities

