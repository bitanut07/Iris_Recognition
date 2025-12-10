"""
Download the MMU iris dataset from Kaggle and arrange it under:

~/STH/datasets/MMU/
  raw/   # original structure from Kaggle
  imgs/  # flattened BMP images (no subfolders)
  masks/ # auto-generated placeholder masks (heuristic)
"""

import os
import shutil
from pathlib import Path

import cv2
import kagglehub
import numpy as np


def generate_mask(img_path: Path) -> np.ndarray:
    """Heuristic mask via Otsu + morphology + largest contour."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image {img_path}")

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 255, thickness=-1)
    return mask


def main():
    print("⬇️  Downloading MMU iris dataset from Kaggle...")
    download_path = Path(kagglehub.dataset_download("naureenmohammad/mmu-iris-dataset"))

    target_root = Path.home() / "STH" / "datasets" / "MMU"
    raw_dir = target_root / "raw"
    imgs_dir = target_root / "imgs"
    masks_dir = target_root / "masks"

    # Keep original structure
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    shutil.copytree(download_path, raw_dir, dirs_exist_ok=True)

    # Flatten images
    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for bmp in raw_dir.rglob("*.bmp"):
        target_img = imgs_dir / bmp.name
        shutil.copy2(bmp, target_img)
        mask = generate_mask(target_img)
        cv2.imwrite(str(masks_dir / f"{target_img.stem}.png"), mask)
        count += 1

    print(f"✅ Downloaded to {raw_dir}")
    print(f"✅ Flattened {count} images into {imgs_dir}")
    print(f"✅ Auto masks saved to {masks_dir} (heuristic; replace with ground-truth if available)")


if __name__ == "__main__":
    main()