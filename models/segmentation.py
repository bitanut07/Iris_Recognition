"""
Segmentation training script for MMU dataset using the U-Net template
from CodeTemplate/Pytorch-UNet.

Expected layout (flattened, no subfolders):
datasets/MMU/
  imgs/   # input images
  masks/  # target masks with matching filenames

If your MMU download is still nested per person/left/right, please copy or
symlink all images into datasets/MMU/imgs and place the corresponding masks
into datasets/MMU/masks using the same base filename.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Allow importing local modules as a package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from unet.unet_model import UNet  # noqa: E402
from utils.data_loading import BasicDataset  # noqa: E402
from utils.dice_score import dice_loss  # noqa: E402
from utils.evaluate import evaluate  # noqa: E402

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "MMU"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "mmu"


def generate_mask_from_image(img_path: Path) -> np.ndarray:
    """Simple heuristic mask: Otsu threshold + morphology + largest contour."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image {img_path}")

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 255, thickness=-1)

    return mask


def prepare_mmu_dataset(base_root: Path, force: bool = False):
    """
    Flatten MMU structure (person/left|right/*.bmp) into imgs/ and auto-generate
    binary masks into masks/ with matching basenames.
    """
    imgs_dir = base_root / "imgs"
    masks_dir = base_root / "masks"

    if imgs_dir.exists() and masks_dir.exists() and not force:
        logging.info("Found existing imgs/ and masks/ - skip auto-prepare.")
        return

    if force:
        shutil.rmtree(imgs_dir, ignore_errors=True)
        shutil.rmtree(masks_dir, ignore_errors=True)

    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for bmp in base_root.rglob("*.bmp"):
        if "imgs" in bmp.parts or "masks" in bmp.parts:
            continue
        target_img = imgs_dir / bmp.name
        if not target_img.exists():
            shutil.copy2(bmp, target_img)
        mask = generate_mask_from_image(target_img)
        mask_path = masks_dir / f"{target_img.stem}.png"
        cv2.imwrite(str(mask_path), mask)
        count += 1

    logging.info(f"Prepared {count} samples into {imgs_dir} and masks/")


def train(
    data_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_percent: float,
    img_scale: float,
    amp: bool,
    bilinear: bool,
    n_channels: int,
    n_classes: int,
    load_ckpt: str | None,
):
    # Primary expected layout: data_root/imgs and data_root/masks
    dir_img = data_root / "imgs"
    dir_mask = data_root / "masks"

    # Fallback: if user passes imgs directly as data_root
    if not dir_img.exists() and data_root.name == "imgs":
        dir_img = data_root
        dir_mask = data_root.parent / "masks"

    if not dir_img.exists() or not dir_mask.exists():
        alt = data_root / "MMU"  # in case of accidental nested path
        if alt.exists() and dir_mask.exists():
            dir_img = alt
        else:
            raise FileNotFoundError(
                f"Expected folders {dir_img} and {dir_mask}. Flatten your MMU images "
                "into imgs/ and masks/ with matching filenames before training."
            )

    try:
        dataset = BasicDataset(dir_img, dir_mask, scale=img_scale)
    except (AssertionError, RuntimeError, IndexError) as exc:
        raise RuntimeError(
            "Failed to load dataset. Ensure imgs/ and masks/ contain matching "
            "files without subfolders."
        ) from exc

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    loader_args = dict(
        batch_size=batch_size, num_workers=min(4, os.cpu_count() or 1), pin_memory=True
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    model = model.to(memory_format=torch.channels_last).to(device=device)

    logging.info(
        f"Network: {model.n_channels} input channels, {model.n_classes} classes, "
        f"{'bilinear' if model.bilinear else 'transposed'} upsampling"
    )

    if load_ckpt:
        state_dict = torch.load(load_ckpt, map_location=device)
        state_dict.pop("mask_values", None)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {load_ckpt}")

    optimizer = optim.RMSprop(
        model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9, foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]

                assert images.shape[1] == model.n_channels, (
                    f"Model expects {model.n_channels} channels but got {images.shape[1]}"
                )

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(
                            torch.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False,
                        )
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            torch.softmax(masks_pred, dim=1).float(),
                            torch.nn.functional.one_hot(true_masks, model.n_classes)
                            .permute(0, 3, 1, 2)
                            .float(),
                            multiclass=True,
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                division_step = max(n_train // (5 * batch_size), 1)
                if global_step % division_step == 0:
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)
                    logging.info(f"Validation Dice: {val_score:.4f}")

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict["mask_values"] = dataset.mask_values
        ckpt_path = CHECKPOINT_DIR / f"mmu_epoch{epoch}.pth"
        torch.save(state_dict, ckpt_path)
        logging.info(f"Checkpoint saved: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on MMU dataset (segment iris masks).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root dir containing imgs/ and masks/")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild imgs/ and masks/ from raw MMU layout")
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-4, dest="lr")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Resize factor for images/masks")
    parser.add_argument("--validation", "-v", type=float, default=10.0, help="Validation split percent (0-100)")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", help="Use bilinear upsampling (default: transposed conv)")
    parser.add_argument("--channels", "-ch", type=int, default=1, help="Input channels (1 for gray, 3 for RGB)")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes")
    parser.add_argument("--load", "-f", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    prepare_mmu_dataset(args.data_root, force=args.force_prepare)

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_percent=args.validation / 100,
        img_scale=args.scale,
        amp=args.amp,
        bilinear=args.bilinear,
        n_channels=args.channels,
        n_classes=args.classes,
        load_ckpt=args.load,
    )