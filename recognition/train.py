"""
Training script for Iris Recognition using Triplet Loss
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Handle imports - works both as script and module
try:
    from .dataset import TripletDataset
    from .loss import TripletLoss
    from .model import IrisEmbeddingModel
except ImportError:
    # If running as script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from recognition.dataset import TripletDataset
    from recognition.loss import TripletLoss
    from recognition.model import IrisEmbeddingModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(
    images_dir: str,
    masks_dir: str,
    labels_file: str = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    embedding_dim: int = 128,
    margin: float = 0.3,
    image_size: int = 224,
    use_crop: bool = True,
    checkpoint_dir: str = "checkpoints/recognition",
    resume_from: str = None,
):
    """
    Train the iris recognition model.
    
    Args:
        images_dir: Directory containing input images
        masks_dir: Directory containing segmentation masks
        labels_file: Optional file mapping image names to identity labels
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for Adam optimizer
        embedding_dim: Dimension of embedding vector
        margin: Margin for triplet loss
        image_size: Target image size
        use_crop: Whether to crop by mask or multiply
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = TripletDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        labels_file=labels_file,
        image_size=image_size,
        use_crop=use_crop,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Number of batches: {len(dataloader)}")
    
    # Create model
    logger.info("Initializing model...")
    model = IrisEmbeddingModel(embedding_dim=embedding_dim, pretrained=True)
    model = model.to(device)
    
    # Create loss function
    criterion = TripletLoss(margin=margin)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        logger.info(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            # Get embeddings for anchor, positive, and negative
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print loss every iteration
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                f"Loss: {loss.item():.6f}"
            )
        
        # Average loss for epoch
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        checkpoint_file = checkpoint_path / f"recognition_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'embedding_dim': embedding_dim,
        }, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Iris Recognition Model with Triplet Loss")
    
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="Directory containing segmentation masks")
    parser.add_argument("--labels-file", type=str, default=None,
                        help="Optional file mapping image names to identity labels")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet loss margin")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Target image size")
    parser.add_argument("--use-crop", action="store_true",
                        help="Crop by mask bounding box (default: multiply by mask)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/recognition",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        labels_file=args.labels_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        image_size=args.image_size,
        use_crop=args.use_crop,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()

