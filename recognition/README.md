# Iris Recognition Module

PyTorch implementation for iris recognition using CNN + Triplet Loss.

## Structure

- `dataset.py`: TripletDataset class for loading masked ROI images with triplet sampling
- `model.py`: ResNet18-based embedding model (IrisEmbeddingModel)
- `loss.py`: Triplet Loss implementation from scratch
- `train.py`: Training script
- `inference.py`: Inference utilities (embeddings, similarity, verification)

## Usage

### Training

```bash
python recognition/train.py \
    --images-dir datasets/MMU/imgs \
    --masks-dir datasets/MMU/masks \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --embedding-dim 128 \
    --margin 0.3 \
    --use-crop
```

### Inference

```python
from recognition import IrisRecognizer

# Initialize recognizer
recognizer = IrisRecognizer(
    checkpoint_path="checkpoints/recognition/recognition_epoch_50.pth",
    embedding_dim=128
)

# Get embeddings
embeddings = recognizer.get_embeddings(
    images=["path/to/image1.bmp", "path/to/image2.bmp"],
    masks=["path/to/mask1.png", "path/to/mask2.png"]
)

# Verify two images
is_match, similarity = recognizer.verify(
    image1="path/to/image1.bmp",
    image2="path/to/image2.bmp",
    mask1="path/to/mask1.png",
    mask2="path/to/mask2.png",
    threshold=0.7
)
```

## Key Features

- **Triplet Sampling**: Automatic anchor/positive/negative sampling
- **Masked ROI**: Applies segmentation masks to extract iris region
- **ResNet18 Backbone**: Pretrained ResNet18 with custom embedding layer
- **L2 Normalization**: Embeddings are L2-normalized
- **Triplet Loss**: Implemented from scratch with Euclidean distance
- **Cosine Similarity**: For verification and matching

