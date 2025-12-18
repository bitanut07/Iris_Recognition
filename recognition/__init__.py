"""
Iris Recognition Module
Provides tools for training and inference using CNN + Triplet Loss
"""

from .dataset import TripletDataset
from .model import IrisEmbeddingModel
from .loss import TripletLoss
from .inference import IrisRecognizer, cosine_similarity

__all__ = [
    'TripletDataset',
    'IrisEmbeddingModel',
    'TripletLoss',
    'IrisRecognizer',
    'cosine_similarity',
]

