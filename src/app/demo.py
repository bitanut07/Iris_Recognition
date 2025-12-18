"""
Demo app for biometric recognition pipeline.

- Uses a mock camera to generate frames.
- Demonstrates:
  - Enrollment for a few users.
  - Runtime recognition for new frames.

MongoDB:
- The environment variable `MONGODB_URL` must point to your MongoDB instance.
"""

import argparse
from typing import List

import torch

from src.pipeline.inference import BiometricPipeline, PipelineConfig


def mock_camera_frame(
    batch_size: int = 1,
    height: int = 480,
    width: int = 640,
) -> torch.Tensor:
    """
    Simple mock camera input.

    Returns:
        frame: Tensor [1, 3, H, W] with random content in [0,1].
    """
    frame = torch.rand(batch_size, 3, height, width, dtype=torch.float32)
    return frame


def enroll_demo_users(
    pipeline: BiometricPipeline,
    num_users: int = 2,
    frames_per_user: int = 5,
) -> None:
    """
    Enroll a few demo users using mock frames.
    """
    print(f"Enrolling {num_users} demo users...")

    for user_idx in range(num_users):
        user_id = f"user_{user_idx + 1}"
        frames: List[torch.Tensor] = []
        for _ in range(frames_per_user):
            frame = mock_camera_frame()
            frames.append(frame)

        emb = pipeline.enroll_user(user_id, frames)
        print(f"Enrolled {user_id}, embedding norm={emb.norm().item():.4f}")


def recognition_loop(
    pipeline: BiometricPipeline,
    num_queries: int = 5,
) -> None:
    """
    Run a simple recognition loop on mock frames.
    """
    print(f"\nStarting recognition loop with {num_queries} queries...")
    for i in range(num_queries):
        frame = mock_camera_frame()
        user_id, score = pipeline.recognize(frame)
        print(f"[Query {i + 1}] -> user_id={user_id}, score={score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biometric Recognition Demo (with MongoDB + FAISS)")

    parser.add_argument(
        "--seg-ckpt",
        type=str,
        required=True,
        help="Path to segmentation model checkpoint (.pth)",
    )
    parser.add_argument(
        "--rec-ckpt",
        type=str,
        required=True,
        help="Path to recognition model checkpoint (.pth)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
        help="Distance metric for vector DB",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for open-set decision (cosine: similarity; l2: distance)",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=2,
        help="Number of demo users to enroll",
    )
    parser.add_argument(
        "--frames-per-user",
        type=int,
        default=5,
        help="Number of frames per user during enrollment",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of recognition queries to run",
    )

    args = parser.parse_args()

    config = PipelineConfig(
        seg_checkpoint=args.seg_ckpt,
        rec_checkpoint=args.rec_ckpt,
        metric=args.metric,
        threshold=args.threshold,
    )

    pipeline = BiometricPipeline(config)

    # Enrollment phase
    enroll_demo_users(
        pipeline,
        num_users=args.num_users,
        frames_per_user=args.frames_per_user,
    )

    # Recognition phase
    recognition_loop(
        pipeline,
        num_queries=args.num_queries,
    )


if __name__ == "__main__":
    main()


