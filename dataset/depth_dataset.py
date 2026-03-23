from __future__ import annotations

"""
depth_dataset.py - PyTorch Dataset and DataLoader for Stereo Depth Estimation.

Đọc stereo pairs (left + right images) và left disparity maps từ processed CSV.
Hỗ trợ normalization, augmentation, batching, shuffling.

Usage:
    from depth_dataset import create_dataloaders

    train_loader, val_loader = create_dataloaders(
        csv_path="processed_data/train.csv",
        root=".",
        batch_size=8,
    )
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StereoDepthDataset(Dataset):
    """
    PyTorch Dataset for stereo depth estimation.

    Reads stereo pairs (left + right RGB images) and left disparity maps
    (16-bit PNG, value/100 = real disp in pixels).
    Applies normalization and optional augmentation.

    Args:
        csv_path: Path to processed CSV index
        root: Dataset root directory
        transform: Optional custom transform for RGB images
        augment: Whether to apply data augmentation (for training)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, csv_path: str, root: str = ".",
                 transform=None, augment: bool = False):
        self.root = Path(root)
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        self.transform = transform or T.ToTensor()

        logger.info(f"Dataset loaded: {len(self.df)} samples (augment={augment})")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load left image (BGR → RGB)
        left_path = str(self.root / row["left_image_path"])
        left_img = cv2.imread(left_path)
        if left_img is None:
            raise FileNotFoundError(f"Cannot read left image: {left_path}")
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

        # Load right image (BGR → RGB)
        right_path = str(self.root / row["right_image_path"])
        right_img = cv2.imread(right_path)
        if right_img is None:
            raise FileNotFoundError(f"Cannot read right image: {right_path}")
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Load left disparity map (16-bit PNG → float32)
        disp_path = str(self.root / row["disparity_path"])
        disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise FileNotFoundError(f"Cannot read disparity: {disp_path}")
        disp = disp.astype(np.float32) / 100.0

        # Augmentation (training only)
        if self.augment:
            left_img, right_img, disp = self._augment(left_img, right_img, disp)

        # Apply transform to both images
        left_tensor = self.transform(left_img)
        right_tensor = self.transform(right_img)

        # Disparity → tensor (1, H, W)
        disp_tensor = torch.from_numpy(disp).unsqueeze(0).float()

        return {
            "left": left_tensor,
            "right": right_tensor,
            "disp": disp_tensor,
            "min_depth": float(row["min_depth"]) if pd.notna(row.get("min_depth")) else 0.0,
            "max_depth": float(row["max_depth"]) if pd.notna(row.get("max_depth")) else 0.0,
            "environment": row.get("environment", "unknown"),
            "left_image_path": row["left_image_path"],
        }

    def _augment(self, left: np.ndarray, right: np.ndarray,
                 disp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random augmentations to stereo pair and disparity.
        Note: horizontal flip is NOT applied for stereo (would swap left/right semantics).
        """
        # Random brightness/contrast (applied identically to both images)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-10, 10)
            left = np.clip(alpha * left.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            right = np.clip(alpha * right.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Random color jitter (applied identically to both images)
        if random.random() > 0.5:
            hue_shift = random.uniform(-10, 10)
            sat_scale = random.uniform(0.8, 1.2)
            for img_ref in [left, right]:
                hsv = cv2.cvtColor(img_ref, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
                img_ref[:] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return left, right, disp


def create_dataloaders(
    train_csv: str,
    val_csv: str = None,
    root: str = ".",
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 4,
    augment_train: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for stereo depth estimation.

    Args:
        train_csv: Path to training CSV (or full CSV if val_csv is None)
        val_csv: Path to validation CSV. If None, split from train_csv using val_split.
        root: Dataset root directory
        batch_size: Batch size
        val_split: Fraction for validation (only used if val_csv is None)
        num_workers: Number of dataloader workers
        augment_train: Apply augmentation to training set
        seed: Random seed for reproducible splits

    Returns:
        (train_loader, val_loader)
    """
    if val_csv is not None:
        # Use pre-split CSVs
        train_dataset = StereoDepthDataset(train_csv, root, augment=augment_train)
        val_dataset = StereoDepthDataset(val_csv, root, augment=False)
    else:
        # Split from single CSV
        full_dataset = StereoDepthDataset(train_csv, root, augment=False)
        total = len(full_dataset)
        val_size = int(total * val_split)
        train_size = total - val_size

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_dataset = StereoDepthDataset(train_csv, root, augment=augment_train)
        val_dataset = StereoDepthDataset(train_csv, root, augment=False)

        train_dataset.df = full_dataset.df.iloc[train_subset.indices].reset_index(drop=True)
        val_dataset.df = full_dataset.df.iloc[val_subset.indices].reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"DataLoaders created: train={len(train_dataset)}, val={len(val_dataset)}, "
                f"batch_size={batch_size}")

    return train_loader, val_loader


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Stereo Depth Dataset")
    parser.add_argument("--csv", type=str, default="processed_data/processed_index.csv")
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train_loader, val_loader = create_dataloaders(
        train_csv=args.csv,
        val_csv=args.val_csv,
        root=args.root,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=0,
    )

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nStereo batch test:")
    print(f"  Left shape:      {batch['left'].shape}")
    print(f"  Right shape:     {batch['right'].shape}")
    print(f"  Disparity shape: {batch['disp'].shape}")
    print(f"  Min depth:       {batch['min_depth']}")
    print(f"  Max depth:       {batch['max_depth']}")
    print(f"  Environment:     {batch['environment']}")
    print(f"  Left range:      [{batch['left'].min():.3f}, {batch['left'].max():.3f}]")
    print(f"  Right range:     [{batch['right'].min():.3f}, {batch['right'].max():.3f}]")
    print(f"  Disp range:      [{batch['disp'].min():.3f}, {batch['disp'].max():.3f}]")

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print("\nStereo dataset test passed!")
