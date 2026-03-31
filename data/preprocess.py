from __future__ import annotations

"""
preprocess.py - Data Preprocessing for Depth Estimation Dataset.

Chức năng:
1. Lọc dữ liệu theo min_depth, max_depth, environment
2. Chuyển đổi depth maps → disparity maps (nếu cần)
3. Resize ảnh và disparity maps về new_size (640x480)
4. Điều chỉnh disparity theo resize ratio
5. Lưu disparity dạng 16-bit PNG (disp * 100 → uint16)
6. Cập nhật CSV với đường dẫn mới

Công thức: disp = (f * B) / depth
Disparity được điều chỉnh: disp_resized = disp * (new_width / old_width)

Usage:
    python preprocess.py --csv dataset_index.csv --new-size 640 480 \
        --min-depth 0.5 --max-depth 10.0 --environment indoor
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import h5py
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Dataset constants ────────────────────────────────────────────────────────
DATASET_INFO = {
    "fat_indoor":      {"type": "depth",     "baseline_m": 0.06,  "depth_scale": 10000.0},
    "fat_outdoor":     {"type": "depth",     "baseline_m": 0.06,  "depth_scale": 10000.0},
    "Stereo2k_indoor": {"type": "disparity", "baseline_m": 0.10,  "disp_scale": 100.0},
    "spring_outdoor":  {"type": "disparity", "baseline_m": 0.065, "disp_scale": None},
}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_fat_depth(path: str) -> np.ndarray:
    """Load FAT 16-bit PNG depth → float32 meters."""
    return np.array(Image.open(path), dtype=np.float32) / 10000.0


def load_stereo2k_disp(path: str) -> np.ndarray:
    """Load Stereo2k 16-bit PNG disparity → float32 pixels."""
    return np.array(Image.open(path), dtype=np.float32) / 100.0


def load_spring_disp(path: str) -> np.ndarray:
    """Load Spring .dsp5 disparity → float32 pixels.
    Downsample 2x to match image resolution (3840→1920).
    Values remain in original pixel units.
    """
    with h5py.File(path, "r") as f:
        disp = f["disparity"][()].astype(np.float32)
    disp = np.ascontiguousarray(disp[::2, ::2])
    disp[disp == np.inf] = -1
    return np.ascontiguousarray(disp)


def get_dataset_source(path: str) -> str:
    """Extract dataset name from image path."""
    return path.split("/")[0]


# ── Core Processing ──────────────────────────────────────────────────────────

def depth_to_disparity(depth: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    """
    Convert depth map to disparity map.
    disp = (f * B) / depth
    Invalid pixels (depth <= 0) → 0
    """
    disp = np.zeros_like(depth)
    valid = depth > 0
    disp[valid] = (focal_px * baseline_m) / depth[valid]
    return disp


def resize_with_pad(img: np.ndarray, new_size: tuple[int, int],
                    interpolation=cv2.INTER_LINEAR,
                    pad_value=0) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image keeping aspect ratio, then pad to target size.
    Preserves spatial structure by not distorting the image.

    Args:
        img: input image (H, W) or (H, W, C)
        new_size: (target_width, target_height)
        interpolation: cv2 interpolation flag
        pad_value: value for padding pixels

    Returns:
        (resized_padded_image, resize_ratio, (pad_top, pad_left))
    """
    target_w, target_h = new_size
    h, w = img.shape[:2]

    # Compute resize ratio (fit inside target, preserving aspect ratio)
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # Pad to target size (center padding)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    if img.ndim == 3:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    else:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_value)

    return padded, ratio, (pad_top, pad_left)


def resize_disparity_with_pad(disp: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """
    Resize disparity map keeping aspect ratio + pad, adjust values by resize ratio.
    Padding pixels get disparity = 0 (invalid).
    """
    padded, ratio, _ = resize_with_pad(disp, new_size,
                                       interpolation=cv2.INTER_NEAREST,
                                       pad_value=0)
    # Scale disparity values by width ratio (same as resize ratio)
    padded = padded * ratio
    return padded


def save_disparity_png(disp: np.ndarray, path: str):
    """Save disparity as 16-bit PNG (disp * 100 → uint16)."""
    disp_uint16 = np.clip(disp * 100.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, disp_uint16)


def process_single_sample(row: pd.Series, root: Path, output_dir: Path,
                          new_size: tuple[int, int]) -> dict | None:
    """
    Process a single stereo sample:
    1. Load left image, right image, and depth/disparity (left)
    2. Convert depth → disparity if needed
    3. Resize all with aspect ratio preservation + padding
    4. Save to output_dir
    Returns updated row dict or None if failed.
    """
    source = get_dataset_source(row["left_image_path"])
    info = DATASET_INFO.get(source)
    if info is None:
        logger.warning(f"Unknown source: {source}")
        return None

    # Load left image
    left_path = root / row["left_image_path"]
    left_img = cv2.imread(str(left_path))
    if left_img is None:
        logger.warning(f"Cannot read left image: {left_path}")
        return None

    # Load right image
    right_img = None
    if pd.notna(row.get("right_image_path")):
        right_path = root / row["right_image_path"]
        right_img = cv2.imread(str(right_path))
        if right_img is None:
            logger.warning(f"Cannot read right image: {right_path}")
            return None

    # Load and compute left disparity
    disp = None
    if info["type"] == "depth" and pd.notna(row.get("depth_path")):
        depth = load_fat_depth(str(root / row["depth_path"]))
        focal_px = depth.shape[1]
        disp = depth_to_disparity(depth, focal_px, info["baseline_m"])

    elif info["type"] == "disparity" and pd.notna(row.get("disparity_path")):
        disp_path = str(root / row["disparity_path"])
        if disp_path.endswith(".dsp5"):
            disp = load_spring_disp(disp_path)
        else:
            disp = load_stereo2k_disp(disp_path)

    if disp is None:
        logger.warning(f"No depth/disparity for: {row['left_image_path']}")
        return None

    # Resize with aspect ratio preservation + padding
    left_resized, _, _ = resize_with_pad(
        left_img, new_size, interpolation=cv2.INTER_LINEAR, pad_value=0)
    disp_resized = resize_disparity_with_pad(disp, new_size)

    # Right image: same resize + pad (must match left exactly)
    right_out_path_str = None
    if right_img is not None:
        right_resized, _, _ = resize_with_pad(
            right_img, new_size, interpolation=cv2.INTER_LINEAR, pad_value=0)

    # Build output paths (flat structure)
    rel_path = row["left_image_path"].replace("/", "_").rsplit(".", 1)[0]
    left_out = output_dir / "left" / f"{rel_path}.png"
    right_out = output_dir / "right" / f"{rel_path}.png"
    disp_out = output_dir / "disparities" / f"{rel_path}_disp.png"

    left_out.parent.mkdir(parents=True, exist_ok=True)
    disp_out.parent.mkdir(parents=True, exist_ok=True)

    # Save left + disparity
    cv2.imwrite(str(left_out), left_resized)
    save_disparity_png(disp_resized, str(disp_out))

    # Save right
    if right_img is not None:
        right_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(right_out), right_resized)
        right_out_path_str = str(right_out.relative_to(root))

    return {
        "left_image_path":  str(left_out.relative_to(root)),
        "right_image_path": right_out_path_str,
        "depth_path":       row.get("depth_path"),
        "disparity_path":   str(disp_out.relative_to(root)),
        "min_depth":        row.get("min_depth"),
        "max_depth":        row.get("max_depth"),
        "environment":      row.get("environment"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess Depth Estimation Dataset")
    parser.add_argument("--csv", type=str, default="dataset_index.csv", help="Input CSV")
    parser.add_argument("--root", type=str, default=".", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="processed_data",
                        help="Output directory for processed data")
    parser.add_argument("--new-size", type=int, nargs=2, default=[640, 480],
                        metavar=("WIDTH", "HEIGHT"), help="Target image size (default: 640 480)")
    parser.add_argument("--min-depth", type=float, default=None,
                        help="Filter: minimum depth in meters")
    parser.add_argument("--max-depth", type=float, default=None,
                        help="Filter: maximum depth in meters")
    parser.add_argument("--environment", type=str, default=None,
                        choices=["indoor", "outdoor", "mixed"],
                        help="Filter: environment type (mixed = keep both indoor and outdoor)")
    parser.add_argument("--split-valid", type=float, default=0.0,
                        help="Fraction of data for validation split (0.0 = no split, e.g. 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output CSV path (default: processed_data/processed_index.csv)")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    new_size = tuple(args.new_size)  # (width, height)

    # Load CSV
    df = pd.read_csv(root / args.csv)
    logger.info(f"Loaded {len(df):,} samples from {args.csv}")

    # Filter by environment ('mixed' keeps both indoor and outdoor)
    if args.environment and args.environment != "mixed":
        df = df[df["environment"] == args.environment]
        logger.info(f"After environment filter ({args.environment}): {len(df):,} samples")
    elif args.environment == "mixed":
        logger.info(f"Environment: mixed (keeping all: {len(df):,} samples)")

    # Filter by depth range
    if args.min_depth is not None:
        df = df[df["min_depth"] >= args.min_depth]
        logger.info(f"After min_depth filter (>= {args.min_depth}m): {len(df):,} samples")

    if args.max_depth is not None:
        df = df[df["max_depth"] <= args.max_depth]
        logger.info(f"After max_depth filter (<= {args.max_depth}m): {len(df):,} samples")

    if len(df) == 0:
        logger.error("No samples left after filtering. Adjust filter parameters.")
        return

    logger.info(f"Processing {len(df):,} samples → {output_dir} (size: {new_size[0]}x{new_size[1]})")

    # Process samples
    processed_records = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        result = process_single_sample(row, root, output_dir, new_size)
        if result:
            processed_records.append(result)
        else:
            failed += 1

    # Save processed CSV
    df_out = pd.DataFrame(processed_records)

    # Train/Val split
    if args.split_valid > 0.0:
        np.random.seed(args.seed)
        indices = np.random.permutation(len(df_out))
        val_size = int(len(df_out) * args.split_valid)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        df_train = df_out.iloc[train_indices].reset_index(drop=True)
        df_val = df_out.iloc[val_indices].reset_index(drop=True)

        train_csv = str(output_dir / "train.csv")
        val_csv = str(output_dir / "val.csv")
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)

        logger.info(f"  Train/Val split: {len(df_train):,} / {len(df_val):,} "
                     f"({1 - args.split_valid:.0%} / {args.split_valid:.0%})")
        logger.info(f"  Train CSV: {train_csv}")
        logger.info(f"  Val CSV:   {val_csv}")

    # Also save the full combined CSV
    out_csv = args.output_csv or str(output_dir / "processed_index.csv")
    df_out.to_csv(out_csv, index=False)

    logger.info(f"\nProcessing complete!")
    logger.info(f"  Processed: {len(processed_records):,}")
    logger.info(f"  Failed:    {failed:,}")
    logger.info(f"  CSV saved: {out_csv}")
    logger.info(f"  Left:      {output_dir / 'left'}")
    logger.info(f"  Right:     {output_dir / 'right'}")
    logger.info(f"  Disparity: {output_dir / 'disparities'}")

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Input:       {args.csv} ({len(df):,} samples after filtering)")
    print(f"Output:      {out_csv} ({len(processed_records):,} samples)")
    print(f"Image size:  {new_size[0]}x{new_size[1]}")
    print(f"Disp format: 16-bit PNG (value * 100)")
    if args.environment:
        print(f"Environment: {args.environment}")
    if args.min_depth:
        print(f"Min depth:   {args.min_depth}m")
    if args.max_depth:
        print(f"Max depth:   {args.max_depth}m")
    if args.split_valid > 0:
        print(f"Split:       train={len(df_train):,} / val={len(df_val):,}")


if __name__ == "__main__":
    main()
