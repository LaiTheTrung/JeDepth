from __future__ import annotations

"""
build_index.py - Quét toàn bộ dataset và tạo file CSV index.

CSV columns: left_image_path, right_image_path, depth_path, disparity_path,
             min_depth, max_depth, environment
Datasets: fat_indoor, fat_outdoor, Stereo2k_indoor, spring_outdoor

Usage:
    python build_index.py [--root DATASET_ROOT] [--output OUTPUT_CSV]
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import h5py
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Dataset-specific constants ───────────────────────────────────────────────
DATASET_CONFIG = {
    "fat_indoor":      {"type": "depth",     "baseline_m": 0.06,  "environment": "indoor"},
    "fat_outdoor":     {"type": "depth",     "baseline_m": 0.06,  "environment": "outdoor"},
    "Stereo2k_indoor": {"type": "disparity", "baseline_m": 0.10,  "environment": "indoor"},
    "spring_outdoor":  {"type": "disparity", "baseline_m": 0.065, "environment": "outdoor"},
}


# ── Helper: load depth/disparity arrays ──────────────────────────────────────

def load_fat_depth(path: Path) -> np.ndarray:
    """Load FAT 16-bit PNG depth map → float32 in meters."""
    arr = np.array(Image.open(path), dtype=np.float32) / 10000.0
    return arr


def load_stereo2k_disp(path: Path) -> np.ndarray:
    """Load Stereo2k 16-bit PNG disparity → float32 in pixels."""
    arr = np.array(Image.open(path), dtype=np.float32) / 100.0
    return arr


def load_spring_disp(path: Path) -> np.ndarray:
    """Load Spring .dsp5 (HDF5) disparity → float32 in pixels.
    Spring disp maps are at 2x image resolution (3840x2160).
    Downsample 2x to match image resolution (1920x1080).
    Disparity values remain in original pixel units (at 3840 resolution).
    """
    with h5py.File(path, "r") as f:
        disp = f["disparity"][()].astype(np.float32)
    # Downsample 2x spatially (keep disparity values unchanged)
    disp = np.ascontiguousarray(disp[::2, ::2])
    disp[disp == np.inf] = -1
    disp = np.ascontiguousarray(disp)
    return disp


def compute_depth_stats(depth: np.ndarray):
    """Compute min/max depth from a depth map (meters). Ignore zeros/invalid."""
    valid = depth[depth > 0]
    if valid.size == 0:
        return None, None
    return float(valid.min()), float(valid.max())


def disp_to_depth_stats(disp: np.ndarray, focal_px: float, baseline_m: float):
    """Convert disparity → depth, then compute min/max depth (meters)."""
    valid = disp[disp > 0]
    if valid.size == 0:
        return None, None
    depths = (focal_px * baseline_m) / valid
    return float(depths.min()), float(depths.max())


# ── Scanners ─────────────────────────────────────────────────────────────────

def scan_fat(top_folder: Path, config: dict) -> list[dict]:
    """
    Scan FAT indoor/outdoor dataset.
    Structure: top_folder/{scene}/{XXXXXX}.{left|right}.jpg + {XXXXXX}.left.depth.png
    """
    env = config["environment"]
    records = []

    scene_folders = sorted([d for d in top_folder.iterdir() if d.is_dir()])
    for scene in tqdm(scene_folders, desc=f"  {top_folder.name}", unit="scene"):
        groups: dict[str, dict] = defaultdict(dict)
        for f in scene.iterdir():
            parts = f.name.split(".")
            if len(parts) == 3 and parts[2] == "jpg" and parts[1] == "left":
                groups[parts[0]]["left_image"] = f
            elif len(parts) == 3 and parts[2] == "jpg" and parts[1] == "right":
                groups[parts[0]]["right_image"] = f
            elif len(parts) == 4 and parts[2] == "depth" and parts[3] == "png" and parts[1] == "left":
                groups[parts[0]]["depth"] = f

        for frame_id, files in sorted(groups.items()):
            left_img = files.get("left_image")
            right_img = files.get("right_image")
            depth_path = files.get("depth")
            if left_img is None:
                continue

            min_d, max_d = (None, None)
            if depth_path and depth_path.exists():
                depth_arr = load_fat_depth(depth_path)
                min_d, max_d = compute_depth_stats(depth_arr)

            records.append({
                "left_image_path":  str(left_img.relative_to(top_folder.parent)),
                "right_image_path": str(right_img.relative_to(top_folder.parent)) if right_img else None,
                "depth_path":       str(depth_path.relative_to(top_folder.parent)) if depth_path else None,
                "disparity_path":   None,
                "min_depth":        min_d,
                "max_depth":        max_d,
                "environment":      env,
            })
    return records


def scan_stereo2k(top_folder: Path, config: dict) -> list[dict]:
    """
    Scan Stereo2k indoor dataset.
    Structure: top_folder/{sample_id}/left.png + right.png + left_disp.png
    """
    env = config["environment"]
    baseline_m = config["baseline_m"]
    records = []

    sample_folders = sorted([d for d in top_folder.iterdir() if d.is_dir()])
    for sample in tqdm(sample_folders, desc=f"  {top_folder.name}", unit="sample"):
        left_img = sample / "left.png"
        right_img = sample / "right.png"
        disp_path = sample / "left_disp.png"

        if not left_img.exists():
            continue

        min_d, max_d = (None, None)
        if disp_path.exists():
            disp_arr = load_stereo2k_disp(disp_path)
            w = disp_arr.shape[1]
            focal_px = w
            min_d, max_d = disp_to_depth_stats(disp_arr, focal_px, baseline_m)

        records.append({
            "left_image_path":  str(left_img.relative_to(top_folder.parent)),
            "right_image_path": str(right_img.relative_to(top_folder.parent)) if right_img.exists() else None,
            "depth_path":       None,
            "disparity_path":   str(disp_path.relative_to(top_folder.parent)) if disp_path.exists() else None,
            "min_depth":        min_d,
            "max_depth":        max_d,
            "environment":      env,
        })
    return records


def scan_spring(top_folder: Path, config: dict) -> list[dict]:
    """
    Scan Spring outdoor dataset.
    Structure: top_folder/{seq}/frame_{left|right}/frame_{left|right}_XXXX.png
               + disp1_left/disp1_left_XXXX.dsp5
    """
    env = config["environment"]
    baseline_m = config["baseline_m"]
    records = []

    seq_folders = sorted([d for d in top_folder.iterdir() if d.is_dir()])
    for seq in tqdm(seq_folders, desc=f"  {top_folder.name}", unit="seq"):
        frame_left_dir = seq / "frame_left"
        frame_right_dir = seq / "frame_right"
        disp_dir = seq / "disp1_left"

        if not frame_left_dir.exists():
            continue

        frame_files = sorted(frame_left_dir.glob("frame_left_*.png"))
        for frame_file in frame_files:
            frame_num = frame_file.stem.replace("frame_left_", "")
            right_file = frame_right_dir / f"frame_right_{frame_num}.png"
            disp_file = disp_dir / f"disp1_left_{frame_num}.dsp5"

            min_d, max_d = (None, None)
            if disp_file.exists():
                disp_arr = load_spring_disp(disp_file)
                focal_px = disp_arr.shape[1]  # 1920 after downsample
                min_d, max_d = disp_to_depth_stats(disp_arr, focal_px, baseline_m)

            records.append({
                "left_image_path":  str(frame_file.relative_to(top_folder.parent)),
                "right_image_path": str(right_file.relative_to(top_folder.parent)) if right_file.exists() else None,
                "depth_path":       None,
                "disparity_path":   str(disp_file.relative_to(top_folder.parent)) if disp_file.exists() else None,
                "min_depth":        min_d,
                "max_depth":        max_d,
                "environment":      env,
            })
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

SCANNER_MAP = {
    "fat_indoor":      scan_fat,
    "fat_outdoor":     scan_fat,
    "Stereo2k_indoor": scan_stereo2k,
    "spring_outdoor":  scan_spring,
}


def build_index(root: Path) -> pd.DataFrame:
    """Scan all datasets and return a unified DataFrame."""
    all_records = []

    for dataset_name, config in DATASET_CONFIG.items():
        folder = root / dataset_name
        if not folder.exists():
            logger.warning(f"Dataset folder not found: {folder}, skipping.")
            continue

        scanner = SCANNER_MAP[dataset_name]
        logger.info(f"Scanning {dataset_name} ({config['type']}, {config['environment']})...")
        records = scanner(folder, config)
        logger.info(f"  → {len(records):,} samples")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df["min_depth"] = pd.to_numeric(df["min_depth"], errors="coerce")
    df["max_depth"] = pd.to_numeric(df["max_depth"], errors="coerce")

    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"  Indoor:  {(df['environment'] == 'indoor').sum():,}")
    logger.info(f"  Outdoor: {(df['environment'] == 'outdoor').sum():,}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build dataset index CSV")
    parser.add_argument("--root", type=str, default=".", help="Dataset root directory")
    parser.add_argument("--output", type=str, default="dataset_index.csv", help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    df = build_index(root)

    output_path = root / args.output
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Dataset Index Summary")
    print("=" * 60)
    print(f"Total samples: {len(df):,}")
    print(f"\nBy environment:")
    print(df["environment"].value_counts().to_string())
    print(f"\nDepth statistics (meters):")
    print(df[["min_depth", "max_depth"]].describe().to_string())
    print(f"\nBy source:")
    df["source"] = df["left_image_path"].apply(lambda x: x.split("/")[0])
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
