from __future__ import annotations

"""
eda.py - Exploratory Data Analysis for Depth Estimation Dataset.

Phân tích:
1. Số lượng ảnh indoor vs outdoor
2. Phạm vi giá trị depth/disparity cho từng dataset
3. Tỷ lệ pixel hợp lệ (valid pixel percentage)
4. Phân bố giá trị depth/disparity (histograms)
5. Thống kê tổng hợp

Usage:
    python eda.py [--csv dataset_index.csv] [--root .] [--sample-size 50]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import h5py
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_depth_or_disp(row: pd.Series, root: Path) -> tuple[np.ndarray | None, str]:
    """Load depth or disparity array from a CSV row. Returns (array, type)."""
    if pd.notna(row.get("depth_path")):
        path = root / row["depth_path"]
        arr = np.array(Image.open(path), dtype=np.float32) / 10000.0
        return arr, "depth"
    elif pd.notna(row.get("disparity_path")):
        path = root / row["disparity_path"]
        if str(path).endswith(".dsp5"):
            with h5py.File(path, "r") as f:
                arr = f["disparity"][()].astype(np.float32)
            # Spring: downsample 2x to match image resolution
            arr = np.ascontiguousarray(arr[::2, ::2])
            arr[arr == np.inf] = -1
        else:
            arr = np.array(Image.open(path), dtype=np.float32) / 100.0
        return arr, "disparity"
    return None, "none"


def get_source(image_path: str) -> str:
    """Extract dataset source name from image path."""
    return image_path.split("/")[0]


# ── EDA Functions ────────────────────────────────────────────────────────────

def analyze_samples(df: pd.DataFrame, root: Path, sample_size: int = 50) -> dict:
    """
    Sample images from each dataset and compute per-pixel statistics.
    Returns dict of {source: {stats...}}.
    """
    sources = df.apply(lambda r: get_source(r["left_image_path"]), axis=1)
    df = df.copy()
    df["source"] = sources

    results = {}

    for source, group in df.groupby("source"):
        logger.info(f"Analyzing {source} ({len(group)} samples, sampling {min(sample_size, len(group))})...")

        sampled = group.sample(n=min(sample_size, len(group)), random_state=42)

        all_values = []
        valid_ratios = []
        data_type = None

        for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc=f"  {source}"):
            arr, dtype = load_depth_or_disp(row, root)
            if arr is None:
                continue
            data_type = dtype

            total_pixels = arr.size
            valid_mask = arr > 0
            valid_pixels = valid_mask.sum()
            valid_ratios.append(100.0 * valid_pixels / total_pixels)

            valid_values = arr[valid_mask]
            if valid_values.size > 0:
                # Subsample to avoid memory issues
                if valid_values.size > 10000:
                    idx = np.random.choice(valid_values.size, 10000, replace=False)
                    valid_values = valid_values[idx]
                all_values.append(valid_values)

        if all_values:
            combined = np.concatenate(all_values)
            results[source] = {
                "data_type": data_type,
                "num_samples": len(group),
                "values": combined,
                "valid_ratio_mean": np.mean(valid_ratios),
                "valid_ratio_std": np.std(valid_ratios),
                "value_min": float(combined.min()),
                "value_max": float(combined.max()),
                "value_mean": float(combined.mean()),
                "value_median": float(np.median(combined)),
                "value_std": float(combined.std()),
            }

    return results


def plot_overview(df: pd.DataFrame, save_path: Path):
    """Plot dataset overview: sample counts and environment distribution."""
    df = df.copy()
    df["source"] = df["left_image_path"].apply(get_source)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Samples per dataset
    source_counts = df["source"].value_counts()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = axes[0].bar(source_counts.index, source_counts.values, color=colors[:len(source_counts)])
    axes[0].set_title("Number of Samples per Dataset", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, source_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                     f"{val:,}", ha="center", va="bottom", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=15)

    # 2. Indoor vs Outdoor
    env_counts = df["environment"].value_counts()
    axes[1].pie(env_counts.values, labels=env_counts.index, autopct="%1.1f%%",
                colors=["#2196F3", "#4CAF50"], startangle=90, textprops={"fontsize": 12})
    axes[1].set_title("Indoor vs Outdoor", fontsize=13, fontweight="bold")

    # 3. Environment per dataset
    cross = pd.crosstab(df["source"], df["environment"])
    cross.plot(kind="bar", ax=axes[2], color=["#2196F3", "#4CAF50"], edgecolor="white")
    axes[2].set_title("Environment per Dataset", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].legend(title="Environment")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved overview plot: {save_path}")


def plot_depth_distributions(analysis: dict, save_path: Path):
    """Plot depth/disparity value distributions for each dataset."""
    n = len(analysis)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(-1, 1)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (source, stats) in enumerate(analysis.items()):
        values = stats["values"]
        dtype = stats["data_type"]
        unit = "m" if dtype == "depth" else "px"
        color = colors[i % len(colors)]

        # Histogram
        axes[0, i].hist(values, bins=100, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        axes[0, i].set_title(f"{source}\n({dtype}, {stats['num_samples']:,} samples)", fontsize=11, fontweight="bold")
        axes[0, i].set_xlabel(f"{dtype.capitalize()} ({unit})")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].axvline(stats["value_mean"], color="red", linestyle="--", label=f"Mean: {stats['value_mean']:.2f}")
        axes[0, i].axvline(stats["value_median"], color="orange", linestyle="--", label=f"Median: {stats['value_median']:.2f}")
        axes[0, i].legend(fontsize=8)

        # Box plot
        bp = axes[1, i].boxplot(values, vert=True, patch_artist=True,
                                boxprops=dict(facecolor=color, alpha=0.5))
        axes[1, i].set_title(f"{source} - Box Plot", fontsize=11)
        axes[1, i].set_ylabel(f"{dtype.capitalize()} ({unit})")

        # Stats annotation
        stats_text = (
            f"Min: {stats['value_min']:.3f}\n"
            f"Max: {stats['value_max']:.3f}\n"
            f"Mean: {stats['value_mean']:.3f}\n"
            f"Std: {stats['value_std']:.3f}\n"
            f"Valid: {stats['valid_ratio_mean']:.1f}%±{stats['valid_ratio_std']:.1f}%"
        )
        axes[1, i].text(0.95, 0.95, stats_text, transform=axes[1, i].transAxes,
                        fontsize=8, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Depth/Disparity Value Distributions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution plot: {save_path}")


def plot_depth_range_scatter(df: pd.DataFrame, save_path: Path):
    """Plot min_depth vs max_depth scatter for each dataset."""
    df = df.copy()
    df["source"] = df["left_image_path"].apply(get_source)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"fat_indoor": "#2196F3", "fat_outdoor": "#4CAF50",
              "Stereo2k_indoor": "#FF9800", "spring_outdoor": "#9C27B0"}

    for source, group in df.groupby("source"):
        valid = group.dropna(subset=["min_depth", "max_depth"])
        ax.scatter(valid["min_depth"], valid["max_depth"],
                   alpha=0.3, s=10, label=f"{source} ({len(valid):,})",
                   color=colors.get(source, "gray"))

    ax.set_xlabel("Min Depth (m)", fontsize=12)
    ax.set_ylabel("Max Depth (m)", fontsize=12)
    ax.set_title("Min vs Max Depth per Sample", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved depth range scatter: {save_path}")


def plot_sample_visualizations(df: pd.DataFrame, root: Path, save_path: Path):
    """Visualize one sample from each dataset: RGB + depth/disparity."""
    df = df.copy()
    df["source"] = df["left_image_path"].apply(get_source)
    sources = df["source"].unique()

    fig, axes = plt.subplots(len(sources), 2, figsize=(12, 4 * len(sources)))
    if len(sources) == 1:
        axes = axes.reshape(1, -1)

    for i, source in enumerate(sorted(sources)):
        row = df[df["source"] == source].iloc[0]

        # Load image
        img = Image.open(root / row["left_image_path"])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{source} - RGB", fontsize=11)
        axes[i, 0].axis("off")

        # Load depth/disparity
        arr, dtype = load_depth_or_disp(row, root)
        if arr is not None:
            im = axes[i, 1].imshow(arr, cmap="inferno")
            unit = "m" if dtype == "depth" else "px"
            axes[i, 1].set_title(f"{source} - {dtype.capitalize()} ({unit})", fontsize=11)
            axes[i, 1].axis("off")
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.suptitle("Sample Visualizations", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sample visualizations: {save_path}")


def print_summary_table(df: pd.DataFrame, analysis: dict):
    """Print a comprehensive summary table."""
    df = df.copy()
    df["source"] = df["left_image_path"].apply(get_source)

    print("\n" + "=" * 80)
    print("EDA SUMMARY")
    print("=" * 80)

    print(f"\n{'Dataset':<20} {'Count':>8} {'Env':<10} {'Type':<10} "
          f"{'Min':>8} {'Max':>8} {'Mean':>8} {'Valid%':>8}")
    print("-" * 80)

    for source in sorted(analysis.keys()):
        stats = analysis[source]
        group = df[df["source"] == source]
        env = group["environment"].iloc[0]
        dtype = stats["data_type"]
        unit = "m" if dtype == "depth" else "px"

        print(f"{source:<20} {stats['num_samples']:>8,} {env:<10} {dtype:<10} "
              f"{stats['value_min']:>7.2f}{unit} {stats['value_max']:>7.2f}{unit} "
              f"{stats['value_mean']:>7.2f}{unit} {stats['valid_ratio_mean']:>7.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':<20} {len(df):>8,}")
    print(f"\nIndoor:  {(df['environment'] == 'indoor').sum():>8,}")
    print(f"Outdoor: {(df['environment'] == 'outdoor').sum():>8,}")

    print("\n--- Depth Range Stats (from CSV) ---")
    for source, group in df.groupby("source"):
        valid = group.dropna(subset=["min_depth", "max_depth"])
        print(f"\n{source}:")
        print(f"  min_depth: [{valid['min_depth'].min():.3f}, {valid['min_depth'].max():.3f}] m")
        print(f"  max_depth: [{valid['max_depth'].min():.3f}, {valid['max_depth'].max():.3f}] m")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA for Depth Estimation Dataset")
    parser.add_argument("--csv", type=str, default="dataset_index.csv", help="Input CSV path")
    parser.add_argument("--root", type=str, default=".", help="Dataset root directory")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of samples per dataset for pixel-level analysis")
    parser.add_argument("--output-dir", type=str, default="eda_output",
                        help="Directory to save EDA plots")
    args = parser.parse_args()

    root = Path(args.root)
    df = pd.read_csv(root / args.csv)
    logger.info(f"Loaded {len(df):,} samples from {args.csv}")

    output_dir = root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # 1. Overview plots
    logger.info("Plotting overview...")
    plot_overview(df, output_dir / "01_overview.png")

    # 2. Pixel-level analysis (sampling)
    logger.info(f"Analyzing pixel-level stats (sample_size={args.sample_size})...")
    analysis = analyze_samples(df, root, sample_size=args.sample_size)

    # 3. Distribution plots
    logger.info("Plotting distributions...")
    plot_depth_distributions(analysis, output_dir / "02_distributions.png")

    # 4. Depth range scatter
    logger.info("Plotting depth range scatter...")
    plot_depth_range_scatter(df, output_dir / "03_depth_range_scatter.png")

    # 5. Sample visualizations
    logger.info("Plotting sample visualizations...")
    plot_sample_visualizations(df, root, output_dir / "04_sample_visualizations.png")

    # 6. Summary
    print_summary_table(df, analysis)

    logger.info(f"\nAll EDA outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
