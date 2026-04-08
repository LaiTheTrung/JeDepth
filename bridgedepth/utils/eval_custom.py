"""Custom evaluation + qualitative inference helpers for the
stereo-smallbaseline pipeline. Uses the metrics defined in
``reference/evaluation/evaluation.py``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .eval_disp import inference_context

# Pull metric definitions from the reference module.
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
from reference.evaluation.evaluation import evaluate as ref_evaluate, eval_names  # noqa: E402

logger = logging.getLogger("waft")


def _model_forward(model, sample, cfg):
    m = model.module if hasattr(model, "module") else model
    try:
        out = m.inference(sample, size=tuple(cfg.DATASETS.CROP_SIZE))
    except Exception:
        out = m(sample)
    return out


@torch.no_grad()
def evaluate_model(model, val_loader, cfg, device):
    """Run evaluation on val_loader and return aggregated metrics dict."""
    sums = {k: 0.0 for k in eval_names}
    counts = {k: 0 for k in eval_names}
    with inference_context(model):
        for sample in tqdm(val_loader, desc="eval", leave=False):
            sample = {k: v.to(device) for k, v in sample.items()}
            out = _model_forward(model, sample, cfg)
            pred = out["disp_pred"]
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            gt = sample["disp"]
            mask = (sample["valid"].bool()) & (gt < float(cfg.WAFT.MAX_DISP)) & (gt > 0)
            if mask.sum() == 0:
                continue
            metrics = ref_evaluate(pred, gt, mask)
            for k, v in metrics.items():
                if v is None or (isinstance(v, float) and (v != v)):  # nan
                    continue
                sums[k] += float(v)
                counts[k] += 1
    results = {k: (sums[k] / counts[k] if counts[k] > 0 else float("nan")) for k in eval_names}
    return results


def _colorize_disp(disp: np.ndarray) -> np.ndarray:
    """Disparity (H, W) → uint8 BGR colormap."""
    valid = disp > 0
    if valid.any():
        vmin, vmax = float(disp[valid].min()), float(disp[valid].max())
    else:
        vmin, vmax = 0.0, 1.0
    norm = np.clip((disp - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    norm = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


@torch.no_grad()
def infer_test_images(model, cfg, writer, global_step, device, max_images: int = 6):
    """Run inference on cfg.TEST.TEST_IMAGES_DIR/{left,right} pairs and log to TB."""
    test_dir = cfg.TEST.TEST_IMAGES_DIR
    if not test_dir:
        return
    test_dir = Path(test_dir)
    if not test_dir.is_absolute():
        test_dir = _REPO_ROOT / test_dir
    left_dir = test_dir / "left"
    right_dir = test_dir / "right"
    if not left_dir.exists() or not right_dir.exists():
        logger.warning(f"test_images dirs missing: {left_dir}, {right_dir}")
        return

    pairs = sorted(left_dir.glob("*.png"))[:max_images]
    if not pairs:
        return

    with inference_context(model):
        for left_path in pairs:
            right_path = right_dir / left_path.name
            if not right_path.exists():
                continue
            left = cv2.cvtColor(cv2.imread(str(left_path)), cv2.COLOR_BGR2RGB)
            right = cv2.cvtColor(cv2.imread(str(right_path)), cv2.COLOR_BGR2RGB)

            # Make divisible by DIVIS_BY (model already pads internally, just keep raw).
            h, w = left.shape[:2]
            divis = int(cfg.DATASETS.DIVIS_BY)
            new_h = (h // divis) * divis
            new_w = (w // divis) * divis
            left = left[:new_h, :new_w]
            right = right[:new_h, :new_w]

            img1 = torch.from_numpy(left).permute(2, 0, 1).float().unsqueeze(0).to(device)
            img2 = torch.from_numpy(right).permute(2, 0, 1).float().unsqueeze(0).to(device)
            sample = {"img1": img1, "img2": img2}
            out = _model_forward(model, sample, cfg)
            pred = out["disp_pred"]
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            disp_np = pred[0].detach().float().cpu().numpy()

            disp_color = _colorize_disp(disp_np)
            disp_color_rgb = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)
            combined = np.concatenate([left, disp_color_rgb], axis=1)
            combined = combined.transpose(2, 0, 1)  # CHW
            writer.add_image(f"test/{left_path.stem}", combined, global_step)
