import json
import logging
import torch
from .disparity_metric import EPE, D1_all, bad_1, bad_2, bad_3

logger = logging.getLogger(__name__)


def evaluate(pred_disp, gt_disp, valid_mask=None):
    """
    Evaluate disparity prediction against ground truth.

    Args:
        pred_disp: (B, 1, H, W) or (B, H, W) predicted disparity
        gt_disp: (B, 1, H, W) or (B, H, W) ground truth disparity
        valid_mask: optional mask (same shape as gt_disp)

    Returns:
        dict of metric name -> value
    """
    # Squeeze to (B, H, W) if needed
    if pred_disp.dim() == 4:
        pred_disp = pred_disp.squeeze(1)
    if gt_disp.dim() == 4:
        gt_disp = gt_disp.squeeze(1)

    if valid_mask is None:
        valid_mask = gt_disp > 0

    results = {}
    # Compute per-sample then average
    batch_size = pred_disp.shape[0]
    epe_sum, d1_sum, b1_sum, b2_sum, b3_sum = 0, 0, 0, 0, 0
    valid_count = 0

    for i in range(batch_size):
        mask_i = valid_mask[i]
        if mask_i.sum() == 0:
            continue
        epe_sum += EPE(pred_disp[i], gt_disp[i], mask_i)
        d1_sum += D1_all(pred_disp[i], gt_disp[i], mask_i)
        b1_sum += bad_1(pred_disp[i], gt_disp[i], mask_i)
        b2_sum += bad_2(pred_disp[i], gt_disp[i], mask_i)
        b3_sum += bad_3(pred_disp[i], gt_disp[i], mask_i)
        valid_count += 1

    if valid_count > 0:
        results['EPE'] = epe_sum / valid_count
        results['D1_all'] = d1_sum / valid_count
        results['bad_1'] = b1_sum / valid_count
        results['bad_2'] = b2_sum / valid_count
        results['bad_3'] = b3_sum / valid_count

    return results


def log_eval(results):
    """Log evaluation results."""
    parts = [f"{k}: {v:.4f}" for k, v in results.items()]
    logger.info(" | ".join(parts))


def save_eval(results, filename):
    """Save evaluation results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filename}")
