import torch
import numpy as np


def mae(pred, gt):
    return torch.mean(torch.abs(pred - gt)).detach().cpu().numpy()


def max_ae(pred, gt):
    return torch.max(torch.abs(pred - gt)).detach().cpu().numpy()


def rmse(pred, gt):
    return torch.sqrt(torch.mean(torch.square(pred - gt))).detach().cpu().numpy()


def absrel(pred, gt):
    valid_mask = gt > 1.0
    if valid_mask.sum() == 0:
        return float('nan')
    return torch.mean(torch.abs(pred[valid_mask] - gt[valid_mask]) / gt[valid_mask]).detach().cpu().numpy()


def sqrel(pred, gt):
    valid_mask = gt > 1.0
    if valid_mask.sum() == 0:
        return float('nan')
    return torch.mean(torch.square(pred[valid_mask] - gt[valid_mask]) / torch.square(gt[valid_mask])).detach().cpu().numpy()


def silog(pred, gt):
    mask = pred > 0
    if mask.sum() == 0:
        return float('nan')
    d = torch.log(pred[mask]) - torch.log(gt[mask])
    return torch.sqrt(torch.mean(torch.square(d)) - torch.square(torch.mean(d))).detach().cpu().numpy()


def pixel_error_pct(th_pixel, pred, gt):
    error = torch.abs(pred - gt)
    count = torch.numel(error[error >= th_pixel])
    total = torch.numel(error)
    return float(100 * count / total)


def D1(pred, gt):
    error = torch.abs(pred - gt)
    count = torch.numel(error[(error >= 3) * (error >= 0.05 * gt)])
    total = torch.numel(error)
    return float(100 * count / total)


def delta_acc(exp, pred, gt):
    error = torch.max(pred / gt, gt / pred)
    count = torch.numel(error[error < 1.25 ** exp])
    total = torch.numel(error)
    return float(100 * count / total)


def threshold_acc(err_pct, pred, gt):
    error = torch.max(pred / gt, gt / pred)
    count = torch.numel(error[error < (1 + err_pct)])
    total = torch.numel(error)
    return float(100 * count / total)


def pixel_accuracy(threshold_px, pred, gt):
    """
    Compute percentage of pixels with error less than threshold (for disparity in pixels).
    
    Args:
        threshold_px: Error threshold in pixels
        pred: Predicted disparity
        gt: Ground truth disparity
        
    Returns:
        Percentage of pixels with error < threshold
    """
    error = torch.abs(pred - gt)
    count = (error < threshold_px).sum().item()
    total = error.numel()
    return float(100 * count / total)


# Metrics for disparity in pixels: MAE, RMSE, D1, AbsRel (gt>1px), threshold accuracies
eval_names = ["MAE(px)", "RMSE(px)", "D1(%)", "AbsRel", "δ<1.25x(%)", "δ<1.56x(%)", "δ<1.95x(%)", "<1px(%)", "<3px(%)", "<5px(%)"]

def evaluate(pred, gt, mask):
    """
    Evaluate predicted disparity against ground truth.
    
    For disparity in pixels, we prioritize:
    - Absolute errors: MAE, RMSE (in pixels)
    - Relative thresholds: δ1, δ2, δ3 (multiplicative)
    - Pixel thresholds: <1px, <3px, <5px (common for stereo)
    
    Note: AbsRel computed only for gt > 1px to avoid division by small values
    
    Args:
        pred: Predicted disparity tensor
        gt: Ground truth disparity tensor
        mask: Valid pixel mask
    
    Returns:
        Dictionary with metric names as keys and computed values as values
    """
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    
    eval_metrics = [
        mae(pred_masked, gt_masked),
        rmse(pred_masked, gt_masked),
        D1(pred_masked, gt_masked),
        absrel(pred_masked, gt_masked),  # Only for gt > 1px
        delta_acc(1.25, pred_masked, gt_masked),  # δ < 1.25x
        delta_acc(1.56, pred_masked, gt_masked),  # δ < 1.56x
        delta_acc(1.95, pred_masked, gt_masked),  # δ < 1.95x
        pixel_accuracy(0.1, pred_masked, gt_masked),  # < 0.1px
        pixel_accuracy(1.0, pred_masked, gt_masked),  # < 1px
        pixel_accuracy(3.0, pred_masked, gt_masked),  # < 3px
    ]
    
    # Return as dictionary for proper access in training loop
    return {name: float(value) for name, value in zip(eval_names, eval_metrics)}

