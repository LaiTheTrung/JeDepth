import torch
import numpy as np


def EPE(pred, gt, valid_mask=None):
    """
    End-Point Error: mean absolute disparity error (in pixels).
    Standard metric for SceneFlow benchmark.
    """
    if valid_mask is None:
        valid_mask = gt > 0
    if valid_mask.sum() == 0:
        return float('nan')
    error = torch.abs(pred[valid_mask] - gt[valid_mask])
    return error.mean().detach().cpu().item()


def D1_all(pred, gt, valid_mask=None):
    """
    D1-all: percentage of disparity outliers over all pixels.
    A pixel is an outlier if error > max(3px, 0.05 * |gt|).
    Standard metric for KITTI 2015 benchmark.
    """
    if valid_mask is None:
        valid_mask = gt > 0
    if valid_mask.sum() == 0:
        return float('nan')
    error = torch.abs(pred[valid_mask] - gt[valid_mask])
    gt_valid = gt[valid_mask]
    outlier = (error > 3.0) & (error > 0.05 * gt_valid.abs())
    return (100.0 * outlier.float().mean()).detach().cpu().item()


def D1_fg(pred, gt, fg_mask, valid_mask=None):
    """
    D1-fg: percentage of disparity outliers in foreground regions.
    fg_mask: binary mask indicating foreground pixels.
    """
    if valid_mask is None:
        valid_mask = gt > 0
    mask = valid_mask & fg_mask
    if mask.sum() == 0:
        return float('nan')
    error = torch.abs(pred[mask] - gt[mask])
    gt_valid = gt[mask]
    outlier = (error > 3.0) & (error > 0.05 * gt_valid.abs())
    return (100.0 * outlier.float().mean()).detach().cpu().item()


def D1_bg(pred, gt, fg_mask, valid_mask=None):
    """
    D1-bg: percentage of disparity outliers in background regions.
    fg_mask: binary mask indicating foreground pixels (bg = ~fg).
    """
    if valid_mask is None:
        valid_mask = gt > 0
    mask = valid_mask & (~fg_mask)
    if mask.sum() == 0:
        return float('nan')
    error = torch.abs(pred[mask] - gt[mask])
    gt_valid = gt[mask]
    outlier = (error > 3.0) & (error > 0.05 * gt_valid.abs())
    return (100.0 * outlier.float().mean()).detach().cpu().item()


def bad_x(pred, gt, threshold_px, valid_mask=None):
    """
    Bad-x: percentage of pixels with absolute disparity error > threshold.
    Common thresholds: 1px, 2px, 3px.
    """
    if valid_mask is None:
        valid_mask = gt > 0
    if valid_mask.sum() == 0:
        return float('nan')
    error = torch.abs(pred[valid_mask] - gt[valid_mask])
    bad = error > threshold_px
    return (100.0 * bad.float().mean()).detach().cpu().item()


# ---- Convenience wrappers ----
def bad_1(pred, gt, valid_mask=None):
    return bad_x(pred, gt, 1.0, valid_mask)

def bad_2(pred, gt, valid_mask=None):
    return bad_x(pred, gt, 2.0, valid_mask)

def bad_3(pred, gt, valid_mask=None):
    return bad_x(pred, gt, 3.0, valid_mask)