# ---- F-Score (Pointcloud-based) ----
import torch

def disparity_to_pointcloud(disp, fx, fy, cx, cy, baseline, valid_mask=None):
    """
    Back-project disparity map to 3D pointcloud.

    Args:
        disp: (H, W) disparity map in pixels
        fx, fy: focal lengths in pixels
        cx, cy: principal point in pixels
        baseline: stereo baseline in meters
        valid_mask: (H, W) boolean mask of valid pixels

    Returns:
        points: (N, 3) tensor of 3D points [X, Y, Z]
    """
    H, W = disp.shape
    if valid_mask is None:
        valid_mask = disp > 0

    # depth = f * B / d
    depth = torch.zeros_like(disp)
    depth[valid_mask] = (fx * baseline) / disp[valid_mask]

    # pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=disp.device, dtype=disp.dtype),
        torch.arange(W, device=disp.device, dtype=disp.dtype),
        indexing='ij'
    )

    # back-project
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    points = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)
    return points[valid_mask]  # (N, 3)


def f_score(pred_disp, gt_disp, fx, fy, cx, cy, baseline, threshold=0.05, valid_mask=None):
    """
    F-Score: harmonic mean of precision and recall between
    predicted and ground-truth pointclouds.

    A predicted point is a "hit" if its nearest GT point is within
    the threshold distance (in meters), and vice versa.

    Args:
        pred_disp: (H, W) predicted disparity
        gt_disp: (H, W) ground truth disparity
        fx, fy, cx, cy: camera intrinsics
        baseline: stereo baseline in meters
        threshold: distance threshold in meters (default 0.05 = 5cm)
        valid_mask: (H, W) boolean mask

    Returns:
        dict with 'f_score', 'precision', 'recall' (all in [0, 1])
    """
    if valid_mask is None:
        valid_mask = (gt_disp > 0) & (pred_disp > 0)

    pred_mask = valid_mask & (pred_disp > 0)
    gt_mask = valid_mask & (gt_disp > 0)

    pred_pts = disparity_to_pointcloud(pred_disp, fx, fy, cx, cy, baseline, pred_mask)
    gt_pts = disparity_to_pointcloud(gt_disp, fx, fy, cx, cy, baseline, gt_mask)

    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return {'f_score': float('nan'), 'precision': float('nan'), 'recall': float('nan')}

    # Compute in chunks to avoid OOM on large pointclouds
    precision = _compute_hit_ratio(pred_pts, gt_pts, threshold)
    recall = _compute_hit_ratio(gt_pts, pred_pts, threshold)

    if precision + recall == 0:
        fs = 0.0
    else:
        fs = 2.0 * precision * recall / (precision + recall)

    return {'f_score': fs, 'precision': precision, 'recall': recall}


def _compute_hit_ratio(src, tgt, threshold, chunk_size=5000):
    """
    For each point in src, find nearest neighbor in tgt.
    Return fraction of src points whose NN distance < threshold.

    Uses chunked computation to avoid OOM.
    """
    threshold_sq = threshold ** 2
    hits = 0
    total = src.shape[0]

    for i in range(0, total, chunk_size):
        src_chunk = src[i:i + chunk_size]  # (C, 3)
        # squared distances: (C, M)
        diff = src_chunk.unsqueeze(1) - tgt.unsqueeze(0)  # (C, M, 3)
        dist_sq = (diff ** 2).sum(dim=-1)  # (C, M)
        min_dist_sq = dist_sq.min(dim=1).values  # (C,)
        hits += (min_dist_sq < threshold_sq).sum().item()

    return hits / total