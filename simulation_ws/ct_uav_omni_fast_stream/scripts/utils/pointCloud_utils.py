import numpy as np

def soft_zbuffer_gating(depth_stack, alpha=0.03, beta=0.01, gamma0=0.02, gamma1=0.01):
    """
    depth_stack: (N, H, W), depth > 0 là hợp lệ
    Trả về: gate (N, H, W) boolean — chỉ giữ các ứng viên gần nhất & nhất quán
    """
    N, H, W = depth_stack.shape
    valid = depth_stack > 0

    # dmin trên các cam (pixel-wise), inf nếu không cam nào hợp lệ
    dmin = np.where(valid, depth_stack, np.inf).min(axis=0)  # (H, W)

    # dải chấp nhận quanh bề mặt gần nhất (pixel-wise)
    delta = gamma0 + gamma1 * np.where(np.isfinite(dmin), dmin, 0.0)  # (H, W)

    # ứng viên đủ gần bề mặt gần nhất
    near_mask = (depth_stack > 0) & (depth_stack <= (dmin + delta)[None, :, :])  # (N,H,W)

    # consistency theo std quanh mean (pixel-wise)
    d = np.where(valid, depth_stack, np.nan)                   # (N,H,W)
    d_mean = np.nanmean(d, axis=0)                            # (H,W)
    # ngưỡng tau = alpha + beta * mean_depth
    tau = alpha + beta * np.nan_to_num(d_mean, nan=0.0)       # (H,W)
    d_dev = np.abs(d - d_mean[None, :, :])                    # (N,H,W)
    cons_mask = np.where(np.isnan(d_dev), False, d_dev <= tau[None, :, :])

    gate = near_mask & cons_mask                               # (N,H,W)

    # Fallback: nếu không còn ứng viên sau gating nhưng có ít nhất 1 valid -> chọn nearest cứng
    any_valid = valid.any(axis=0)                              # (H,W)
    none_after_gate = ~gate.any(axis=0)                        # (H,W)
    fallback = none_after_gate & any_valid                     # (H,W)

    if np.any(fallback):
        # chỉ số cam gần nhất theo pixel
        idx = np.argmin(np.where(valid, depth_stack, np.inf), axis=0)  # (H,W)
        h_idx, w_idx = np.nonzero(fallback)                            # (K,), (K,)
        cam_idx = idx[h_idx, w_idx]                                    # (K,)

        # đảm bảo tại các pixel fallback, chỉ cam được chọn là True
        gate[:, h_idx, w_idx] = False
        gate[cam_idx, h_idx, w_idx] = True

    return gate