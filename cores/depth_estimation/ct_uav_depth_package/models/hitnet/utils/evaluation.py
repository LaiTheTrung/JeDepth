import numpy as np


def mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))


def max_ae(pred, gt):
    return float(np.max(np.abs(pred - gt)))


def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def absrel(pred, gt):
    mask = gt > 0
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask]))


def sqrel(pred, gt):
    mask = gt > 0
    return float(np.mean(((pred[mask] - gt[mask]) ** 2) / (gt[mask] ** 2)))


def silog(pred, gt):
    # sqrt of the silog (following KITTI)
    mask1 = gt > 0
    mask2 = pred > 0
    mask = mask1 & mask2
    d = np.log(pred[mask]) - np.log(gt[mask])
    return float(np.sqrt(np.mean(d ** 2) - (np.mean(d) ** 2)))


def pixel_error_pct(th_pixel, pred, gt):
    error = np.abs(pred - gt)
    return float(100.0 * np.count_nonzero(error >= th_pixel) / error.size)


def D1(th_pixel, th_pct, pred, gt):
    error = np.abs(pred - gt)
    mask = (error >= th_pixel) & (error >= th_pct * gt)
    return float(100.0 * np.count_nonzero(mask) / error.size)


def delta_acc(exp, pred, gt):
    error = np.maximum(pred / gt, gt / pred)
    return float(100.0 * np.count_nonzero(error < (1.25 ** exp)) / error.size)


def threshold_acc(err_pct, pred, gt):
    error = np.maximum(pred / gt, gt / pred)
    return float(100.0 * np.count_nonzero(error < (1.0 + err_pct)) / error.size)


eval_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]


def evaluate(pred, gt, mask):
    pred_m = pred[mask]
    gt_m = gt[mask]
    eval_metrics = [
        mae(pred_m, gt_m),
        rmse(pred_m, gt_m),
        absrel(pred_m, gt_m),
        sqrel(pred_m, gt_m),
        silog(pred_m, gt_m),
        delta_acc(1, pred_m, gt_m),
        delta_acc(2, pred_m, gt_m),
        delta_acc(3, pred_m, gt_m),
    ]
    return np.array(eval_metrics, dtype=np.float32)

