"""
train.py - Huấn luyện mô hình JeDepth stereo depth estimation với dữ liệu custom.

Sử dụng:
    python train.py --cfg cfgs/iinet/iinet_custom.yaml
    python train.py --cfg cfgs/iinet/iinet_custom.yaml --resume output/iinet_custom/ckpt/checkpoint_epoch_10.pth

Tính năng:
    - Huấn luyện với AMP (Automatic Mixed Precision) tùy chọn
    - Đánh giá mỗi N epoch trên tập validation, lưu checkpoint
    - Lưu best model dựa trên MAE metric
    - TensorBoard logging (loss, metrics, visualization)
    - tqdm progress bars cho epoch và iteration
"""

import argparse
import os
import sys
import datetime
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jedepth.dataset.depth_dataset import StereoDepthDataset
from jedepth.model.iinet import JeDepth
from jedepth.evaluation.evaluation import evaluate as compute_metrics
from jedepth.utils.common_utils import (
    config_loader, set_random_seed, create_logger,
    save_checkpoint, load_params_from_file,
    color_map_tensorboard, write_tensorboard, get_pos_fullres,
)


# ── Argument Parser ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train JeDepth Stereo Depth Model")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Đường dẫn tới file config YAML")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Thư mục gốc lưu kết quả")
    parser.add_argument("--workers", type=int, default=4,
                        help="Số worker cho DataLoader")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed để tái tạo kết quả")
    parser.add_argument("--resume", type=str, default=None,
                        help="Đường dẫn tới checkpoint để resume training")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    return parser.parse_args()


# ── Dataset Wrappers ─────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def random_crop(left, right, disp, crop_h, crop_w):
    """Random crop stereo pair và disparity về kích thước (crop_h, crop_w).
    Tự động pad nếu ảnh nhỏ hơn crop size.
    """
    _, h, w = left.shape
    # Pad nếu ảnh nhỏ hơn crop size
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)
    if pad_h > 0 or pad_w > 0:
        left = F.pad(left, (0, pad_w, 0, pad_h))
        right = F.pad(right, (0, pad_w, 0, pad_h))
        disp = F.pad(disp, (0, pad_w, 0, pad_h))
        _, h, w = left.shape

    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    return (left[:, y:y + crop_h, x:x + crop_w],
            right[:, y:y + crop_h, x:x + crop_w],
            disp[:, y:y + crop_h, x:x + crop_w])


def center_crop(left, right, disp, crop_h, crop_w):
    """Center crop stereo pair và disparity về kích thước (crop_h, crop_w).
    Tự động pad nếu ảnh nhỏ hơn crop size.
    """
    _, h, w = left.shape
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)
    if pad_h > 0 or pad_w > 0:
        left = F.pad(left, (0, pad_w, 0, pad_h))
        right = F.pad(right, (0, pad_w, 0, pad_h))
        disp = F.pad(disp, (0, pad_w, 0, pad_h))
        _, h, w = left.shape

    y = (h - crop_h) // 2
    x = (w - crop_w) // 2
    return (left[:, y:y + crop_h, x:x + crop_w],
            right[:, y:y + crop_h, x:x + crop_w],
            disp[:, y:y + crop_h, x:x + crop_w])


class TrainDataset(StereoDepthDataset):
    """Dataset wrapper cho training: ImageNet normalization + random crop."""

    def __init__(self, csv_path, root, crop_size, augment=True):
        super().__init__(csv_path, root, augment=augment)
        self.crop_size = crop_size  # (H, W)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        left = sample["left"]    # (3, H, W) float [0, 1]
        right = sample["right"]
        disp = sample["disp"]    # (1, H, W) float

        # ImageNet normalization
        left = (left - IMAGENET_MEAN) / IMAGENET_STD
        right = (right - IMAGENET_MEAN) / IMAGENET_STD

        # Random crop về kích thước cố định (bắt buộc cho batching)
        left, right, disp = random_crop(
            left, right, disp, self.crop_size[0], self.crop_size[1]
        )

        return {"left": left, "right": right, "disp": disp}


class ValDataset(StereoDepthDataset):
    """Dataset wrapper cho validation: ImageNet normalization + center crop."""

    def __init__(self, csv_path, root, crop_size):
        super().__init__(csv_path, root, augment=False)
        self.crop_size = crop_size  # (H, W)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        left = sample["left"]
        right = sample["right"]
        disp = sample["disp"]

        # ImageNet normalization
        left = (left - IMAGENET_MEAN) / IMAGENET_STD
        right = (right - IMAGENET_MEAN) / IMAGENET_STD

        # Center crop cho validation
        left, right, disp = center_crop(
            left, right, disp, self.crop_size[0], self.crop_size[1]
        )

        return {"left": left, "right": right, "disp": disp}


# ── Training Logic ───────────────────────────────────────────────────────────

def prepare_batch(data, cfgs, device):
    """Di chuyển batch lên GPU và chuẩn bị format cho model/loss.

    - Đổi 'disp' → 'disp_pyr' (Criterion sẽ tự tạo pyramid)
    - Tính position grid nếu cần (UNCER_ONLY=False → normal loss cần pos)
    """
    for k, v in data.items():
        if torch.is_tensor(v):
            data[k] = v.to(device)

    # Criterion expects key 'disp_pyr' chứa raw disparity (B, 1, H, W)
    data["disp_pyr"] = data["disp"]

    # Position grid cho normal loss (chỉ cần khi UNCER_ONLY=False)
    if not cfgs.TRAINER.UNCER_ONLY:
        B, _, H, W = data["disp"].shape
        fx = W / 2.0  # approximate focal length
        pos = get_pos_fullres(fx, W, H)
        pos = torch.from_numpy(pos).unsqueeze(0).expand(B, -1, -1, -1).to(device)
        data["pos"] = pos

    return data


def train_one_epoch(model, loader, optimizer, scaler, cfgs, device, epoch,
                    total_epochs, logger, tb_writer):
    """Huấn luyện 1 epoch. Trả về average loss."""
    model.train()
    total_loss = 0.0
    log_interval = cfgs.TRAINER.LOG_INTERVAL

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Train Epoch {epoch}/{total_epochs}",
                dynamic_ncols=True, leave=False)

    for i, data in pbar:
        global_step = epoch * len(loader) + i
        data = prepare_batch(data, cfgs, device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
            model_output = model(data, only_uncer=cfgs.TRAINER.UNCER_ONLY)
            loss, loss_info = model.get_loss(cfgs, data, model_output)

        # Backward pass với gradient scaling (AMP)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        lr = optimizer.param_groups[0]["lr"]

        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        # TensorBoard logging
        if i % log_interval == 0 and tb_writer is not None:
            loss_info["scalar/train/lr"] = lr
            write_tensorboard(tb_writer, loss_info, global_step)

            # Visualization: stereo pair và disparity prediction
            if cfgs.TRAINER.TRAIN_VISUALIZATION and "disp_pred" in model_output:
                tb_info = {
                    "image/train/stereo": torch.cat(
                        [data["left"][0], data["right"][0]], dim=1
                    ) / 256,
                    "image/train/disp": color_map_tensorboard(
                        data["disp"][0].squeeze(0),
                        model_output["disp_pred"][0].squeeze(0) * cfgs.MODEL.DISP_SCALE,
                    ),
                }
                write_tensorboard(tb_writer, tb_info, global_step)

        # Log message
        if i % log_interval == 0:
            logger.info(
                f"Epoch {epoch}/{total_epochs} Iter {i}/{len(loader)} "
                f"Loss: {loss.item():.6f} ({avg_loss:.6f}) LR: {lr:.4e}"
            )

    return total_loss / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, cfgs, device, epoch, logger, tb_writer):
    """Đánh giá trên tập validation. Trả về dict metrics trung bình."""
    model.eval()
    max_disp = cfgs.MODEL.MAX_DISP
    disp_scale = cfgs.MODEL.DISP_SCALE

    # Accumulate metrics từ tất cả batch
    all_metrics = {}
    num_samples = 0

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Eval  Epoch {epoch}", dynamic_ncols=True, leave=False)

    for i, data in pbar:
        data = prepare_batch(data, cfgs, device)

        with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
            model_output = model(data, only_uncer=cfgs.TRAINER.UNCER_ONLY)

        # Disparity prediction (scale về pixel space)
        if "disp_pred" in model_output:
            disp_pred = model_output["disp_pred"].squeeze(1) * disp_scale  # (B, H, W)
        else:
            # Fallback: dùng coarse disparity nếu chưa có disp_pred (UNCER_ONLY phase)
            disp_pred = model_output["coarse_disp"].squeeze(1) * disp_scale

        disp_gt = data["disp"].squeeze(1)  # (B, H, W)
        mask = (disp_gt > 0) & (disp_gt < max_disp)

        # Tính metrics cho từng sample trong batch
        B = disp_pred.shape[0]
        for b in range(B):
            if mask[b].sum() == 0:
                continue
            metrics_b = compute_metrics(disp_pred[b], disp_gt[b], mask[b])

            for k, v in metrics_b.items():
                if k not in all_metrics:
                    all_metrics[k] = 0.0
                all_metrics[k] += v
            num_samples += 1

        # Visualization cho batch đầu tiên
        if i == 0 and cfgs.TRAINER.EVAL_VISUALIZATION and tb_writer is not None:
            tb_info = {
                "image/eval/disp": color_map_tensorboard(
                    disp_gt[0], disp_pred[0], disp_max=max_disp
                ),
            }
            write_tensorboard(tb_writer, tb_info, epoch)

    # Tính trung bình metrics
    avg_metrics = {}
    if num_samples > 0:
        for k, v in all_metrics.items():
            avg_metrics[k] = v / num_samples

    # Log metrics
    logger.info(f"{'='*60}")
    logger.info(f"Epoch {epoch} Validation Results ({num_samples} samples):")
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"{'='*60}")

    # TensorBoard: ghi metrics
    if tb_writer is not None:
        tb_info = {}
        for k, v in avg_metrics.items():
            # Sanitize metric name cho tensorboard
            safe_name = k.replace("(", "").replace(")", "").replace("%", "pct").replace("<", "lt")
            tb_info[f"scalar/val/{safe_name}"] = v
        write_tensorboard(tb_writer, tb_info, epoch)

    return avg_metrics


def manage_checkpoints(ckpt_dir, max_keep):
    """Xóa checkpoint cũ nhất nếu vượt quá số lượng tối đa."""
    ckpt_list = sorted(
        glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth")),
        key=os.path.getmtime,
    )
    # Giữ lại best_model.pth, chỉ quản lý checkpoint_epoch_*.pth
    while len(ckpt_list) > max_keep:
        os.remove(ckpt_list.pop(0))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfgs = EasyDict(config_loader(args.cfg))

    # ── Setup ────────────────────────────────────────────────────────────────
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Output directories
    exp_name = os.path.basename(args.cfg).replace(".yaml", "")
    output_dir = os.path.join(args.output_dir, exp_name)
    ckpt_dir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Logger
    log_file = os.path.join(
        output_dir,
        f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger = create_logger(log_file)
    logger.info(f"Config: {args.cfg}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    # Log config
    for key, val in vars(args).items():
        logger.info(f"  {key}: {val}")
    logger.info(f"Config contents:")
    for section in cfgs:
        logger.info(f"  [{section}] {cfgs[section]}")

    # Lưu config vào output dir
    import shutil
    shutil.copy2(args.cfg, os.path.join(output_dir, os.path.basename(args.cfg)))

    # ── Data ─────────────────────────────────────────────────────────────────
    crop_size = tuple(cfgs.DATA.CROP_SIZE)  # (H, W)
    logger.info(f"Loading data from: {cfgs.DATA.ROOT}")
    logger.info(f"Crop size: {crop_size}")

    train_dataset = TrainDataset(
        csv_path=cfgs.DATA.TRAIN_CSV,
        root=cfgs.DATA.ROOT,
        crop_size=crop_size,
        augment=cfgs.DATA.AUGMENT,
    )
    val_dataset = ValDataset(
        csv_path=cfgs.DATA.VAL_CSV,
        root=cfgs.DATA.ROOT,
        crop_size=crop_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfgs.OPTIMIZATION.BATCH_SIZE,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfgs.OPTIMIZATION.BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # ── Model ────────────────────────────────────────────────────────────────
    model = JeDepth(EasyDict(cfgs.MODEL)).to(device)

    # Load pretrained weights nếu có
    if cfgs.MODEL.PRETRAINED_MODEL:
        logger.info(f"Loading pretrained: {cfgs.MODEL.PRETRAINED_MODEL}")
        load_params_from_file(
            model, cfgs.MODEL.PRETRAINED_MODEL,
            device=str(device), dist_mode=False, logger=logger, strict=False,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfgs.MODEL.NAME}")
    logger.info(f"  Total params:     {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    opt_cfg = cfgs.OPTIMIZATION.OPTIMIZER
    optimizer = getattr(torch.optim, opt_cfg.NAME)(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg.LR,
        weight_decay=opt_cfg.get("WEIGHT_DECAY", 0),
    )

    sch_cfg = cfgs.OPTIMIZATION.SCHEDULER
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sch_cfg.MILESTONES,
        gamma=sch_cfg.GAMMA,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfgs.OPTIMIZATION.AMP)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_mae = float("inf")

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", float("inf"))
        logger.info(f"Resumed at epoch {start_epoch}, best MAE: {best_mae:.4f}")

    # ── Training Loop ────────────────────────────────────────────────────────
    num_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
    eval_interval = cfgs.TRAINER.EVAL_INTERVAL
    ckpt_interval = cfgs.TRAINER.CKPT_SAVE_INTERVAL

    logger.info(f"Starting training: {num_epochs} epochs, eval every {eval_interval} epochs")
    logger.info(f"UNCER_ONLY: {cfgs.TRAINER.UNCER_ONLY}")

    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training",
                      dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{num_epochs}")

        # ── Train 1 epoch ────────────────────────────────────────────────────
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, cfgs, device,
            epoch, num_epochs, logger, tb_writer,
        )
        logger.info(f"Epoch {epoch} avg train loss: {avg_loss:.6f}")

        # Step scheduler (per-epoch)
        if sch_cfg.ON_EPOCH:
            scheduler.step()

        # ── Evaluate & Save Checkpoint ───────────────────────────────────────
        should_eval = ((epoch + 1) % eval_interval == 0) or (epoch == num_epochs - 1)
        should_save = ((epoch + 1) % ckpt_interval == 0) or (epoch == num_epochs - 1)

        if should_eval:
            metrics = eval_one_epoch(
                model, val_loader, cfgs, device, epoch, logger, tb_writer,
            )

            # Lưu best model dựa trên MAE
            current_mae = metrics.get("MAE(px)", float("inf"))
            if current_mae < best_mae:
                best_mae = current_mae
                best_path = os.path.join(ckpt_dir, "best_model.pth")
                save_checkpoint(model, optimizer, scheduler, scaler,
                                False, epoch, filename=best_path)
                logger.info(f"New best model saved! MAE: {best_mae:.4f}")

        if should_save:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler,
                            False, epoch, filename=ckpt_path)
            manage_checkpoints(ckpt_dir, cfgs.TRAINER.MAX_CKPT_SAVE_NUM)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    tb_writer.close()
    logger.info(f"Training completed! Best MAE: {best_mae:.4f}")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
