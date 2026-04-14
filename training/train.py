"""
train.py - Training script for FastFoundationStereo stereo depth estimation.

Usage:
    python train.py --cfg cfgs/ffstereo_custom.yaml --output_dir output
    python train.py --cfg cfgs/ffstereo_custom.yaml --resume output/ckpt_epoch_010.pth
"""

import os

# Auto-disable torch.compile on Kaggle (Triton ptxas binary on /kaggle/input is
# mounted without execute permissions). Can be overridden by exporting
# TORCH_COMPILE_DISABLE=0 explicitly.
if os.path.exists('/kaggle/input') and 'TORCH_COMPILE_DISABLE' not in os.environ:
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

import sys
import logging
import csv
import argparse
import glob
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict
import torchvision.transforms as T

from jedepth.core.foundation_stereo import FastFoundationStereo
from jedepth.core.utils.utils import InputPadder
from jedepth.dataset.depth_dataset import create_dataloaders
from jedepth.evaluation import evaluate
from jedepth.utils.utils import vis_disparity, set_seed, set_logging_format

logger = logging.getLogger(__name__)

# Utility packages (pre-installed from jedepth-utility-script kernel)
UTILITY_PATH = "/kaggle/input/notebooks/laithetrung/hitnet-utility-script"
if os.path.exists(UTILITY_PATH):
    sys.path.insert(0, UTILITY_PATH)

# ============================================================================
# Config & CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFoundationStereo")
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model weights (fine-tune)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--test_images", type=str, default="test_images", help="Path to test stereo pairs directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_config(cfg_path):
    """Load YAML config file and return EasyDict."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


# ============================================================================
# Model
# ============================================================================

def build_model(cfg):
    """Build FastFoundationStereo model with required attributes for get_loss().
    NOTE: model stays on CPU. Call model.to(device) AFTER loading pretrained weights.
    """
    model = FastFoundationStereo(cfg)
    # get_loss() uses self.max_disp (not set in __init__)
    model.max_disp = cfg.max_disp
    # get_loss() uses self.logger for warning messages
    model.logger = logging.getLogger('FastFoundationStereo')

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model built: {n_params:.1f}M parameters")
    return model


# ============================================================================
# Optimizer & Scheduler
# ============================================================================

def build_optimizer(model, cfg):
    """Build AdamW optimizer, MultiStepLR scheduler, and GradScaler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.lr_milestones,
        gamma=cfg.lr_gamma,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.mixed_precision)

    return optimizer, scheduler, scaler


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_epe, metrics):
    """Save full training checkpoint."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_epe': best_epe,
        'metrics': metrics,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """
    Load checkpoint. Returns (start_epoch, best_epe).
    If optimizer/scheduler/scaler are None, only loads model weights.
    """
    logger.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt.get('epoch', -1) + 1
        best_epe = ckpt.get('best_epe', float('inf'))

        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        logger.info(f"Resumed from epoch {start_epoch}, best EPE: {best_epe:.4f}")
    else:
        # Pretrained weights only (e.g. serialized model from NVlabs)
        model.load_state_dict(ckpt, strict=False)
        start_epoch = 0
        best_epe = float('inf')
        logger.info("Loaded pretrained weights (model only)")

    return start_epoch, best_epe


def load_pretrained(model, path, device='cpu'):
    """
    Load pretrained weights. Auto-detects format:
      1. NVlabs serialized model (nn.Module with named_children)
      2. Full training checkpoint (dict with 'model_state_dict')
      3. Raw state_dict (dict of tensors)

    Logs matched/mismatched/skipped keys for debugging.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained model not found: {path}")

    logger.info(f"Loading pretrained model: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Case 1: NVlabs serialized model (a full nn.Module)
    if isinstance(ckpt, torch.nn.Module):
        replaced, skipped = [], []
        for name, module in ckpt.named_children():
            if hasattr(model, name):
                setattr(model, name, module)
                replaced.append(name)
            else:
                skipped.append(name)
        logger.info(f"[Serialized model] Replaced {len(replaced)} modules: {replaced}")
        if skipped:
            logger.warning(f"[Serialized model] Skipped {len(skipped)} modules: {skipped}")
        return

    # Case 2: Full training checkpoint
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        epoch_info = f" (from epoch {ckpt.get('epoch', '?')})"
    elif isinstance(ckpt, dict):
        # Case 3: Raw state_dict
        state_dict = ckpt
        epoch_info = ""
    else:
        raise ValueError(f"Unknown checkpoint format: {type(ckpt)}")

    # Load with detailed logging
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Handle 'module.' prefix from DDP checkpoints
    if all(k.startswith('module.') for k in ckpt_keys):
        logger.info("Stripping 'module.' prefix from DDP checkpoint")
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        ckpt_keys = set(state_dict.keys())

    matched = model_keys & ckpt_keys
    missing_in_ckpt = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    # Filter by shape mismatch
    shape_mismatch = []
    loadable = {}
    model_state = model.state_dict()
    for k in matched:
        if model_state[k].shape == state_dict[k].shape:
            loadable[k] = state_dict[k]
        else:
            shape_mismatch.append(
                f"  {k}: model={model_state[k].shape} vs ckpt={state_dict[k].shape}"
            )

    model.load_state_dict(loadable, strict=False)

    logger.info(
        f"[State dict{epoch_info}] "
        f"Loaded {len(loadable)}/{len(model_keys)} params"
    )
    if missing_in_ckpt:
        logger.info(f"  Missing in checkpoint ({len(missing_in_ckpt)}): "
                     f"{list(missing_in_ckpt)[:5]}{'...' if len(missing_in_ckpt) > 5 else ''}")
    if unexpected:
        logger.info(f"  Unexpected in checkpoint ({len(unexpected)}): "
                     f"{list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
    if shape_mismatch:
        logger.warning(f"  Shape mismatch ({len(shape_mismatch)}):")
        for m in shape_mismatch[:5]:
            logger.warning(m)


def cleanup_checkpoints(output_dir, keep_last=3):
    """Keep only the last N epoch checkpoints + best_model.pth."""
    ckpts = sorted(glob.glob(os.path.join(output_dir, 'ckpt_epoch_*.pth')))
    if len(ckpts) > keep_last:
        for old_ckpt in ckpts[:-keep_last]:
            os.remove(old_ckpt)


# ============================================================================
# Test Images
# ============================================================================

def load_test_images(test_dir):
    """Load test stereo pairs for inference visualization."""
    left_dir = os.path.join(test_dir, 'left_rect')
    right_dir = os.path.join(test_dir, 'right_rect')

    if not os.path.exists(left_dir):
        logger.warning(f"Test images not found at {left_dir}")
        return []

    left_paths = sorted(glob.glob(os.path.join(left_dir, '*.png')))
    to_tensor = T.ToTensor()
    test_images = []

    for lp in left_paths:
        name = Path(lp).stem
        rp = os.path.join(right_dir, Path(lp).name)
        if not os.path.exists(rp):
            continue

        left_img = cv2.imread(lp)
        right_img = cv2.imread(rp)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        left_tensor = to_tensor(left_img).unsqueeze(0)   # (1,3,H,W)
        right_tensor = to_tensor(right_img).unsqueeze(0)  # (1,3,H,W)

        # Pad to divisible by 32
        padder = InputPadder(left_tensor.shape[-2:], divis_by=32)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)

        test_images.append({
            'left': left_padded,
            'right': right_padded,
            'padder': padder,
            'name': name,
            'left_orig': to_tensor(left_img),  # unpadded for visualization
        })

    logger.info(f"Loaded {len(test_images)} test stereo pairs")
    return test_images


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch, writer, global_step):
    """Train for one epoch. Returns (avg_loss, global_step)."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        left = batch['left'].to(device)
        right = batch['right'].to(device)
        # get_loss() expects (B,H,W) because it calls unsqueeze(1) internally
        disp_gt = batch['disp'].squeeze(1).to(device)

        model_input = {'left': left, 'right': right}
        loss_input = {'disp': disp_gt, 'name': batch['left_image_path'][0]}

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=cfg.mixed_precision, dtype=torch.float16):
            model_pred = model(model_input)
            loss, loss_info = model.get_loss(model_pred, loss_input)

        # NaN/inf detection
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/inf loss at epoch {epoch}, step {batch_idx}. Skipping batch.")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # TensorBoard logging
        if global_step % cfg.log_interval == 0 and writer is not None:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            for k, v in loss_info.items():
                writer.add_scalar(k, v, global_step)

    avg_loss = epoch_loss / max(num_batches, 1)
    return avg_loss, global_step


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate(model, val_loader, cfg, device):
    """Run validation and return averaged metrics."""
    model.eval()
    all_metrics = defaultdict(float)
    count = 0

    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for batch in pbar:
        left = batch['left'].to(device)
        right = batch['right'].to(device)
        disp_gt = batch['disp'].to(device)  # (B,1,H,W)

        with torch.amp.autocast('cuda', enabled=cfg.mixed_precision, dtype=torch.float16):
            pred = model({'left': left, 'right': right})

        pred_disp = pred['disp_pred']  # (B,1,H,W)

        valid_mask = (disp_gt.squeeze(1) > 0) & (disp_gt.squeeze(1) < cfg.max_disp)
        metrics = evaluate(pred_disp, disp_gt, valid_mask)

        for k, v in metrics.items():
            if not np.isnan(v):
                all_metrics[k] += v
        count += 1

    avg_metrics = {k: v / max(count, 1) for k, v in all_metrics.items()}
    return avg_metrics


# ============================================================================
# Test Inference Visualization
# ============================================================================

@torch.no_grad()
def run_test_inference(model, test_images, cfg, device, writer, epoch):
    """Run inference on test images and log visualizations to TensorBoard."""
    if not test_images or writer is None:
        return

    model.eval()

    for sample in test_images:
        left = sample['left'].to(device)
        right = sample['right'].to(device)

        with torch.amp.autocast('cuda', enabled=cfg.mixed_precision, dtype=torch.float16):
            pred = model({'left': left, 'right': right})

        disp = sample['padder'].unpad(pred['disp_pred'])  # (1,1,H,W)
        disp_np = disp.squeeze().cpu().float().numpy()       # (H,W)

        # Colorized disparity
        vis = vis_disparity(disp_np)  # (H,W,3) uint8
        writer.add_image(f'test/{sample["name"]}/disparity', vis, epoch, dataformats='HWC')

        # Left input image for reference
        left_vis = (sample['left_orig'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        writer.add_image(f'test/{sample["name"]}/left_input', left_vis, epoch, dataformats='HWC')


# ============================================================================
# CSV Logging
# ============================================================================

def log_to_csv(csv_path, epoch, train_loss, val_metrics, lr):
    """Append epoch results to experiments CSV."""
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer_csv = csv.writer(f)
        if not file_exists:
            header = ['timestamp', 'epoch', 'train_loss', 'EPE', 'D1_all', 'bad_1', 'bad_2', 'bad_3', 'lr']
            writer_csv.writerow(header)

        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            epoch,
            f"{train_loss:.6f}",
            f"{val_metrics.get('EPE', 0):.4f}",
            f"{val_metrics.get('D1_all', 0):.4f}",
            f"{val_metrics.get('bad_1', 0):.4f}",
            f"{val_metrics.get('bad_2', 0):.4f}",
            f"{val_metrics.get('bad_3', 0):.4f}",
            f"{lr:.6f}",
        ]
        writer_csv.writerow(row)


def save_experiment_json(output_dir, cfg, best_epe, best_metrics, total_epochs, args, n_params):
    """Save experiment summary as JSON for cross-experiment comparison.

    Creates/updates experiments.json with one entry per training run.
    Each entry contains: experiment name, date, model config, best metrics.
    """
    import json

    json_path = os.path.join(output_dir, 'experiments.json')

    # Load existing experiments
    experiments = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            experiments = json.load(f)

    # Build experiment entry
    entry = {
        'name': f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': {
            'name': 'FastFoundationStereo',
            'max_disp': cfg.max_disp,
            'vit_size': cfg.get('vit_size', 'vitl'),
            'train_iters': cfg.train_iters,
            'valid_iters': cfg.valid_iters,
            'params_M': round(n_params, 1),
        },
        'training': {
            'epochs': total_epochs,
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'crop_size': list(cfg.crop_size) if cfg.get('crop_size') else None,
            'pretrained': cfg.get('pretrained_model', ''),
            'seed': args.seed,
        },
        'best_metrics': {
            'EPE': best_metrics.get('EPE', None),
            'D1_all': best_metrics.get('D1_all', None),
            'bad_1': best_metrics.get('bad_1', None),
            'bad_2': best_metrics.get('bad_2', None),
            'bad_3': best_metrics.get('bad_3', None),
        },
    }

    experiments.append(entry)

    with open(json_path, 'w') as f:
        json.dump(experiments, f, indent=2)

    logger.info(f"Experiment saved to {json_path}: {entry['name']}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    # Setup
    set_logging_format()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.cfg}")

    # TensorBoard
    tb_dir = os.path.join(args.output_dir, 'tb_logs')
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"TensorBoard logs: {tb_dir}")

    # Build model on CPU first
    model = build_model(cfg)

    # Load pretrained weights BEFORE moving to GPU (serialized models need this)
    start_epoch = 0
    best_epe = float('inf')
    pretrained_path = args.pretrained or cfg.get('pretrained_model', '')
    if not args.resume and pretrained_path:
        load_pretrained(model, pretrained_path)

    # Move model to GPU
    model = model.to(device)

    # Build optimizer, scheduler, scaler (after model is on GPU)
    optimizer, scheduler, scaler = build_optimizer(model, cfg)

    # Resume full training state (model + optimizer + scheduler + epoch)
    if args.resume:
        start_epoch, best_epe = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    # Data
    logger.info("Loading datasets...")
    crop_size = tuple(cfg.crop_size) if cfg.get('crop_size') else None
    train_loader, val_loader = create_dataloaders(
        train_csv=cfg.train_csv,
        val_csv=cfg.get('val_csv', None),
        root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=args.workers,
        augment_train=True,
        crop_size=crop_size,
        seed=args.seed,
    )

    # Test images
    test_images = load_test_images(args.test_images)

    # Training loop
    global_step = start_epoch * len(train_loader)
    logger.info(f"Starting training from epoch {start_epoch} to {cfg.epochs}")
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    best_metrics = {}

    for epoch in range(start_epoch, cfg.epochs):
        # Warmup LR
        if epoch < cfg.warmup_epochs:
            warmup_lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, cfg, device,
            epoch, writer, global_step,
        )

        # Step scheduler (after warmup period)
        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, cfg, device)

        # Log to TensorBoard
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'epoch/val_{k}', v, epoch)

        # Test inference visualization
        run_test_inference(model, test_images, cfg, device, writer, epoch)

        # Log to CSV
        log_to_csv(
            os.path.join(args.output_dir, 'experiments.csv'),
            epoch, train_loss, val_metrics, optimizer.param_groups[0]['lr'],
        )

        # Epoch summary
        epe = val_metrics.get('EPE', float('inf'))
        logger.info(
            f"Epoch {epoch}/{cfg.epochs}: "
            f"loss={train_loss:.4f}, EPE={epe:.4f}, "
            f"D1={val_metrics.get('D1_all', 0):.2f}%, "
            f"bad1={val_metrics.get('bad_1', 0):.2f}%, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save checkpoint
        is_best = epe < best_epe
        if is_best:
            best_epe = epe
            best_metrics = val_metrics.copy()

        ckpt_path = os.path.join(args.output_dir, f'ckpt_epoch_{epoch:03d}.pth')
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, best_epe, val_metrics)

        if is_best:
            best_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_epe, val_metrics)
            logger.info(f"New best model! EPE: {best_epe:.4f}")

        # Keep only last 3 checkpoints
        cleanup_checkpoints(args.output_dir, keep_last=3)

    writer.close()

    # Save experiment summary JSON
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    save_experiment_json(args.output_dir, cfg, best_epe, best_metrics, cfg.epochs, args, n_params)

    logger.info(f"Training complete. Best EPE: {best_epe:.4f}")


if __name__ == '__main__':
    main()
