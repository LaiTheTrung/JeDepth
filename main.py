#----- Thêm utility packages vào path (cần thiết khi chạy trên Kaggle) -----#
import os, sys
UTILITY_PATH = "/kaggle/input/waft-utility/kaggle/working"
if os.path.exists(UTILITY_PATH):
    sys.path.append(UTILITY_PATH)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
# ------------------------Code starts here------------------------ #
import logging
import os
import argparse
import sys
import json
import copy

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from algorithms.waft import WAFT
from bridgedepth.config import export_model_config
from bridgedepth.dataloader.depth_dataset import build_custom_loaders
from bridgedepth.loss import build_criterion
from bridgedepth.utils import misc
import bridgedepth.utils.dist_utils as comm
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils.launch import launch
from bridgedepth.utils.eval_disp import eval_disp
from bridgedepth.utils.eval_custom import evaluate_model, infer_test_images

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_args_parser():
    parser = argparse.ArgumentParser(
        f"""
        Examples:

        Run on single machine:
            $ {sys.argv[0]} --num-gpus 8

        Change some config options:
            $ {sys.argv[0]} SOLVER.IMS_PER_BATCH 8

        Run on multiple machines:
            (machine 0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine 1)$ {sys.argv[1]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action='store_true')
    parser.add_argument("--ckpt", default=None, help='path to the checkpoint file or model name when eval_only is True')
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    # distributed training
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details."
    )
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pair.
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER
    )

    return parser

def build_optimizer(model, cfg):
    base_lr = cfg.SOLVER.BASE_LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5, eps=1e-8)
    return optimizer


def _setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the bridgedepth logger
    2. Log basic information about environment, cmdline arguments, git commit, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    data_name = args.config_file.split('/')[-2]
    alg_name = args.config_file.split('/')[-1].split('.')[0]
    args.checkpoint_dir = f"ckpts/{data_name}/{alg_name}/{args.seed}"
    checkpoint_dir = args.checkpoint_dir
    if comm.is_main_process() and checkpoint_dir:
        misc.check_path(checkpoint_dir)

    rank = comm.get_rank()
    logger = setup_logger(checkpoint_dir, distributed_rank=rank, name='waft')

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + misc.collect_env_info())

    logger.info("git:\n {}\n".format(misc.get_sha()))
    logger.info("Command line arguments: " + str(args))

    if comm.is_main_process() and checkpoint_dir:
        path = os.path.join(checkpoint_dir, "config.yaml")
        with open(path, 'w') as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    misc.seed_all_rng(None if args.seed < 0 else args.seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def setup(args):
    """
    Create config and perform basic setups.
    """
    from bridgedepth.config import get_cfg
    cfg = get_cfg()
    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    _setup(cfg, args)
    comm.setup_for_distributed(comm.is_main_process())
    return cfg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def macs_profiler(model):
    input = torch.randn(1, 3, 544, 960).cuda()
    sample = {
        "img1": input,
        "img2": input,
    }
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_flops=True) as prof:
                output = model(sample)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    events = prof.events()
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: ", forward_MACs / 2 / 1e9, "G")
    print("Number of parameters: ", count_parameters(model) / 1e6, "M")


def main(args):
    cfg = setup(args)
    model = WAFT(cfg)
    model = model.to(torch.device("cuda"))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if comm.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # evaluate
    if args.eval_only:
        test_model = copy.deepcopy(model)
        macs_profiler(test_model)
        print('Load checkpoint: %s' % args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_without_ddp.load_state_dict(weights, strict=False)
        eval_disp(model, cfg)
        return

    logger = logging.getLogger("waft")
    optimizer = build_optimizer(model_without_ddp, cfg)
    criterion = build_criterion(cfg)
    device = torch.device("cuda")

    # Init weights từ --ckpt (dùng cho training kernel offline để bỏ HF download)
    if args.ckpt:
        logger.info('Load init weights from --ckpt: %s' % args.ckpt)
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        weights = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        missing, unexpected = model_without_ddp.load_state_dict(weights, strict=False)
        logger.info(f"  loaded; missing={len(missing)} unexpected={len(unexpected)}")

    # resume checkpoints
    start_epoch = 0
    start_step = 0
    resume = cfg.SOLVER.RESUME
    no_resume_optimizer = cfg.SOLVER.NO_RESUME_OPTIMIZER
    if resume:
        logger.info('Load checkpoint: %s' % resume)
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_without_ddp.load_state_dict(weights, strict=False)
        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not no_resume_optimizer:
            logger.info('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

    # training dataset (custom CSV-based)
    train_loader, val_loader, train_sampler = build_custom_loaders(cfg)

    max_epoch = cfg.SOLVER.MAX_EPOCH if cfg.SOLVER.MAX_EPOCH > 0 else 100
    total_iters = max_epoch * len(train_loader)

    # training scheduler
    last_epoch = start_step - 1 if resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg.SOLVER.BASE_LR,
        total_iters + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear',
        last_epoch=last_epoch
    )

    writer = None
    if comm.is_main_process():
        tb_dir = os.path.join(args.checkpoint_dir, "tb")
        misc.check_path(tb_dir)
        writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard logs → {tb_dir}")

    total_steps = start_step
    logger.info(f'Start training: {max_epoch} epochs, {len(train_loader)} iters/epoch, total {total_iters}')

    print_freq = 20
    avg_dict = {}
    for epoch in range(start_epoch, max_epoch):
        model.train()
        if comm.get_world_size() > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not comm.is_main_process())
        for sample in pbar:
            sample = {k: v.to(device) for k, v in sample.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.SOLVER.MIX_PRECISION):
                result_dict = model(sample)
                loss_dict, metrics = criterion(result_dict, sample, log=True)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            for param in model_without_ddp.parameters():
                param.grad = None
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()

            if comm.is_main_process() and sample["valid"].sum() > 0:
                for k, v in metrics.items():
                    meter = avg_dict.get(k, AverageMeter())
                    meter.update(float(v))
                    avg_dict[k] = meter

            total_steps += 1

            if total_steps % print_freq == 0 and comm.is_main_process():
                for k, meter in avg_dict.items():
                    writer.add_scalar(f"train/{k}", meter.avg, total_steps)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], total_steps)
                pbar.set_postfix({k: f"{m.avg:.3f}" for k, m in avg_dict.items()})
                avg_dict = {}

        # ── End of epoch: eval + checkpoint ──
        is_eval_epoch = ((epoch + 1) % cfg.TEST.EVAL_EPOCH_PERIOD == 0) or (epoch + 1 == max_epoch)
        if is_eval_epoch:
            logger.info(f'Validation @ epoch {epoch + 1}')
            results = evaluate_model(model, val_loader, cfg, device)
            if comm.is_main_process():
                for k, v in results.items():
                    writer.add_scalar(f"val/{k}", v, total_steps)
                logger.info("val: " + " | ".join(f"{k}={v:.4f}" for k, v in results.items()))
                infer_test_images(model, cfg, writer, total_steps, device)

                ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1:03d}.pth")
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'model_config': export_model_config(cfg),
                    'optimizer': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch + 1,
                    'val_metrics': results,
                }, ckpt_path)
                latest = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'model_config': export_model_config(cfg),
                    'optimizer': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch + 1,
                }, latest)
                logger.info(f"Saved checkpoint → {ckpt_path}")
            model.train()

    if comm.is_main_process() and writer is not None:
        writer.close()
    logger.info('Training done')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )