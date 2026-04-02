"""
DDP (Distributed Data Parallel) 多卡训练版本
结构对齐 BasicIRSTD-main/train.py，功能保持不变
"""
import argparse
import time
import os
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import SPIRENet
from utils.dataset import IRST
from utils.experiment_utils import build_experiment_name, resolve_output_subdir, write_text_block
from utils import transforms
from utils.train_eval_utils import train_one_epoch, validate_one_epoch, nrx_evaluate
from utils.checkpoint_save import save_eval_metric_checkpoints
from utils.plot_curve import plot_loss_and_lr, plot_val_loss

from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch SPIRE DDP train")
parser.add_argument("--world_size", default=-1, type=int, help="number of GPUs")
parser.add_argument("--local_rank", "--local-rank", default=-1, type=int, help="local rank")
parser.add_argument("--device", default='cuda', type=str, help="device")
parser.add_argument("--data_path", default=r'N:\AcademicResearchs\Datasets\RemoteDatasets\planedatasets',
                    type=str, help="dataset root directory")
parser.add_argument("--person_det", type=str, default=None)
parser.add_argument("--fixed_size", default=[640, 640], nargs='+', type=int, help="input size [H, W]")
parser.add_argument("--num_joints", default=1, type=int, help="number of keypoint joints")
parser.add_argument("--batchSize", type=int, default=2, help="batch size per GPU")
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument("--method_name", default='SPIRENet', type=str, help="method name used in result folder naming")
parser.add_argument("--save", default='./train_results', type=str, help="root directory used to store training results")
parser.add_argument("--resume", default='', type=str, help="resume from checkpoint")
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
parser.add_argument("--eval_interval", default=10, type=int, help="evaluation interval (epochs)")
parser.add_argument("--min_save_weight_epochs", default=0, type=int, help="minimum epochs before saving weights")
parser.add_argument("--threshold", type=float, default=0.35, help="detection threshold")
parser.add_argument("--value_range", type=float, default=0.35, help="value range for NMS")
parser.add_argument("--max_num_targets", default=8, type=int, help="max number of targets")
parser.add_argument("--tp_distance", default=5, type=int, help="TP distance threshold")
parser.add_argument("--amp", action="store_true", help="use mixed precision training")

global opt
opt = parser.parse_args()
opt.save = resolve_output_subdir(
    opt.save,
    build_experiment_name(
        opt.method_name, opt.fixed_size, opt.data_path, opt.lr, opt.batchSize, "ddp"
    ),
)


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    backend = 'nccl' if torch.cuda.is_available() and hasattr(dist, "is_nccl_available") and dist.is_nccl_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://',
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_value(value, world_size, device):
    if isinstance(value, torch.Tensor):
        rt = value.detach().clone().to(device)
    else:
        rt = torch.tensor(value, device=device, dtype=torch.float32)
    if not dist.is_available() or not dist.is_initialized():
        return rt
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def create_model():
    model = SPIRENet(base_channel=32, num_joints=opt.num_joints)
    return model


def build_init_log_lines(world_size, device, num_workers, train_dataset, val_dataset):
    return [
        "Initialization Info",
        f"Configuration: {opt}",
        f"Available GPUs: {world_size}",
        f"Using {world_size} GPU(s) for training.",
        f"System CPU cores: {os.cpu_count() or 1}, GPUs: {world_size}, workers/GPU: {num_workers}",
        f"Main process device: {device}",
        f"Train dataset size: {len(train_dataset)}",
        f"Validation dataset size: {len(val_dataset)}",
        f"Save directory: {opt.save}",
        "=================================================",
    ]


def train(rank, world_size):
    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process():
        if not os.path.exists(opt.save):
            os.makedirs(opt.save)
        print(f"Using {world_size} GPU(s) for training.")
    if use_ddp:
        dist.barrier()

    cpu_count = os.cpu_count() or 1
    reserved_cores = max(2, world_size)
    available_cores = max(1, cpu_count - reserved_cores)
    nw = max(1, min(available_cores // world_size, 8))
    if is_main_process():
        print(f'System CPU cores: {cpu_count}, GPUs: {world_size}, workers/GPU: {nw}')

    fixed_size = opt.fixed_size
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)

    data_transform = {
        "train": transforms.Compose([
            transforms.AffineTransform(scale=(1, 1.15), fixed_size=fixed_size),
            transforms.nrxKeypointToHeatMap_targetEnhance(heatmap_hw=heatmap_hw, gaussian_sigma=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.AffineTransform(scale=(1, 1), fixed_size=fixed_size),
            transforms.nrxKeypointToHeatMap_targetEnhance(heatmap_hw=heatmap_hw, gaussian_sigma=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_dataset = IRST(opt.data_path, "train", transforms=data_transform["train"], fixed_size=opt.fixed_size)
    val_dataset = IRST(opt.data_path, "test", transforms=data_transform["test"], fixed_size=opt.fixed_size,
                       det_json_path=opt.person_det)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=not use_ddp,
                              sampler=train_sampler, pin_memory=True, num_workers=nw,
                              collate_fn=train_dataset.collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False,
                            sampler=val_sampler, pin_memory=True, num_workers=nw,
                            collate_fn=val_dataset.collate_fn)

    eval_loader = None
    if is_main_process():
        eval_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False,
                                 pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    model = create_model()
    val_result_path = os.path.join(opt.save, 'val_result.txt')
    if is_main_process():
        write_text_block(
            val_result_path,
            build_init_log_lines(world_size, device, nw, train_dataset, val_dataset),
        )
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    writer = None
    if is_main_process():
        writer = SummaryWriter(os.path.join(opt.save, 'tensorboard_logs'))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    if opt.resume != "":
        checkpoint = torch.load(opt.resume, map_location=device)
        (model.module if use_ddp else model).load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        opt.start_epoch = checkpoint['epoch'] + 1
        if opt.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if is_main_process():
            print(f"Resuming training from epoch {opt.start_epoch}...")

    train_loss = []
    val_loss = []
    learning_rate = []
    best_f1_score = -1
    best_epoch = 0
    metric_bests = {'f1': -1.0, 'rec': -1.0, 'pre': -1.0}
    best_metrics = {
        "epoch": 0, "total_tp": 0, "total_fp": 0, "total_fn": 0,
        "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "val_mean_loss": float('inf')
    }

    for epoch in range(opt.start_epoch, opt.nEpochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch(model, optimizer, train_loader,
                                        device=device, epoch=epoch,
                                        print_freq=50, warmup=True, scaler=scaler)
        mean_loss_reduced = reduce_value(mean_loss, world_size, device)

        if is_main_process():
            train_loss.append(mean_loss_reduced.item())
            learning_rate.append(lr)
            writer.add_scalar('Loss/train', mean_loss_reduced.item(), epoch)
            writer.add_scalar('Learning Rate', lr, epoch)

        if epoch % opt.eval_interval == 0:
            val_mean_loss = validate_one_epoch(model, val_loader, device, scaler)
            val_loss_reduced = reduce_value(val_mean_loss, world_size, device)
            lr_scheduler.step(val_loss_reduced.item())

            if is_main_process():
                val_loss.append(val_loss_reduced.item())
                writer.add_scalar('Loss/val', val_loss_reduced.item(), epoch)

                total_tp, total_fp, total_fn, precision, recall, f1_score = test(
                    model.module if use_ddp else model, eval_loader, device, scaler, epoch
                )

                writer.add_scalar('Val/TP', total_tp, epoch)
                writer.add_scalar('Val/FP', total_fp, epoch)
                writer.add_scalar('Val/FN', total_fn, epoch)
                writer.add_scalar('Val/Precision', precision, epoch)
                writer.add_scalar('Val/Recall', recall, epoch)
                writer.add_scalar('Val/F1', f1_score, epoch)

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_metrics.update({
                        "epoch": best_epoch, "total_tp": total_tp, "total_fp": total_fp,
                        "total_fn": total_fn, "precision": precision, "recall": recall,
                        "f1_score": f1_score, "val_mean_loss": val_loss_reduced.item()
                    })

                save_eval_metric_checkpoints(
                    opt.save, epoch, (model.module if use_ddp else model).state_dict(), optimizer, lr_scheduler, scaler,
                    precision, recall, f1_score, metric_bests, opt.min_save_weight_epochs,
                )

                with open(val_result_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Epoch: {epoch}\n")
                    log_file.write(f"TP: {total_tp} FP: {total_fp} FN: {total_fn}\n")
                    log_file.write(f"Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1_score:.4f}\n")
                    log_file.write(f"Validation Mean Loss: {val_loss_reduced.item():.4f}\n")
                    log_file.write(
                        f"Best F1: {best_f1_score:.4f} @ {best_epoch} | "
                        f"slot bests P/R/F1: {metric_bests['pre']:.4f} / {metric_bests['rec']:.4f} / {metric_bests['f1']:.4f}\n"
                    )
                    log_file.write("=================================================\n")

        if use_ddp:
            dist.barrier()

    if is_main_process():
        print(f"Best F1 Score: {best_f1_score:.4f} at Epoch: {best_epoch}")
        print(f"Metrics at best F1: TP: {best_metrics['total_tp']}, FP: {best_metrics['total_fp']}, "
              f"FN: {best_metrics['total_fn']}, Precision: {best_metrics['precision']:.4f}, "
              f"Recall: {best_metrics['recall']:.4f}, F1: {best_metrics['f1_score']:.4f}")
        print(f"Metric bests (checkpoint slots): P/R/F1 = "
              f"{metric_bests['pre']:.4f} / {metric_bests['rec']:.4f} / {metric_bests['f1']:.4f}")
        print(f"Checkpoints (last + best slots): {opt.save}")
        writer.close()

        if len(train_loss) != 0 and len(learning_rate) != 0:
            plot_loss_and_lr(train_loss, learning_rate, opt.save)
        if len(val_loss) != 0:
            plot_val_loss(val_loss, opt.save)

    cleanup_ddp()


def test(model, data_loader, device, scaler, epoch):
    total_tp, total_fp, total_fn, precision, recall, f1_score = nrx_evaluate(
        model, data_loader, device, scaler, opt.threshold, opt.value_range,
        opt.max_num_targets, opt.save, opt.data_path, opt.tp_distance
    )
    return total_tp, total_fp, total_fn, precision, recall, f1_score


if __name__ == '__main__':
    if opt.world_size == -1:
        opt.world_size = torch.cuda.device_count()

    print(f"Configuration: {opt}")
    print(f"Available GPUs: {opt.world_size}")

    if opt.world_size < 2:
        print("Warning: Less than 2 GPUs available. Consider using single-GPU version (train.py).")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        train(rank, world_size)
    elif 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', opt.world_size))
        train(rank, world_size)
    elif opt.world_size <= 1:
        train(0, 1)
    else:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(opt.world_size,), nprocs=opt.world_size, join=True)
