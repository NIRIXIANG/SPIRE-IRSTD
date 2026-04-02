"""仅在验证节点保存至多四个权重槽位：last / f1_best / rec_best / pre_best。"""
import glob
import os
import torch


def _unlink_glob(directory, pattern):
    for path in glob.glob(os.path.join(directory, pattern)):
        try:
            os.remove(path)
        except OSError:
            pass


def build_checkpoint_state(model_state, optimizer, lr_scheduler, epoch, scaler=None):
    state = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    return state


def save_eval_metric_checkpoints(
    save_dir,
    epoch,
    model_state,
    optimizer,
    lr_scheduler,
    scaler,
    precision,
    recall,
    f1_score,
    metric_bests,
    min_epoch,
):
    """
    与旧版相同的 checkpoint 字段（model / optimizer / lr_scheduler / epoch / scaler）。

    每个槽位目录下仅保留一个文件；更新时用 glob 删除同前缀旧文件再写入新 epoch 文件名。
    """
    if epoch < min_epoch:
        return

    ckpt = build_checkpoint_state(model_state, optimizer, lr_scheduler, epoch, scaler)

    _unlink_glob(save_dir, 'last_epoch*.pth')
    torch.save(ckpt, os.path.join(save_dir, f'last_epoch{epoch}.pth'))

    if f1_score > metric_bests['f1']:
        metric_bests['f1'] = float(f1_score)
        _unlink_glob(save_dir, 'f1_best_epoch*.pth')
        torch.save(ckpt, os.path.join(save_dir, f'f1_best_epoch{epoch}.pth'))

    if recall > metric_bests['rec']:
        metric_bests['rec'] = float(recall)
        _unlink_glob(save_dir, 'rec_best_epoch*.pth')
        torch.save(ckpt, os.path.join(save_dir, f'rec_best_epoch{epoch}.pth'))

    if precision > metric_bests['pre']:
        metric_bests['pre'] = float(precision)
        _unlink_glob(save_dir, 'pre_best_epoch*.pth')
        torch.save(ckpt, os.path.join(save_dir, f'pre_best_epoch{epoch}.pth'))
