import os
import random
import math
import numpy as np
import torch
import torch.multiprocessing
import torch.distributed as dist
from configs import cfg, args
from networks import make_network
from train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from datasets import make_data_loader
from utils.net_utils import load_model, save_model, load_network, load_pretrain
from evaluators import make_evaluator


if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def train(cfg, network):
    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)
    if cfg.skip_eval:
        val_loader = None
    else:
        val_loader = make_data_loader(cfg, is_train=False)
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg) if cfg.local_rank == 0 else None

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    if begin_epoch == 0 and cfg.train.pretrain != '':
        load_pretrain(network, cfg.train.pretrain)

    set_lr_scheduler(cfg, scheduler)
    
    # 提前计算 20% 对应的 warmup epoch 节点
    warmup_epochs = max(1, int(cfg.train.epoch * 0.20))
    force_occ_full = bool(getattr(cfg.train, 'force_occ_full', False))

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_loader.dataset.epoch = epoch

        if force_occ_full:
            current_step_ratio = 1.0
            current_epoch_ratio = 1.0
        else:
            # =======================================================
            # [控制 1] 掩码强度控制：20% 非线性课程学习 (Cosine S-Curve)
            # =======================================================
            # 1. 计算局部线性进度 (0.0 -> 1.0)，一旦 epoch >= warmup_epochs，锁定为 1.0
            linear_progress = min(1.0, epoch / warmup_epochs)
            # 2. 映射为平滑的余弦 S 型非线性曲线
            current_step_ratio = (1.0 - math.cos(math.pi * linear_progress)) / 2.0
            
            # =======================================================
            # [控制 2] 触发开关控制：全局 Epoch 比例 (用于 network 里的 5% 阈值)
            # =======================================================
            current_epoch_ratio = float(epoch) / max(1.0, float(cfg.train.epoch))
            # =======================================================

        # 将两个控制参数显式传给 trainer
        trainer.train(epoch, train_loader, optimizer, recorder, 
                      step_ratio=current_step_ratio, 
                      epoch_ratio=current_epoch_ratio)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        if not cfg.skip_eval and (epoch + 1) % cfg.eval_ep == 0:
            # All ranks must enter eval to keep DDP collectives in the same order.
            trainer.val(epoch, val_loader, evaluator, recorder if cfg.local_rank == 0 else None,
                        step_ratio=current_step_ratio,
                        epoch_ratio=current_epoch_ratio)
            synchronize()

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    # 测试阶段直接火力全开，传入 1.0 激活全量掩码剔除
    trainer.val(epoch, val_loader, evaluator, step_ratio=1.0, epoch_ratio=1.0)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",  init_method="env://")
        synchronize()

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)
    if cfg.local_rank == 0:
        print('Success!')
        print('='*80)


if __name__ == "__main__":
    main()
