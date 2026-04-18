import random
import time
import argparse
from itertools import chain, combinations

import numpy as np
import torch
import yaml
from decorator import decorator
import wandb


def print_exp_info(args, config, epoch):
    s = (
        f'epoch: [{epoch+1}/{config["num_epochs"]}] {args.device} {args.task_name} '
        f'method: {args.method} mech: {args.mechanism} K: {args.num_clients} p_miss_train: {args.p_miss_train} '
        f'seed: {args.seed} lr: {config["lr"]} decay: {config["weight_decay"]} '
        f'mom: {config["momentum"]}'
    )
    sep = "-" * len(s)
    print(f"{sep}\n{s}\n{sep}")


def init_wandb(args, config):
    wandb.init(
        project=args.project,
        config={
            "task_name": args.task_name,
            "mechanism": args.mechanism,
            "cut_dim": config.get("cut_dim"),
            "dataset": config["dataset"],
            "architecture": config["model"],
            "cuda": args.cuda_id,
            "method": args.method,
            "lr": config["lr"],
            "n_epochs": config["num_epochs"],
            "seed": args.seed,
            "num_clients": args.num_clients,
            "batch_size": config["batch_size"],
            "weight_decay": config["weight_decay"],
            "momentum": config["momentum"],
            "p_miss_train": args.p_miss_train,
        },
        name = (
            args.wandb_name 
            if args.wandb_name 
            else f"{args.task_name}_{args.method}_K{args.num_clients}_p_miss_train{args.p_miss_train}_s{args.seed}"
        ),
    )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@decorator
def time_decorator(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    print(f"time to run {func.__name__}(): {time.time() - start:.2f}s.")
    return result


def powerset_except_empty(n):
    return list(chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1)))


def setup_task(args):
    # 延迟导入，防止循环依赖
    from models import get_model 
    from optimizers import get_optimizer
    from criterions import get_criterion
    from schedulers import get_scheduler

    with open("configs/task_config.yaml", encoding='utf-8') as file:
        config = yaml.safe_load(file)[args.method_type][args.task_name][args.num_clients][args.p_miss_train]

    if args.cut_dim is not None:
        config['cut_dim'] = args.cut_dim

    # ===============================================================
    # [核心逻辑] 根据配置判断模型类型
    # diabetes6 如果在 yaml 中配置为 resnet/cnn，将自动走 else 分支
    # ===============================================================
    if config["model"] == "mlp":
        from models.mlp import MLPLaserModel, MLPDecoupledModel
        target_cut_dim = config.get('cut_dim', None)
        
        if args.method_type == "laser":
            # LASER 只需要一个大模型
            model = MLPLaserModel(config["dataset"], args.num_clients, cut_dim=target_cut_dim)
            model.to(args.device)
        else:
            # Decoupled (Local, Combinatorial) 需要一组模型列表
            model_list = []
            print(f"[Setup] Initializing {len(args.blocks_in_tasks_t)} independent models for {args.method}...")
            for clients_subset in args.blocks_in_tasks_t:
                m = MLPDecoupledModel(
                    config["dataset"], 
                    args, 
                    clients_in_model=list(clients_subset), 
                    cut_dim=target_cut_dim
                )
                m.to(args.device)
                model_list.append(m)
            model = model_list 
    else:
        # 非 MLP 模型（如 ResNet）走原有逻辑
        # 这确保了 diabetes6 (图像任务) 能正确调用 get_model 加载 ResNet
        model = get_model(args.method_type, config["model"], config["dataset"], args, config)
        
        # 👇👇👇 [修复核心] 确保 ResNet 等图像模型也被推送到显卡 👇👇👇
        if isinstance(model, list):
            for m in model: 
                m.to(args.device)
        else:
            model.to(args.device)
        # 👆👆👆 ========================================= 👆👆👆

    # ===============================================================
    # [优化器处理] 支持 List 或 Single Model
    # ===============================================================
    if isinstance(model, list):
        optimizer = [
            get_optimizer(args.method_type, config["optimizer"], m, config) 
            for m in model
        ]
        scheduler = []
        for opt in optimizer:
            sch = get_scheduler(args.method_type, config["scheduler"], opt, config)
            if sch is not None:
                scheduler.append(sch)
        if len(scheduler) == 0: scheduler = None
    else:
        optimizer = get_optimizer(args.method_type, config["optimizer"], model, config)
        scheduler = get_scheduler(args.method_type, config["scheduler"], optimizer, config)

    criterion = get_criterion(config["criterion"])
    
    method_map = {
        "laser": ("methods.laser", "train_laser", "test_laser"),
        "decoupled": ("methods.decoupled", "train_decoupled", "test_decoupled"),
        "ensemble": ("methods.ensemble", "train_ensemble", "test_ensemble"),
        "plug": ("methods.plug", "train_plug", "test_plug"),
    }
    module_name, train_name, test_name = method_map[args.method_type]
    mod = __import__(module_name, fromlist=[train_name, test_name])
    train, test = getattr(mod, train_name), getattr(mod, test_name)

    return config, model, optimizer, scheduler, criterion, train, test

def process_method(args):
    if args.method in ("local", "ensemble"):
        args.blocks_in_tasks_t =  [(i,) for i in range(args.num_clients)]
    elif args.method in ("svfl", "plug"):
        args.blocks_in_tasks_t =  [tuple(range(args.num_clients))]
    elif args.method in ("combinatorial", "laser"):
        args.blocks_in_tasks_t =  powerset_except_empty(args.num_clients)
    else:
        raise ValueError("Invalid method.")
    
    args.method_type = "decoupled" if args.method in ("local", "svfl", "combinatorial") else args.method


def get_metrics(train_metrics, test_metrics, compute_f1, blocks_in_tasks_t):
    return {
        str(clients): {
            "train_loss": train_metrics["train_loss"][i],
            "train_acc": train_metrics["train_acc"][i],
            "test_loss": test_metrics["test_loss"][i],
            "test_acc": test_metrics["test_acc"][i],
            **(
                {"train_f1": train_metrics["train_f1"][i], "test_f1": test_metrics["test_f1"][i]}
                if compute_f1 else {}
            ),
        }
        for i, clients in enumerate(blocks_in_tasks_t)
    }


def none_or_float(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float or 'None'.")