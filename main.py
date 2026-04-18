import argparse
import wandb
import torch
import sys
import os
import traceback

sys.path.append('results') 

from utils import (time_decorator, print_exp_info, init_wandb, setup_task,
                   set_seed, process_method, get_metrics, none_or_float)
from data.data_utils import get_dataloaders

try:
    from tsne_utils import visualize_tsne
except ImportError:
    def visualize_tsne(*args, **kwargs): pass

# 安全获取指标函数
def safe_get_metrics(train_m, test_m, compute_f1, blocks):
    try:
        return get_metrics(train_m, test_m, compute_f1, blocks)
    except (KeyError, IndexError, ValueError) as e:
        safe = {}
        try:
            if "train_acc" in train_m: 
                val = train_m["train_acc"]
                safe["train_acc"] = val[0] if isinstance(val, list) else val
            if "test_acc" in test_m: 
                val = test_m["test_acc"]
                safe["test_acc"] = val[0] if isinstance(val, list) else val
        except:
            pass
        return safe

@time_decorator
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    process_method(args)
    
    try:
        config, model, optimizer, scheduler, criterion, train, test = setup_task(args)
    except Exception as e:
        print(f"[FATAL] Setup failed: {e}")
        print(traceback.format_exc())
        return

    # [兼容性修复] 统一转为列表处理，适配 diabetes6 等多模型场景
    models = model if isinstance(model, list) else [model]
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    
    schedulers = []
    if scheduler is not None:
        if isinstance(scheduler, list):
            for item in scheduler:
                if isinstance(item, list): schedulers.extend(item)
                else: schedulers.append(item)
        else:
            schedulers = [scheduler]
    
    if args.cut_dim is not None: config['cut_dim'] = args.cut_dim
    config['p_miss_train'] = args.p_miss_train
    
    # [核心修复] 如果命令行传入了 p_miss (测试缺失率)，则覆盖默认配置
    if hasattr(args, 'p_miss') and args.p_miss is not None:
        # 注意：这里我们只影响测试集的数据加载
        print(f"[Main] Overriding test missing rate with --p_miss={args.p_miss}")
        p_miss_test_init = args.p_miss
    else:
        p_miss_test_init = 0.0
        
    print(f"[Main] Loading Training Data with p_miss_train={args.p_miss_train}...")
    try:
        # 尝试传递 p_miss_test 参数 (适配新版 get_dataloaders)
        train_loader, test_loader_init = get_dataloaders(args, config, p_miss_test=p_miss_test_init)
    except TypeError:
        # 回退兼容旧版
        train_loader, test_loader_init = get_dataloaders(args, config)
    
    if args.use_wandb: init_wandb(args, config)
    
    # [核心修复] 确保 diabetes6 被识别为需要计算 F1 的任务
    compute_f1 = True if args.task_name in ["mimic4", "credit", "diabetes5", "diabetes6"] else False
    
    try:
        for epoch in range(config["num_epochs"]):
            
            # 1. 训练
            train_metrics = train(train_loader, models, optimizers, criterion, args, compute_f1=compute_f1)
            
            # 2. 测试 (使用初始测试集)
            test_metrics = test(test_loader_init, models, criterion, args, compute_f1=compute_f1)
            
            # 3. 打印 Loss 和 Acc
            # 提取 Train Loss
            tr_loss_list = train_metrics.get("train_loss", [0.0])
            if isinstance(tr_loss_list, list) and len(tr_loss_list) > 0:
                avg_tr_loss = sum(tr_loss_list) / len(tr_loss_list)
            else:
                avg_tr_loss = tr_loss_list if isinstance(tr_loss_list, (float, int)) else 0.0
            
            # 提取 Test Acc
            te_acc_list = test_metrics.get("test_acc", [0.0])
            if isinstance(te_acc_list, list) and len(te_acc_list) > 0:
                avg_te_acc = sum(te_acc_list) / len(te_acc_list)
            else:
                avg_te_acc = te_acc_list if isinstance(te_acc_list, (float, int)) else 0.0

            print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
                  f"Train Loss: {avg_tr_loss:.4f} | Test Acc: {avg_te_acc:.2f}%")

            # 更新学习率
            for sch in schedulers:
                if hasattr(sch, 'step'): sch.step()

            if args.use_wandb: 
                wandb.log(safe_get_metrics(train_metrics, test_metrics, compute_f1, args.blocks_in_tasks_t))

        print("\n" + "="*50 + "\n[Final Test] Running Sensitivity Analysis...\n" + "="*50)

        # 如果命令行指定了单一 p_miss，优先测这个；否则测默认列表
        test_miss_rates = getattr(args, 'final_p_miss_test_l', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
        if hasattr(args, 'p_miss') and args.p_miss is not None:
             # 如果指定了特定缺失率，确保它被包含在最终测试列表中
             if args.p_miss not in test_miss_rates:
                 test_miss_rates.append(args.p_miss)

        for p_miss_test in test_miss_rates:
            if p_miss_test is None: continue
            print(f" -> Testing with p_miss={p_miss_test}...")
            
            try:
                _, final_test_loader = get_dataloaders(args, config, p_miss_test)
            except TypeError:
                _, final_test_loader = get_dataloaders(args, config)

            # [关键] 传入 is_final=True 以便让 metric key 带有 final_ 前缀
            test_metrics = test(final_test_loader, models, criterion, args, compute_f1=compute_f1, is_final=True)
            
            acc = test_metrics.get("final_test_acc", 0.0)
            print_str = (f"(p_miss_test {p_miss_test}) final_test_acc: {acc:.2f}%")
            
            # 记录到 WandB (带后缀以便区分)
            metrics = {f'final_test_acc_{p_miss_test}': acc}
            
            # 同时记录 Loss (如果有)
            if "final_test_loss" in test_metrics:
                metrics[f'final_test_loss_{p_miss_test}'] = test_metrics["final_test_loss"]
            
            if compute_f1:
                f1 = test_metrics.get("final_test_f1", 0.0)
                metrics[f'final_test_f1_{p_miss_test}'] = f1
                print_str += f" | final_test_f1: {f1:.4f}"
                
            if args.use_wandb: wandb.log(metrics)
            print(print_str)
            
        # === [新增] t-SNE 隐空间特征可视化 ===
        if args.tsne:
            # 导入我们刚刚新建的模块
            from tsne_visualizer import plot_tsne
            # 因为这里可能是 LASER，所以取列表里的第一个（或主体）模型
            main_model = models[0] if isinstance(models, list) else models
            plot_tsne(main_model, test_loader_init, args.device, args.method, args.task_name, p_miss_test_init)

    except Exception as e:
        print("\n" + "!"*50)
        print(f"[CRITICAL ERROR] Training crashed: {e}")
        print(traceback.format_exc())
        print("!"*50 + "\n")
    
    finally:
        if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--wandb_name', help='Name of the run.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0]) # 此时 seeds 只是个列表
    # [修正] 单次运行的 seed 参数，通常由 external loop 传入
    parser.add_argument('--seed', type=int, default=None, help='Single seed for this run')
    
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--p_miss_train', type=none_or_float, default=0.0)
    
    # [核心修复] 添加 --p_miss 参数 (用于测试时的缺失率)
    parser.add_argument('--p_miss', type=float, default=None, help='Test missing rate for ablation study')
    
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb')
    parser.add_argument('--method', choices=['local', 'svfl', 'ensemble', 'combinatorial', 'plug', 'laser'], required=True)
    parser.add_argument('--p_drop', type=float, default=0.0)
    parser.add_argument('--project', type=str, default='laser')
    parser.add_argument('--mechanism', type=str, default='uniform')
    parser.add_argument('--cut_dim', type=int, default=None)
    parser.add_argument('--viz_tsne', action='store_true')
    
    # [新增] t-SNE 触发开关
    parser.add_argument('--tsne', action='store_true', help='Generate t-SNE visualization after training')

    args = parser.parse_args()
    args.final_p_miss_test_l = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9] 
    args.device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')

    # 逻辑修正：如果命令行指定了 --seed，则只跑这一个；否则跑 --seeds 列表
    if args.seed is not None:
        # 单次运行模式 (通常由 .bat 或并行脚本调用)
        main(args)
    else:
        # 批量运行模式 (旧逻辑)
        for seed in args.seeds:
            args.seed = seed
            main(args)