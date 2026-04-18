import math
import torch
import torch.nn.functional as F
from utils import time_decorator
from methods.method_utils import get_f1

# =========================================================================
# [修复核心] 定义递归辅助函数
# 作用：自动处理单个优化器、优化器列表、甚至嵌套列表
# 这样 Dataset 1-5 (单/简单列表) 和 Dataset 6 (复杂列表) 都能通用
# =========================================================================
def zero_grad_all(optimizers):
    """递归地对所有优化器执行 zero_grad()"""
    if isinstance(optimizers, (list, tuple)):
        for opt in optimizers:
            zero_grad_all(opt)
    else:
        # 只有当对象真正具备 zero_grad 方法时才调用，防止报错
        if hasattr(optimizers, 'zero_grad'):
            optimizers.zero_grad()

def step_all(optimizers):
    """递归地对所有优化器执行 step()"""
    if isinstance(optimizers, (list, tuple)):
        for opt in optimizers:
            step_all(opt)
    else:
        if hasattr(optimizers, 'step'):
            optimizers.step()

@time_decorator
def test_laser(dataloader, models, criterion, args, is_final=False, compute_f1=False, is_train_data=False):
    model = models[0] if isinstance(models, list) else models
    model.eval()
    num_samples = len(dataloader.dataset)
    
    total_correct = 0.0
    total_loss = 0.0
    
    if compute_f1:
        total_tp, total_fp, total_fn = 0.0, 0.0, 0.0

    # [消融实验开关] 保持不变
    force_simple_average = False
    if hasattr(args, 'wandb_name') and args.wandb_name and 'no_weight' in args.wandb_name:
        force_simple_average = True
        if not hasattr(test_laser, "_has_printed_ablation_msg"):
            print("\n" + "!"*60)
            print("[Ablation Mode] Detected 'no_weight' in run name.")
            print("              -> Forcing SIMPLE AVERAGE aggregation (Weight=1.0).")
            print("!"*60 + "\n")
            test_laser._has_printed_ablation_msg = True

    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs = [t.to(args.device) for t in inputs]
            targets = targets.to(args.device)
            mask = mask.to(args.device)
            
            masked_inputs_list = []
            for k, inp in enumerate(inputs):
                if mask.dim() > 1:
                    m = mask[:, k].view([-1] + [1] * (inp.dim() - 1))
                    masked_inputs_list.append(inp * m)
                else:
                    masked_inputs_list.append(inp * mask[k])
            
            full_input = torch.cat(masked_inputs_list, dim=1)
            
            outputs_per_head_l = model(full_input, training=False)
            
            ensemble_probs = torch.zeros(targets.size(0), 2, device=args.device)
            total_weight_per_sample = torch.zeros(targets.size(0), 1, device=args.device)

            for outputs_per_task_d in outputs_per_head_l:
                 for clients_subset, outputs in outputs_per_task_d.items():
                    if len(clients_subset) > 0:
                        subset_mask_cols = mask[:, list(clients_subset)]
                        is_valid = subset_mask_cols.all(dim=1, keepdim=True).float()
                    else:
                        is_valid = torch.zeros(targets.size(0), 1, device=args.device)

                    if force_simple_average:
                        base_weight = 1.0
                    else:
                        base_weight = len(clients_subset)
                    
                    probs = F.softmax(outputs, dim=1)
                    weighted_probs = base_weight * probs * is_valid
                    ensemble_probs += weighted_probs
                    total_weight_per_sample += (base_weight * is_valid)
            
            final_probs = ensemble_probs / (total_weight_per_sample + 1e-10)
            
            # 计算 Loss
            log_probs = torch.log(final_probs + 1e-10)
            batch_loss = F.nll_loss(log_probs, targets, reduction='sum')
            total_loss += batch_loss.item()

            predicted = final_probs.argmax(1)
            total_correct += (predicted == targets).float().sum().item()
            
            if compute_f1:
                total_tp += ((predicted == 1) & (targets == 1)).sum().item()
                total_fp += ((predicted == 1) & (targets == 0)).sum().item()
                total_fn += ((predicted == 0) & (targets == 1)).sum().item()

    final_acc = 100 * total_correct / max(1, num_samples)
    final_avg_loss = total_loss / max(1, num_samples)
    
    data_split_type = "train" if is_train_data else "test"
    metrics = {}
    metrics[f"final_{data_split_type}_acc"] = final_acc
    metrics[f"final_{data_split_type}_loss"] = final_avg_loss
    
    if compute_f1:
        metrics[f"final_{data_split_type}_f1"] = get_f1(total_tp, total_fp, total_fn)
        
    if not is_final:
        metrics[f"{data_split_type}_loss"] = [final_avg_loss] 
        metrics[f"{data_split_type}_acc"] = [final_acc]
        if compute_f1:
            metrics[f"{data_split_type}_f1"] = [metrics[f"final_{data_split_type}_f1"]]

    return metrics

@time_decorator
def train_laser(dataloader, models, optimizers, criterion, args, compute_f1=False):
    # [修改点 1] 不再尝试硬提取单个 optimizer，保持原样传入
    # 旧代码: optimizer = optimizers[0] ... (这行删除了)
    
    model = models[0] if isinstance(models, list) else models
    device = args.device
    model.train()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    loss_d = {clients_l: 0.0 for clients_l in model.powerset}
    correct_d = {clients_l: 0.0 for clients_l in model.powerset}
    if compute_f1:
        tp_d = {k: 0.0 for k in model.powerset}
        fp_d = {k: 0.0 for k in model.powerset}
        fn_d = {k: 0.0 for k in model.powerset}

    num_heads = 0
    for batch_num, batch in enumerate(dataloader):
        *inputs, targets, mask = batch
        inputs = [t.to(device) for t in inputs]
        targets = targets.to(device)
        mask = mask.to(device)
        if torch.sum(mask).item() == 0: continue
        
        # [修改点 2] 使用递归函数清空梯度
        # 这对 list 或 object 都有效
        zero_grad_all(optimizers)
        
        total_batch_loss = 0
        if mask.dim() > 1:
            unique_masks, inverse_indices = torch.unique(mask, dim=0, return_inverse=True)
        else:
            unique_masks = mask.unsqueeze(0)
            inverse_indices = torch.zeros(mask.shape[0], dtype=torch.long, device=device)
        for group_idx, m_pattern in enumerate(unique_masks):
            if m_pattern.sum() == 0: continue
            sample_indices = (inverse_indices == group_idx).nonzero().flatten()
            if len(sample_indices) == 0: continue
            sub_inputs_list = [inp[sample_indices] for inp in inputs]
            sub_targets = targets[sample_indices]
            sub_inputs_cat = torch.cat(sub_inputs_list, dim=1)
            observed_blocks = torch.nonzero(m_pattern).flatten().tolist()
            outputs_per_head_l = model(sub_inputs_cat, training=True, observed_blocks=observed_blocks)
            if num_heads == 0: num_heads = len(outputs_per_head_l)
            for outputs_per_task_d in outputs_per_head_l:
                for clients_subset, outputs in outputs_per_task_d.items():
                    loss = criterion(outputs, sub_targets)
                    norm_constant = 1.0 / len(clients_subset)
                    n, k = (len(observed_blocks)-1, len(clients_subset)-1)
                    weight = math.comb(n, k) if n >= k >= 0 else 0
                    weighted_loss = loss * norm_constant * weight
                    total_batch_loss += weighted_loss
                    loss_d[clients_subset] += loss.item()
                    predicted = outputs.argmax(1)
                    correct_d[clients_subset] += (predicted == sub_targets).float().sum().item()
                    if compute_f1:
                        tp_d[clients_subset] += ((predicted == 1) & (sub_targets == 1)).sum().item()
                        fp_d[clients_subset] += ((predicted == 1) & (sub_targets == 0)).sum().item()
                        fn_d[clients_subset] += ((predicted == 0) & (sub_targets == 1)).sum().item()
        
        total_batch_loss.backward()
        
        # [修改点 3] 使用递归函数更新参数
        step_all(optimizers)

    avg_loss = [l / max(1, num_batches) for l in loss_d.values()]
    acc = [100 * c / max(1, num_samples) for c in correct_d.values()]
    final_avg_loss = sum(avg_loss)/len(avg_loss) if avg_loss else 0.0
    final_avg_acc = sum(acc)/len(acc) if acc else 0.0
    if num_heads == 0: num_heads = 1
    metrics = {"train_loss": [final_avg_loss] * num_heads, "train_acc": [final_avg_acc] * num_heads}
    if compute_f1:
        f1_vals = [get_f1(tp_d[k], fp_d[k], fn_d[k]) for k in model.powerset]
        avg_f1 = sum(f1_vals)/len(f1_vals) if f1_vals else 0.0
        metrics["train_f1"] = [avg_f1] * num_heads
    return metrics