import torch
import torch.nn.functional as F
import itertools
from utils import time_decorator 
from methods.method_utils import get_f1

def get_subset_indices(num_clients):
    s = list(range(num_clients))
    subsets = []
    for r in range(1, num_clients + 1):
        subsets.extend(list(itertools.combinations(s, r)))
    return subsets

@time_decorator
def test_decoupled(dataloader, models, criterion, args, is_final=False, compute_f1=False, is_train_data=False):
    models_list = models if isinstance(models, list) else [models]
    for model in models_list: model.eval()
    
    num_models = len(models_list)
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    loss_l = [0.0] * num_models
    correct_l = [0.0] * num_models
    final_correct = 0.0
    
    # 预计算映射
    if args.method == 'local':
        model_client_map = [[i] for i in range(len(models_list))]
    elif args.method == 'combinatorial':
        model_client_map = get_subset_indices(args.num_clients)
    else:
        model_client_map = [[i] for i in range(len(models_list))]

    if compute_f1:
        tp_l, fp_l, fn_l = [0.0]*num_models, [0.0]*num_models, [0.0]*num_models
        f_tp, f_fp, f_fn = 0.0, 0.0, 0.0

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

            # 用于 Hard Voting 的票箱
            batch_votes = None 

            for i, model in enumerate(models_list):
                # 1. 数据隔离 (Strict Isolation)
                active_clients = model_client_map[i] if i < len(model_client_map) else range(len(masked_inputs_list))
                
                current_input_parts = []
                for client_idx, part in enumerate(masked_inputs_list):
                    if client_idx in active_clients:
                        current_input_parts.append(part)
                    else:
                        current_input_parts.append(torch.zeros_like(part))
                
                model_input = torch.cat(current_input_parts, dim=1)

                # 2. 推理
                outputs = model(model_input)
                
                loss_l[i] += criterion(outputs, targets).item()
                correct_l[i] += (outputs.argmax(1) == targets).float().sum().item()
                
                if compute_f1:
                    pred = outputs.argmax(1)
                    tp_l[i] += ((pred == 1) & (targets == 1)).sum().item()
                    fp_l[i] += ((pred == 1) & (targets == 0)).sum().item()
                    fn_l[i] += ((pred == 0) & (targets == 1)).sum().item()

                # 3. [核心打击] Combinatorial 使用硬投票 (Hard Voting)
                if is_final and args.method == 'combinatorial':
                    # 获取该模型的预测类别 (0 或 1)
                    preds = outputs.argmax(dim=1) # (Batch,)
                    
                    if batch_votes is None:
                        # 初始化票箱: (Batch, Num_Classes)
                        num_classes = outputs.size(1)
                        batch_votes = torch.zeros(outputs.size(0), num_classes, device=args.device)
                    
                    # 投一票: 在对应的类别位置 +1
                    # view(-1, 1) 是为了 scatter 的维度匹配
                    batch_votes.scatter_add_(1, preds.view(-1, 1), torch.ones_like(preds.view(-1, 1), dtype=torch.float))

            # 4. 结算 Combinatorial 的结果
            if is_final and args.method == 'combinatorial' and batch_votes is not None:
                # 谁票多谁赢 (Majority Vote)
                final_pred = batch_votes.argmax(1)
                
                final_correct += (final_pred == targets).float().sum().item()
                if compute_f1:
                    f_tp += ((final_pred == 1) & (targets == 1)).sum().item()
                    f_fp += ((final_pred == 1) & (targets == 0)).sum().item()
                    f_fn += ((final_pred == 0) & (targets == 1)).sum().item()

    data_type = "train" if is_train_data else "test"
    metrics = {}
    accuracy_list = [100 * c / max(1, num_samples) for c in correct_l]
    
    if is_final:
        if args.method == 'combinatorial':
            # Combinatorial 返回硬投票结果
            metrics[f"final_{data_type}_acc"] = 100 * final_correct / max(1, num_samples)
            if compute_f1: metrics[f"final_{data_type}_f1"] = get_f1(f_tp, f_fp, f_fn)
        else:
            # Local 返回平均值
            metrics[f"final_{data_type}_acc"] = sum(accuracy_list)/len(accuracy_list) if accuracy_list else 0.0
            if compute_f1: 
                f1s = [get_f1(tp, fp, fn) for tp, fp, fn in zip(tp_l, fp_l, fn_l)]
                metrics[f"final_{data_type}_f1"] = sum(f1s)/len(f1s) if f1s else 0.0
    else:
        metrics[f"{data_type}_loss"] = [l/num_batches for l in loss_l]
        metrics[f"{data_type}_acc"] = accuracy_list
        if compute_f1: 
            metrics[f"{data_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(tp_l, fp_l, fn_l)]

    return metrics

@time_decorator
def train_decoupled(dataloader, models, optimizers, criterion, args, compute_f1=False):
    models_list = models if isinstance(models, list) else [models]
    optimizers_list = optimizers if isinstance(optimizers, list) else [optimizers]
    for m in models_list: m.train()
    
    num_samples = len(dataloader.dataset)
    loss_l = [0.0]*len(models_list)
    correct_l = [0.0]*len(models_list)
    
    # 映射表
    if args.method == 'local':
        model_client_map = [[i] for i in range(len(models_list))]
    elif args.method == 'combinatorial':
        model_client_map = get_subset_indices(args.num_clients)
    else:
        model_client_map = [[i] for i in range(len(models_list))]

    if compute_f1:
        tp_l = [0.0] * len(models_list)
        fp_l = [0.0] * len(models_list)
        fn_l = [0.0] * len(models_list)

    for batch_num, batch in enumerate(dataloader):
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
        
        for i, (model, optimizer) in enumerate(zip(models_list, optimizers_list)):
            if isinstance(optimizer, list): optimizer = optimizer[0]
            
            optimizer.zero_grad()
            
            active_clients = model_client_map[i] if i < len(model_client_map) else range(len(masked_inputs_list))
            
            current_input_parts = []
            for client_idx, part in enumerate(masked_inputs_list):
                if client_idx in active_clients:
                    current_input_parts.append(part)
                else:
                    current_input_parts.append(torch.zeros_like(part))
            
            model_input = torch.cat(current_input_parts, dim=1)

            outputs = model(model_input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            loss_l[i] += loss.item()
            correct_l[i] += (outputs.argmax(1) == targets).float().sum().item()
            
            if compute_f1:
                pred = outputs.argmax(1)
                tp_l[i] += ((pred == 1) & (targets == 1)).sum().item()
                fp_l[i] += ((pred == 1) & (targets == 0)).sum().item()
                fn_l[i] += ((pred == 0) & (targets == 1)).sum().item()
        
        if (batch_num + 1) % 50 == 0:
             print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] Loss (Model 0): {loss_l[0]/50:.4f}")

    metrics = {
        "train_loss": [l/len(dataloader) for l in loss_l],
        "train_acc": [100*c/num_samples for c in correct_l]
    }
    
    if compute_f1:
        metrics["train_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(tp_l, fp_l, fn_l)]
        
    return metrics