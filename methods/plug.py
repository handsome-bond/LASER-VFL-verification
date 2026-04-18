import torch
from utils import time_decorator
from methods.method_utils import get_f1

@time_decorator
def test_plug(dataloader, models, criterion, args, is_final=False, compute_f1=False, is_train_data=False):
    models_list = models if isinstance(models, list) else [models]
    for m in models_list: m.eval()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    loss_l, correct_l = [0.0]*len(models_list), [0.0]*len(models_list)
    final_correct = 0.0
    
    # [修复] 初始化 F1
    if compute_f1: 
        f_tp, f_fp, f_fn = 0.0, 0.0, 0.0
        # 针对非 final 阶段的每个模型统计
        tp_l, fp_l, fn_l = [0.0]*len(models_list), [0.0]*len(models_list), [0.0]*len(models_list)

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

            for i, model in enumerate(models_list):
                try:
                    out = model(full_input, mask, args.p_drop)
                except:
                    out = model(masked_inputs_list, mask, args.p_drop)

                loss_l[i] += criterion(out, targets).item()
                pred = out.argmax(1)
                correct_l[i] += (pred == targets).float().sum().item()
                
                # F1 统计
                if compute_f1:
                    tp_l[i] += ((pred == 1) & (targets == 1)).sum().item()
                    fp_l[i] += ((pred == 1) & (targets == 0)).sum().item()
                    fn_l[i] += ((pred == 0) & (targets == 1)).sum().item()
                
                if is_final and i == 0:
                    final_correct += (pred == targets).float().sum().item()
                    if compute_f1:
                        f_tp += ((pred == 1) & (targets == 1)).sum().item()
                        f_fp += ((pred == 1) & (targets == 0)).sum().item()
                        f_fn += ((pred == 0) & (targets == 1)).sum().item()

    data_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_type}_acc"] = 100 * final_correct / max(1, num_samples)
        if compute_f1: metrics[f"final_{data_type}_f1"] = get_f1(f_tp, f_fp, f_fn)
    else:
        metrics[f"{data_type}_loss"] = [l/num_batches for l in loss_l]
        metrics[f"{data_type}_acc"] = [100*c/num_samples for c in correct_l]
        # [关键修复] 返回 test_f1
        if compute_f1:
             metrics[f"{data_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(tp_l, fp_l, fn_l)]
             
    return metrics

@time_decorator
def train_plug(dataloader, models, optimizers, criterion, args, compute_f1=False):
    models_list = models if isinstance(models, list) else [models]
    optimizers_list = optimizers if isinstance(optimizers, list) else [optimizers]
    for m in models_list: m.train()
    
    num_samples = len(dataloader.dataset)
    correct_sum, loss_sum = 0.0, 0.0
    
    # [修复] 初始化 F1
    if compute_f1: tp, fp, fn = 0.0, 0.0, 0.0
    
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

        model = models_list[0]
        optimizer = optimizers_list[0]
        if isinstance(optimizer, list): optimizer = optimizer[0]

        optimizer.zero_grad()
        try:
            out = model(full_input, mask, args.p_drop)
        except:
            out = model(masked_inputs_list, mask, args.p_drop)
            
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        pred = out.argmax(1)
        correct_sum += (pred == targets).float().sum().item()
        
        # [修复] 计算 F1
        if compute_f1:
            tp += ((pred == 1) & (targets == 1)).sum().item()
            fp += ((pred == 1) & (targets == 0)).sum().item()
            fn += ((pred == 0) & (targets == 1)).sum().item()

    metrics = {
        "train_acc": [100*correct_sum/num_samples], 
        "train_loss": [loss_sum/len(dataloader)]
    }
    
    # [关键修复] 返回 train_f1
    if compute_f1:
        metrics["train_f1"] = [get_f1(tp, fp, fn)]
        
    return metrics