import torch
import torch.nn.functional as F
from utils import time_decorator
from methods.method_utils import get_f1
from methods.decoupled import train_decoupled as train_ensemble

@time_decorator
def test_ensemble(dataloader, models, criterion, args, is_final=False, compute_f1=False, is_train_data=False):
    models_list = models if isinstance(models, list) else [models]
    for m in models_list: m.eval()
    
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_sum, correct_sum = 0.0, 0.0
    if compute_f1: total_tp, total_fp, total_fn = 0.0, 0.0, 0.0

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

            # 初始化概率累加器
            num_classes = 2
            if hasattr(dataloader.dataset, 'classes'): num_classes = len(dataloader.dataset.classes)
            ensemble_probs = torch.zeros(targets.size(0), num_classes, device=args.device)

            for i, model in enumerate(models_list):
                # [核心修复] 严格数据隔离
                # Ensemble 的第 i 个模型只能看到第 i 个客户端的数据
                current_input_parts = []
                for client_idx, part in enumerate(masked_inputs_list):
                    if client_idx == i:
                        current_input_parts.append(part)
                    else:
                        current_input_parts.append(torch.zeros_like(part)) # 其他置零
                
                model_input = torch.cat(current_input_parts, dim=1)

                # 推理
                out = model(model_input)
                
                # 累加 Softmax 概率
                ensemble_probs += F.softmax(out, dim=1)

            # 取平均 (或直接 Argmax)
            final_probs = ensemble_probs / len(models_list)
            
            # 计算 Loss (近似)
            loss = F.nll_loss(torch.log(final_probs + 1e-10), targets)
            loss_sum += loss.item()
            
            predicted = final_probs.argmax(1)
            correct_sum += (predicted == targets).float().sum().item()
            
            if compute_f1:
                total_tp += ((predicted == 1) & (targets == 1)).sum().item()
                total_fp += ((predicted == 1) & (targets == 0)).sum().item()
                total_fn += ((predicted == 0) & (targets == 1)).sum().item()

    acc = 100 * correct_sum / max(1, num_samples)
    data_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_type}_acc"] = acc
        if compute_f1: metrics[f"final_{data_type}_f1"] = get_f1(total_tp, total_fp, total_fn)
    else:
        # 为了兼容 utils.py 的绘图格式，返回列表
        metrics[f"{data_type}_loss"] = [loss_sum/num_batches] * len(models_list)
        metrics[f"{data_type}_acc"] = [acc] * len(models_list)
        if compute_f1: metrics[f"{data_type}_f1"] = [get_f1(total_tp, total_fp, total_fn)] * len(models_list)
            
    return metrics