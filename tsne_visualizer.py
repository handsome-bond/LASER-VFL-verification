import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.font_manager as fm

def configure_style():
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams['axes.unicode_minus'] = False
    cjk_fonts = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
    sys_fonts = set(f.name for f in fm.fontManager.ttflist)
    found = [f for f in cjk_fonts if f in sys_fonts]
    if found:
        plt.rcParams['font.sans-serif'] = found + plt.rcParams['font.sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams.update({'font.size': 14})

def plot_tsne(model, test_loader, device, method, task_name, p_miss):
    print(f"\n=======================================================")
    print(f"[t-SNE] 正在为 {method.upper()} 提取并精修隐空间特征...")
    model.eval()
    
    features_list = []
    
    # 1. 寻找最后一个线性层
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            
    if last_linear is None: return
        
    def hook_fn(module, input, output):
        feat = input[0].detach().cpu().numpy()
        features_list.append(feat)

    handle = last_linear.register_forward_hook(hook_fn)
    
    final_features = []
    final_labels = []
    
    # 2. 提取特征
    with torch.no_grad():
        for batch in test_loader:
            features_list.clear()
            *inputs, targets, mask = batch
            inputs = [t.to(device) for t in inputs]
            targets = targets.to(device)
            mask = mask.to(device)

            masked_inputs_list = []
            for k, inp in enumerate(inputs):
                if mask.dim() > 1:
                    m = mask[:, k].view([-1] + [1] * (inp.dim() - 1))
                    masked_inputs_list.append(inp * m)
                else:
                    masked_inputs_list.append(inp * mask[k])

            full_input = torch.cat(masked_inputs_list, dim=1)

            if method == 'laser':
                _ = model(full_input, training=False)
            else:
                _ = model(full_input)
                
            if len(features_list) > 0:
                final_features.append(features_list[0])
                final_labels.append(targets.cpu().numpy())

    handle.remove()
    
    if not final_features: return
        
    X_all = np.concatenate(final_features, axis=0)
    y_all = np.concatenate(final_labels, axis=0)
    
    # ==========================================================
    # 💎 学术级精修处理核心区 
    # ==========================================================
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    max_samples_per_class = 400 
    indices_to_keep = []
    for cls in np.unique(y_all):
        cls_indices = np.where(y_all == cls)[0]
        if len(cls_indices) > max_samples_per_class:
            np.random.seed(42)
            sampled_indices = np.random.choice(cls_indices, max_samples_per_class, replace=False)
            indices_to_keep.extend(sampled_indices)
        else:
            indices_to_keep.extend(cls_indices)
            
    X_subset = X_all[indices_to_keep]
    y_subset = y_all[indices_to_keep]
    
    print(f"[t-SNE] 优化完毕！采样后绘制 {len(y_subset)} 个数据点以保证视觉清晰度。")
    
    tsne = TSNE(n_components=2, perplexity=35, early_exaggeration=12.0, 
                random_state=42, n_iter=2000, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X_subset)
    
    # ==========================================================
    
    configure_style()
    plt.figure(figsize=(7, 6), dpi=300)
    
    # 【医疗配色修正】: 翠绿(0)代表无病健康，正红(1)代表患病风险
    palette = ["#2ECC71", "#E74C3C"] 
    
    scatter = sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=y_subset, palette=palette, 
        style=y_subset, s=35, alpha=0.85, edgecolor='white', linewidth=0.6
    )
    
    # --- [修改部分] 深度契合“糖尿病预测”任务的医学图表标注 ---
    method_name = "LASER-VFL" if method.lower() == "laser" else method.upper()
    
    # 标题点明是“糖尿病隐空间分布”
    plt.title(f'{method_name} 糖尿病预测隐空间分布 (特征缺失率={p_miss})', fontsize=16, pad=15, fontweight='bold')
    plt.xlabel('t-SNE 维度 1', fontsize=14)
    plt.ylabel('t-SNE 维度 2', fontsize=14)
    
    handles, labels = scatter.get_legend_handles_labels()
    
    # 按照 0(健康) 和 1(患病) 映射直观的医学标签
    cn_labels = ['无糖尿病 (Class 0)', '患糖尿病 (Class 1)']
    
    plt.legend(handles=handles, labels=cn_labels, 
               title='真实诊断标签', loc='upper right', frameon=True)
    # --------------------------------------
               
    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True, bottom=True)
    
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"TSNE_{task_name}_{method}_pmiss_{p_miss}_refined.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[t-SNE] 完美医学精修图已保存至: {save_path}")
    print(f"=======================================================\n")
    plt.close()