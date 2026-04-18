import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import matplotlib as mpl

# --- 风格设置 (保持一致) ---
try:
    import scienceplots
    plt.style.use(['science', 'ieee', 'no-latex'])
except ImportError:
    plt.style.use('seaborn-v0_8-paper')

mpl.rcParams['axes.unicode_minus'] = False
from sys import platform
if platform == "win32":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Times New Roman']
elif platform == "darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

def visualize_tsne(models, dataloader, args, title_suffix=""):
    """
    运行 t-SNE 可视化
    models: 模型列表
    dataloader: 测试集加载器
    args: 参数
    title_suffix: 标题后缀 (如 "LASER" 或 "Local")
    """
    model = models[0]
    model.eval()
    device = args.device

    # 1. 注册 Hook：截获最后一个全连接层之前的输入 (即特征)
    features_list = []
    
    def hook_fn(module, input, output):
        # input[0] 是进入该层的张量，即我们需要的 Feature
        # detach() 去掉梯度，cpu() 转到 CPU
        features_list.append(input[0].detach().cpu())

    # 自动寻找最后一个 Linear 层 (通常是分类头)
    target_layer = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            target_layer = m
    
    if target_layer is None:
        print("[Error] 无法找到 Linear 层用于提取特征，跳过 t-SNE。")
        return

    # 注册钩子
    handle = target_layer.register_forward_hook(hook_fn)

    # 2. 运行推理 (收集特征)
    labels_list = []
    
    print(f"[{title_suffix}] 正在提取特征以进行 t-SNE 可视化...")
    
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs = [x.to(device) for x in inputs]
            mask = mask.to(device)
            
            # 简单的 Forward，Hook 会自动把特征存入 features_list
            # 注意：这里我们只做推理，不需要计算 Loss
            # 对于 LASER，需要传入 observed_blocks
            if args.method == 'laser':
                if mask.dim() > 1:
                    observed_blocks = torch.nonzero(mask.sum(dim=0) > 0).reshape(-1).tolist()
                else:
                    observed_blocks = torch.nonzero(mask).reshape(-1).tolist()
                
                if not observed_blocks: continue
                model(inputs, training=False, observed_blocks=observed_blocks)
            else:
                # 其他方法 (Local, Ensemble 等)
                model(inputs)
            
            # 收集标签 (只收集一次，因为 LASER 可能会有多个 head 输出，但 hook 每次 forward 都会触发)
            # 这里的逻辑稍微复杂：由于 LASER 可能在一个 batch 里多次调用 forward (针对不同 head)，
            # 我们只需要保证 feature 和 label 数量对齐。
            # 简化策略：我们只取最后一次 forward 的结果，或者展平。
            
            # 为了简单起见，我们假设 batch_size 对齐。
            # 实际上，features_list 的长度可能会多于 batch 数 (因为 LASER 内部可能有循环)
            # 但 targets 是固定的。
            # **修正策略**：为了避免 LASER 内部多次调用导致错位，我们直接用最后一层输出的 shape[0] 来对齐
            
            current_feat_batch_size = features_list[-1].shape[0]
            # 复制对应数量的标签 (通常 batch size 是一样的，这里是为了防止 LASER 内部机制导致的差异)
            labels_list.append(targets[:current_feat_batch_size].cpu())

    # 移除钩子
    handle.remove()

    # 3. 数据整理
    # 将列表拼接成大张量
    try:
        X = torch.cat(features_list, dim=0).numpy()
        y = torch.cat(labels_list, dim=0).numpy()
    except Exception as e:
        print(f"[Warning] 特征拼接失败 (可能是 LASER 多头导致形状不一): {e}")
        return

    # 为了画图快一点，如果数据太多，随机采样 2000 个点
    if len(X) > 2000:
        indices = np.random.choice(len(X), 2000, replace=False)
        X = X[indices]
        y = y[indices]

    # 4. 运行 t-SNE
    print(f"[{title_suffix}] 正在计算 t-SNE (样本数: {len(X)})...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # 5. 绘图
    plt.figure(figsize=(6, 5), dpi=300)
    
    # 定义颜色：0(Negative/Healthy) -> 蓝色, 1(Positive/Diabetes) -> 红色
    colors = ['#005EB8' if label == 0 else '#d62728' for label in y]
    
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, alpha=0.6, s=10)
    
    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#005EB8', label='Negative (Healthy)', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label='Positive (Diabetes)', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f'Feature Space Visualization ({title_suffix})')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()

    # 保存
    save_dir = Path("./results/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"tSNE_{args.task_name}_{args.method}_{args.p_miss_train}.png"
    plt.savefig(save_dir / filename)
    plt.savefig(save_dir / filename.replace(".png", ".pdf"))
    
    print(f"[Success] t-SNE 图已保存至: {save_dir / filename}")