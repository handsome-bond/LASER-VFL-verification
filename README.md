# LASER-VFL: 面向缺失特征场景的垂直联邦学习框架

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-Supported-FFBE00.svg)](https://wandb.ai/)

本项目为《面向缺失特征场景的LASER-VFL模型在糖尿病风险预测中的应用研究》的官方代码实现。

针对垂直联邦学习（VFL）在实际医疗场景（如糖尿病风险预测）中普遍存在的**“特征缺失”**与**“数据孤岛”**难题，本项目实现了 **LASER-VFL** 框架。该框架打破了传统分离神经网络（SplitNN）对数据视图完整性的强依赖，能在训练和推理阶段平稳应对任意客户端或特征块的缺失情况。

## ✨ 核心特性与贡献

- **免受“维度灾难”困扰**：采用独创的**参数共享机制**，使不同特征子集的预测器协同共用底层表示网络，完美化解了因特征组合爆炸带来的计算量和参数量呈指数级增长的问题。
- **高效的任务采样策略**：在训练阶段通过随机任务采样，将计算复杂度降低至亚指数或线性级别，极大缩减了训练耗时（比穷举组合法快约 27.3%）。
- **零额外通信开销**：保持与标准垂直联邦学习（SVFL）一致的通信负载，无需复杂的跨客户端额外信息传输，非常适合带宽受限的医疗物联网边缘计算环境。
- **多模态与异构数据支持**：全面兼容低维结构化临床表格数据（如 Mendeley Data, NHANES）与高维非结构化图像数据（如 IDRiD 眼底图像）。
- **内置隐空间特征可视化**：提供 t-SNE 特征降维可视化功能，直观展示模型在极端数据缺失下的表征解耦与抗干扰能力。

## 🚀 支持的方法与基线 (Baselines)

本项目内置了多种联邦学习架构以供对比消融实验：
- `local`: 本地单视图基准（无联邦协同）
- `svfl`: 标准垂直联邦学习（Standard VFL，基于分离神经网络）
- `ensemble`: 集成投票策略（无交叉训练的后期融合）
- `combinatorial`: 穷举组合方法（独立训练所有可能的特征子集）
- `plug`: 借鉴 PlugVFL 的可插拔零填充架构
- `laser`: **本项目提出的 LASER-VFL 核心架构**

## 🛠️ 环境依赖与安装

推荐使用 Python 3.8 及以上版本。请在环境中安装以下依赖包：

```bash
pip install torch torchvision wandb scikit-learn matplotlib
```

*注：本项目默认使用 [Weights & Biases (WandB)](https://wandb.ai/) 进行实验指标追踪与可视化，建议提前注册并登录 WandB 账号 (`wandb login`)。*

## 💻 快速开始

可以通过命令行传递参数来直接运行 `main.py`。

### 1. 运行标准 LASER-VFL 训练与测试
模拟训练缺失率为 30%，测试缺失率为 50% 的极端场景：

```bash
python main.py \
    --task_name diabetes6 \
    --method laser \
    --p_miss_train 0.3 \
    --p_miss 0.5 \
    --num_clients 4 \
    --cuda_id 0
```

### 2. 运行对比基线并生成 t-SNE 可视化
运行标准 SVFL 并输出隐空间特征的可视化图表：

```bash
python main.py \
    --task_name diabetes6 \
    --method svfl \
    --p_miss_train 0.0 \
    --p_miss 0.5 \
    --tsne
```

### 3. 禁用 WandB 本地运行调试
如果您只想在本地测试代码流而不上传日志到云端，可以添加 `--no_wandb` 标志：

```bash
python main.py --task_name mimic4 --method laser --no_wandb
```

## ⚙️ 核心命令行参数说明

| 参数名 | 类型 | 默认值 | 描述说明 |
| :--- | :---: | :---: | :--- |
| `--task_name` | `str` | **必填** | 任务或数据集名称 (如 `diabetes5`, `diabetes6`, `mimic4`, `credit`) |
| `--method` | `str` | **必填** | 联邦学习算法选型 (`local`, `svfl`, `ensemble`, `combinatorial`, `plug`, `laser`) |
| `--num_clients` | `int` | `4` | 参与垂直联邦学习的客户端/机构数量 |
| `--p_miss_train` | `float` | `0.0` | 训练阶段的特征缺失率 (0.0 ~ 1.0) |
| `--p_miss` | `float` | `None` | 测试阶段的特征缺失率 (若指定，将覆盖默认的敏感度消融测试列表) |
| `--tsne` | `flag` | `False` | 训练结束后是否触发隐空间特征的 t-SNE 降维可视化 |
| `--no_wandb` | `flag` | `False` | 传入此标志以禁用 Weights & Biases 实验记录 |
| `--cuda_id` | `int` | `0` | 指定运行的 GPU 设备号 (`cuda:0`, `cuda:1` 等) |
| `--cut_dim` | `int` | `None` | 通信维度控制 (用于调节通信开销与性能的平衡) |

## 📊 实验结果亮点

根据研究论文结果，LASER-VFL 在多项异构数据集中表现卓越：
- **极端缺失抗性**：在测试数据丢失达 50% 的极端工况下，LASER 仍能保持稳健的预测性能，远超标准 SVFL（后者往往会面临特征维度不匹配导致的崩溃或大幅降维）。
- **高维图像优势**：在 IDRiD 糖尿病视网膜眼底图像任务中，即使面临高维缺失，其准确率仍比前沿基准方法高出 4.9%，显现出极强的特征互补挖掘能力。

## 📝 作者与版权信息

* **作者**: handsome-bond
* **许可证**: MIT License