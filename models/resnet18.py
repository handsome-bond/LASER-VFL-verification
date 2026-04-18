import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import BasicBlock

from models.model_utils import BaseLaserModel, BaseDecoupledModel, FusionModel, task_to_hyperparameters
from data.cifar_partitions import CIFAR_PARTITIONS

# =============================================================================
# 1. 切分配置
# =============================================================================
DIABETES6_PARTITIONS = {
    4: {
        0: ((0, 112), (0, 112)),      # 左上
        1: ((0, 112), (112, 224)),    # 右上
        2: ((112, 224), (0, 112)),    # 左下
        3: ((112, 224), (112, 224)),  # 右下
    },
    2: {
        0: ((0, 224), (0, 112)),      # 左半
        1: ((0, 224), (112, 224)),    # 右半
    }
}

def get_idx_to_partition_map(dataset: str, num_clients: int) -> dict:
    if dataset == "diabetes6":
        if num_clients not in DIABETES6_PARTITIONS:
             H, W = 224, 224
             return {i: ((0, H), (0, W)) for i in range(num_clients)}
        return DIABETES6_PARTITIONS[num_clients]

    if dataset not in ("cifar10", "cifar100"):
        pass
        
    if num_clients not in CIFAR_PARTITIONS:
        raise NotImplementedError(f"No partition map for num_clients={num_clients} in {dataset}")
    return CIFAR_PARTITIONS[num_clients]


# =============================================================================
# 2. 核心辅助函数
# =============================================================================
def slice_image_block(x, coords):
    (r_start, r_end), (c_start, c_end) = coords
    return x[:, :, r_start:r_end, c_start:c_end]


# =============================================================================
# 3. 特征提取器
# =============================================================================
class Resnet18FeatureExtractor(nn.Module):
    def __init__(self, cut_dim, dataset):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(weights=None)
        self.dataset = dataset

        if dataset == 'cifar10' or dataset == 'cifar100':
            self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet18.maxpool = nn.Identity()

        self.resnet18.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, cut_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = self.projection(x)
        return x


# =============================================================================
# 4. 模型定义
# =============================================================================
class Resnet18LaserModel(BaseLaserModel):
    def __init__(self, dataset, num_clients, cut_dim=None):
        try:
            num_classes, _, default_cut_dim, _ = task_to_hyperparameters(dataset)
            if cut_dim is None: cut_dim = default_cut_dim
        except:
            num_classes = 2 if 'diabetes' in dataset else 10
            if cut_dim is None: cut_dim = 64

        self.map_idx_to_partition = get_idx_to_partition_map(dataset, num_clients)
        feature_extractors = nn.ModuleList([Resnet18FeatureExtractor(cut_dim, dataset) for _ in range(num_clients)])
        
        # [核心修复] 将 "uniform" 改为 "sum"，确保特征列表被正确相加合并
        fusion_models = nn.ModuleList([FusionModel(cut_dim, num_classes, "sum", num_clients)])
        super().__init__(feature_extractors, fusion_models, num_clients)

    def get_block(self, x, i):
        if isinstance(x, (list, tuple)):
            x = x[0]
        
        if x.shape[1] == 3 * self.num_clients:
            start_c = i * 3
            end_c = (i + 1) * 3
            client_x = x[:, start_c:end_c, :, :]
        else:
            client_x = x
            
        return slice_image_block(client_x, self.map_idx_to_partition[i])


class Resnet18DecoupledModel(BaseDecoupledModel):
    def __init__(self, dataset, args, clients_in_model=None, cut_dim=None):
        num_clients = args.num_clients
        try:
            num_classes, _, default_cut_dim, _ = task_to_hyperparameters(dataset)
            if cut_dim is None: cut_dim = default_cut_dim
        except:
            num_classes = 2 if 'diabetes' in dataset else 10
            if cut_dim is None: cut_dim = 64
            
        self.map_idx_to_partition = get_idx_to_partition_map(dataset, num_clients)
        self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        feature_extractors = nn.ModuleList([Resnet18FeatureExtractor(cut_dim, dataset) for _ in self.clients_in_model])
        
        # [核心修复] 同上，将 "uniform" 改为 "sum"
        fusion_model = FusionModel(cut_dim, num_classes, "sum", args.num_clients)
        super().__init__(feature_extractors, fusion_model, num_clients)

    def get_block(self, x, i):
        if isinstance(x, (list, tuple)):
            x = x[0]
        
        if x.shape[1] == 3 * self.num_clients:
            start_c = i * 3
            end_c = (i + 1) * 3
            client_x = x[:, start_c:end_c, :, :]
        else:
            client_x = x
            
        return slice_image_block(client_x, self.map_idx_to_partition[i])