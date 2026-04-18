import torch.nn as nn
import numpy as np
from models.model_utils import (BaseLaserModel, BaseDecoupledModel,
                                FusionModel, task_to_hyperparameters)


class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_size, cut_dim, hidden_dim):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cut_dim),
        )

    def forward(self, x):
        return self.hidden_layers(x)


class MLPLaserModel(BaseLaserModel):
    # [核心修改] 增加 cut_dim 参数
    def __init__(self, dataset, num_clients, cut_dim=None):
        num_classes, input_size, default_cut_dim, hidden_dim = task_to_hyperparameters(dataset)
        
        # 优先使用传入的 cut_dim
        self.cut_dim = cut_dim if cut_dim is not None else default_cut_dim
        
        # 使用 numpy 精确切分
        self.split_sizes = [len(s) for s in np.array_split(np.arange(input_size), num_clients)]
        
        feature_extractors = nn.ModuleList([
            MLPFeatureExtractor(self.split_sizes[i], self.cut_dim, hidden_dim) 
            for i in range(num_clients)
        ])
        
        fusion_models = nn.ModuleList([FusionModel(self.cut_dim, num_classes) for _ in range(num_clients)])
        super().__init__(feature_extractors, fusion_models, num_clients)
    
    def get_block(self, x, i):
        input_tensor = x[0] if isinstance(x, (list, tuple)) else x
        start = sum(self.split_sizes[:i])
        end = start + self.split_sizes[i]
        return input_tensor[:, start:end]


class MLPDecoupledModel(BaseDecoupledModel):
    # [核心修改] 增加 cut_dim 参数
    def __init__(self, dataset, args, clients_in_model=None, aggregation="mean", cut_dim=None): 
        num_clients = args.num_clients
        self.num_clients = num_clients
        num_classes, input_size, default_cut_dim, hidden_dim = task_to_hyperparameters(dataset)
        
        self.cut_dim = cut_dim if cut_dim is not None else default_cut_dim

        self.split_sizes = [len(s) for s in np.array_split(np.arange(input_size), num_clients)]
        self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        feature_extractors = nn.ModuleList([
            MLPFeatureExtractor(self.split_sizes[i], self.cut_dim, hidden_dim) 
            for i in self.clients_in_model
        ])
        
        fusion_model = FusionModel(self.cut_dim, num_classes, aggregation, args.num_clients)
        super().__init__(feature_extractors, fusion_model, num_clients)

    def get_block(self, x, i):
        input_tensor = x[0] if isinstance(x, (list, tuple)) else x
        start = sum(self.split_sizes[:i])
        end = start + self.split_sizes[i]
        return input_tensor[:, start:end]