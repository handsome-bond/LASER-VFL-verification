import torch
import torchvision.models as models_torch
import torch.nn as nn

from models.resnet18 import Resnet18LaserModel
from models.resnet18 import Resnet18DecoupledModel
from models.lstm import DecoupledModel as DecoupledLstm
from models.lstm import LaserModel as LaserLstm
from models.mlp import MLPDecoupledModel
from models.mlp import MLPLaserModel

models_dict = {
    "laser": {
        "resnet18": Resnet18LaserModel,
        "lstm": LaserLstm,
        "mlp": MLPLaserModel,
    },
    "decoupled": {
        "resnet18": Resnet18DecoupledModel,
        "lstm": DecoupledLstm,
        "mlp": MLPDecoupledModel,
    },
    "plug": {
        "resnet18": Resnet18DecoupledModel, # Reusing Decoupled structure for simplicity
        "lstm": DecoupledLstm,
        "mlp": MLPDecoupledModel,
    },
    "ensemble": {
        "resnet18": Resnet18DecoupledModel,
        "lstm": DecoupledLstm,
        "mlp": MLPDecoupledModel,
    },
}

def get_model(method_type, model_name, dataset, args, config):
    # [新增逻辑] 强制让 diabetes6 使用 ResNet18，即使 config 写错了
    if dataset == 'diabetes6':
        model_name = 'resnet18'
    
    try:
        Model = models_dict[method_type][model_name]
        
        # 特殊处理 LSTM (MIMIC)
        if model_name == "lstm":
            from .mimic_model_utils import init as init_mimic
            vocab_d = init_mimic(True, False, False, True, False, False)
            if method_type == "laser":
                 return [Model(dataset, args, vocab_d, config).to(args.device)]
            else:
                 return [Model(dataset, args, vocab_d, config, clients_in_model).to(args.device) for clients_in_model in args.blocks_in_tasks_t]
        
        # 标准处理 (MLP / ResNet)
        else:
            if method_type == "laser":
                return [Model(dataset, args.num_clients).to(args.device)]
            else:
                return [Model(dataset, args, clients_in_model).to(args.device) for clients_in_model in args.blocks_in_tasks_t]

    except KeyError:
        raise ValueError(f"Unknown model name ({model_name}) or method name ({method_type})")