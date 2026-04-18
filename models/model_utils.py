import random
import torch
import torch.nn as nn
from utils import powerset_except_empty

class FusionModel(nn.Module):
    def __init__(self, cut_dim, num_classes, aggregation="mean", num_clients=None):
        super().__init__()
        self.aggregation = aggregation
        if aggregation == 'conc':
            assert num_clients is not None
        fusion_input_dim = cut_dim * num_clients if aggregation == "conc" else cut_dim
        self.classifier = nn.Linear(fusion_input_dim, num_classes)

    def forward(self, x):
        # x is a list of tensors
        if self.aggregation == 'sum':
            x = torch.stack(x).sum(dim=0)
        elif self.aggregation == 'mean':
            x = torch.stack(x).mean(dim=0)
        elif self.aggregation == 'conc':
            x = torch.cat(x, dim=1)
        pooled_view = self.classifier(x)
        return pooled_view

class BaseLaserModel(nn.Module):
    def __init__(self, feature_extractors, fusion_models, num_clients):
        super().__init__()
        self.num_clients = num_clients
        self.powerset = powerset_except_empty(num_clients)
        self.feature_extractors = feature_extractors
        self.fusion_models = fusion_models

    def forward(self, x, training=True, observed_blocks=None):
        if observed_blocks is None: # NOTE this is be the case e.g. for test data
            observed_blocks = list(range(self.num_clients))

        embeddings = {}
        for i in observed_blocks:
            local_input = self.get_block(x, i)
            embeddings[i] = self.feature_extractors[i](local_input)

        if training:
            outputs = []
            for i, fusion_model in enumerate(self.fusion_models):
                if i not in observed_blocks:
                    continue
                sets_considered_by_head = [clients_l for clients_l in self.powerset if i in clients_l and set(clients_l).issubset(set(observed_blocks))]
                head_output = {}
                for num_clients_in_agg in range(1, len(observed_blocks) + 1):
                    set_to_sample = [client_set for client_set in sets_considered_by_head if len(client_set) == num_clients_in_agg]
                    if not set_to_sample: continue # Handle edge cases
                    [sample] = random.sample(set_to_sample, 1)
                    head_output[sample] = fusion_model([embeddings[j] for j in sample])
                outputs.append(head_output)
        else:
            outputs = [{clients_l: fusion_model([embeddings[j] for j in clients_l]) for clients_l in self.powerset if i in clients_l} for i, fusion_model in enumerate(self.fusion_models)]
        return outputs
    
    def get_block(self, x, i):
        raise NotImplementedError("Subclasses must override this method.")

class BaseDecoupledModel(nn.Module):
    def __init__(self, feature_extractors, fusion_model, num_clients):
        super().__init__()
        self.num_clients = num_clients
        self.feature_extractors = feature_extractors
        self.fusion_model = fusion_model

    def forward(self, x, plug_mask=None, p_drop=0.0):
        # [Fix] Rewritten to handle 2D masks (Batch processing) correctly
        embeddings = []
        
        # Calculate mask with dropout if needed
        final_mask = None
        if plug_mask is not None:
            final_mask = drop_mask(plug_mask, p_drop)

        for i, j in enumerate(self.clients_in_model):
            # 1. Compute embedding for this client
            client_input = self.get_block(x, j)
            out = self.feature_extractors[i](client_input)
            
            # 2. Apply Masking (Multiply by 0 if missing)
            if final_mask is not None:
                if final_mask.dim() > 1:
                    # 2D Case: (Batch, Clients)
                    # Extract column for client i -> Shape (Batch, 1)
                    # We assume plug_mask aligns with feature_extractors indices
                    client_mask = final_mask[:, i].view(-1, 1)
                    out = out * client_mask
                else:
                    # 1D Case: (Clients,)
                    if not final_mask[i]:
                        out = torch.zeros_like(out)
            
            embeddings.append(out)

        # 3. Fuse embeddings
        return self.fusion_model(embeddings)
    
    def _get_dummy_output(self, feature_extractor, input_tensor):
        # Deprecated: The multiplication logic handles this naturally
        with torch.no_grad():
            return feature_extractor(input_tensor)

    def get_block(self, x, i):
        raise NotImplementedError("Subclasses must override this method.")

def drop_mask(plug_mask: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Return a new mask, dropping elements with probability p_drop.
    The last element (Active Party) is never dropped."""
    if p_drop == 0.0:
        return plug_mask

    keep = torch.rand_like(plug_mask, dtype=torch.float32) >= p_drop
    
    # [Fix] Correctly handle the Active Party index for 2D Batches
    if plug_mask.dim() > 1:
        # (Batch, Clients): Keep last column
        keep[:, -1] = True
    else:
        # (Clients,): Keep last element
        keep[-1] = True
        
    return plug_mask & keep

def task_to_hyperparameters(dataset):
    # Returns: num_classes, input_size, cut_dim, hidden_dim
    
    # Existing Datasets
    if dataset == 'diabetes1':
        return 2, 3, 16, 32
    elif dataset == 'diabetes2':
        return 2, 4, 16, 32
    elif dataset == 'diabetes3':
        return 2, 5, 16, 32
    elif dataset == 'diabetes4':
        return 2, 3, 16, 32
        
    # --- Newly Added Datasets ---
    elif dataset == 'diabetes5':
        # NHANES Data: 109 Features (A-DE) -> Important for MLP split
        return 2, 104, 10, 256
        
    elif dataset == 'diabetes6':
        # IDRiD Image Data: 224*224*3
        return 2, 150528, 10, 256

    else:
        raise ValueError(f"Unexpected dataset {dataset}")