from dataclasses import dataclass
import torch

@dataclass
class KGraph:
    num_nodes: int
    # For now, one CSR per property
    # Example: edges for property p_idx: 
    #   neighbors_p[p_idx][offsets_p[p_idx][v] : offsets_p[p_idx][v+1]]
    offsets_p: list[torch.Tensor]  # list of int32 tensors of shape [num_nodes+1]
    neighbors_p: list[torch.Tensor]  # list of int32 tensors, concatenated neighbors
    # Types: node -> bitmask or multi-hot vector over classes
    node_types: torch.Tensor  # [num_nodes, num_classes] or [num_nodes] with class IDs
    # Class hierarchy / other metadata can be added later
