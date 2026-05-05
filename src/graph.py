from dataclasses import dataclass, field
from typing import Optional
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
    # Datatype support for lifted literal nodes.
    literal_datatype_idx: Optional[torch.Tensor] = None  # [num_nodes], -1 for non-literals / untyped
    literal_numeric_value: Optional[torch.Tensor] = None  # [num_nodes], NaN where non-numeric
    reflexive_prop_mask: Optional[torch.Tensor] = None  # [num_props], 1 where property is reflexive
    transitive_prop_families: Optional[list[torch.Tensor]] = None  # per prop, property indices participating in transitive closure
    src_index_p: Optional[list[torch.Tensor]] = None
    dst_index_p: Optional[list[torch.Tensor]] = None
    segment_layout_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = field(default_factory=dict)
    transitive_prop_family_indices: Optional[list[tuple[int, ...]]] = None


def prepare_kgraph_for_device(
    graph: KGraph,
    device: str = "cpu",
) -> KGraph:
    target_device = torch.device(device)

    offsets_p = [off.to(target_device) for off in graph.offsets_p]
    neighbors_p = [neigh.to(target_device) for neigh in graph.neighbors_p]
    node_types = graph.node_types.to(target_device)
    literal_datatype_idx = (
        None if graph.literal_datatype_idx is None else graph.literal_datatype_idx.to(target_device)
    )
    literal_numeric_value = (
        None if graph.literal_numeric_value is None else graph.literal_numeric_value.to(target_device)
    )
    reflexive_prop_mask = (
        None if graph.reflexive_prop_mask is None else graph.reflexive_prop_mask.to(target_device)
    )
    transitive_prop_families = (
        None
        if graph.transitive_prop_families is None
        else [family.to(target_device) for family in graph.transitive_prop_families]
    )
    transitive_prop_family_indices = (
        None
        if transitive_prop_families is None
        else [
            tuple(int(idx) for idx in family.detach().cpu().tolist()) if family.numel() > 0 else tuple()
            for family in transitive_prop_families
        ]
    )

    nodes = torch.arange(graph.num_nodes, device=target_device, dtype=torch.long)
    src_index_p: list[torch.Tensor] = []
    dst_index_p: list[torch.Tensor] = []
    for offsets, neighbors in zip(offsets_p, neighbors_p):
        deg = offsets[1:] - offsets[:-1]
        src_index = torch.repeat_interleave(nodes, deg)
        src_index_p.append(src_index)
        dst_index_p.append(neighbors.long())

    return KGraph(
        num_nodes=graph.num_nodes,
        offsets_p=offsets_p,
        neighbors_p=neighbors_p,
        node_types=node_types,
        literal_datatype_idx=literal_datatype_idx,
        literal_numeric_value=literal_numeric_value,
        reflexive_prop_mask=reflexive_prop_mask,
        transitive_prop_families=transitive_prop_families,
        src_index_p=src_index_p,
        dst_index_p=dst_index_p,
        segment_layout_cache={},
        transitive_prop_family_indices=transitive_prop_family_indices,
    )
