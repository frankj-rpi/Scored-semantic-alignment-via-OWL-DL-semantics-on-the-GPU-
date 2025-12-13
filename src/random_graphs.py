from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

from .graph import KGraph


@dataclass
class PathPattern:
    """
    Simple specification of a typed path pattern:

      steps: list of (prop_idx, class_idx)
             i.e., follow prop_idx, land on a node of class_idx

      normalize: whether we *intend* to normalize this path's score
                 later by len(steps) (the engine doesn't enforce this yet;
                 it just tells the compiler what to do).
    """
    steps: List[Tuple[int, int]]
    normalize: bool = True


def make_rng(seed: Optional[int] = None) -> torch.Generator:
    """
    Helper: create a CPU RNG with an optional seed.
    """
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    return g


def generate_random_kgraph(
    num_nodes: int,
    num_props: int,
    num_classes: int,
    avg_degree_per_prop: float = 2.0,
    seed: Optional[int] = None,
) -> KGraph:
    """
    Generate a random KGraph with:

      - num_nodes nodes
      - num_props properties (p0..p_{num_props-1})
      - num_classes classes (C0..C_{num_classes-1})
      - approx avg_degree_per_prop outgoing edges *per property* per node

    Structure:
      - For each property p, we create about num_nodes * avg_degree_per_prop edges
        with sources and targets chosen uniformly at random.
      - Each node is assigned exactly one class (one-hot) by default.

    All outputs are on CPU (you can move them to CUDA in the evaluator).
    """
    rng = make_rng(seed)

    # --- Node types: one-hot assignment of a single class per node ---
    node_classes = torch.randint(
        low=0,
        high=num_classes,
        size=(num_nodes,),
        generator=rng,
    )
    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    node_types[torch.arange(num_nodes), node_classes] = 1.0

    offsets_p: List[torch.Tensor] = []
    neighbors_p: List[torch.Tensor] = []

    nodes = torch.arange(num_nodes, dtype=torch.long)

    for _p in range(num_props):
        # Number of edges for this property
        num_edges = int(num_nodes * avg_degree_per_prop)

        if num_edges == 0:
            # Degenerate case: no edges at all
            offsets = torch.zeros(num_nodes + 1, dtype=torch.int32)
            neighbors = torch.empty(0, dtype=torch.int32)
            offsets_p.append(offsets)
            neighbors_p.append(neighbors)
            continue

        # Random src, dst for each edge
        src = torch.randint(
            low=0,
            high=num_nodes,
            size=(num_edges,),
            generator=rng,
        )
        dst = torch.randint(
            low=0,
            high=num_nodes,
            size=(num_edges,),
            generator=rng,
        )

        # Sort edges by source to build CSR
        sorted_src, perm = torch.sort(src)
        sorted_dst = dst[perm]

        # deg[v] = number of outgoing edges from v
        deg = torch.bincount(sorted_src, minlength=num_nodes)  # [num_nodes]

        offsets = torch.zeros(num_nodes + 1, dtype=torch.int32)
        offsets[1:] = deg.cumsum(0).to(torch.int32)
        neighbors = sorted_dst.to(torch.int32)

        offsets_p.append(offsets)
        neighbors_p.append(neighbors)

    return KGraph(
        num_nodes=num_nodes,
        offsets_p=offsets_p,
        neighbors_p=neighbors_p,
        node_types=node_types,
    )


def generate_random_path_pattern(
    num_steps: int,
    num_props: int,
    num_classes: int,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> PathPattern:
    """
    Generate a random typed path pattern:

      steps: [(p0, Ck0), (p1, Ck1), ..., (p_{L-1}, Ck_{L-1})]

    where:
      - each prop_idx ∈ [0, num_props)
      - each class_idx ∈ [0, num_classes)

    This does *not* compile to a DAG yet; it just produces a spec that
    you can later feed into a compiler that constructs PATH_STEP chains.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be >= 1 for a non-empty path.")

    rng = make_rng(seed)

    prop_indices = torch.randint(
        low=0,
        high=num_props,
        size=(num_steps,),
        generator=rng,
    )
    class_indices = torch.randint(
        low=0,
        high=num_classes,
        size=(num_steps,),
        generator=rng,
    )

    steps: List[Tuple[int, int]] = list(
        zip(prop_indices.tolist(), class_indices.tolist())
    )

    return PathPattern(steps=steps, normalize=normalize)
