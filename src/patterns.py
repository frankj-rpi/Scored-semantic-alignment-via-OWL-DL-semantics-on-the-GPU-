from __future__ import annotations

from typing import List

from .constraints import ConstraintNode, ConstraintType, ConstraintDAG
from .random_graphs import PathPattern


def compile_path_pattern_to_dag(pattern: PathPattern) -> ConstraintDAG:
    """
    Compile a typed path pattern into a ConstraintDAG built from PATH_STEP nodes.

    Pattern:
      steps = [(p0, Ck0), (p1, Ck1), ..., (p_{L-1}, Ck_{L-1})]

    Semantics:
      - We follow the path in order: p0 then p1 then ... p_{L-1}.
      - For each step, we require that the target node has the given class.
      - Each step contributes +1 to the raw path score if matched.
      - The dynamic programming is implemented exactly as:
          score(v, i) = max_{(v, p_i, u)} [ local_match(u, Cki) + score(u, i+1) ]
        where for the last step L-1, score(u, L) = 0.

    DAG layout:
      - Node i is PATH_STEP for step i.
      - Node L-1 has no child (base case).
      - Node i (0 <= i < L-1) has child_indices = [i+1].
      - Root is node 0.
      - Layers go from leaves upward: [ [L-1], [L-2], ..., [0] ].

    Normalization:
      - If pattern.normalize == True, we set scale_factor = 1 / L on the root node.
        So a fully matching path gets score 1.0, partial matches get (steps_matched / L).
    """
    steps = pattern.steps
    L = len(steps)
    if L <= 0:
        raise ValueError("PathPattern must have at least one step.")

    # Build PATH_STEP chain
    nodes: List[ConstraintNode] = [None] * L  # type: ignore

    for i in range(L - 1, -1, -1):
        prop_idx, class_idx = steps[i]
        if i == L - 1:
            # Last step: no child, base case score(u, L) = 0
            child_indices = None
        else:
            child_indices = [i + 1]

        node = ConstraintNode(
            idx=i,
            ctype=ConstraintType.PATH_STEP,
            prop_idx=prop_idx,
            class_idx=class_idx,
            child_indices=child_indices,
            scale_factor=None,  # set below for root if we normalize
        )
        nodes[i] = node

    # Optional normalization on the root node (step 0)
    if pattern.normalize:
        nodes[0].scale_factor = 1.0 / float(L)

    # Layers: leaves first, root last
    # layer 0: [L-1], layer 1: [L-2], ..., layer L-1: [0]
    layers: List[List[int]] = []
    for i in range(L - 1, -1, -1):
        layers.append([i])

    root_idx = 0

    return ConstraintDAG(nodes=nodes, root_idx=root_idx, layers=layers)
