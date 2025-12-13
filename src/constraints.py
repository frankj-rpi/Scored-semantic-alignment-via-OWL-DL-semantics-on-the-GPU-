from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class ConstraintType(Enum):
    CONST = auto()               # Constant score
    ATOMIC_CLASS = auto()        # node has (or is similar to) a given class
    EXISTS_RESTRICTION = auto()  # ∃ R.D
    INTERSECTION = auto()        # A ⊓ B
    UNION = auto()               # A ⊔ B
    PATH_STEP = auto()          # A→P→B
    # More later: FORALL_RESTRICTION, CARDINALITY, etc.

class IntersectionAgg(Enum):
    MIN = "min"
    MEAN = "mean"

@dataclass
class ConstraintNode:
    """
    One node in the constraint DAG.

    Convention:
    - class_idx: index into the class dimension for ATOMIC_CLASS.
    - prop_idx: index into graph.offsets_p / neighbors_p for EXISTS nodes.
    - child_indices: indices into the DAG's node list for dependent constraints.
    """
    idx: int
    ctype: ConstraintType
    class_idx: Optional[int] = None
    prop_idx: Optional[int] = None
    child_indices: Optional[List[int]] = None

    # For INTERSECTION nodes: how to aggregate child scores
    intersection_agg: Optional[IntersectionAgg] = None

    # Optional scale factor
    scale_factor: Optional[float] = None



@dataclass
class ConstraintDAG:
    """
    Constraint DAG for a single concept.

    - nodes: list of all constraint nodes (indexed by .idx)
    - root_idx: index of the node whose score is the concept score.
    - layers: list of layers, each a list of node indices.

    Layering convention (important for evaluation):
    - layers[0] are "leaf" constraints (no dependencies or only on data),
    - layers[1] depend on layers[0],
    - ...
    - layers[-1] contains the root constraint node.
    """
    nodes: List[ConstraintNode]
    root_idx: int
    layers: List[List[int]]
