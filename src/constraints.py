from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class ConstraintType(Enum):
    CONST = auto()               # Constant score
    ATOMIC_CLASS = auto()        # Node has, or is similar to, a given class
    NOMINAL = auto()             # Node is exactly a given RDF individual
    DATATYPE_CONSTRAINT = auto() # Literal node satisfies a datatype predicate
    NEGATION = auto()            # Not D
    HAS_SELF_RESTRICTION = auto()  # Exists R.Self
    EXISTS_RESTRICTION = auto()  # Exists R.D
    EXISTS_TRANSITIVE_RESTRICTION = auto()  # Exists R+.D
    MIN_CARDINALITY_RESTRICTION = auto()    # At least n R.(D)
    MAX_CARDINALITY_RESTRICTION = auto()    # At most n R.(D)
    EXACT_CARDINALITY_RESTRICTION = auto()  # Exactly n R.(D)
    FORALL_RESTRICTION = auto()  # Forall R.D
    INTERSECTION = auto()        # A and B
    UNION = auto()               # A or B
    PATH_STEP = auto()           # Typed path step
    # More later: CARDINALITY, etc.


class TraversalDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class IntersectionAgg(Enum):
    MIN = "min"
    MEAN = "mean"


class CardinalityAgg(Enum):
    STRICT = "strict"
    FUZZY = "fuzzy"


@dataclass
class ConstraintNode:
    """
    One node in the constraint DAG.

    Convention:
    - class_idx: index into the class dimension for ATOMIC_CLASS.
    - prop_idx: index into graph.offsets_p / neighbors_p for EXISTS/FORALL nodes.
    - child_indices: indices into the DAG's node list for dependent constraints.
    """

    idx: int
    ctype: ConstraintType
    class_idx: Optional[int] = None
    node_idx: Optional[int] = None
    datatype_idx: Optional[int] = None
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    min_inclusive: bool = True
    max_inclusive: bool = True
    prop_idx: Optional[int] = None
    prop_direction: TraversalDirection = TraversalDirection.FORWARD
    cardinality_target: Optional[int] = None
    cardinality_delta: Optional[float] = None
    cardinality_agg: CardinalityAgg = CardinalityAgg.STRICT
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

    Layering convention:
    - layers[0] are leaf constraints,
    - layers[1] depend on layers[0],
    - ...
    - layers[-1] contains the root node.
    """

    nodes: List[ConstraintNode]
    root_idx: int
    layers: List[List[int]]
