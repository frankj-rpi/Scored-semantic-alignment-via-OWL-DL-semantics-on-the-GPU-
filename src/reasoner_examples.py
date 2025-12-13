from __future__ import annotations

import torch

from .graph import KGraph
from .constraints import (
    ConstraintNode,
    ConstraintDAG,
    ConstraintType,
    IntersectionAgg,
)
from .dag_reasoner import DAGReasoner


# ---------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------


def make_toy_graph() -> KGraph:
    """
    Toy graph:

      Nodes:
        0: FurnitureA
        1: Wheel1
        2: Wheel2
        3: FurnitureB

      Property p0 = hasPart:
        0 -> 1
        0 -> 2
        3 -> 2

      Classes:
        0: (unused)
        1: Wheel
        2: Furniture
    """

    num_nodes = 4
    num_classes = 3

    # CSR for p0 (hasPart)
    neighbors_p0 = torch.tensor([1, 2, 2], dtype=torch.int32)
    # offsets: 0->[1,2], 1->[], 2->[], 3->[2]
    offsets_p0 = torch.tensor([0, 2, 2, 2, 3], dtype=torch.int32)

    offsets_p = [offsets_p0]
    neighbors_p = [neighbors_p0]

    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    # Wheels
    node_types[1, 1] = 1.0
    node_types[2, 1] = 1.0
    # Furniture
    node_types[0, 2] = 1.0
    node_types[3, 2] = 1.0

    return KGraph(
        num_nodes=num_nodes,
        offsets_p=offsets_p,
        neighbors_p=neighbors_p,
        node_types=node_types,
    )


def make_vehicle_graph() -> KGraph:
    """
    Vehicle example:

      Vehicle ≡ ∃hasPart.Wheel ⊓ ∃hasPart.Seat

      Nodes:
        0: Bicycle
        1: Stool
        2: WheelPart
        3: SeatPart

      hasPart (p0):
        0 -> 2 (WheelPart)
        0 -> 3 (SeatPart)
        1 -> 3 (SeatPart)

      Classes:
        0: Wheel
        1: Seat
    """

    num_nodes = 4
    num_classes = 2

    neighbors_p0 = torch.tensor([2, 3, 3], dtype=torch.int32)
    offsets_p0 = torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32)

    offsets_p = [offsets_p0]
    neighbors_p = [neighbors_p0]

    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    # WheelPart
    node_types[2, 0] = 1.0
    # SeatPart
    node_types[3, 1] = 1.0

    return KGraph(
        num_nodes=num_nodes,
        offsets_p=offsets_p,
        neighbors_p=neighbors_p,
        node_types=node_types,
    )


# ---------------------------------------------------------------------
# Constraint DAG builders
# ---------------------------------------------------------------------


def make_has_wheel_dag() -> ConstraintDAG:
    """
    C1 ≡ ∃ hasPart.Wheel

    class_idx 1 = Wheel
    prop_idx 0 = hasPart
    """

    atomic_wheel = ConstraintNode(
        idx=0,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=1,
    )

    exists_hasPart_wheel = ConstraintNode(
        idx=1,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        prop_idx=0,
        child_indices=[0],
    )

    nodes = [atomic_wheel, exists_hasPart_wheel]
    layers = [
        [0],  # leaf
        [1],  # root
    ]

    return ConstraintDAG(nodes=nodes, root_idx=1, layers=layers)


def make_furniture_with_wheel_dag() -> ConstraintDAG:
    """
    C2 ≡ Furniture ⊓ ∃ hasPart.Wheel

    class_idx 1 = Wheel
    class_idx 2 = Furniture
    prop_idx 0 = hasPart
    """

    atomic_wheel = ConstraintNode(
        idx=0,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=1,
    )

    exists_hasPart_wheel = ConstraintNode(
        idx=1,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        prop_idx=0,
        child_indices=[0],
    )

    atomic_furniture = ConstraintNode(
        idx=2,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=2,
    )

    intersection_node = ConstraintNode(
        idx=3,
        ctype=ConstraintType.INTERSECTION,
        child_indices=[1, 2],
        intersection_agg=IntersectionAgg.MIN,  # strict logical-style AND
    )

    nodes = [atomic_wheel, exists_hasPart_wheel, atomic_furniture, intersection_node]
    layers = [
        [0, 2],  # atomic
        [1],     # exists
        [3],     # intersection root
    ]

    return ConstraintDAG(nodes=nodes, root_idx=3, layers=layers)


def make_vehicle_dag() -> ConstraintDAG:
    """
    Vehicle ≡ ∃hasPart.Wheel ⊓ ∃hasPart.Seat

    Using "fractional" intersection via MEAN:
      - if both restrictions are satisfied → 1.0
      - if only one is → 0.5
    """

    # atomic Wheel (class_idx 0)
    atomic_wheel = ConstraintNode(
        idx=0,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=0,
    )

    exists_hasPart_wheel = ConstraintNode(
        idx=1,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        prop_idx=0,
        child_indices=[0],
    )

    # atomic Seat (class_idx 1)
    atomic_seat = ConstraintNode(
        idx=2,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=1,
    )

    exists_hasPart_seat = ConstraintNode(
        idx=3,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        prop_idx=0,
        child_indices=[2],
    )

    intersection_vehicle = ConstraintNode(
        idx=4,
        ctype=ConstraintType.INTERSECTION,
        child_indices=[1, 3],
        intersection_agg=IntersectionAgg.MEAN,  # fractional coverage
    )

    nodes = [
        atomic_wheel,
        exists_hasPart_wheel,
        atomic_seat,
        exists_hasPart_seat,
        intersection_vehicle,
    ]

    layers = [
        [0, 2],  # ATOMIC_CLASS
        [1, 3],  # EXISTS
        [4],     # INTERSECTION root
    ]

    return ConstraintDAG(nodes=nodes, root_idx=4, layers=layers)

# --------
# Multi-hop constraint matching
# --------

def make_multihop_graph() -> KGraph:
    """
    Multi-hop typed path example:

    Pattern we want to score:

      P ≡ (step1) via p0 to a Mid node (class 0)
           then
           (step2) via p1 from that Mid node to an EndGood node (class 1)

    Nodes:
      0: StartComplete
      1: StartPartial
      2: MidA
      3: MidB
      4: EndGood
      5: EndBad

    Classes:
      0: MidType
      1: EndGoodType

    Edges:
      p0 (index 0):
        0 -> 2
        1 -> 3
      p1 (index 1):
        2 -> 4
        3 -> 5
    """

    num_nodes = 6
    num_classes = 2

    # p0 adjacency: 0->2, 1->3
    neighbors_p0 = torch.tensor([2, 3], dtype=torch.int32)
    offsets_p0 = torch.tensor([0, 1, 2, 2, 2, 2, 2], dtype=torch.int32)
    # v0: neighbors[0:1] = [2]
    # v1: neighbors[1:2] = [3]
    # v2..v5: []

    # p1 adjacency: 2->4, 3->5
    neighbors_p1 = torch.tensor([4, 5], dtype=torch.int32)
    offsets_p1 = torch.tensor([0, 0, 0, 1, 2, 2, 2], dtype=torch.int32)
    # v0,v1: []
    # v2: neighbors[0:1] = [4]
    # v3: neighbors[1:2] = [5]
    # v4,v5: []

    offsets_p = [offsets_p0, offsets_p1]
    neighbors_p = [neighbors_p0, neighbors_p1]

    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    # Mid nodes
    node_types[2, 0] = 1.0  # MidType
    node_types[3, 0] = 1.0
    # EndGood
    node_types[4, 1] = 1.0  # EndGoodType
    # EndBad has no EndGoodType

    return KGraph(
        num_nodes=num_nodes,
        offsets_p=offsets_p,
        neighbors_p=neighbors_p,
        node_types=node_types,
    )

def make_two_step_path_dag() -> ConstraintDAG:
    """
    Two-step typed path pattern:

      Raw pattern P_raw:
        PATH_STEP(p0, MidType) ◦ PATH_STEP(p1, EndGoodType)

      Normalized pattern P:
        score_norm = score_raw / 2

    Implementation:
      - Node 0: PATH_STEP over p1 to EndGoodType, child = None (base case)
      - Node 1: PATH_STEP over p0 to MidType,    child = Node 0, scale_factor = 1/2

    Raw scores:
      - v=0 (StartComplete): 2.0
      - v=1 (StartPartial):  1.0

    Normalized scores:
      - v=0: 1.0
      - v=1: 0.5
    """

    # Node 0: second / last step: via p1 to EndGoodType (class_idx 1)
    # child_indices = None → base case: child_score(u) = 0
    step2 = ConstraintNode(
        idx=0,
        ctype=ConstraintType.PATH_STEP,
        prop_idx=1,        # p1
        class_idx=1,       # EndGoodType
        child_indices=None,
    )

    # Node 1: first step: via p0 to MidType (class_idx 0), root with scaling
    step1_root = ConstraintNode(
        idx=1,
        ctype=ConstraintType.PATH_STEP,
        prop_idx=0,        # p0
        class_idx=0,       # MidType
        child_indices=[0],
        scale_factor=0.5,  # 1 / path_length (2)
    )

    nodes = [step2, step1_root]
    layers = [
        [0],  # step2 (depends on no other nodes)
        [1],  # step1_root depends on 0 (root)
    ]

    return ConstraintDAG(nodes=nodes, root_idx=1, layers=layers)




# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------


def main():
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) Toy graph: Wheels + Furniture
    print("\n=== Toy graph: Wheels + Furniture ===")
    graph = make_toy_graph()

    print("\nCSR adjacency for p0 (hasPart):")
    print("  offsets:", graph.offsets_p[0])
    print("  neighbors:", graph.neighbors_p[0])

    print("\nNode types [node, class]:")
    print(graph.node_types)

    reasoner = DAGReasoner(graph, device=device, sim_class=None)
    reasoner.add_concept("HasWheel", make_has_wheel_dag())
    reasoner.add_concept("FurnitureWithWheel", make_furniture_with_wheel_dag())

    scores = reasoner.evaluate_all()  # [num_nodes, 2]

    print("\nScores matrix [node, concept]:")
    print("Concepts:", reasoner.concept_names)
    print(scores)

    print("\nTop-2 nodes for 'HasWheel':")
    for node_id, score in reasoner.top_k_for_concept("HasWheel", k=2):
        print(f"  node {node_id}: {score:.4f}")

    print("\nTop-2 concepts for node 0:")
    for cname, score in reasoner.top_k_for_node(0, k=2):
        print(f"  {cname}: {score:.4f}")

    # 2) Vehicle example with fractional intersection
    print("\n=== Vehicle example: fractional intersection ===")
    v_graph = make_vehicle_graph()
    v_reasoner = DAGReasoner(v_graph, device=device, sim_class=None)
    v_reasoner.add_concept("Vehicle", make_vehicle_dag())

    v_scores = v_reasoner.evaluate_all()  # [4, 1]

    print("\nScores s(v, Vehicle) for each node v (0=Bicycle, 1=Stool, 2=WheelPart, 3=SeatPart):")
    for v in range(v_graph.num_nodes):
        print(f"  v = {v}: s(v, Vehicle) = {v_scores[v, 0].item():.4f}")

    print("\nExpected behavior:")
    print("  - v=0 (Bicycle) has WheelPart + SeatPart → Vehicle score 1.0")
    print("  - v=1 (Stool)   has only SeatPart       → Vehicle score 0.5")
    print("  - v=2,3 (parts) have no hasPart edges   → Vehicle score 0.0")

    # 3) Multi-hop typed path example
    print("\n=== Two-step typed path example ===")
    mh_graph = make_multihop_graph()
    mh_reasoner = DAGReasoner(mh_graph, device=device, sim_class=None)
    mh_reasoner.add_concept("TwoStepPattern", make_two_step_path_dag())

    mh_scores = mh_reasoner.evaluate_all()  # [6, 1]

    print("\nScores s(v, TwoStepPattern) for each node v:")
    for v in range(mh_graph.num_nodes):
        print(f"  v = {v}: s(v, TwoStepPattern) = {mh_scores[v, 0].item():.4f}")

    print("\nInterpretation:")
    print("  - v=0 (StartComplete): p0->MidA and p1->EndGood → score 1.0")
    print("  - v=1 (StartPartial):  p0->MidB but p1->EndBad → score 0.5")
    print("  - v=2..5: no full start path → score 0.0")



if __name__ == "__main__":
    main()
