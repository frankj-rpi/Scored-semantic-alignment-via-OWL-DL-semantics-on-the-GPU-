from __future__ import annotations
import torch

from .graph import KGraph
from .constraints import ConstraintNode, ConstraintDAG, ConstraintType, IntersectionAgg
from .dag_eval import eval_dag_scores

def make_class_similarity_matrix(num_classes: int) -> torch.Tensor:
    """
    Construct a toy class similarity matrix sim_class[i, j] in [0,1],
    where i is the stored class and j is the constraint class.

    Here:
      - class 1 = Wheel
      - class 2 = Furniture-like
    We will say Furniture-like has some similarity to Wheel.
    """
    sim = torch.eye(num_classes, dtype=torch.float32)

    # Make Furniture-like (2) partially similar to Wheel (1)
    # sim_class[stored, constraint]
    sim[2, 1] = 0.3  # a Furniture-like node partially matches 'Wheel' constraint

    return sim


def make_toy_graph() -> KGraph:
    """
    Construct the small example graph:
        Nodes: 0, 1, 2, 3
        Property p0 = hasPart
        Edges:
            0 -> 1
            0 -> 2
            3 -> 2
        Node types: a toy one-hot matrix of size [4 x 3].
        - class 0: Thing (unused)
        - class 1: Wheel
        - class 2: Furniture-like
    """

    num_nodes = 4
    num_classes = 3  # Just a toy dimensionality

    # CSR adjacency for property 0 (hasPart)
    # neighbors = [1, 2, 2]
    # offsets  = [0, 2, 2, 2, 3]
    neighbors_p0 = torch.tensor([1, 2, 2], dtype=torch.int32)
    offsets_p0 = torch.tensor([0, 2, 2, 2, 3], dtype=torch.int32)

    offsets_p = [offsets_p0]
    neighbors_p = [neighbors_p0]

    # Node types: [num_nodes, num_classes]
    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    # nodes 1 and 2 are Wheels (class 1)
    node_types[1, 1] = 1.0
    node_types[2, 1] = 1.0
    #node_types[1, 2] = 1.0
    #node_types[2, 2] = 1.0
    # nodes 0 and 3 are Furniture-like (class 2)
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
    Example:
      Vehicle ≡ ∃hasPart.Wheel ⊓ ∃hasPart.Seat

      :Bicycle hasPart wheelNode, seatNode.
      :Stool   hasPart seatNodeOnly.

    We want:
      score(Bicycle, Vehicle) = 1.0
      score(Stool,   Vehicle) = 0.5
    """

    # Nodes:
    # 0: Bicycle
    # 1: Stool
    # 2: WheelPart
    # 3: SeatPart
    num_nodes = 4

    # Classes:
    # 0: Wheel
    # 1: Seat
    num_classes = 2

    # hasPart edges (prop_idx 0):
    #   0 -> 2 (WheelPart)
    #   0 -> 3 (SeatPart)
    #   1 -> 3 (SeatPart)
    neighbors_p0 = torch.tensor([2, 3, 3], dtype=torch.int32)
    offsets_p0 = torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32)
    offsets_p = [offsets_p0]
    neighbors_p = [neighbors_p0]

    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    # WheelPart is a Wheel
    node_types[2, 0] = 1.0
    # SeatPart is a Seat
    node_types[3, 1] = 1.0
    # Bicycle and Stool have no direct type here; we're inferring Vehicle score structurally.

    return KGraph(
        num_nodes=num_nodes,
        offsets_p=offsets_p,
        neighbors_p=neighbors_p,
        node_types=node_types,
    )


def make_toy_constraint_dag() -> ConstraintDAG:
    """
    Build a constraint DAG for the concept:

        C ≡ ∃ hasPart.Wheel

    Using:
        - class_idx 1 for Wheel
        - prop_idx 0 for hasPart
    """

    # Node 0: ATOMIC_CLASS Wheel
    atomic_wheel = ConstraintNode(
        idx=0,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=1,   # 'Wheel'
        prop_idx=None,
        child_indices=None,
    )

    # Node 1: EXISTS hasPart.Wheel
    exists_hasPart_wheel = ConstraintNode(
        idx=1,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        class_idx=None,
        prop_idx=0,          # hasPart
        child_indices=[0],   # child is atomic_wheel
    )

    nodes = [atomic_wheel, exists_hasPart_wheel]

    # Layering convention:
    # layers[0] = leaves (atomic)
    # layers[1] = nodes depending on them (exists)
    layers = [
        [0],  # ATOMIC_CLASS Wheel
        [1],  # EXISTS hasPart.Wheel
    ]

    return ConstraintDAG(nodes=nodes, root_idx=1, layers=layers)

def make_furniture_with_wheel_dag() -> ConstraintDAG:
    """
    Build a constraint DAG for the concept:

        C2 ≡ FurnitureLike ⊓ ∃ hasPart.Wheel

    Where:
        - class_idx 1 = Wheel
        - class_idx 2 = Furniture-like
        - prop_idx 0 = hasPart
    """

    # Node 0: ATOMIC_CLASS Wheel
    atomic_wheel = ConstraintNode(
        idx=0,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=1,   # 'Wheel'
        prop_idx=None,
        child_indices=None,
    )

    # Node 1: EXISTS hasPart.Wheel
    exists_hasPart_wheel = ConstraintNode(
        idx=1,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        class_idx=None,
        prop_idx=0,          # hasPart
        child_indices=[0],
    )

    # Node 2: ATOMIC_CLASS Furniture-like
    atomic_furniture = ConstraintNode(
        idx=2,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=2,   # 'Furniture-like'
        prop_idx=None,
        child_indices=None,
    )

    # Node 3: INTERSECTION(Furniture-like, EXISTS hasPart.Wheel)
    intersection_node = ConstraintNode(
        idx=3,
        ctype=ConstraintType.INTERSECTION,
        class_idx=None,
        prop_idx=None,
        child_indices=[1, 2],  # depends on EXISTS node and Furniture-like
    )

    nodes = [atomic_wheel, exists_hasPart_wheel, atomic_furniture, intersection_node]

    # layers:
    #   layer 0: atomic leaves (Wheel, Furniture-like)
    #   layer 1: EXISTS hasPart.Wheel
    #   layer 2: INTERSECTION
    layers = [
        [0, 2],  # leaf constraints
        [1],     # exists
        [3],     # intersection root
    ]

    return ConstraintDAG(nodes=nodes, root_idx=3, layers=layers)

def make_vehicle_dag() -> ConstraintDAG:
    """
    Vehicle ≡ ∃hasPart.Wheel ⊓ ∃hasPart.Seat

    Using "fractional" intersection (mean of child scores).
    """

    # Node 0: ATOMIC_CLASS Wheel (class_idx = 0)
    atomic_wheel = ConstraintNode(
        idx=0,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=0,
    )

    # Node 1: EXISTS hasPart.Wheel
    exists_hasPart_wheel = ConstraintNode(
        idx=1,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        prop_idx=0,          # hasPart
        child_indices=[0],
    )

    # Node 2: ATOMIC_CLASS Seat (class_idx = 1)
    atomic_seat = ConstraintNode(
        idx=2,
        ctype=ConstraintType.ATOMIC_CLASS,
        class_idx=1,
    )

    # Node 3: EXISTS hasPart.Seat
    exists_hasPart_seat = ConstraintNode(
        idx=3,
        ctype=ConstraintType.EXISTS_RESTRICTION,
        prop_idx=0,          # hasPart
        child_indices=[2],
    )

    # Node 4: INTERSECTION( ∃hasPart.Wheel, ∃hasPart.Seat )
    intersection_vehicle = ConstraintNode(
        idx=4,
        ctype=ConstraintType.INTERSECTION,
        child_indices=[1, 3],
        intersection_agg=IntersectionAgg.MEAN,  # <-- fractional coverage
    )

    nodes = [atomic_wheel, exists_hasPart_wheel,
             atomic_seat, exists_hasPart_seat,
             intersection_vehicle]

    # layers:
    #   layer 0: atomic class tests
    #   layer 1: exists restrictions
    #   layer 2: intersection
    layers = [
        [0, 2],  # ATOMIC_CLASS
        [1, 3],  # EXISTS
        [4],     # INTERSECTION root
    ]

    return ConstraintDAG(nodes=nodes, root_idx=4, layers=layers)


def main():
    print("=== Sanity check: constructing toy graph ===")

    graph = make_toy_graph()

    print("\nNumber of nodes:", graph.num_nodes)

    print("\nCSR adjacency for property p0 (hasPart):")
    print("offsets:", graph.offsets_p[0])
    print("neighbors:", graph.neighbors_p[0])

    print("\nNode types matrix [num_nodes x num_classes]:")
    print(graph.node_types)

    print("\nNeighbors per node (via hasPart):")
    offsets = graph.offsets_p[0]
    neigh = graph.neighbors_p[0]
    for v in range(graph.num_nodes):
        start = int(offsets[v].item())
        end = int(offsets[v + 1].item())
        print(f"  {v} -> {neigh[start:end].tolist()}")

    num_classes = graph.node_types.shape[1]
    sim_class = make_class_similarity_matrix(num_classes)

    # You can pick device = "cuda" or "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing device:", device)

    # Concept 1: C1 ≡ ∃ hasPart.Wheel
    print("\n=== Evaluating C1: C1 ≡ ∃ hasPart.Wheel (exact) ===")
    dag_c1 = make_toy_constraint_dag()
    scores_c1_exact = eval_dag_scores(graph, dag_c1, device=device)

    print("\nScores s(v, C1) [exact] for each node v:")
    for v in range(graph.num_nodes):
        print(f"  v = {v}: s_exact(v, C1) = {scores_c1_exact[v].item():.4f}")

    print("\n=== Evaluating C1: C1 ≡ ∃ hasPart.Wheel (with class similarity) ===")
    scores_c1_sim = eval_dag_scores(graph, dag_c1, device=device, sim_class=sim_class)

    print("\nScores s(v, C1) [similarity] for each node v:")
    for v in range(graph.num_nodes):
        print(f"  v = {v}: s_sim(v, C1) = {scores_c1_sim[v].item():.4f}")

    # Concept 2: C2 ≡ FurnitureLike ⊓ ∃ hasPart.Wheel
    print("\n=== Evaluating C2: C2 ≡ FurnitureLike ⊓ ∃ hasPart.Wheel (with class similarity) ===")
    dag_c2 = make_furniture_with_wheel_dag()
    scores_c2_sim = eval_dag_scores(graph, dag_c2, device=device, sim_class=sim_class)

    print("\nScores s(v, C2) [similarity] for each node v:")
    for v in range(graph.num_nodes):
        print(f"  v = {v}: s_sim(v, C2) = {scores_c2_sim[v].item():.4f}")


    # Vehicle example with aggregation intersection
    print("\n=== Vehicle example: fractional intersection ===")

    v_graph = make_vehicle_graph()
    v_dag = make_vehicle_dag()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores_vehicle = eval_dag_scores(v_graph, v_dag, device=device)

    print("\nScores s(v, Vehicle) for each node v (0=Bicycle, 1=Stool, 2=WheelPart, 3=SeatPart):")
    for v in range(v_graph.num_nodes):
        print(f"  v = {v}: s(v, Vehicle) = {scores_vehicle[v].item():.4f}")

    print("\nExpected behavior:")
    print("  - v=0 (Bicycle) has a WheelPart and a SeatPart → scores 1.0")
    print("  - v=1 (Stool)   has only a SeatPart           → scores 0.5")
    print("  - v=2,3 (parts) have no hasPart edges         → score 0.0")



# def main():
#     print("=== Sanity check: constructing toy graph ===")

#     graph = make_toy_graph()

#     print("\nNumber of nodes:", graph.num_nodes)

#     print("\nCSR adjacency for property p0 (hasPart):")
#     print("offsets:", graph.offsets_p[0])
#     print("neighbors:", graph.neighbors_p[0])

#     print("\nNode types matrix [num_nodes x num_classes]:")
#     print(graph.node_types)

#     print("\nNeighbors per node (via hasPart):")
#     offsets = graph.offsets_p[0]
#     neigh = graph.neighbors_p[0]
#     for v in range(graph.num_nodes):
#         start = int(offsets[v].item())
#         end = int(offsets[v + 1].item())
#         print(f"  {v} -> {neigh[start:end].tolist()}")

#     # Concept 1: C1 ≡ ∃ hasPart.Wheel
#     print("\n=== Evaluating C1: C1 ≡ ∃ hasPart.Wheel ===")
#     dag_c1 = make_toy_constraint_dag()
#     scores_c1 = eval_dag_scores(graph, dag_c1, device="cuda")

#     print("\nScores s(v, C1) for each node v:")
#     for v in range(graph.num_nodes):
#         print(f"  v = {v}: s(v, C1) = {scores_c1[v].item():.4f}")

#     # Concept 2: C2 ≡ FurnitureLike ⊓ ∃ hasPart.Wheel
#     print("\n=== Evaluating C2: C2 ≡ FurnitureLike ⊓ ∃ hasPart.Wheel ===")
#     dag_c2 = make_furniture_with_wheel_dag()
#     scores_c2 = eval_dag_scores(graph, dag_c2, device="cuda")

#     print("\nScores s(v, C2) for each node v:")
#     for v in range(graph.num_nodes):
#         print(f"  v = {v}: s(v, C2) = {scores_c2[v].item():.4f}")

#     print("\nInterpretation:")
#     print("  C1: nodes with at least one hasPart neighbor of type Wheel.")
#     print("  C2: nodes that are Furniture-like AND satisfy C1.")
#     print("  In this toy graph:")
#     print("    - Node 0: Furniture-like, hasPart [1,2] which are Wheels → C1=1, C2=1.")
#     print("    - Node 3: Furniture-like, hasPart [2] which is Wheel → C1=1, C2=1.")
#     print("    - Nodes 1,2: Wheels, no hasPart edges → C1=0, C2=0.\n")

#     print("Done ✓")



if __name__ == "__main__":
    main()
