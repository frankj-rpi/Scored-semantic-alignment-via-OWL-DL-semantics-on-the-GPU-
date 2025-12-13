from __future__ import annotations
import torch

from .graph import KGraph
from .constraints import ConstraintDAG, ConstraintType, IntersectionAgg


def eval_dag_scores(
    graph: KGraph,
    dag: ConstraintDAG,
    device: str = "cpu",
    sim_class: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Evaluate a single ConstraintDAG over all nodes in the graph.

    Returns:
        scores_for_concept: tensor of shape [num_nodes],
        where scores_for_concept[v] is the score s(v, C) for the DAG's concept.
    """
    # Move graph data to device
    offsets_p = [off.to(device) for off in graph.offsets_p]
    neighbors_p = [neigh.to(device) for neigh in graph.neighbors_p]
    node_types = graph.node_types.to(device)
    num_nodes = graph.num_nodes
    num_constraints = len(dag.nodes)
    num_classes = node_types.shape[1]

    if sim_class is None:
        # Default: exact matching (identity matrix)
        sim_class = torch.eye(num_classes, device=device)
    else:
        sim_class = sim_class.to(device)
        assert sim_class.shape == (num_classes, num_classes), \
            f"sim_class must have shape ({num_classes}, {num_classes}), got {sim_class.shape}"

    # NEW: precompute src indices for each property p
    src_index_p: list[torch.Tensor] = []
    nodes = torch.arange(num_nodes, device=device, dtype=torch.long)

    for offsets in offsets_p:
        # deg[v] = number of outgoing edges on this property
        deg = offsets[1:] - offsets[:-1]          # [num_nodes]
        # repeat each node id by its degree → one entry per edge
        src_index = torch.repeat_interleave(nodes, deg)  # [num_edges_p]
        src_index_p.append(src_index)

    # scores[v, k] = score for node v at constraint node k
    scores = torch.zeros((num_nodes, num_constraints), device=device)

    # Evaluate layer by layer.
    # Convention: layers[0] = leaves, layers[-1] = root.
    for layer in dag.layers:
        for node_idx in layer:
            cnode = dag.nodes[node_idx]

            if cnode.ctype == ConstraintType.CONST:
                # Base case: constant value
                scores[:, node_idx] = 1.0
            elif cnode.ctype == ConstraintType.ATOMIC_CLASS:
                # Exact: 1.0 if node has this class, else 0.0.
                # Here node_types is [num_nodes, num_classes].
                #scores[:, node_idx] = node_types[:, cnode.class_idx]

                # Graded class matching using sim_class.
                # node_types: [num_nodes, num_classes] (one-hot or multi-hot)
                # sim_class:  [num_classes, num_classes], where sim_class[i, j]
                #   = similarity between stored class i and constraint class j.
                # For constraint class_idx = k, we compute:
                #   scores[:, k] = node_types @ sim_class[:, k]
                class_idx = cnode.class_idx
                # Column vector [num_classes] for this target class
                col = sim_class[:, class_idx]  # [num_classes]
                # For each node v, score = sum_i node_types[v, i] * sim_class[i, class_idx]
                scores[:, node_idx] = node_types @ col

            elif cnode.ctype == ConstraintType.EXISTS_RESTRICTION:
                # ∃ R.D :
                # For each node v, look at neighbors via property prop_idx
                # and take max of child constraint scores at those neighbors.
                if not cnode.child_indices or len(cnode.child_indices) != 1:
                    raise ValueError("EXISTS_RESTRICTION must have exactly one child index.")

                child_idx = cnode.child_indices[0]
                child_scores = scores[:, child_idx]  # [num_nodes]

                prop_idx = cnode.prop_idx
                offsets = offsets_p[prop_idx]
                neigh = neighbors_p[prop_idx]

                out = torch.zeros(num_nodes, device=device)

                # # Naive implementation for now (we can optimize later)
                # for v in range(num_nodes):
                #     start = int(offsets[v].item())
                #     end = int(offsets[v + 1].item())
                #     if start == end:
                #         out[v] = 0.0
                #     else:
                #         u_indices = neigh[start:end].long()  # ensure long dtype for indexing
                #         out[v] = child_scores[u_indices].max()

                # GPU implementation:

                # Edge-level view for this property
                dst_index = neighbors_p[prop_idx].long()   # [num_edges_p]
                src_index = src_index_p[prop_idx]          # [num_edges_p], long

                # For each edge e: candidate score is just child_scores[dst_e]
                edge_values = child_scores[dst_index]      # [num_edges_p]

                # PyTorch 2.x: scatter_reduce_ is in-place
                # reduce="amax" means elementwise max over all edges with same src_index
                out.scatter_reduce_(
                    dim=0,
                    index=src_index,
                    src=edge_values,
                    reduce="amax",
                    include_self=True,
                )

                scores[:, node_idx] = out

            elif cnode.ctype == ConstraintType.INTERSECTION:
                # A ⊓ B ⊓ ...
                if not cnode.child_indices or len(cnode.child_indices) < 2:
                    raise ValueError("INTERSECTION must have at least two child indices.")

                child_tensors = [scores[:, ci] for ci in cnode.child_indices]
                stacked = torch.stack(child_tensors, dim=1)  # [num_nodes, num_children]

                agg_mode = cnode.intersection_agg or IntersectionAgg.MIN

                if agg_mode == IntersectionAgg.MIN:
                    scores[:, node_idx] = stacked.min(dim=1).values
                elif agg_mode == IntersectionAgg.MEAN:
                    scores[:, node_idx] = stacked.mean(dim=1)
                else:
                    raise ValueError(f"Unknown intersection agg mode: {agg_mode}")

            elif cnode.ctype == ConstraintType.UNION:
                # A ⊔ B ⊔ ...
                if not cnode.child_indices or len(cnode.child_indices) < 2:
                    raise ValueError("UNION must have at least two child indices.")

                child_tensors = [scores[:, ci] for ci in cnode.child_indices]
                stacked = torch.stack(child_tensors, dim=1)  # [num_nodes, num_children]
                scores[:, node_idx] = stacked.max(dim=1).values

            elif cnode.ctype == ConstraintType.PATH_STEP:
                # Multi-hop typed path step:
                #   score(v, step_k) = max_{(v, p, u)} (local_match(v->u) + score(u, child))
                #
                # where:
                #   local_match(v->u) = node_types[u, required_class]
                # and "child" is the suffix of the path starting at u.
                #
                # Base case: if no child_indices, child_score(u) = 0 for all u.
                
                prop_idx = cnode.prop_idx
                required_class = cnode.class_idx

                if not cnode.child_indices or len(cnode.child_indices) == 0:
                    # Base step: child_score(u) = 0
                    child_scores = torch.zeros(num_nodes, device=device)
                elif len(cnode.child_indices) == 1:
                    child_idx = cnode.child_indices[0]
                    child_scores = scores[:, child_idx]
                else:
                    raise ValueError("PATH_STEP must have at most one child index.")

                # Naive:
                # offsets = offsets_p[prop_idx]
                # neigh = neighbors_p[prop_idx]

                # out = torch.zeros(num_nodes, device=device)

                # for v in range(num_nodes):
                #     start = int(offsets[v].item())
                #     end = int(offsets[v + 1].item())
                #     if start == end:
                #         out[v] = 0.0
                #     else:
                #         best = 0.0
                #         for e in range(start, end):
                #             u = int(neigh[e].item())
                #             # local_match is 1.0 if u has the required class, else 0.0
                #             local_match = node_types[u, required_class]
                #             candidate = local_match + child_scores[u]
                #             if candidate > best:
                #                 best = candidate
                #         out[v] = best
                
                # GPU:

                # Edge-level view for this property
                dst_index = neighbors_p[prop_idx].long()   # [num_edges_p]
                src_index = src_index_p[prop_idx]          # [num_edges_p], precomputed

                # local_match(u) = node_types[u, required_class]
                local_match = node_types[dst_index, required_class]  # [num_edges_p]

                # edge_values[e] = local_match(u) + child_scores[u]
                edge_values = local_match + child_scores[dst_index]  # [num_edges_p]

                # Scatter-reduce with max over src_index
                out = torch.zeros(num_nodes, device=device)

                out.scatter_reduce_(
                    dim=0,
                    index=src_index,
                    src=edge_values,
                    reduce="amax",
                    include_self=True,
                )

                scores[:, node_idx] = out

            else:
                raise NotImplementedError(f"Constraint type {cnode.ctype} not supported yet.")

            # Optional generic scaling for any node
            if cnode.scale_factor is not None:
                scores[:, node_idx] *= float(cnode.scale_factor)


    # Concept score is score at the root constraint node
    root_idx = dag.root_idx
    return scores[:, root_idx]
