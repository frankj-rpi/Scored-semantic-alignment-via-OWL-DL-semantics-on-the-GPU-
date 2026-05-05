from __future__ import annotations

import torch

from .constraints import (
    CardinalityAgg,
    ConstraintDAG,
    ConstraintType,
    IntersectionAgg,
    TraversalDirection,
)
from .graph import KGraph


def eval_dag_score_matrix(
    graph: KGraph,
    dag: ConstraintDAG,
    device: str = "cpu",
    sim_class: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Evaluate every DAG node over all graph nodes.

    Returns:
        scores: tensor of shape [num_nodes, num_constraints],
        where scores[v, i] is the score of DAG node i at graph node v.
    """

    device = torch.device(device)
    offsets_p = graph.offsets_p
    neighbors_p = graph.neighbors_p
    node_types = graph.node_types
    if node_types.device != device:
        offsets_p = [off.to(device) for off in graph.offsets_p]
        neighbors_p = [neigh.to(device) for neigh in graph.neighbors_p]
        node_types = graph.node_types.to(device)
    num_nodes = graph.num_nodes
    num_constraints = len(dag.nodes)
    num_classes = node_types.shape[1]

    use_identity_sim = sim_class is None
    if sim_class is None:
        sim_class = torch.eye(num_classes, device=device)
    else:
        sim_class = sim_class.to(device)
        assert sim_class.shape == (num_classes, num_classes), (
            f"sim_class must have shape ({num_classes}, {num_classes}), got {sim_class.shape}"
        )

    if graph.src_index_p is not None and graph.dst_index_p is not None and node_types.device == device:
        src_index_p = graph.src_index_p
        dst_index_p = graph.dst_index_p
    else:
        src_index_p = []
        dst_index_p = []
        nodes = torch.arange(num_nodes, device=device, dtype=torch.long)

        for offsets, neighbors in zip(offsets_p, neighbors_p):
            deg = offsets[1:] - offsets[:-1]
            src_index = torch.repeat_interleave(nodes, deg)
            src_index_p.append(src_index)
            dst_index_p.append(neighbors.long())

    scores = torch.zeros((num_nodes, num_constraints), device=device)
    segment_layout_cache = graph.segment_layout_cache

    def oriented_edges(
        prop_idx: int,
        direction: TraversalDirection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_src = src_index_p[prop_idx]
        forward_dst = dst_index_p[prop_idx]

        if direction == TraversalDirection.FORWARD:
            return forward_src, forward_dst
        if direction == TraversalDirection.BACKWARD:
            return forward_dst, forward_src
        raise ValueError(f"Unknown traversal direction: {direction}")

    def is_reflexive_property(prop_idx: int) -> bool:
        if graph.reflexive_prop_mask is None:
            return False
        return bool(graph.reflexive_prop_mask[prop_idx].item())

    def segment_layout(
        prop_idx: int,
        direction: TraversalDirection,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cache_key = (prop_idx, direction.value)
        cached = segment_layout_cache.get(cache_key)
        if cached is not None:
            return cached

        reduce_index, next_nodes = oriented_edges(prop_idx, direction)
        if reduce_index.numel() == 0:
            cached = (
                reduce_index,
                next_nodes,
                torch.zeros((0,), device=device, dtype=torch.long),
                torch.zeros((0,), device=device, dtype=torch.long),
                0,
            )
            segment_layout_cache[cache_key] = cached
            return cached

        perm = torch.argsort(reduce_index, stable=True)
        sorted_reduce = reduce_index[perm]
        sorted_next = next_nodes[perm]
        counts = torch.bincount(sorted_reduce, minlength=num_nodes)
        starts = torch.zeros_like(counts)
        if counts.numel() > 1:
            starts[1:] = torch.cumsum(counts[:-1], dim=0)
        local_pos = torch.arange(sorted_reduce.numel(), device=device, dtype=torch.long)
        local_pos = local_pos - starts[sorted_reduce]
        max_width = int(counts.max().item()) if counts.numel() > 0 else 0

        cached = (sorted_reduce, sorted_next, perm, local_pos, max_width)
        segment_layout_cache[cache_key] = cached
        return cached

    def transitive_family_props(prop_idx: int) -> list[int]:
        if graph.transitive_prop_family_indices is None:
            return [prop_idx]
        family_indices = graph.transitive_prop_family_indices[prop_idx]
        if not family_indices:
            return [prop_idx]
        return list(family_indices)

    def topk_edge_scores(
        prop_idx: int,
        direction: TraversalDirection,
        edge_scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        if k <= 0:
            return torch.zeros((num_nodes, 0), device=device, dtype=edge_scores.dtype)

        sorted_reduce, _sorted_next, perm, local_pos, max_width = segment_layout(
            prop_idx,
            direction,
        )
        width = max(max_width, k)
        dense = torch.zeros((num_nodes, width), device=device, dtype=edge_scores.dtype)
        if perm.numel() == 0:
            return dense[:, :k]

        sorted_scores = edge_scores[perm]
        dense[sorted_reduce, local_pos] = sorted_scores
        return torch.topk(dense, k=k, dim=1).values

    for layer in dag.layers:
        for node_idx in layer:
            cnode = dag.nodes[node_idx]

            if cnode.ctype == ConstraintType.CONST:
                scores[:, node_idx] = 1.0

            elif cnode.ctype == ConstraintType.ATOMIC_CLASS:
                class_idx = cnode.class_idx
                if use_identity_sim:
                    scores[:, node_idx] = node_types[:, class_idx]
                else:
                    col = sim_class[:, class_idx]
                    scores[:, node_idx] = node_types @ col

            elif cnode.ctype == ConstraintType.NOMINAL:
                if cnode.node_idx is None:
                    raise ValueError("NOMINAL must provide node_idx.")
                out = torch.zeros(num_nodes, device=device)
                out[cnode.node_idx] = 1.0
                scores[:, node_idx] = out

            elif cnode.ctype == ConstraintType.DATATYPE_CONSTRAINT:
                if graph.literal_datatype_idx is None or graph.literal_numeric_value is None:
                    raise ValueError(
                        "DATATYPE_CONSTRAINT requires literal datatype metadata in KGraph."
                    )

                literal_datatype_idx = graph.literal_datatype_idx.to(device)
                literal_numeric_value = graph.literal_numeric_value.to(device)
                out = torch.ones(num_nodes, dtype=torch.float32, device=device)

                if cnode.datatype_idx is not None:
                    out = out * (literal_datatype_idx == cnode.datatype_idx).to(torch.float32)
                else:
                    out = out * (literal_datatype_idx >= 0).to(torch.float32)

                needs_numeric = cnode.numeric_min is not None or cnode.numeric_max is not None
                if needs_numeric:
                    numeric_mask = ~torch.isnan(literal_numeric_value)
                    out = out * numeric_mask.to(torch.float32)
                    if cnode.numeric_min is not None:
                        if cnode.min_inclusive:
                            out = out * (literal_numeric_value >= float(cnode.numeric_min)).to(torch.float32)
                        else:
                            out = out * (literal_numeric_value > float(cnode.numeric_min)).to(torch.float32)
                    if cnode.numeric_max is not None:
                        if cnode.max_inclusive:
                            out = out * (literal_numeric_value <= float(cnode.numeric_max)).to(torch.float32)
                        else:
                            out = out * (literal_numeric_value < float(cnode.numeric_max)).to(torch.float32)

                scores[:, node_idx] = out

            elif cnode.ctype == ConstraintType.NEGATION:
                if not cnode.child_indices or len(cnode.child_indices) != 1:
                    raise ValueError("NEGATION must have exactly one child index.")
                child_idx = cnode.child_indices[0]
                scores[:, node_idx] = 1.0 - scores[:, child_idx]

            elif cnode.ctype == ConstraintType.HAS_SELF_RESTRICTION:
                prop_idx = cnode.prop_idx
                reduce_index, next_nodes = oriented_edges(prop_idx, cnode.prop_direction)
                edge_values = (reduce_index == next_nodes).to(torch.float32)

                out = torch.zeros(num_nodes, device=device)
                out.scatter_reduce_(
                    dim=0,
                    index=reduce_index,
                    src=edge_values,
                    reduce="amax",
                    include_self=True,
                )
                if is_reflexive_property(prop_idx):
                    out = torch.ones(num_nodes, device=device)
                scores[:, node_idx] = out

            elif cnode.ctype == ConstraintType.EXISTS_RESTRICTION:
                if not cnode.child_indices or len(cnode.child_indices) != 1:
                    raise ValueError("EXISTS_RESTRICTION must have exactly one child index.")

                child_idx = cnode.child_indices[0]
                child_scores = scores[:, child_idx]
                prop_idx = cnode.prop_idx
                reduce_index, next_nodes = oriented_edges(prop_idx, cnode.prop_direction)
                edge_values = child_scores[next_nodes]

                out = torch.zeros(num_nodes, device=device)
                out.scatter_reduce_(
                    dim=0,
                    index=reduce_index,
                    src=edge_values,
                    reduce="amax",
                    include_self=True,
                )
                if is_reflexive_property(prop_idx):
                    out = torch.maximum(out, child_scores)
                scores[:, node_idx] = out

            elif cnode.ctype == ConstraintType.EXISTS_TRANSITIVE_RESTRICTION:
                if not cnode.child_indices or len(cnode.child_indices) != 1:
                    raise ValueError(
                        "EXISTS_TRANSITIVE_RESTRICTION must have exactly one child index."
                    )

                child_idx = cnode.child_indices[0]
                child_scores = scores[:, child_idx]
                prop_idx = cnode.prop_idx
                family_prop_indices = transitive_family_props(prop_idx)

                def one_hop(values: torch.Tensor) -> torch.Tensor:
                    propagated = torch.zeros(num_nodes, device=device)
                    for family_prop_idx in family_prop_indices:
                        reduce_index, next_nodes = oriented_edges(
                            family_prop_idx,
                            cnode.prop_direction,
                        )
                        edge_values = values[next_nodes]
                        propagated.scatter_reduce_(
                            dim=0,
                            index=reduce_index,
                            src=edge_values,
                            reduce="amax",
                            include_self=True,
                        )
                    return propagated

                seen = one_hop(child_scores)
                if is_reflexive_property(prop_idx):
                    seen = torch.maximum(seen, child_scores)
                frontier = seen.clone()

                while True:
                    propagated = one_hop(frontier)
                    new_frontier = torch.where(
                        propagated > seen,
                        propagated,
                        torch.zeros_like(propagated),
                    )
                    if float(new_frontier.max().item()) <= 0.0:
                        break
                    seen = torch.maximum(seen, new_frontier)
                    frontier = new_frontier

                scores[:, node_idx] = seen

            elif cnode.ctype in {
                ConstraintType.MIN_CARDINALITY_RESTRICTION,
                ConstraintType.MAX_CARDINALITY_RESTRICTION,
                ConstraintType.EXACT_CARDINALITY_RESTRICTION,
            }:
                if cnode.cardinality_agg != CardinalityAgg.STRICT:
                    raise NotImplementedError(
                        "Fuzzy cardinality operators are currently disabled."
                    )

                prop_idx = cnode.prop_idx
                target = cnode.cardinality_target
                if target is None:
                    raise ValueError(f"{cnode.ctype.name} requires cardinality_target.")

                _reduce_index, next_nodes = oriented_edges(prop_idx, cnode.prop_direction)

                if not cnode.child_indices or len(cnode.child_indices) == 0:
                    # The property-similarity term is currently an implicit 1.0.
                    edge_scores = torch.ones_like(next_nodes, dtype=torch.float32, device=device)
                elif len(cnode.child_indices) == 1:
                    child_scores = scores[:, cnode.child_indices[0]]
                    edge_scores = child_scores[next_nodes].to(torch.float32)
                else:
                    raise ValueError(f"{cnode.ctype.name} must have at most one child index.")

                def at_least_score(k: int) -> torch.Tensor:
                    if k <= 0:
                        return torch.ones(num_nodes, dtype=torch.float32, device=device)
                    top_values = topk_edge_scores(prop_idx, cnode.prop_direction, edge_scores, k)
                    return top_values[:, -1]

                at_least_k = at_least_score(target)
                at_most_k = 1.0 - at_least_score(target + 1)

                if cnode.ctype == ConstraintType.MIN_CARDINALITY_RESTRICTION:
                    out = at_least_k
                elif cnode.ctype == ConstraintType.MAX_CARDINALITY_RESTRICTION:
                    out = at_most_k
                else:
                    out = torch.minimum(at_least_k, at_most_k)

                scores[:, node_idx] = torch.clamp(out, min=0.0, max=1.0)

            elif cnode.ctype == ConstraintType.FORALL_RESTRICTION:
                if not cnode.child_indices or len(cnode.child_indices) != 1:
                    raise ValueError("FORALL_RESTRICTION must have exactly one child index.")

                child_idx = cnode.child_indices[0]
                child_scores = scores[:, child_idx]
                prop_idx = cnode.prop_idx
                reduce_index, next_nodes = oriented_edges(prop_idx, cnode.prop_direction)
                edge_values = child_scores[next_nodes]

                out = torch.full((num_nodes,), float("inf"), device=device)
                out.scatter_reduce_(
                    dim=0,
                    index=reduce_index,
                    src=edge_values,
                    reduce="amin",
                    include_self=True,
                )
                out = torch.where(torch.isinf(out), torch.ones_like(out), out)
                if is_reflexive_property(prop_idx):
                    out = torch.minimum(out, child_scores)
                scores[:, node_idx] = out

            elif cnode.ctype == ConstraintType.INTERSECTION:
                if not cnode.child_indices or len(cnode.child_indices) < 2:
                    raise ValueError("INTERSECTION must have at least two child indices.")

                child_tensors = [scores[:, ci] for ci in cnode.child_indices]
                stacked = torch.stack(child_tensors, dim=1)
                agg_mode = cnode.intersection_agg or IntersectionAgg.MIN

                if agg_mode == IntersectionAgg.MIN:
                    scores[:, node_idx] = stacked.min(dim=1).values
                elif agg_mode == IntersectionAgg.MEAN:
                    scores[:, node_idx] = stacked.mean(dim=1)
                else:
                    raise ValueError(f"Unknown intersection agg mode: {agg_mode}")

            elif cnode.ctype == ConstraintType.UNION:
                if not cnode.child_indices or len(cnode.child_indices) < 2:
                    raise ValueError("UNION must have at least two child indices.")

                child_tensors = [scores[:, ci] for ci in cnode.child_indices]
                stacked = torch.stack(child_tensors, dim=1)
                scores[:, node_idx] = stacked.max(dim=1).values

            elif cnode.ctype == ConstraintType.PATH_STEP:
                prop_idx = cnode.prop_idx
                required_class = cnode.class_idx
                direction = cnode.prop_direction

                if not cnode.child_indices or len(cnode.child_indices) == 0:
                    child_scores = torch.zeros(num_nodes, device=device)
                elif len(cnode.child_indices) == 1:
                    child_scores = scores[:, cnode.child_indices[0]]
                else:
                    raise ValueError("PATH_STEP must have at most one child index.")

                reduce_index, next_nodes = oriented_edges(prop_idx, direction)
                local_match = node_types[next_nodes, required_class]
                edge_values = local_match + child_scores[next_nodes]

                out = torch.zeros(num_nodes, device=device)
                out.scatter_reduce_(
                    dim=0,
                    index=reduce_index,
                    src=edge_values,
                    reduce="amax",
                    include_self=True,
                )
                scores[:, node_idx] = out

            else:
                raise NotImplementedError(f"Constraint type {cnode.ctype} not supported yet.")

            if cnode.scale_factor is not None:
                scores[:, node_idx] *= float(cnode.scale_factor)

    return scores


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

    return eval_dag_score_matrix(
        graph,
        dag,
        device=device,
        sim_class=sim_class,
    )[:, dag.root_idx]
