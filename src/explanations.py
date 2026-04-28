from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from rdflib import URIRef
from rdflib.term import Identifier

from .constraints import ConstraintDAG, ConstraintNode, ConstraintType, TraversalDirection
from .dag_eval import eval_dag_score_matrix
from .ontology_parse import (
    RDFKGraphMapping,
    ReasoningDataset,
    compile_class_to_dag,
    load_reasoning_dataset,
    plan_reasoning_preprocessing,
)

_TIE_EPSILON = 1e-6
_MAX_TIED_CHILDREN = 5


@dataclass
class QueryExplanation:
    target_class: URIRef
    node_term: Identifier
    score: float
    text: str


def _render_term(term: Identifier) -> str:
    return term.n3() if hasattr(term, "n3") else str(term)


def _oriented_neighbors(
    dataset: ReasoningDataset,
    *,
    node_idx: int,
    prop_idx: int,
    direction: TraversalDirection,
) -> List[int]:
    if direction == TraversalDirection.FORWARD:
        offsets = dataset.kg.offsets_p[prop_idx]
        neighbors = dataset.kg.neighbors_p[prop_idx]
        start = int(offsets[node_idx].item())
        end = int(offsets[node_idx + 1].item())
        return [int(neighbors[pos].item()) for pos in range(start, end)]

    reverse_neighbors: List[int] = []
    offsets = dataset.kg.offsets_p[prop_idx]
    neighbors = dataset.kg.neighbors_p[prop_idx]
    for src_idx in range(dataset.kg.num_nodes):
        start = int(offsets[src_idx].item())
        end = int(offsets[src_idx + 1].item())
        for pos in range(start, end):
            if int(neighbors[pos].item()) == node_idx:
                reverse_neighbors.append(src_idx)
    return reverse_neighbors


def _find_transitive_witness(
    dataset: ReasoningDataset,
    *,
    start_idx: int,
    prop_idx: int,
    direction: TraversalDirection,
    child_scores: torch.Tensor,
) -> Optional[Tuple[List[int], float]]:
    queue: deque[int] = deque([start_idx])
    parent: Dict[int, Optional[int]] = {start_idx: None}
    best_target: Optional[int] = None
    best_score = 0.0

    while queue:
        current = queue.popleft()
        for neighbor in _oriented_neighbors(
            dataset,
            node_idx=current,
            prop_idx=prop_idx,
            direction=direction,
        ):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
            candidate_score = float(child_scores[neighbor].item())
            if candidate_score > best_score:
                best_score = candidate_score
                best_target = neighbor

    if best_target is None or best_score <= 0.0:
        return None

    path: List[int] = [best_target]
    cursor = best_target
    while parent[cursor] is not None:
        cursor = parent[cursor]
        path.append(cursor)
    path.reverse()
    return path, best_score


def _indent(lines: Iterable[str], prefix: str) -> List[str]:
    return [prefix + line for line in lines]


def _tied_child_indices(
    child_scores: Sequence[Tuple[int, float]],
    *,
    pick_max: bool,
) -> List[Tuple[int, float]]:
    if not child_scores:
        return []
    target_value = max(score for _idx, score in child_scores) if pick_max else min(
        score for _idx, score in child_scores
    )
    return [
        (child_idx, child_score)
        for child_idx, child_score in child_scores
        if abs(child_score - target_value) <= _TIE_EPSILON
    ]


def _explain_node(
    dataset: ReasoningDataset,
    dag: ConstraintDAG,
    score_matrix: torch.Tensor,
    *,
    graph_node_idx: int,
    dag_node_idx: int,
    indent: str,
    visited: Set[Tuple[int, int]],
) -> List[str]:
    key = (graph_node_idx, dag_node_idx)
    if key in visited:
        return [f"{indent}- recursion stop on DAG node {dag_node_idx} at {_render_term(dataset.mapping.node_terms[graph_node_idx])}"]

    visited = set(visited)
    visited.add(key)

    cnode = dag.nodes[dag_node_idx]
    node_term = dataset.mapping.node_terms[graph_node_idx]
    score = float(score_matrix[graph_node_idx, dag_node_idx].item())
    header = (
        f"{indent}- DAG node {dag_node_idx} {cnode.ctype.name} "
        f"at {_render_term(node_term)} => {score:.4f}"
    )
    lines = [header]

    if cnode.ctype == ConstraintType.CONST:
        lines.append(f"{indent}  constant 1.0")
        return lines

    if cnode.ctype == ConstraintType.ATOMIC_CLASS:
        class_term = dataset.mapping.class_terms[cnode.class_idx]
        has_type = bool(dataset.kg.node_types[graph_node_idx, cnode.class_idx].item() > 0.0)
        lines.append(
            f"{indent}  requires atomic class {_render_term(class_term)}; "
            f"node_types match={'yes' if has_type else 'no'}"
        )
        return lines

    if cnode.ctype == ConstraintType.NOMINAL:
        nominal_term = dataset.mapping.node_terms[cnode.node_idx]
        lines.append(
            f"{indent}  exact nominal match against {_render_term(nominal_term)}"
        )
        return lines

    if cnode.ctype == ConstraintType.NEGATION:
        child_idx = cnode.child_indices[0]
        child_score = float(score_matrix[graph_node_idx, child_idx].item())
        lines.append(f"{indent}  negates child score {child_score:.4f}")
        lines.extend(
            _explain_node(
                dataset,
                dag,
                score_matrix,
                graph_node_idx=graph_node_idx,
                dag_node_idx=child_idx,
                indent=indent + "    ",
                visited=visited,
            )
        )
        return lines

    if cnode.ctype in (ConstraintType.INTERSECTION, ConstraintType.UNION):
        child_scores = [
            (child_idx, float(score_matrix[graph_node_idx, child_idx].item()))
            for child_idx in (cnode.child_indices or [])
        ]
        child_scores.sort(key=lambda item: (item[1], item[0]))
        label = "minimum" if cnode.ctype == ConstraintType.INTERSECTION else "maximum"
        lines.append(
            f"{indent}  aggregate uses {label} over child scores: "
            + ", ".join(f"{child_idx}={child_score:.4f}" for child_idx, child_score in child_scores)
        )
        tied_children = _tied_child_indices(
            child_scores,
            pick_max=(cnode.ctype == ConstraintType.UNION),
        )
        if tied_children:
            lines.append(
                f"{indent}  tied decisive children: "
                + ", ".join(f"{child_idx}={child_score:.4f}" for child_idx, child_score in tied_children[:_MAX_TIED_CHILDREN])
            )
            if len(tied_children) > _MAX_TIED_CHILDREN:
                lines.append(
                    f"{indent}  ... plus {len(tied_children) - _MAX_TIED_CHILDREN} more tied child(ren)"
                )
            for chosen_child, _chosen_score in tied_children[:_MAX_TIED_CHILDREN]:
                lines.extend(
                    _explain_node(
                        dataset,
                        dag,
                        score_matrix,
                        graph_node_idx=graph_node_idx,
                        dag_node_idx=chosen_child,
                        indent=indent + "    ",
                        visited=visited,
                    )
                )
        return lines

    if cnode.ctype in (
        ConstraintType.HAS_SELF_RESTRICTION,
        ConstraintType.EXISTS_RESTRICTION,
        ConstraintType.FORALL_RESTRICTION,
        ConstraintType.MIN_CARDINALITY_RESTRICTION,
        ConstraintType.MAX_CARDINALITY_RESTRICTION,
        ConstraintType.EXACT_CARDINALITY_RESTRICTION,
        ConstraintType.PATH_STEP,
        ConstraintType.EXISTS_TRANSITIVE_RESTRICTION,
    ):
        prop_term = dataset.mapping.prop_terms[cnode.prop_idx]
        neighbor_indices = _oriented_neighbors(
            dataset,
            node_idx=graph_node_idx,
            prop_idx=cnode.prop_idx,
            direction=cnode.prop_direction,
        )
        child_idx = cnode.child_indices[0] if cnode.child_indices else None
        child_scores = (
            score_matrix[:, child_idx]
            if child_idx is not None
            else torch.ones((dataset.kg.num_nodes,), dtype=torch.float32)
        )

        if cnode.ctype == ConstraintType.HAS_SELF_RESTRICTION:
            has_self_edge = graph_node_idx in neighbor_indices
            lines.append(
                f"{indent}  property {_render_term(prop_term)} ({cnode.prop_direction.value}) "
                f"{'has' if has_self_edge else 'does not have'} a self-edge at "
                f"{_render_term(dataset.mapping.node_terms[graph_node_idx])}"
            )
            return lines

        if cnode.ctype == ConstraintType.EXISTS_TRANSITIVE_RESTRICTION:
            witness = _find_transitive_witness(
                dataset,
                start_idx=graph_node_idx,
                prop_idx=cnode.prop_idx,
                direction=cnode.prop_direction,
                child_scores=child_scores,
            )
            lines.append(
                f"{indent}  transitive restriction over {_render_term(prop_term)} "
                f"({cnode.prop_direction.value})"
            )
            if witness is None:
                lines.append(f"{indent}  no reachable witness with positive child score")
                return lines
            path, witness_score = witness
            rendered_path = " -> ".join(_render_term(dataset.mapping.node_terms[idx]) for idx in path)
            lines.append(f"{indent}  witness path: {rendered_path} (child score {witness_score:.4f})")
            if child_idx is not None:
                lines.extend(
                    _explain_node(
                        dataset,
                        dag,
                        score_matrix,
                        graph_node_idx=path[-1],
                        dag_node_idx=child_idx,
                        indent=indent + "    ",
                        visited=visited,
                    )
                )
            return lines

        scored_neighbors = [
            (neighbor_idx, float(child_scores[neighbor_idx].item()))
            for neighbor_idx in neighbor_indices
        ]
        scored_neighbors.sort(key=lambda item: (item[1], _render_term(dataset.mapping.node_terms[item[0]])))

        lines.append(
            f"{indent}  property {_render_term(prop_term)} ({cnode.prop_direction.value}) "
            f"has {len(scored_neighbors)} immediate neighbor(s)"
        )
        if scored_neighbors:
            lines.append(
                f"{indent}  neighbor child scores: "
                + ", ".join(
                    f"{_render_term(dataset.mapping.node_terms[idx])}={neighbor_score:.4f}"
                    for idx, neighbor_score in scored_neighbors[:6]
                )
            )

        if cnode.ctype == ConstraintType.EXISTS_RESTRICTION:
            if scored_neighbors:
                witness_idx, witness_score = scored_neighbors[-1]
                lines.append(
                    f"{indent}  best witness is {_render_term(dataset.mapping.node_terms[witness_idx])} "
                    f"with child score {witness_score:.4f}"
                )
                if child_idx is not None:
                    lines.extend(
                        _explain_node(
                            dataset,
                            dag,
                            score_matrix,
                            graph_node_idx=witness_idx,
                            dag_node_idx=child_idx,
                            indent=indent + "    ",
                            visited=visited,
                        )
                    )
            return lines

        if cnode.ctype == ConstraintType.FORALL_RESTRICTION:
            if scored_neighbors:
                worst_idx, worst_score = scored_neighbors[0]
                lines.append(
                    f"{indent}  worst neighbor is {_render_term(dataset.mapping.node_terms[worst_idx])} "
                    f"with child score {worst_score:.4f}"
                )
                if child_idx is not None:
                    lines.extend(
                        _explain_node(
                            dataset,
                            dag,
                            score_matrix,
                            graph_node_idx=worst_idx,
                            dag_node_idx=child_idx,
                            indent=indent + "    ",
                            visited=visited,
                        )
                    )
            else:
                lines.append(f"{indent}  vacuous truth: there are no neighbors on this property")
            return lines

        if cnode.ctype in (
            ConstraintType.MIN_CARDINALITY_RESTRICTION,
            ConstraintType.MAX_CARDINALITY_RESTRICTION,
            ConstraintType.EXACT_CARDINALITY_RESTRICTION,
        ):
            target = cnode.cardinality_target
            descending = sorted(scored_neighbors, key=lambda item: item[1], reverse=True)
            kth = descending[target - 1][1] if target and len(descending) >= target else 0.0
            kp1 = descending[target][1] if target is not None and len(descending) >= target + 1 else 0.0
            lines.append(
                f"{indent}  top-k cardinality view with target={target}: "
                f"kth={kth:.4f}, (k+1)th={kp1:.4f}"
            )
            return lines

        if cnode.ctype == ConstraintType.PATH_STEP:
            lines.append(f"{indent}  typed path step explanation currently reports only local neighbor scores")
            return lines

    if cnode.ctype == ConstraintType.DATATYPE_CONSTRAINT:
        datatype_term = (
            dataset.mapping.datatype_terms[cnode.datatype_idx]
            if cnode.datatype_idx is not None
            else None
        )
        dtype_parts: List[str] = []
        if datatype_term is not None:
            dtype_parts.append(f"datatype={_render_term(datatype_term)}")
        if dataset.kg.literal_numeric_value is not None:
            numeric_value = float(dataset.kg.literal_numeric_value[graph_node_idx].item())
            if not torch.isnan(torch.tensor(numeric_value)):
                dtype_parts.append(f"value={numeric_value}")
        lines.append(f"{indent}  literal check: " + ", ".join(dtype_parts))
        return lines

    lines.append(f"{indent}  explanation detail not implemented for this node type")
    return lines


def explain_dataset_query(
    dataset: ReasoningDataset,
    *,
    target_class: str | URIRef,
    node_term: str | Identifier,
    augment_property_domain_range: bool = False,
    device: str = "cpu",
) -> QueryExplanation:
    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    query_plan = plan_reasoning_preprocessing(
        dataset.ontology_graph,
        target_classes=[target_term],
        augment_property_domain_range=augment_property_domain_range,
    )
    lookup = {_render_term(term): idx for idx, term in enumerate(dataset.mapping.node_terms)}
    if isinstance(node_term, Identifier) and node_term in dataset.mapping.node_terms:
        try:
            graph_node_idx = dataset.mapping.node_terms.index(node_term)
        except ValueError as exc:
            raise KeyError(f"Node term not found in dataset mapping: {node_term}") from exc
        resolved_node_term = node_term
    elif isinstance(node_term, str):
        if node_term not in lookup:
            raise KeyError(f"Node term not found in dataset mapping: {node_term}")
        graph_node_idx = lookup[node_term]
        resolved_node_term = dataset.mapping.node_terms[graph_node_idx]
    else:
        raise KeyError(f"Node term not found in dataset mapping: {node_term}")

    dag = compile_class_to_dag(
        dataset.ontology_graph,
        dataset.mapping,
        target_term,
        augment_property_domain_range=query_plan.augment_property_domain_range.enabled,
    )
    device_to_use = device
    if device_to_use == "cuda" and not torch.cuda.is_available():
        device_to_use = "cpu"
    score_matrix = eval_dag_score_matrix(dataset.kg, dag, device=device_to_use).detach().cpu()
    score = float(score_matrix[graph_node_idx, dag.root_idx].item())

    header = [
        f"Explanation for target {_render_term(target_term)} at node {_render_term(resolved_node_term)}",
        f"Final score: {score:.4f}",
        "",
    ]
    lines = header + _explain_node(
        dataset,
        dag,
        score_matrix,
        graph_node_idx=graph_node_idx,
        dag_node_idx=dag.root_idx,
        indent="",
        visited=set(),
    )
    return QueryExplanation(
        target_class=target_term,
        node_term=resolved_node_term,
        score=score,
        text="\n".join(lines),
    )


def explain_loaded_query(
    *,
    schema_paths: Sequence[str],
    data_paths: Sequence[str],
    target_class: str,
    node_term: str,
    include_literals: bool = False,
    include_type_edges: bool = False,
    materialize_hierarchy: bool = True,
    materialize_atomic_domain_range: bool = False,
    materialize_horn_safe_domain_range: bool = False,
    materialize_target_roles: bool = False,
    augment_property_domain_range: bool = False,
    device: str = "cpu",
) -> QueryExplanation:
    dataset = load_reasoning_dataset(
        schema_paths=schema_paths,
        data_paths=data_paths,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_atomic_domain_range=materialize_atomic_domain_range,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_target_roles=materialize_target_roles,
        target_classes=[target_class],
    )
    return explain_dataset_query(
        dataset,
        target_class=target_class,
        node_term=node_term,
        augment_property_domain_range=augment_property_domain_range,
        device=device,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Explain a single node/class score by reconstructing local witnesses from the compiled DAG."
    )
    parser.add_argument("--schema", nargs="+", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--node", required=True, help="Node IRI / rendered term to explain.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--include-literals", action="store_true")
    parser.add_argument("--include-type-edges", action="store_true")
    hierarchy_group = parser.add_mutually_exclusive_group()
    hierarchy_group.add_argument("--materialize-hierarchy", dest="materialize_hierarchy", action="store_true")
    hierarchy_group.add_argument("--no-materialize-hierarchy", dest="materialize_hierarchy", action="store_false")
    roles_group = parser.add_mutually_exclusive_group()
    roles_group.add_argument("--materialize-target-roles", dest="materialize_target_roles", action="store_true")
    roles_group.add_argument("--no-materialize-target-roles", dest="materialize_target_roles", action="store_false")
    atomic_domain_group = parser.add_mutually_exclusive_group()
    atomic_domain_group.add_argument("--materialize-domain-range-atomic", dest="materialize_domain_range_atomic", action="store_true")
    atomic_domain_group.add_argument("--no-materialize-domain-range-atomic", dest="materialize_domain_range_atomic", action="store_false")
    horn_domain_group = parser.add_mutually_exclusive_group()
    horn_domain_group.add_argument("--materialize-domain-range-horn", dest="materialize_domain_range_horn", action="store_true")
    horn_domain_group.add_argument("--no-materialize-domain-range-horn", dest="materialize_domain_range_horn", action="store_false")
    augment_group = parser.add_mutually_exclusive_group()
    augment_group.add_argument("--augment-domain-range", dest="augment_domain_range", action="store_true")
    augment_group.add_argument("--no-augment-domain-range", dest="augment_domain_range", action="store_false")
    parser.set_defaults(
        materialize_hierarchy=None,
        materialize_target_roles=None,
        materialize_domain_range_atomic=None,
        materialize_domain_range_horn=None,
        augment_domain_range=None,
    )
    args = parser.parse_args()

    explanation = explain_loaded_query(
        schema_paths=args.schema,
        data_paths=args.data,
        target_class=args.target_class,
        node_term=args.node,
        device=args.device,
        include_literals=args.include_literals,
        include_type_edges=args.include_type_edges,
        materialize_hierarchy=args.materialize_hierarchy,
        materialize_atomic_domain_range=args.materialize_domain_range_atomic,
        materialize_horn_safe_domain_range=args.materialize_domain_range_horn,
        materialize_target_roles=args.materialize_target_roles,
        augment_property_domain_range=args.augment_domain_range,
    )
    print(explanation.text)


if __name__ == "__main__":
    main()
