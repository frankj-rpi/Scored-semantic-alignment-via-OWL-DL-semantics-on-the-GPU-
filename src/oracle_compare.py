from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
import shutil
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import contextlib
import io
import os
import subprocess
import tempfile
import uuid

import torch
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.term import Identifier

from .constraints import ConstraintDAG, ConstraintType
from .dag_reasoner import DAGReasoner
from .ontology_parse import (
    MaterializationIterationTiming,
    NamedClassDependencyAnalysis,
    OntologyCompileContext,
    PreprocessingTimings,
    analyze_named_class_dependencies,
    build_ontology_compile_context,
    build_rdflib_mapping,
    ConflictPolicy,
    ReasoningDataset,
    StratifiedMaterializationResult,
    build_reasoning_dataset_from_graphs,
    collect_referenced_named_classes_for_class,
    get_named_class_dependency_cycle_component,
    collect_inferable_named_classes,
    collect_named_class_terms,
    compile_class_to_dag,
    compile_sufficient_condition_dag,
    load_rdflib_graph,
    materialize_stratified_class_inferences,
    materialize_supported_class_inferences,
    plan_reasoning_preprocessing,
    query_target_is_obviously_supported,
    summarize_loaded_kgraph,
)


DEFAULT_OWLAPI_HOME = Path(__file__).resolve().parents[1] / "comparison" / "owlapi-5.5.1"


ORACLE_NS = Namespace("urn:dag-oracle:")


@dataclass
class BackendQueryResult:
    backend: str
    elapsed_ms: float
    members_by_target: Dict[URIRef, Set[Identifier]]
    status: str = "ok"
    consistent: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class EngineQueryResult(BackendQueryResult):
    dataset: Optional[ReasoningDataset] = None
    scores_by_target: Optional[Dict[URIRef, Dict[Identifier, float]]] = None
    materialization_iterations: Optional[int] = None
    engine_mode: str = "query"
    conflict_policy: Optional[str] = None
    stratified_result: Optional[StratifiedMaterializationResult] = None
    filtered_query_result: Optional["FilteredQueryResult"] = None
    ontology_merge_elapsed_ms: float = 0.0
    stratified_initial_data_copy_elapsed_ms: float = 0.0
    stratified_initial_ontology_merge_elapsed_ms: float = 0.0
    schema_cache_elapsed_ms: float = 0.0
    preprocessing_plan_elapsed_ms: float = 0.0
    sufficient_rule_extraction_elapsed_ms: float = 0.0
    sufficient_rule_index_elapsed_ms: float = 0.0
    dependency_closure_elapsed_ms: float = 0.0
    sameas_state_init_elapsed_ms: float = 0.0
    dataset_build_elapsed_ms: float = 0.0
    hierarchy_elapsed_ms: float = 0.0
    atomic_domain_range_elapsed_ms: float = 0.0
    horn_safe_domain_range_elapsed_ms: float = 0.0
    sameas_elapsed_ms: float = 0.0
    reflexive_elapsed_ms: float = 0.0
    target_role_elapsed_ms: float = 0.0
    kgraph_build_elapsed_ms: float = 0.0
    mapping_vocab_collect_elapsed_ms: float = 0.0
    mapping_graph_scan_elapsed_ms: float = 0.0
    mapping_sort_elapsed_ms: float = 0.0
    mapping_index_elapsed_ms: float = 0.0
    kgraph_edge_bucket_elapsed_ms: float = 0.0
    kgraph_negative_helper_elapsed_ms: float = 0.0
    kgraph_literal_feature_elapsed_ms: float = 0.0
    kgraph_adjacency_elapsed_ms: float = 0.0
    sameas_passes_elapsed_ms: Optional[List[float]] = None
    stratified_positive_reasoner_setup_elapsed_ms: float = 0.0
    dag_compile_elapsed_ms: float = 0.0
    dag_eval_elapsed_ms: float = 0.0
    stratified_positive_assertion_update_elapsed_ms: float = 0.0
    stratified_positive_total_elapsed_ms: float = 0.0
    stratified_positive_iterations: int = 0
    stratified_positive_avg_dataset_build_elapsed_ms: float = 0.0
    stratified_positive_avg_dag_compile_elapsed_ms: float = 0.0
    stratified_positive_avg_dag_eval_elapsed_ms: float = 0.0
    stratified_negative_blocker_elapsed_ms: float = 0.0
    stratified_assignment_status_elapsed_ms: float = 0.0
    stratified_conflict_policy_elapsed_ms: float = 0.0
    stratified_reporting_compile_elapsed_ms: float = 0.0
    stratified_iteration_timings: Optional[List[MaterializationIterationTiming]] = None
    stratified_final_dataset_timing: Optional[PreprocessingTimings] = None


@dataclass
class QueryEvaluationSnapshot:
    dataset: ReasoningDataset
    members_by_target: Dict[URIRef, Set[Identifier]]
    scores_by_target: Dict[URIRef, Dict[Identifier, float]]
    dataset_build_elapsed_ms: float
    hierarchy_elapsed_ms: float
    atomic_domain_range_elapsed_ms: float
    horn_safe_domain_range_elapsed_ms: float
    sameas_elapsed_ms: float
    reflexive_elapsed_ms: float
    target_role_elapsed_ms: float
    kgraph_build_elapsed_ms: float
    mapping_vocab_collect_elapsed_ms: float
    mapping_graph_scan_elapsed_ms: float
    mapping_sort_elapsed_ms: float
    mapping_index_elapsed_ms: float
    kgraph_edge_bucket_elapsed_ms: float
    kgraph_negative_helper_elapsed_ms: float
    kgraph_literal_feature_elapsed_ms: float
    kgraph_adjacency_elapsed_ms: float
    dag_compile_elapsed_ms: float
    dag_eval_elapsed_ms: float


@dataclass
class FilteredQueryResult:
    raw_members_by_target: Dict[URIRef, Set[Identifier]]
    necessary_stable_members_by_target: Dict[URIRef, Set[Identifier]]
    closure_blocked_members_by_target: Dict[URIRef, Set[Identifier]]
    final_members_by_target: Dict[URIRef, Set[Identifier]]
    raw_candidate_count: int
    necessary_retraction_count: int
    closure_blocked_retraction_count: int
    final_emitted_count: int
    necessary_fixpoint_iterations: int
    stratified_result: StratifiedMaterializationResult


@dataclass
class TargetResolutionResult:
    requested_specs: Tuple[str, ...]
    resolved_targets: List[URIRef]
    skipped_targets: List[Tuple[str, str]]


def _render_term(term: Identifier) -> str:
    return term.n3() if hasattr(term, "n3") else str(term)


def _sorted_terms(terms: Iterable[Identifier]) -> List[Identifier]:
    return sorted(terms, key=lambda term: _render_term(term))


def _copy_graph(graph: Graph) -> Graph:
    copied = Graph()
    for triple in graph:
        copied.add(triple)
    return copied


def _clone_members_by_target(
    members_by_target: Dict[URIRef, Set[Identifier]],
) -> Dict[URIRef, Set[Identifier]]:
    return {
        target_term: set(members)
        for target_term, members in members_by_target.items()
    }


def _add_type_assignments(
    data_graph: Graph,
    members_by_target: Dict[URIRef, Set[Identifier]],
) -> Graph:
    augmented = _copy_graph(data_graph)
    for target_term, members in members_by_target.items():
        for node_term in members:
            augmented.add((node_term, RDF.type, target_term))
    return augmented


def _count_members_by_target(
    members_by_target: Dict[URIRef, Set[Identifier]],
) -> int:
    return sum(len(members) for members in members_by_target.values())


def _build_binary_scores(
    node_terms: Sequence[Identifier],
    target_classes: Sequence[URIRef],
    members_by_target: Dict[URIRef, Set[Identifier]],
) -> Dict[URIRef, Dict[Identifier, float]]:
    scores_by_target: Dict[URIRef, Dict[Identifier, float]] = {}
    for target_term in target_classes:
        members = members_by_target.get(target_term, set())
        scores_by_target[target_term] = {
            node_term: (1.0 if node_term in members else 0.0)
            for node_term in node_terms
        }
    return scores_by_target


def format_engine_timing_breakdown(engine_result: EngineQueryResult) -> str:
    lines = ["Engine timing breakdown:"]
    if (
        engine_result.schema_cache_elapsed_ms
        or engine_result.preprocessing_plan_elapsed_ms
        or engine_result.sufficient_rule_extraction_elapsed_ms
        or engine_result.sufficient_rule_index_elapsed_ms
        or engine_result.dependency_closure_elapsed_ms
        or engine_result.sameas_state_init_elapsed_ms
    ):
        lines.extend(
            [
                (
                    "  - stage 1 schema cache extraction: "
                    f"{engine_result.schema_cache_elapsed_ms:.3f} ms"
                ),
                (
                    "  - stage 2 preprocessing plan selection: "
                    f"{engine_result.preprocessing_plan_elapsed_ms:.3f} ms"
                ),
                (
                    "  - stage 3 initial graph prep: "
                    f"data copy={engine_result.stratified_initial_data_copy_elapsed_ms:.3f} ms, "
                    f"ontology merge={engine_result.stratified_initial_ontology_merge_elapsed_ms:.3f} ms"
                ),
                (
                    "  - stage 5 target/dependency prep: "
                    f"rule extraction={engine_result.sufficient_rule_extraction_elapsed_ms:.3f} ms, "
                    f"rule index={engine_result.sufficient_rule_index_elapsed_ms:.3f} ms, "
                    f"dependency closure={engine_result.dependency_closure_elapsed_ms:.3f} ms, "
                    f"sameAs state init={engine_result.sameas_state_init_elapsed_ms:.3f} ms"
                ),
            ]
        )
    lines.extend(
        [
            (
                "  - preprocessing/dataset build: "
                f"{engine_result.dataset_build_elapsed_ms:.3f} ms"
            ),
            (
                "      merge="
                f"{engine_result.ontology_merge_elapsed_ms:.3f} ms, "
                "hierarchy="
                f"{engine_result.hierarchy_elapsed_ms:.3f} ms, "
                "atomic domain/range="
                f"{engine_result.atomic_domain_range_elapsed_ms:.3f} ms, "
                "horn-safe domain/range="
                f"{engine_result.horn_safe_domain_range_elapsed_ms:.3f} ms, "
                f"sameAs={engine_result.sameas_elapsed_ms:.3f} ms, "
                f"reflexive={engine_result.reflexive_elapsed_ms:.3f} ms, "
                f"target roles={engine_result.target_role_elapsed_ms:.3f} ms, "
                f"kgraph build={engine_result.kgraph_build_elapsed_ms:.3f} ms"
            ),
            (
                "      mapping: vocab collect="
                f"{engine_result.mapping_vocab_collect_elapsed_ms:.3f} ms, "
                f"graph scan={engine_result.mapping_graph_scan_elapsed_ms:.3f} ms, "
                f"sort={engine_result.mapping_sort_elapsed_ms:.3f} ms, "
                f"index={engine_result.mapping_index_elapsed_ms:.3f} ms"
            ),
            (
                "      kgraph internals: edge buckets="
                f"{engine_result.kgraph_edge_bucket_elapsed_ms:.3f} ms, "
                f"negative helpers={engine_result.kgraph_negative_helper_elapsed_ms:.3f} ms, "
                f"literal features={engine_result.kgraph_literal_feature_elapsed_ms:.3f} ms, "
                f"adjacency={engine_result.kgraph_adjacency_elapsed_ms:.3f} ms"
            ),
            f"  - DAG compile: {engine_result.dag_compile_elapsed_ms:.3f} ms",
            f"  - DAG eval: {engine_result.dag_eval_elapsed_ms:.3f} ms",
        ]
    )
    if engine_result.sameas_passes_elapsed_ms:
        lines.append(
            "      sameAs passes: "
            + ", ".join(f"{elapsed_ms:.3f} ms" for elapsed_ms in engine_result.sameas_passes_elapsed_ms)
        )
    if engine_result.engine_mode == "stratified" and engine_result.stratified_positive_iterations:
        positive_accounted_elapsed_ms = (
            engine_result.stratified_initial_data_copy_elapsed_ms
            + engine_result.stratified_initial_ontology_merge_elapsed_ms
            + engine_result.dataset_build_elapsed_ms
            + engine_result.stratified_positive_reasoner_setup_elapsed_ms
            + engine_result.dag_compile_elapsed_ms
            + engine_result.dag_eval_elapsed_ms
            + engine_result.stratified_positive_assertion_update_elapsed_ms
        )
        positive_overhead_elapsed_ms = max(
            0.0,
            engine_result.stratified_positive_total_elapsed_ms - positive_accounted_elapsed_ms,
        )
        lines.extend(
            [
                (
                    "  - stratified positive OWA loop: "
                    f"{engine_result.stratified_positive_total_elapsed_ms:.3f} ms "
                    f"over {engine_result.stratified_positive_iterations} iterations"
                ),
                (
                    "      avg/iter: dataset build="
                    f"{engine_result.stratified_positive_avg_dataset_build_elapsed_ms:.3f} ms, "
                    "reasoner setup="
                    f"{(engine_result.stratified_positive_reasoner_setup_elapsed_ms / engine_result.stratified_positive_iterations if engine_result.stratified_positive_iterations else 0.0):.3f} ms, "
                    "dag compile="
                    f"{engine_result.stratified_positive_avg_dag_compile_elapsed_ms:.3f} ms, "
                    "dag eval="
                    f"{engine_result.stratified_positive_avg_dag_eval_elapsed_ms:.3f} ms, "
                    "assertion update="
                    f"{(engine_result.stratified_positive_assertion_update_elapsed_ms / engine_result.stratified_positive_iterations if engine_result.stratified_positive_iterations else 0.0):.3f} ms"
                ),
                (
                    "      one-off setup/overhead="
                    f"{positive_overhead_elapsed_ms:.3f} ms "
                    "(includes pre-loop rule/cache prep and loop-control work)"
                ),
            ]
        )
        for iteration_timing in engine_result.stratified_iteration_timings or ():
            lines.append(
                "      iter "
                f"{iteration_timing.iteration}: refreshes={iteration_timing.dataset_refresh_count}, "
                f"dataset build={iteration_timing.dataset_build_elapsed_ms:.3f} ms, "
                f"merge={iteration_timing.ontology_merge_elapsed_ms:.3f} ms, "
                f"hierarchy={iteration_timing.hierarchy_elapsed_ms:.3f} ms, "
                f"atomic={iteration_timing.atomic_domain_range_elapsed_ms:.3f} ms, "
                f"horn-safe={iteration_timing.horn_safe_domain_range_elapsed_ms:.3f} ms, "
                f"sameAs={iteration_timing.sameas_elapsed_ms:.3f} ms, "
                f"reflexive={iteration_timing.reflexive_elapsed_ms:.3f} ms, "
                f"target roles={iteration_timing.target_role_elapsed_ms:.3f} ms, "
                f"kgraph={iteration_timing.kgraph_build_elapsed_ms:.3f} ms, "
                f"reasoner setup={iteration_timing.reasoner_setup_elapsed_ms:.3f} ms, "
                f"dag compile={iteration_timing.dag_compile_elapsed_ms:.3f} ms, "
                f"dag eval={iteration_timing.dag_eval_elapsed_ms:.3f} ms, "
                f"assertion update={iteration_timing.assertion_update_elapsed_ms:.3f} ms"
            )
            if iteration_timing.sameas_passes_elapsed_ms:
                lines.append(
                    "          sameAs passes: "
                    + ", ".join(
                        f"{elapsed_ms:.3f} ms"
                        for elapsed_ms in iteration_timing.sameas_passes_elapsed_ms
                    )
                )
        if engine_result.stratified_final_dataset_timing is not None:
            final_timing = engine_result.stratified_final_dataset_timing
            lines.append(
                "      final dataset refresh: "
                f"dataset build={final_timing.dataset_build_elapsed_ms:.3f} ms, "
                f"merge={final_timing.ontology_merge_elapsed_ms:.3f} ms, "
                f"hierarchy={final_timing.hierarchy_elapsed_ms:.3f} ms, "
                f"atomic={final_timing.atomic_domain_range_elapsed_ms:.3f} ms, "
                f"horn-safe={final_timing.horn_safe_domain_range_elapsed_ms:.3f} ms, "
                f"sameAs={final_timing.sameas_elapsed_ms:.3f} ms, "
                f"reflexive={final_timing.reflexive_elapsed_ms:.3f} ms, "
                f"target roles={final_timing.target_role_elapsed_ms:.3f} ms, "
                f"kgraph={final_timing.kgraph_build_elapsed_ms:.3f} ms"
            )
            if final_timing.sameas_passes_elapsed_ms:
                lines.append(
                    "          sameAs passes: "
                    + ", ".join(
                        f"{elapsed_ms:.3f} ms"
                        for elapsed_ms in final_timing.sameas_passes_elapsed_ms
                    )
                )
        lines.extend(
            [
                (
                    "  - stratified negative/blocker: "
                    f"{engine_result.stratified_negative_blocker_elapsed_ms:.3f} ms"
                ),
                (
                    "  - stratified assignment status: "
                    f"{engine_result.stratified_assignment_status_elapsed_ms:.3f} ms"
                ),
                (
                    "  - stratified conflict policy: "
                    f"{engine_result.stratified_conflict_policy_elapsed_ms:.3f} ms"
                ),
                (
                    "  - stratified final reporting compile: "
                    f"{engine_result.stratified_reporting_compile_elapsed_ms:.3f} ms"
                ),
            ]
        )
    return "\n".join(lines)


def format_skipped_target_warnings(
    resolution: TargetResolutionResult,
    *,
    warning_limit: int = 5,
    verbose: bool = False,
) -> str:
    if not resolution.skipped_targets:
        return ""

    lines = [
        (
            f"Warning: skipped {len(resolution.skipped_targets)} target classes "
            "that did not compile cleanly."
        )
    ]
    display_count = len(resolution.skipped_targets) if verbose else min(warning_limit, len(resolution.skipped_targets))
    for target_text, reason in resolution.skipped_targets[:display_count]:
        lines.append(f"  - {target_text}: {reason}")
    if not verbose and len(resolution.skipped_targets) > display_count:
        lines.append(f"  ... and {len(resolution.skipped_targets) - display_count} others skipped")
    return "\n".join(lines)


def _dag_is_positive_monotone(dag: ConstraintDAG) -> bool:
    allowed_types = {
        ConstraintType.CONST,
        ConstraintType.ATOMIC_CLASS,
        ConstraintType.EXISTS_RESTRICTION,
        ConstraintType.EXISTS_TRANSITIVE_RESTRICTION,
        ConstraintType.FORALL_RESTRICTION,
        ConstraintType.INTERSECTION,
        ConstraintType.UNION,
        ConstraintType.NOMINAL,
        ConstraintType.DATATYPE_CONSTRAINT,
        ConstraintType.MIN_CARDINALITY_RESTRICTION,
        ConstraintType.HAS_SELF_RESTRICTION,
    }
    return all(node.ctype in allowed_types for node in dag.nodes)


def resolve_target_classes(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    target_class_specs: Sequence[str],
    engine_mode: str = "query",
    include_literals: bool = False,
    include_type_edges: bool = False,
    materialize_hierarchy: Optional[bool] = None,
    augment_property_domain_range: Optional[bool] = None,
) -> TargetResolutionResult:
    ontology_graph = _copy_graph(schema_graph)
    for triple in data_graph:
        ontology_graph.add(triple)

    requested_specs = tuple(target_class_specs)
    candidate_targets: Set[URIRef] = set()
    for spec in requested_specs:
        normalized = spec.strip()
        lowered = normalized.lower()
        if lowered in {"all", "all-named-classes"}:
            candidate_targets.update(collect_named_class_terms(ontology_graph))
        elif lowered in {"all-defined-classes", "all-inferable-classes"}:
            candidate_targets.update(collect_inferable_named_classes(ontology_graph))
        else:
            candidate_targets.add(URIRef(normalized))

    ordered_candidates = sorted(candidate_targets, key=str)
    if not ordered_candidates:
        return TargetResolutionResult(
            requested_specs=requested_specs,
            resolved_targets=[],
            skipped_targets=[],
        )

    mapping = build_rdflib_mapping(
        data_graph,
        vocab_source=ontology_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
    )
    compile_context = build_ontology_compile_context(ontology_graph)
    dependency_analysis = compile_context.dependency_analysis

    resolved_targets: List[URIRef] = []
    skipped_targets: List[Tuple[str, str]] = []
    for target_term in ordered_candidates:
        try:
            if engine_mode == "stratified":
                compile_sufficient_condition_dag(
                    ontology_graph,
                    mapping,
                    target_term,
                )
            else:
                query_plan = plan_reasoning_preprocessing(
                    ontology_graph,
                    target_classes=[target_term],
                    augment_property_domain_range=augment_property_domain_range,
                )
                target_canonical = dependency_analysis.canonical_map.get(target_term, target_term)
                has_reachable_cycle = dependency_analysis.reaches_cycle_by_class.get(target_canonical, False)
                has_nonmonotone_cycle = False
                if has_reachable_cycle:
                    referenced_classes = collect_referenced_named_classes_for_class(
                        ontology_graph,
                        target_term,
                        dependency_analysis=dependency_analysis,
                    )
                    for referenced_class in referenced_classes:
                        cycle_component = get_named_class_dependency_cycle_component(
                            ontology_graph,
                            referenced_class,
                            dependency_analysis=dependency_analysis,
                        )
                        if not cycle_component:
                            continue
                        cycle_query_plan = plan_reasoning_preprocessing(
                            ontology_graph,
                            target_classes=[referenced_class],
                            augment_property_domain_range=augment_property_domain_range,
                        )
                        cycle_dag = compile_class_to_dag(
                            ontology_graph,
                            mapping,
                            referenced_class,
                            augment_property_domain_range=cycle_query_plan.augment_property_domain_range.enabled,
                            dependency_analysis=dependency_analysis,
                            compile_context=compile_context,
                        )
                        if not _dag_is_positive_monotone(cycle_dag):
                            has_nonmonotone_cycle = True
                            break
                if has_nonmonotone_cycle:
                    skipped_targets.append(
                        (
                            _render_term(target_term),
                            "reachable named-class cycle is not in the current positive/monotone query fragment",
                        )
                    )
                    continue

                if (
                    not has_reachable_cycle
                    and query_target_is_obviously_supported(
                        ontology_graph,
                        mapping,
                        target_term,
                        augment_property_domain_range=query_plan.augment_property_domain_range.enabled,
                        compile_context=compile_context,
                    )
                ):
                    resolved_targets.append(target_term)
                    continue

                dag = compile_class_to_dag(
                    ontology_graph,
                    mapping,
                    target_term,
                    augment_property_domain_range=query_plan.augment_property_domain_range.enabled,
                    dependency_analysis=dependency_analysis,
                    compile_context=compile_context,
                )
                if (
                    get_named_class_dependency_cycle_component(
                        ontology_graph,
                        target_term,
                        dependency_analysis=dependency_analysis,
                    )
                    and not _dag_is_positive_monotone(dag)
                ):
                    skipped_targets.append(
                        (
                            _render_term(target_term),
                            "reachable named-class cycle is not in the current positive/monotone query fragment",
                        )
                    )
                    continue
            resolved_targets.append(target_term)
        except Exception as exc:
            skipped_targets.append((_render_term(target_term), str(exc)))

    return TargetResolutionResult(
        requested_specs=requested_specs,
        resolved_targets=resolved_targets,
        skipped_targets=skipped_targets,
    )


def _collect_query_root_expressions(
    ontology_graph: Graph,
    target_term: URIRef,
) -> List[Identifier]:
    exprs: List[Identifier] = []

    for expr in ontology_graph.objects(target_term, RDFS.subClassOf):
        exprs.append(expr)

    for expr in ontology_graph.objects(target_term, OWL.equivalentClass):
        if expr != target_term:
            exprs.append(expr)

    for expr in ontology_graph.objects(target_term, OWL.intersectionOf):
        if not isinstance(expr, BNode):
            continue
        for member in Collection(ontology_graph, expr):
            exprs.append(member)

    seen: Set[Tuple[str, str]] = set()
    deduped: List[Identifier] = []
    for expr in exprs:
        key = (type(expr).__name__, str(expr))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(expr)
    return deduped


def _build_query_expression_term(
    query_graph: Graph,
    target_term: URIRef,
) -> Identifier:
    root_exprs = _collect_query_root_expressions(query_graph, target_term)

    for expr in query_graph.objects(target_term, OWL.disjointWith):
        neg_expr = BNode()
        query_graph.add((neg_expr, OWL.complementOf, expr))
        query_graph.add((neg_expr, RDF.type, OWL.Class))
        root_exprs.append(neg_expr)

    for expr in query_graph.subjects(OWL.disjointWith, target_term):
        if not isinstance(expr, Identifier) or expr == target_term:
            continue
        neg_expr = BNode()
        query_graph.add((neg_expr, OWL.complementOf, expr))
        query_graph.add((neg_expr, RDF.type, OWL.Class))
        root_exprs.append(neg_expr)

    if not root_exprs:
        return target_term
    if len(root_exprs) == 1:
        return root_exprs[0]

    inter_expr = BNode()
    list_head = BNode()
    query_graph.add((inter_expr, RDF.type, OWL.Class))
    query_graph.add((inter_expr, OWL.intersectionOf, list_head))
    Collection(query_graph, list_head, root_exprs)
    return inter_expr


def add_definitional_bridge_axioms(query_graph: Graph) -> Graph:
    """
    Add `equivalentClass` bridge axioms for supported named classes.

    This mirrors the engine's current Horn-style fixpoint interpretation, where
    supported named classes can be materialized from their defining constraints.
    """

    bridged_graph = _copy_graph(query_graph)
    for class_term in collect_inferable_named_classes(bridged_graph):
        expr = _build_query_expression_term(bridged_graph, class_term)
        bridged_graph.add((class_term, OWL.equivalentClass, expr))
    return bridged_graph


def build_oracle_query_graph(
    ontology_graph: Graph,
    target_classes: Sequence[str | URIRef],
    *,
    mode: str = "query",
    bridge_supported_definitions: bool = False,
) -> Tuple[Graph, Dict[URIRef, URIRef]]:
    """
    Build a graph that can be handed to classical reasoners for comparison.

    In `query` mode, each target class is wrapped in a synthetic named class:

      Query_C equivalentClass (root conditions used by compile_class_to_dag)

    This mirrors the engine's current "constraint query" interpretation for a
    named class, which is often stronger than native OWL instance membership
    when the ontology uses only `subClassOf` necessary conditions.

    In `native` mode, targets are compared against direct inferred membership in
    the original target classes.
    """

    query_graph = _copy_graph(ontology_graph)
    if mode == "query" and bridge_supported_definitions:
        query_graph = add_definitional_bridge_axioms(query_graph)
    query_class_by_target: Dict[URIRef, URIRef] = {}

    for target in target_classes:
        target_term = URIRef(target) if isinstance(target, str) else target

        if mode == "native":
            query_class_by_target[target_term] = target_term
            continue

        query_hash = sha1(str(target_term).encode("utf-8")).hexdigest()
        query_class = URIRef(ORACLE_NS[f"query/{query_hash}"])
        query_graph.add((query_class, RDF.type, OWL.Class))
        query_graph.add((query_class, RDFS.label, Literal(f"OracleQuery({target_term})")))

        query_graph.add((query_class, OWL.equivalentClass, _build_query_expression_term(query_graph, target_term)))

        query_class_by_target[target_term] = query_class

    return query_graph, query_class_by_target


def _extract_dataset_timing_breakdown(
    dataset: ReasoningDataset,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    hierarchy_elapsed_ms = 0.0
    atomic_domain_range_elapsed_ms = 0.0
    horn_safe_domain_range_elapsed_ms = 0.0
    sameas_elapsed_ms = 0.0
    reflexive_elapsed_ms = 0.0
    target_role_elapsed_ms = 0.0
    kgraph_build_elapsed_ms = 0.0
    mapping_vocab_collect_elapsed_ms = 0.0
    mapping_graph_scan_elapsed_ms = 0.0
    mapping_sort_elapsed_ms = 0.0
    mapping_index_elapsed_ms = 0.0
    kgraph_edge_bucket_elapsed_ms = 0.0
    kgraph_negative_helper_elapsed_ms = 0.0
    kgraph_literal_feature_elapsed_ms = 0.0
    kgraph_adjacency_elapsed_ms = 0.0
    dataset_build_elapsed_ms = 0.0
    mapping_vocab_collect_elapsed_ms = 0.0
    mapping_graph_scan_elapsed_ms = 0.0
    mapping_sort_elapsed_ms = 0.0
    mapping_index_elapsed_ms = 0.0
    kgraph_edge_bucket_elapsed_ms = 0.0
    kgraph_negative_helper_elapsed_ms = 0.0
    kgraph_literal_feature_elapsed_ms = 0.0
    kgraph_adjacency_elapsed_ms = 0.0
    if dataset.preprocessing_timings is not None:
        dataset_build_elapsed_ms = dataset.preprocessing_timings.dataset_build_elapsed_ms
        hierarchy_elapsed_ms = dataset.preprocessing_timings.hierarchy_elapsed_ms
        atomic_domain_range_elapsed_ms = dataset.preprocessing_timings.atomic_domain_range_elapsed_ms
        horn_safe_domain_range_elapsed_ms = dataset.preprocessing_timings.horn_safe_domain_range_elapsed_ms
        sameas_elapsed_ms = dataset.preprocessing_timings.sameas_elapsed_ms
        reflexive_elapsed_ms = dataset.preprocessing_timings.reflexive_elapsed_ms
        target_role_elapsed_ms = dataset.preprocessing_timings.target_role_elapsed_ms
        kgraph_build_elapsed_ms = dataset.preprocessing_timings.kgraph_build_elapsed_ms
        mapping_vocab_collect_elapsed_ms = dataset.preprocessing_timings.mapping_vocab_collect_elapsed_ms
        mapping_graph_scan_elapsed_ms = dataset.preprocessing_timings.mapping_graph_scan_elapsed_ms
        mapping_sort_elapsed_ms = dataset.preprocessing_timings.mapping_sort_elapsed_ms
        mapping_index_elapsed_ms = dataset.preprocessing_timings.mapping_index_elapsed_ms
        kgraph_edge_bucket_elapsed_ms = dataset.preprocessing_timings.kgraph_edge_bucket_elapsed_ms
        kgraph_negative_helper_elapsed_ms = dataset.preprocessing_timings.kgraph_negative_helper_elapsed_ms
        kgraph_literal_feature_elapsed_ms = dataset.preprocessing_timings.kgraph_literal_feature_elapsed_ms
        kgraph_adjacency_elapsed_ms = dataset.preprocessing_timings.kgraph_adjacency_elapsed_ms
    return (
        dataset_build_elapsed_ms,
        hierarchy_elapsed_ms,
        atomic_domain_range_elapsed_ms,
        horn_safe_domain_range_elapsed_ms,
        sameas_elapsed_ms,
        reflexive_elapsed_ms,
        target_role_elapsed_ms,
        kgraph_build_elapsed_ms,
        mapping_vocab_collect_elapsed_ms,
        mapping_graph_scan_elapsed_ms,
        mapping_sort_elapsed_ms,
        mapping_index_elapsed_ms,
        kgraph_edge_bucket_elapsed_ms,
        kgraph_negative_helper_elapsed_ms,
        kgraph_literal_feature_elapsed_ms,
        kgraph_adjacency_elapsed_ms,
    )


def _compile_and_evaluate_query_dataset(
    *,
    dataset: ReasoningDataset,
    target_classes: Sequence[URIRef],
    device: str,
    threshold: float,
    augment_property_domain_range: Optional[bool],
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
    compile_context: Optional[OntologyCompileContext] = None,
) -> Tuple[Dict[URIRef, Set[Identifier]], Dict[URIRef, Dict[Identifier, float]], Dict[URIRef, ConstraintDAG], float, float]:
    query_plan = plan_reasoning_preprocessing(
        dataset.ontology_graph,
        target_classes=target_classes,
        augment_property_domain_range=augment_property_domain_range,
    )

    reasoner = DAGReasoner(dataset.kg, device=device)
    dags_by_target: Dict[URIRef, ConstraintDAG] = {}
    compile_t0 = perf_counter()
    for target_term in target_classes:
        dag = compile_class_to_dag(
            dataset.ontology_graph,
            dataset.mapping,
            target_term,
            augment_property_domain_range=query_plan.augment_property_domain_range.enabled,
            dependency_analysis=dependency_analysis,
            compile_context=compile_context,
        )
        dags_by_target[target_term] = dag
        reasoner.add_concept(str(target_term), dag)
    dag_compile_elapsed_ms = (perf_counter() - compile_t0) * 1000.0

    eval_t0 = perf_counter()
    score_matrix = reasoner.evaluate_all().detach().cpu()
    dag_eval_elapsed_ms = (perf_counter() - eval_t0) * 1000.0

    members_by_target: Dict[URIRef, Set[Identifier]] = {}
    scores_by_target: Dict[URIRef, Dict[Identifier, float]] = {}
    for class_col, target_term in enumerate(target_classes):
        members: Set[Identifier] = set()
        scores: Dict[Identifier, float] = {}
        for node_idx, node_term in enumerate(dataset.mapping.node_terms):
            score = float(score_matrix[node_idx, class_col].item())
            scores[node_term] = score
            if score >= threshold:
                members.add(node_term)
        members_by_target[target_term] = members
        scores_by_target[target_term] = scores

    return members_by_target, scores_by_target, dags_by_target, dag_compile_elapsed_ms, dag_eval_elapsed_ms


def _evaluate_query_snapshot(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    target_classes: Sequence[URIRef],
    device: str,
    threshold: float,
    include_literals: bool,
    include_type_edges: bool,
    materialize_hierarchy: Optional[bool],
    materialize_sameas: Optional[bool],
    materialize_haskey_equality: Optional[bool],
    materialize_target_roles: Optional[bool],
    augment_property_domain_range: Optional[bool],
) -> QueryEvaluationSnapshot:
    dataset_build_elapsed_ms = 0.0
    hierarchy_elapsed_ms = 0.0
    atomic_domain_range_elapsed_ms = 0.0
    horn_safe_domain_range_elapsed_ms = 0.0
    sameas_elapsed_ms = 0.0
    reflexive_elapsed_ms = 0.0
    target_role_elapsed_ms = 0.0
    kgraph_build_elapsed_ms = 0.0
    dag_compile_elapsed_ms = 0.0
    dag_eval_elapsed_ms = 0.0

    initial_dataset = build_reasoning_dataset_from_graphs(
        schema_graph=schema_graph,
        data_graph=data_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_sameas=materialize_sameas,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_target_roles=materialize_target_roles,
        target_classes=target_classes,
    )
    (
        build_ms,
        hierarchy_ms,
        atomic_ms,
        horn_ms,
        sameas_ms,
        reflexive_ms,
        role_ms,
        kgraph_ms,
        mapping_vocab_ms,
        mapping_scan_ms,
        mapping_sort_ms,
        mapping_index_ms,
        edge_bucket_ms,
        negative_helper_ms,
        literal_feature_ms,
        adjacency_ms,
    ) = _extract_dataset_timing_breakdown(initial_dataset)
    dataset_build_elapsed_ms += build_ms
    hierarchy_elapsed_ms += hierarchy_ms
    atomic_domain_range_elapsed_ms += atomic_ms
    horn_safe_domain_range_elapsed_ms += horn_ms
    sameas_elapsed_ms += sameas_ms
    reflexive_elapsed_ms += reflexive_ms
    target_role_elapsed_ms += role_ms
    kgraph_build_elapsed_ms += kgraph_ms
    mapping_vocab_collect_elapsed_ms += mapping_vocab_ms
    mapping_graph_scan_elapsed_ms += mapping_scan_ms
    mapping_sort_elapsed_ms += mapping_sort_ms
    mapping_index_elapsed_ms += mapping_index_ms
    kgraph_edge_bucket_elapsed_ms += edge_bucket_ms
    kgraph_negative_helper_elapsed_ms += negative_helper_ms
    kgraph_literal_feature_elapsed_ms += literal_feature_ms
    kgraph_adjacency_elapsed_ms += adjacency_ms
    compile_context = build_ontology_compile_context(
        initial_dataset.ontology_graph,
        schema_graph=schema_graph,
        sameas_source_graph=initial_dataset.ontology_graph,
    )
    dependency_analysis = compile_context.dependency_analysis

    helper_cycle_classes: Set[URIRef] = set()
    for target_term in target_classes:
        referenced_classes = collect_referenced_named_classes_for_class(
            initial_dataset.ontology_graph,
            target_term,
            dependency_analysis=dependency_analysis,
        )
        for referenced_class in referenced_classes:
            component = get_named_class_dependency_cycle_component(
                initial_dataset.ontology_graph,
                referenced_class,
                dependency_analysis=dependency_analysis,
            )
            if component:
                helper_cycle_classes.update(component)

    monotone_cycle_classes: Set[URIRef] = set()
    if helper_cycle_classes:
        eval_targets = sorted(helper_cycle_classes, key=str)
        members_by_target, _scores_by_target, dags_by_target, compile_ms, eval_ms = _compile_and_evaluate_query_dataset(
            dataset=initial_dataset,
            target_classes=eval_targets,
            device=device,
            threshold=threshold,
            augment_property_domain_range=augment_property_domain_range,
            dependency_analysis=dependency_analysis,
            compile_context=compile_context,
        )
        dag_compile_elapsed_ms += compile_ms
        dag_eval_elapsed_ms += eval_ms
        for class_term, dag in dags_by_target.items():
            if _dag_is_positive_monotone(dag):
                monotone_cycle_classes.add(class_term)
        helper_cycle_classes = {
            class_term
            for class_term in helper_cycle_classes
            if class_term in monotone_cycle_classes
        }
    if helper_cycle_classes:
        current_cycle_members: Dict[URIRef, Set[Identifier]] = {
            class_term: set(initial_dataset.mapping.node_terms)
            for class_term in helper_cycle_classes
        }
        final_dataset = initial_dataset
        max_iterations = 25
        for _iteration in range(max_iterations):
            augmented_data_graph = _add_type_assignments(data_graph, current_cycle_members)
            cycle_dataset = build_reasoning_dataset_from_graphs(
                schema_graph=schema_graph,
                data_graph=augmented_data_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=materialize_hierarchy,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_target_roles=materialize_target_roles,
                target_classes=sorted(set(target_classes) | helper_cycle_classes, key=str),
            )
            (
                build_ms,
                hierarchy_ms,
                atomic_ms,
                horn_ms,
                sameas_ms,
                reflexive_ms,
                role_ms,
                kgraph_ms,
                mapping_vocab_ms,
                mapping_scan_ms,
                mapping_sort_ms,
                mapping_index_ms,
                edge_bucket_ms,
                negative_helper_ms,
                literal_feature_ms,
                adjacency_ms,
            ) = _extract_dataset_timing_breakdown(cycle_dataset)
            dataset_build_elapsed_ms += build_ms
            hierarchy_elapsed_ms += hierarchy_ms
            atomic_domain_range_elapsed_ms += atomic_ms
            horn_safe_domain_range_elapsed_ms += horn_ms
            sameas_elapsed_ms += sameas_ms
            reflexive_elapsed_ms += reflexive_ms
            target_role_elapsed_ms += role_ms
            kgraph_build_elapsed_ms += kgraph_ms
            mapping_vocab_collect_elapsed_ms += mapping_vocab_ms
            mapping_graph_scan_elapsed_ms += mapping_scan_ms
            mapping_sort_elapsed_ms += mapping_sort_ms
            mapping_index_elapsed_ms += mapping_index_ms
            kgraph_edge_bucket_elapsed_ms += edge_bucket_ms
            kgraph_negative_helper_elapsed_ms += negative_helper_ms
            kgraph_literal_feature_elapsed_ms += literal_feature_ms
            kgraph_adjacency_elapsed_ms += adjacency_ms

            cycle_targets = sorted(helper_cycle_classes, key=str)
            members_by_target, _scores_by_target, _dags_by_target, compile_ms, eval_ms = _compile_and_evaluate_query_dataset(
                dataset=cycle_dataset,
                target_classes=cycle_targets,
                device=device,
                threshold=threshold,
                augment_property_domain_range=augment_property_domain_range,
                dependency_analysis=dependency_analysis,
                compile_context=compile_context,
            )
            dag_compile_elapsed_ms += compile_ms
            dag_eval_elapsed_ms += eval_ms

            next_cycle_members = {
                class_term: set(members_by_target.get(class_term, set()))
                for class_term in helper_cycle_classes
            }
            final_dataset = cycle_dataset
            if next_cycle_members == current_cycle_members:
                break
            current_cycle_members = next_cycle_members

        final_augmented_data_graph = _add_type_assignments(data_graph, current_cycle_members)
        dataset = build_reasoning_dataset_from_graphs(
            schema_graph=schema_graph,
            data_graph=final_augmented_data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_target_roles=materialize_target_roles,
            target_classes=target_classes,
        )
        (
            build_ms,
            hierarchy_ms,
            atomic_ms,
            horn_ms,
            sameas_ms,
            reflexive_ms,
            role_ms,
            kgraph_ms,
            mapping_vocab_ms,
            mapping_scan_ms,
            mapping_sort_ms,
            mapping_index_ms,
            edge_bucket_ms,
            negative_helper_ms,
            literal_feature_ms,
            adjacency_ms,
        ) = _extract_dataset_timing_breakdown(dataset)
        dataset_build_elapsed_ms += build_ms
        hierarchy_elapsed_ms += hierarchy_ms
        atomic_domain_range_elapsed_ms += atomic_ms
        horn_safe_domain_range_elapsed_ms += horn_ms
        sameas_elapsed_ms += sameas_ms
        reflexive_elapsed_ms += reflexive_ms
        target_role_elapsed_ms += role_ms
        kgraph_build_elapsed_ms += kgraph_ms
        mapping_vocab_collect_elapsed_ms += mapping_vocab_ms
        mapping_graph_scan_elapsed_ms += mapping_scan_ms
        mapping_sort_elapsed_ms += mapping_sort_ms
        mapping_index_elapsed_ms += mapping_index_ms
        kgraph_edge_bucket_elapsed_ms += edge_bucket_ms
        kgraph_negative_helper_elapsed_ms += negative_helper_ms
        kgraph_literal_feature_elapsed_ms += literal_feature_ms
        kgraph_adjacency_elapsed_ms += adjacency_ms
    else:
        dataset = initial_dataset

    members_by_target, scores_by_target, _dags_by_target, compile_ms, eval_ms = _compile_and_evaluate_query_dataset(
        dataset=dataset,
        target_classes=target_classes,
        device=device,
        threshold=threshold,
        augment_property_domain_range=augment_property_domain_range,
        dependency_analysis=dependency_analysis,
        compile_context=compile_context,
    )
    dag_compile_elapsed_ms += compile_ms
    dag_eval_elapsed_ms += eval_ms

    return QueryEvaluationSnapshot(
        dataset=dataset,
        members_by_target=members_by_target,
        scores_by_target=scores_by_target,
        dataset_build_elapsed_ms=dataset_build_elapsed_ms,
        hierarchy_elapsed_ms=hierarchy_elapsed_ms,
        atomic_domain_range_elapsed_ms=atomic_domain_range_elapsed_ms,
        horn_safe_domain_range_elapsed_ms=horn_safe_domain_range_elapsed_ms,
        sameas_elapsed_ms=sameas_elapsed_ms,
        reflexive_elapsed_ms=reflexive_elapsed_ms,
        target_role_elapsed_ms=target_role_elapsed_ms,
        kgraph_build_elapsed_ms=kgraph_build_elapsed_ms,
        mapping_vocab_collect_elapsed_ms=mapping_vocab_collect_elapsed_ms,
        mapping_graph_scan_elapsed_ms=mapping_graph_scan_elapsed_ms,
        mapping_sort_elapsed_ms=mapping_sort_elapsed_ms,
        mapping_index_elapsed_ms=mapping_index_elapsed_ms,
        kgraph_edge_bucket_elapsed_ms=kgraph_edge_bucket_elapsed_ms,
        kgraph_negative_helper_elapsed_ms=kgraph_negative_helper_elapsed_ms,
        kgraph_literal_feature_elapsed_ms=kgraph_literal_feature_elapsed_ms,
        kgraph_adjacency_elapsed_ms=kgraph_adjacency_elapsed_ms,
        dag_compile_elapsed_ms=dag_compile_elapsed_ms,
        dag_eval_elapsed_ms=dag_eval_elapsed_ms,
    )


def run_engine_queries(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    target_classes: Sequence[URIRef],
    device: str = "cuda",
    threshold: float = 0.999,
    include_literals: bool = False,
    include_type_edges: bool = False,
    materialize_hierarchy: Optional[bool] = None,
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
    materialize_target_roles: Optional[bool] = None,
    materialize_supported_types: bool = False,
    augment_property_domain_range: Optional[bool] = None,
    engine_mode: str = "query",
    conflict_policy: str = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
) -> EngineQueryResult:
    device_to_use = device
    if device_to_use == "cuda" and not torch.cuda.is_available():
        device_to_use = "cpu"

    t0 = perf_counter()
    iterations: Optional[int] = None
    stratified_result: Optional[StratifiedMaterializationResult] = None
    filtered_query_result: Optional[FilteredQueryResult] = None
    stratified_positive_total_elapsed_ms = 0.0
    stratified_positive_iterations = 0
    stratified_positive_avg_dataset_build_elapsed_ms = 0.0
    stratified_positive_avg_dag_compile_elapsed_ms = 0.0
    stratified_positive_avg_dag_eval_elapsed_ms = 0.0
    stratified_positive_reasoner_setup_elapsed_ms = 0.0
    stratified_positive_assertion_update_elapsed_ms = 0.0
    stratified_negative_blocker_elapsed_ms = 0.0
    stratified_assignment_status_elapsed_ms = 0.0
    stratified_conflict_policy_elapsed_ms = 0.0
    stratified_reporting_compile_elapsed_ms = 0.0
    ontology_merge_elapsed_ms = 0.0
    stratified_initial_data_copy_elapsed_ms = 0.0
    stratified_initial_ontology_merge_elapsed_ms = 0.0
    schema_cache_elapsed_ms = 0.0
    preprocessing_plan_elapsed_ms = 0.0
    sufficient_rule_extraction_elapsed_ms = 0.0
    sufficient_rule_index_elapsed_ms = 0.0
    dependency_closure_elapsed_ms = 0.0
    sameas_state_init_elapsed_ms = 0.0
    dataset_build_elapsed_ms = 0.0
    hierarchy_elapsed_ms = 0.0
    atomic_domain_range_elapsed_ms = 0.0
    horn_safe_domain_range_elapsed_ms = 0.0
    sameas_elapsed_ms = 0.0
    reflexive_elapsed_ms = 0.0
    target_role_elapsed_ms = 0.0
    kgraph_build_elapsed_ms = 0.0
    mapping_vocab_collect_elapsed_ms = 0.0
    mapping_graph_scan_elapsed_ms = 0.0
    mapping_sort_elapsed_ms = 0.0
    mapping_index_elapsed_ms = 0.0
    kgraph_edge_bucket_elapsed_ms = 0.0
    kgraph_negative_helper_elapsed_ms = 0.0
    kgraph_literal_feature_elapsed_ms = 0.0
    kgraph_adjacency_elapsed_ms = 0.0
    sameas_passes_elapsed_ms: List[float] = []
    dag_compile_elapsed_ms = 0.0
    dag_eval_elapsed_ms = 0.0
    stratified_iteration_timings: Optional[List[MaterializationIterationTiming]] = None
    stratified_final_dataset_timing: Optional[PreprocessingTimings] = None

    if engine_mode == "stratified":
        stratified_result = materialize_stratified_class_inferences(
            schema_graph=schema_graph,
            data_graph=data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_target_roles=False,
            target_classes=target_classes,
            threshold=threshold,
            device=device_to_use,
            conflict_policy=ConflictPolicy(conflict_policy),
        )
        dataset = stratified_result.positive_result.dataset
        iterations = stratified_result.positive_result.iterations
        if dataset.preprocessing_timings is not None:
            dataset_build_elapsed_ms = dataset.preprocessing_timings.dataset_build_elapsed_ms
            ontology_merge_elapsed_ms = dataset.preprocessing_timings.ontology_merge_elapsed_ms
            schema_cache_elapsed_ms = dataset.preprocessing_timings.schema_cache_elapsed_ms
            preprocessing_plan_elapsed_ms = dataset.preprocessing_timings.preprocessing_plan_elapsed_ms
            hierarchy_elapsed_ms = dataset.preprocessing_timings.hierarchy_elapsed_ms
            atomic_domain_range_elapsed_ms = dataset.preprocessing_timings.atomic_domain_range_elapsed_ms
            horn_safe_domain_range_elapsed_ms = dataset.preprocessing_timings.horn_safe_domain_range_elapsed_ms
            sameas_elapsed_ms = dataset.preprocessing_timings.sameas_elapsed_ms
            reflexive_elapsed_ms = dataset.preprocessing_timings.reflexive_elapsed_ms
            target_role_elapsed_ms = dataset.preprocessing_timings.target_role_elapsed_ms
            kgraph_build_elapsed_ms = dataset.preprocessing_timings.kgraph_build_elapsed_ms
            mapping_vocab_collect_elapsed_ms = dataset.preprocessing_timings.mapping_vocab_collect_elapsed_ms
            mapping_graph_scan_elapsed_ms = dataset.preprocessing_timings.mapping_graph_scan_elapsed_ms
            mapping_sort_elapsed_ms = dataset.preprocessing_timings.mapping_sort_elapsed_ms
            mapping_index_elapsed_ms = dataset.preprocessing_timings.mapping_index_elapsed_ms
            kgraph_edge_bucket_elapsed_ms = dataset.preprocessing_timings.kgraph_edge_bucket_elapsed_ms
            kgraph_negative_helper_elapsed_ms = dataset.preprocessing_timings.kgraph_negative_helper_elapsed_ms
            kgraph_literal_feature_elapsed_ms = dataset.preprocessing_timings.kgraph_literal_feature_elapsed_ms
            kgraph_adjacency_elapsed_ms = dataset.preprocessing_timings.kgraph_adjacency_elapsed_ms
            sameas_passes_elapsed_ms.extend(dataset.preprocessing_timings.sameas_passes_elapsed_ms)
    elif engine_mode == "filtered_query":
        raw_snapshot = _evaluate_query_snapshot(
            schema_graph=schema_graph,
            data_graph=data_graph,
            target_classes=target_classes,
            device=device_to_use,
            threshold=threshold,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_target_roles=materialize_target_roles,
            augment_property_domain_range=augment_property_domain_range,
        )
        raw_members_by_target = _clone_members_by_target(raw_snapshot.members_by_target)
        current_members_by_target = _clone_members_by_target(raw_members_by_target)
        stable_snapshot = raw_snapshot
        dataset_build_elapsed_ms += raw_snapshot.dataset_build_elapsed_ms
        hierarchy_elapsed_ms += raw_snapshot.hierarchy_elapsed_ms
        atomic_domain_range_elapsed_ms += raw_snapshot.atomic_domain_range_elapsed_ms
        horn_safe_domain_range_elapsed_ms += raw_snapshot.horn_safe_domain_range_elapsed_ms
        sameas_elapsed_ms += raw_snapshot.sameas_elapsed_ms
        reflexive_elapsed_ms += raw_snapshot.reflexive_elapsed_ms
        target_role_elapsed_ms += raw_snapshot.target_role_elapsed_ms
        kgraph_build_elapsed_ms += raw_snapshot.kgraph_build_elapsed_ms
        mapping_vocab_collect_elapsed_ms += raw_snapshot.mapping_vocab_collect_elapsed_ms
        mapping_graph_scan_elapsed_ms += raw_snapshot.mapping_graph_scan_elapsed_ms
        mapping_sort_elapsed_ms += raw_snapshot.mapping_sort_elapsed_ms
        mapping_index_elapsed_ms += raw_snapshot.mapping_index_elapsed_ms
        kgraph_edge_bucket_elapsed_ms += raw_snapshot.kgraph_edge_bucket_elapsed_ms
        kgraph_negative_helper_elapsed_ms += raw_snapshot.kgraph_negative_helper_elapsed_ms
        kgraph_literal_feature_elapsed_ms += raw_snapshot.kgraph_literal_feature_elapsed_ms
        kgraph_adjacency_elapsed_ms += raw_snapshot.kgraph_adjacency_elapsed_ms
        dag_compile_elapsed_ms += raw_snapshot.dag_compile_elapsed_ms
        dag_eval_elapsed_ms += raw_snapshot.dag_eval_elapsed_ms
        if raw_snapshot.dataset.preprocessing_timings is not None:
            ontology_merge_elapsed_ms += raw_snapshot.dataset.preprocessing_timings.ontology_merge_elapsed_ms
            schema_cache_elapsed_ms += raw_snapshot.dataset.preprocessing_timings.schema_cache_elapsed_ms
            preprocessing_plan_elapsed_ms += raw_snapshot.dataset.preprocessing_timings.preprocessing_plan_elapsed_ms
            sameas_passes_elapsed_ms.extend(raw_snapshot.dataset.preprocessing_timings.sameas_passes_elapsed_ms)

        necessary_fixpoint_iterations = 0
        while True:
            necessary_fixpoint_iterations += 1
            augmented_data_graph = _add_type_assignments(data_graph, current_members_by_target)
            stable_snapshot = _evaluate_query_snapshot(
                schema_graph=schema_graph,
                data_graph=augmented_data_graph,
                target_classes=target_classes,
                device=device_to_use,
                threshold=threshold,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=materialize_hierarchy,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_target_roles=materialize_target_roles,
                augment_property_domain_range=augment_property_domain_range,
            )
            dataset_build_elapsed_ms += stable_snapshot.dataset_build_elapsed_ms
            hierarchy_elapsed_ms += stable_snapshot.hierarchy_elapsed_ms
            atomic_domain_range_elapsed_ms += stable_snapshot.atomic_domain_range_elapsed_ms
            horn_safe_domain_range_elapsed_ms += stable_snapshot.horn_safe_domain_range_elapsed_ms
            sameas_elapsed_ms += stable_snapshot.sameas_elapsed_ms
            reflexive_elapsed_ms += stable_snapshot.reflexive_elapsed_ms
            target_role_elapsed_ms += stable_snapshot.target_role_elapsed_ms
            kgraph_build_elapsed_ms += stable_snapshot.kgraph_build_elapsed_ms
            mapping_vocab_collect_elapsed_ms += stable_snapshot.mapping_vocab_collect_elapsed_ms
            mapping_graph_scan_elapsed_ms += stable_snapshot.mapping_graph_scan_elapsed_ms
            mapping_sort_elapsed_ms += stable_snapshot.mapping_sort_elapsed_ms
            mapping_index_elapsed_ms += stable_snapshot.mapping_index_elapsed_ms
            kgraph_edge_bucket_elapsed_ms += stable_snapshot.kgraph_edge_bucket_elapsed_ms
            kgraph_negative_helper_elapsed_ms += stable_snapshot.kgraph_negative_helper_elapsed_ms
            kgraph_literal_feature_elapsed_ms += stable_snapshot.kgraph_literal_feature_elapsed_ms
            kgraph_adjacency_elapsed_ms += stable_snapshot.kgraph_adjacency_elapsed_ms
            dag_compile_elapsed_ms += stable_snapshot.dag_compile_elapsed_ms
            dag_eval_elapsed_ms += stable_snapshot.dag_eval_elapsed_ms
            if stable_snapshot.dataset.preprocessing_timings is not None:
                ontology_merge_elapsed_ms += stable_snapshot.dataset.preprocessing_timings.ontology_merge_elapsed_ms
                schema_cache_elapsed_ms += stable_snapshot.dataset.preprocessing_timings.schema_cache_elapsed_ms
                preprocessing_plan_elapsed_ms += stable_snapshot.dataset.preprocessing_timings.preprocessing_plan_elapsed_ms
                sameas_passes_elapsed_ms.extend(stable_snapshot.dataset.preprocessing_timings.sameas_passes_elapsed_ms)

            next_members_by_target: Dict[URIRef, Set[Identifier]] = {}
            for target_term in target_classes:
                previous_members = current_members_by_target.get(target_term, set())
                scores = stable_snapshot.scores_by_target.get(target_term, {})
                next_members_by_target[target_term] = {
                    node_term
                    for node_term in previous_members
                    if scores.get(node_term, 0.0) >= threshold
                }
            if next_members_by_target == current_members_by_target:
                current_members_by_target = next_members_by_target
                break
            current_members_by_target = next_members_by_target

        necessary_stable_members_by_target = _clone_members_by_target(current_members_by_target)
        filtered_data_graph = _add_type_assignments(data_graph, necessary_stable_members_by_target)
        stratified_result = materialize_stratified_class_inferences(
            schema_graph=schema_graph,
            data_graph=filtered_data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_target_roles=False,
            target_classes=target_classes,
            threshold=threshold,
            device=device_to_use,
            conflict_policy=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED,
        )

        closure_blocked_members_by_target: Dict[URIRef, Set[Identifier]] = {
            target_term: set()
            for target_term in target_classes
        }
        for blocked in stratified_result.negative_result.blocked_assertions:
            target_term = blocked.target_class
            if (
                target_term in closure_blocked_members_by_target
                and blocked.node_term in necessary_stable_members_by_target.get(target_term, set())
            ):
                closure_blocked_members_by_target[target_term].add(blocked.node_term)

        final_members_by_target: Dict[URIRef, Set[Identifier]] = {}
        for target_term in target_classes:
            final_members_by_target[target_term] = (
                necessary_stable_members_by_target.get(target_term, set())
                - closure_blocked_members_by_target.get(target_term, set())
            )

        filtered_query_result = FilteredQueryResult(
            raw_members_by_target=raw_members_by_target,
            necessary_stable_members_by_target=necessary_stable_members_by_target,
            closure_blocked_members_by_target=closure_blocked_members_by_target,
            final_members_by_target=final_members_by_target,
            raw_candidate_count=_count_members_by_target(raw_members_by_target),
            necessary_retraction_count=(
                _count_members_by_target(raw_members_by_target)
                - _count_members_by_target(necessary_stable_members_by_target)
            ),
            closure_blocked_retraction_count=_count_members_by_target(closure_blocked_members_by_target),
            final_emitted_count=_count_members_by_target(final_members_by_target),
            necessary_fixpoint_iterations=necessary_fixpoint_iterations,
            stratified_result=stratified_result,
        )
        dataset = stable_snapshot.dataset
        members_by_target = final_members_by_target
        scores_by_target = _build_binary_scores(
            dataset.mapping.node_terms,
            target_classes,
            final_members_by_target,
        )
        iterations = necessary_fixpoint_iterations
    elif materialize_supported_types:
        materialized = materialize_supported_class_inferences(
            schema_graph=schema_graph,
            data_graph=data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            target_classes=target_classes,
            threshold=threshold,
            device=device_to_use,
        )
        dataset = materialized.dataset
        iterations = materialized.iterations
        if dataset.preprocessing_timings is not None:
            dataset_build_elapsed_ms = dataset.preprocessing_timings.dataset_build_elapsed_ms
            ontology_merge_elapsed_ms = dataset.preprocessing_timings.ontology_merge_elapsed_ms
            schema_cache_elapsed_ms = dataset.preprocessing_timings.schema_cache_elapsed_ms
            preprocessing_plan_elapsed_ms = dataset.preprocessing_timings.preprocessing_plan_elapsed_ms
            hierarchy_elapsed_ms = dataset.preprocessing_timings.hierarchy_elapsed_ms
            atomic_domain_range_elapsed_ms = dataset.preprocessing_timings.atomic_domain_range_elapsed_ms
            horn_safe_domain_range_elapsed_ms = dataset.preprocessing_timings.horn_safe_domain_range_elapsed_ms
            sameas_elapsed_ms = dataset.preprocessing_timings.sameas_elapsed_ms
            reflexive_elapsed_ms = dataset.preprocessing_timings.reflexive_elapsed_ms
            target_role_elapsed_ms = dataset.preprocessing_timings.target_role_elapsed_ms
            kgraph_build_elapsed_ms = dataset.preprocessing_timings.kgraph_build_elapsed_ms
            sameas_passes_elapsed_ms.extend(dataset.preprocessing_timings.sameas_passes_elapsed_ms)
    else:
        snapshot = _evaluate_query_snapshot(
            schema_graph=schema_graph,
            data_graph=data_graph,
            target_classes=target_classes,
            device=device_to_use,
            threshold=threshold,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_target_roles=materialize_target_roles,
            augment_property_domain_range=augment_property_domain_range,
        )
        dataset = snapshot.dataset
        members_by_target = snapshot.members_by_target
        scores_by_target = snapshot.scores_by_target
        dataset_build_elapsed_ms = snapshot.dataset_build_elapsed_ms
        ontology_merge_elapsed_ms = (
            snapshot.dataset.preprocessing_timings.ontology_merge_elapsed_ms
            if snapshot.dataset.preprocessing_timings is not None
            else 0.0
        )
        schema_cache_elapsed_ms = (
            snapshot.dataset.preprocessing_timings.schema_cache_elapsed_ms
            if snapshot.dataset.preprocessing_timings is not None
            else 0.0
        )
        preprocessing_plan_elapsed_ms = (
            snapshot.dataset.preprocessing_timings.preprocessing_plan_elapsed_ms
            if snapshot.dataset.preprocessing_timings is not None
            else 0.0
        )
        hierarchy_elapsed_ms = snapshot.hierarchy_elapsed_ms
        atomic_domain_range_elapsed_ms = snapshot.atomic_domain_range_elapsed_ms
        horn_safe_domain_range_elapsed_ms = snapshot.horn_safe_domain_range_elapsed_ms
        sameas_elapsed_ms = snapshot.sameas_elapsed_ms
        reflexive_elapsed_ms = snapshot.reflexive_elapsed_ms
        target_role_elapsed_ms = snapshot.target_role_elapsed_ms
        kgraph_build_elapsed_ms = snapshot.kgraph_build_elapsed_ms
        mapping_vocab_collect_elapsed_ms = snapshot.mapping_vocab_collect_elapsed_ms
        mapping_graph_scan_elapsed_ms = snapshot.mapping_graph_scan_elapsed_ms
        mapping_sort_elapsed_ms = snapshot.mapping_sort_elapsed_ms
        mapping_index_elapsed_ms = snapshot.mapping_index_elapsed_ms
        kgraph_edge_bucket_elapsed_ms = snapshot.kgraph_edge_bucket_elapsed_ms
        kgraph_negative_helper_elapsed_ms = snapshot.kgraph_negative_helper_elapsed_ms
        kgraph_literal_feature_elapsed_ms = snapshot.kgraph_literal_feature_elapsed_ms
        kgraph_adjacency_elapsed_ms = snapshot.kgraph_adjacency_elapsed_ms
        if snapshot.dataset.preprocessing_timings is not None:
            sameas_passes_elapsed_ms.extend(snapshot.dataset.preprocessing_timings.sameas_passes_elapsed_ms)
        dag_compile_elapsed_ms = snapshot.dag_compile_elapsed_ms
        dag_eval_elapsed_ms = snapshot.dag_eval_elapsed_ms

    if engine_mode == "stratified":
        stratified_positive_total_elapsed_ms = 0.0
        stratified_positive_iterations = 0
        stratified_positive_avg_dataset_build_elapsed_ms = 0.0
        stratified_positive_avg_dag_compile_elapsed_ms = 0.0
        stratified_positive_avg_dag_eval_elapsed_ms = 0.0
        stratified_negative_blocker_elapsed_ms = 0.0
        stratified_assignment_status_elapsed_ms = 0.0
        stratified_conflict_policy_elapsed_ms = 0.0
        stratified_reporting_compile_elapsed_ms = 0.0
        if stratified_result is not None and stratified_result.timings is not None:
            positive_timings = stratified_result.timings.positive_timings
            ontology_merge_elapsed_ms = positive_timings.ontology_merge_elapsed_ms
            stratified_initial_data_copy_elapsed_ms = positive_timings.initial_data_copy_elapsed_ms
            stratified_initial_ontology_merge_elapsed_ms = positive_timings.initial_ontology_merge_elapsed_ms
            schema_cache_elapsed_ms = positive_timings.schema_cache_elapsed_ms
            preprocessing_plan_elapsed_ms = positive_timings.preprocessing_plan_elapsed_ms
            sufficient_rule_extraction_elapsed_ms = positive_timings.rule_extraction_elapsed_ms
            sufficient_rule_index_elapsed_ms = positive_timings.rule_index_elapsed_ms
            dependency_closure_elapsed_ms = positive_timings.dependency_closure_elapsed_ms
            sameas_state_init_elapsed_ms = positive_timings.sameas_state_init_elapsed_ms
            hierarchy_elapsed_ms = positive_timings.hierarchy_elapsed_ms
            atomic_domain_range_elapsed_ms = positive_timings.atomic_domain_range_elapsed_ms
            horn_safe_domain_range_elapsed_ms = positive_timings.horn_safe_domain_range_elapsed_ms
            sameas_elapsed_ms = positive_timings.sameas_elapsed_ms
            reflexive_elapsed_ms = positive_timings.reflexive_elapsed_ms
            target_role_elapsed_ms = positive_timings.target_role_elapsed_ms
            kgraph_build_elapsed_ms = positive_timings.kgraph_build_elapsed_ms
            mapping_vocab_collect_elapsed_ms = positive_timings.mapping_vocab_collect_elapsed_ms
            mapping_graph_scan_elapsed_ms = positive_timings.mapping_graph_scan_elapsed_ms
            mapping_sort_elapsed_ms = positive_timings.mapping_sort_elapsed_ms
            mapping_index_elapsed_ms = positive_timings.mapping_index_elapsed_ms
            kgraph_edge_bucket_elapsed_ms = positive_timings.kgraph_edge_bucket_elapsed_ms
            kgraph_negative_helper_elapsed_ms = positive_timings.kgraph_negative_helper_elapsed_ms
            kgraph_literal_feature_elapsed_ms = positive_timings.kgraph_literal_feature_elapsed_ms
            kgraph_adjacency_elapsed_ms = positive_timings.kgraph_adjacency_elapsed_ms
            dataset_build_elapsed_ms = positive_timings.dataset_build_elapsed_ms
            dag_compile_elapsed_ms = positive_timings.dag_compile_elapsed_ms
            dag_eval_elapsed_ms = positive_timings.dag_eval_elapsed_ms
            stratified_positive_total_elapsed_ms = positive_timings.total_elapsed_ms
            stratified_positive_iterations = positive_timings.iterations
            stratified_positive_avg_dataset_build_elapsed_ms = positive_timings.avg_dataset_build_elapsed_ms
            stratified_positive_avg_dag_compile_elapsed_ms = positive_timings.avg_dag_compile_elapsed_ms
            stratified_positive_avg_dag_eval_elapsed_ms = positive_timings.avg_dag_eval_elapsed_ms
            stratified_positive_reasoner_setup_elapsed_ms = positive_timings.reasoner_setup_elapsed_ms
            stratified_positive_assertion_update_elapsed_ms = positive_timings.assertion_update_elapsed_ms
            stratified_iteration_timings = positive_timings.iteration_timings
            stratified_final_dataset_timing = positive_timings.final_dataset_timings
            stratified_negative_blocker_elapsed_ms = stratified_result.timings.negative_blocker_elapsed_ms
            stratified_assignment_status_elapsed_ms = stratified_result.timings.assignment_status_elapsed_ms
            stratified_conflict_policy_elapsed_ms = stratified_result.timings.conflict_policy_elapsed_ms
        members_by_target = {}
        scores_by_target = {}
        compile_t0 = perf_counter()
        for target_term in target_classes:
            compile_sufficient_condition_dag(
                dataset.ontology_graph,
                dataset.mapping,
                target_term,
            )
        stratified_reporting_compile_elapsed_ms = (perf_counter() - compile_t0) * 1000.0
        dag_compile_elapsed_ms += stratified_reporting_compile_elapsed_ms

        emitted_members_by_target: Dict[URIRef, Set[Identifier]] = {target: set() for target in target_classes}
        if stratified_result is not None:
            for status in stratified_result.policy_result.emitted_assignments:
                if status.target_class in emitted_members_by_target:
                    emitted_members_by_target[status.target_class].add(status.node_term)

        for target_term in target_classes:
            members = emitted_members_by_target.get(target_term, set())
            scores: Dict[Identifier, float] = {}
            for node_term in dataset.mapping.node_terms:
                score = 1.0 if node_term in members else 0.0
                scores[node_term] = score
            members_by_target[target_term] = set(members)
            scores_by_target[target_term] = scores
    elapsed_ms = (perf_counter() - t0) * 1000.0

    return EngineQueryResult(
        backend="engine",
        elapsed_ms=elapsed_ms,
        members_by_target=members_by_target,
        dataset=dataset,
        scores_by_target=scores_by_target,
        materialization_iterations=iterations,
        engine_mode=engine_mode,
        conflict_policy=(conflict_policy if engine_mode in {"stratified", "filtered_query"} else None),
        stratified_result=stratified_result,
        filtered_query_result=filtered_query_result,
        consistent=None,
        ontology_merge_elapsed_ms=ontology_merge_elapsed_ms,
        stratified_initial_data_copy_elapsed_ms=stratified_initial_data_copy_elapsed_ms,
        stratified_initial_ontology_merge_elapsed_ms=stratified_initial_ontology_merge_elapsed_ms,
        schema_cache_elapsed_ms=schema_cache_elapsed_ms,
        preprocessing_plan_elapsed_ms=preprocessing_plan_elapsed_ms,
        sufficient_rule_extraction_elapsed_ms=sufficient_rule_extraction_elapsed_ms,
        sufficient_rule_index_elapsed_ms=sufficient_rule_index_elapsed_ms,
        dependency_closure_elapsed_ms=dependency_closure_elapsed_ms,
        sameas_state_init_elapsed_ms=sameas_state_init_elapsed_ms,
        dataset_build_elapsed_ms=dataset_build_elapsed_ms,
        hierarchy_elapsed_ms=hierarchy_elapsed_ms,
        atomic_domain_range_elapsed_ms=atomic_domain_range_elapsed_ms,
        horn_safe_domain_range_elapsed_ms=horn_safe_domain_range_elapsed_ms,
        sameas_elapsed_ms=sameas_elapsed_ms,
        reflexive_elapsed_ms=reflexive_elapsed_ms,
        target_role_elapsed_ms=target_role_elapsed_ms,
        kgraph_build_elapsed_ms=kgraph_build_elapsed_ms,
        mapping_vocab_collect_elapsed_ms=mapping_vocab_collect_elapsed_ms,
        mapping_graph_scan_elapsed_ms=mapping_graph_scan_elapsed_ms,
        mapping_sort_elapsed_ms=mapping_sort_elapsed_ms,
        mapping_index_elapsed_ms=mapping_index_elapsed_ms,
        kgraph_edge_bucket_elapsed_ms=kgraph_edge_bucket_elapsed_ms,
        kgraph_negative_helper_elapsed_ms=kgraph_negative_helper_elapsed_ms,
        kgraph_literal_feature_elapsed_ms=kgraph_literal_feature_elapsed_ms,
        kgraph_adjacency_elapsed_ms=kgraph_adjacency_elapsed_ms,
        sameas_passes_elapsed_ms=sameas_passes_elapsed_ms,
        stratified_positive_reasoner_setup_elapsed_ms=stratified_positive_reasoner_setup_elapsed_ms,
        dag_compile_elapsed_ms=dag_compile_elapsed_ms,
        dag_eval_elapsed_ms=dag_eval_elapsed_ms,
        stratified_positive_assertion_update_elapsed_ms=stratified_positive_assertion_update_elapsed_ms,
        stratified_positive_total_elapsed_ms=stratified_positive_total_elapsed_ms,
        stratified_positive_iterations=stratified_positive_iterations,
        stratified_positive_avg_dataset_build_elapsed_ms=stratified_positive_avg_dataset_build_elapsed_ms,
        stratified_positive_avg_dag_compile_elapsed_ms=stratified_positive_avg_dag_compile_elapsed_ms,
        stratified_positive_avg_dag_eval_elapsed_ms=stratified_positive_avg_dag_eval_elapsed_ms,
        stratified_negative_blocker_elapsed_ms=stratified_negative_blocker_elapsed_ms,
        stratified_assignment_status_elapsed_ms=stratified_assignment_status_elapsed_ms,
        stratified_conflict_policy_elapsed_ms=stratified_conflict_policy_elapsed_ms,
        stratified_reporting_compile_elapsed_ms=stratified_reporting_compile_elapsed_ms,
        stratified_iteration_timings=stratified_iteration_timings,
        stratified_final_dataset_timing=stratified_final_dataset_timing,
    )


def run_owlrl_queries(
    query_graph: Graph,
    query_class_by_target: Dict[URIRef, URIRef],
    candidate_terms: Set[Identifier],
) -> BackendQueryResult:
    try:
        from owlrl import DeductiveClosure, OWLRL_Semantics
    except ImportError as exc:
        return BackendQueryResult(
            backend="owlrl",
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            error=str(exc),
        )

    expanded = _copy_graph(query_graph)
    t0 = perf_counter()
    DeductiveClosure(OWLRL_Semantics).expand(expanded)
    elapsed_ms = (perf_counter() - t0) * 1000.0

    members_by_target: Dict[URIRef, Set[Identifier]] = {}
    for target_term, query_class in query_class_by_target.items():
        members = {
            subj
            for subj, _pred, _obj in expanded.triples((None, RDF.type, query_class))
            if subj in candidate_terms
        }
        members_by_target[target_term] = members

    return BackendQueryResult(
        backend="owlrl",
        elapsed_ms=elapsed_ms,
        members_by_target=members_by_target,
    )


def run_owlready2_queries(
    query_graph: Graph,
    query_class_by_target: Dict[URIRef, URIRef],
    candidate_terms: Set[Identifier],
    *,
    reasoner_name: str = "hermit",
) -> BackendQueryResult:
    try:
        from owlready2 import World, sync_reasoner, sync_reasoner_pellet
        from owlready2.base import OwlReadyInconsistentOntologyError
    except ImportError as exc:
        return BackendQueryResult(
            backend="owlready2",
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            error=str(exc),
        )

    temp_path: Optional[str] = None
    try:
        handle = tempfile.NamedTemporaryFile(suffix=".owl", delete=False)
        temp_path = handle.name
        handle.close()
        query_graph.serialize(destination=temp_path, format="xml")

        world = World()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            ontology = world.get_ontology(temp_path).load()

            t0 = perf_counter()
            if reasoner_name == "pellet":
                sync_reasoner_pellet(
                    [ontology],
                    infer_property_values=False,
                    infer_data_property_values=False,
                    debug=0,
                )
            else:
                sync_reasoner([ontology], infer_property_values=False, debug=0)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        rdflib_graph = world.as_rdflib_graph()
        members_by_target: Dict[URIRef, Set[Identifier]] = {}
        for target_term, query_class in query_class_by_target.items():
            members = {
                subj
                for subj, _pred, _obj in rdflib_graph.triples((None, RDF.type, query_class))
                if subj in candidate_terms
            }
            members_by_target[target_term] = members

        return BackendQueryResult(
            backend="owlready2",
            elapsed_ms=elapsed_ms,
            members_by_target=members_by_target,
            consistent=True,
        )

    except OwlReadyInconsistentOntologyError:
        return BackendQueryResult(
            backend="owlready2",
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="inconsistent",
            consistent=False,
            error="Ontology reported inconsistent by owlready2.",
        )
    except Exception as exc:
        return BackendQueryResult(
            backend="owlready2",
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            consistent=None,
            error=str(exc),
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _resolve_elk_classpath(
    *,
    elk_classpath: Optional[str],
    elk_jar: Optional[str],
    owlapi_home: Optional[str] = None,
) -> Optional[str]:
    if elk_classpath:
        return elk_classpath
    if elk_jar:
        elk_path = Path(elk_jar)
        if elk_path.is_dir():
            jar_paths = sorted(
                {
                    str(path)
                    for path in elk_path.rglob("*.jar")
                    if path.is_file()
                }
            )
            if jar_paths:
                return os.pathsep.join(jar_paths)
        return elk_jar
    owlapi_root = Path(owlapi_home) if owlapi_home else DEFAULT_OWLAPI_HOME
    classpath_file = owlapi_root / "classpath.txt"
    if classpath_file.exists():
        entries = [
            entry.strip()
            for entry in classpath_file.read_text(encoding="utf-8").split(os.pathsep)
            if entry.strip()
        ]
        if entries and all(Path(entry).exists() for entry in entries):
            return os.pathsep.join(entries)
    maven_repo = owlapi_root / "maven-repo"
    if maven_repo.exists():
        jar_paths = sorted(str(path) for path in maven_repo.rglob("*.jar") if path.is_file())
        if jar_paths:
            return os.pathsep.join(jar_paths)
    return os.environ.get("ELK_CLASSPATH") or os.environ.get("ELK_JAR")


def _compile_elk_helper(
    *,
    classpath: str,
    javac_command: str = "javac",
) -> Path:
    source_path = Path(__file__).resolve().parent / "java" / "ElkOracleRunner.java"
    build_dir = Path(tempfile.gettempdir()) / "dag_elk_oracle_helper"
    class_file = build_dir / "ElkOracleRunner.class"

    needs_compile = not class_file.exists() or source_path.stat().st_mtime > class_file.stat().st_mtime
    if not needs_compile:
        return build_dir

    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        javac_command,
        "--release",
        "11",
        "-cp",
        classpath,
        "-d",
        str(build_dir),
        str(source_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return build_dir


def validate_elk_backend(
    *,
    elk_classpath: Optional[str] = None,
    elk_jar: Optional[str] = None,
    owlapi_home: Optional[str] = None,
    java_command: str = "java",
    javac_command: str = "javac",
    java_options: Optional[Sequence[str]] = None,
) -> Tuple[bool, str, Optional[str]]:
    classpath = _resolve_elk_classpath(
        elk_classpath=elk_classpath,
        elk_jar=elk_jar,
        owlapi_home=owlapi_home,
    )
    if not classpath:
        return (
            False,
            (
                "ELK backend requested but no OWLAPI/ELK classpath could be resolved. "
                "Use comparison/owlapi-5.5.1, --owlapi-home, --elk-classpath, --elk-jar, "
                "or set ELK_CLASSPATH / ELK_JAR."
            ),
            None,
        )
    try:
        helper_dir = _compile_elk_helper(classpath=classpath, javac_command=javac_command)
        run_classpath = os.pathsep.join([str(helper_dir), classpath])
        cmd = [java_command]
        if java_options:
            cmd.extend(java_options)
        cmd.extend(["-cp", run_classpath, "ElkOracleRunner", "--probe"])
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if "PROBE_OK" not in completed.stdout:
            return False, "ELK probe did not return PROBE_OK.", classpath
        return True, "", classpath
    except subprocess.CalledProcessError as exc:
        error_text = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return False, error_text, classpath
    except Exception as exc:
        return False, str(exc), classpath


def run_elk_queries(
    query_graph: Graph,
    query_class_by_target: Dict[URIRef, URIRef],
    candidate_terms: Set[Identifier],
    *,
    elk_classpath: Optional[str] = None,
    elk_jar: Optional[str] = None,
    owlapi_home: Optional[str] = None,
    java_command: str = "java",
    javac_command: str = "javac",
    java_options: Optional[Sequence[str]] = None,
) -> BackendQueryResult:
    classpath = _resolve_elk_classpath(
        elk_classpath=elk_classpath,
        elk_jar=elk_jar,
        owlapi_home=owlapi_home,
    )
    if not classpath:
        return BackendQueryResult(
            backend="elk",
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            error=(
                "ELK backend requested but no OWLAPI/ELK classpath was provided. "
                "Use comparison/owlapi-5.5.1, --owlapi-home, --elk-classpath, "
                "--elk-jar, or set ELK_CLASSPATH / ELK_JAR."
            ),
        )

    workspace_temp_root = Path.cwd() / ".tmp" / "elk-oracle"
    workspace_temp_root.mkdir(parents=True, exist_ok=True)
    temp_root = workspace_temp_root / f"dag-elk-oracle-{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=False)
    try:
        helper_dir = _compile_elk_helper(classpath=classpath, javac_command=javac_command)
        ontology_path = temp_root / "query.owl"
        targets_path = temp_root / "targets.tsv"
        candidates_path = temp_root / "candidates.txt"

        ontology_xml = query_graph.serialize(format="xml")
        if isinstance(ontology_xml, bytes):
            ontology_path.write_bytes(ontology_xml)
        else:
            ontology_path.write_text(ontology_xml, encoding="utf-8")
        targets_path.write_text(
            "".join(
                f"{str(target_term)}\t{str(query_class)}\n"
                for target_term, query_class in query_class_by_target.items()
            ),
            encoding="utf-8",
        )
        candidates_path.write_text(
            "".join(
                f"{str(term)}\n"
                for term in _sorted_terms(candidate_terms)
                if isinstance(term, URIRef)
            ),
            encoding="utf-8",
        )

        run_classpath = os.pathsep.join([str(helper_dir), classpath])
        cmd = [java_command]
        if java_options:
            cmd.extend(java_options)
        cmd.extend(
            [
                "-cp",
                run_classpath,
                "ElkOracleRunner",
                str(ontology_path),
                str(targets_path),
                str(candidates_path),
            ]
        )
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        elapsed_ms = 0.0
        members_by_target: Dict[URIRef, Set[Identifier]] = {
            target: set() for target in query_class_by_target
        }
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if parts[0] == "ELAPSED_MS" and len(parts) == 2:
                elapsed_ms = float(parts[1])
            elif parts[0] == "MEMBER" and len(parts) == 3:
                target_term = URIRef(parts[1])
                member_term = URIRef(parts[2])
                members_by_target.setdefault(target_term, set()).add(member_term)

        return BackendQueryResult(
            backend="elk",
            elapsed_ms=elapsed_ms,
            members_by_target=members_by_target,
            consistent=True,
        )
    except subprocess.CalledProcessError as exc:
        error_text = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return BackendQueryResult(
            backend="elk",
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            error=error_text,
        )
    except Exception as exc:
        return BackendQueryResult(
            backend="elk",
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            error=str(exc),
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _format_member_rows(
    terms: Iterable[Identifier],
    *,
    scores: Optional[Dict[Identifier, float]] = None,
    max_items: int = 20,
) -> str:
    rows: List[str] = []
    for term in _sorted_terms(terms)[:max_items]:
        if scores is None:
            rows.append(f"  - {_render_term(term)}")
        else:
            rows.append(f"  - {_render_term(term)}: {scores.get(term, 0.0):.4f}")
    if not rows:
        return "  (none)"
    return "\n".join(rows)


def print_comparison_report(
    *,
    engine_result: EngineQueryResult,
    oracle_results: Sequence[BackendQueryResult],
    target_classes: Sequence[URIRef],
    query_class_by_target: Dict[URIRef, URIRef],
    threshold: float,
    show_matches: bool = False,
    show_engine_scores: bool = False,
    max_diff_items: int = 20,
    show_timing_breakdown: bool = False,
) -> None:
    print("=== Oracle Comparison ===")
    print(f"Engine threshold: {threshold}")
    print(f"Engine time: {engine_result.elapsed_ms:.3f} ms")
    if show_timing_breakdown:
        print(format_engine_timing_breakdown(engine_result))
    if engine_result.materialization_iterations is not None:
        print(f"Engine materialization iterations: {engine_result.materialization_iterations}")
    if engine_result.dataset is not None:
        print("")
        print(summarize_loaded_kgraph(engine_result.dataset.kg, engine_result.dataset.mapping, max_items=10))

    for backend in oracle_results:
        status = backend.status
        if backend.consistent is True:
            status += ", consistent"
        elif backend.consistent is False:
            status += ", inconsistent"
        print(f"{backend.backend} time: {backend.elapsed_ms:.3f} ms ({status})")
        if backend.error:
            print(f"  error: {backend.error}")

    for target_term in target_classes:
        print("")
        print(f"Target class: {_render_term(target_term)}")
        print(f"Oracle query class: {_render_term(query_class_by_target[target_term])}")

        engine_members = engine_result.members_by_target[target_term]
        print(f"Engine matches: {len(engine_members)}")
        if show_matches or show_engine_scores:
            print(_format_member_rows(
                engine_members,
                scores=engine_result.scores_by_target.get(target_term) if show_engine_scores else None,
                max_items=max_diff_items,
            ))

        for backend in oracle_results:
            backend_members = backend.members_by_target.get(target_term, set())
            only_in_engine = engine_members - backend_members
            only_in_backend = backend_members - engine_members
            agree = backend.status == "ok" and not only_in_engine and not only_in_backend

            print(
                f"{backend.backend}: matches={len(backend_members)}, "
                f"agreement={'yes' if agree else 'no'}"
            )
            if backend.status != "ok":
                continue
            if show_matches:
                print(_format_member_rows(backend_members, max_items=max_diff_items))
            if only_in_engine:
                print("  only in engine:")
                print(_format_member_rows(only_in_engine, max_items=max_diff_items))
            if only_in_backend:
                print(f"  only in {backend.backend}:")
                print(_format_member_rows(only_in_backend, max_items=max_diff_items))


def run_oracle_comparison(
    *,
    schema_paths: Sequence[str],
    data_paths: Sequence[str],
    target_class_specs: Sequence[str],
    device: str = "cuda",
    threshold: float = 0.999,
    include_literals: bool = False,
    include_type_edges: bool = False,
    materialize_hierarchy: bool = True,
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
    materialize_target_roles: Optional[bool] = None,
    materialize_supported_types: bool = False,
    engine_mode: str = "query",
    conflict_policy: str = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
    query_mode: str = "query",
    bridge_supported_definitions: bool = False,
    oracle_backends: Sequence[str] = ("owlrl", "owlready2"),
    owlready2_reasoner: str = "hermit",
    owlapi_home: Optional[str] = None,
    elk_classpath: Optional[str] = None,
    elk_jar: Optional[str] = None,
    elk_java_command: str = "java",
    elk_javac_command: str = "javac",
    elk_java_options: Optional[Sequence[str]] = None,
    show_matches: bool = False,
    show_engine_scores: bool = False,
    max_diff_items: int = 20,
    show_timing_breakdown: bool = False,
    verbose: bool = False,
) -> None:
    resolved_elk_classpath = elk_classpath
    if "elk" in oracle_backends:
        ok, error_text, resolved_classpath = validate_elk_backend(
            elk_classpath=elk_classpath,
            elk_jar=elk_jar,
            owlapi_home=owlapi_home,
            java_command=elk_java_command,
            javac_command=elk_javac_command,
            java_options=elk_java_options,
        )
        if not ok:
            print("=== Oracle Comparison ===")
            print("ELK preflight failed before engine execution.")
            print(error_text)
            return
        resolved_elk_classpath = resolved_classpath

    schema_graph = load_rdflib_graph(schema_paths)
    data_graph = load_rdflib_graph(data_paths)
    resolution = resolve_target_classes(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_class_specs=target_class_specs,
        engine_mode=engine_mode,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
    )
    target_terms = resolution.resolved_targets
    warning_text = format_skipped_target_warnings(resolution, verbose=verbose)
    if warning_text:
        print(warning_text)
        print("")
    if not target_terms:
        print("No target classes resolved for comparison.")
        return

    engine_result = run_engine_queries(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_classes=target_terms,
        device=device,
        threshold=threshold,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_sameas=materialize_sameas,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_target_roles=materialize_target_roles,
        materialize_supported_types=materialize_supported_types,
        engine_mode=engine_mode,
        conflict_policy=conflict_policy,
    )

    candidate_terms = set(engine_result.dataset.mapping.node_terms) if engine_result.dataset else set()
    ontology_graph = Graph()
    for triple in schema_graph:
        ontology_graph.add(triple)
    for triple in data_graph:
        ontology_graph.add(triple)

    query_graph, query_class_by_target = build_oracle_query_graph(
        ontology_graph,
        target_terms,
        mode=query_mode,
        bridge_supported_definitions=bridge_supported_definitions,
    )

    oracle_results: List[BackendQueryResult] = []
    for backend in oracle_backends:
        if backend == "owlrl":
            oracle_results.append(
                run_owlrl_queries(
                    query_graph=query_graph,
                    query_class_by_target=query_class_by_target,
                    candidate_terms=candidate_terms,
                )
            )
        elif backend == "owlready2":
            oracle_results.append(
                run_owlready2_queries(
                    query_graph=query_graph,
                    query_class_by_target=query_class_by_target,
                    candidate_terms=candidate_terms,
                    reasoner_name=owlready2_reasoner,
                )
            )
        elif backend == "elk":
            oracle_results.append(
                run_elk_queries(
                    query_graph=query_graph,
                    query_class_by_target=query_class_by_target,
                    candidate_terms=candidate_terms,
                    elk_classpath=resolved_elk_classpath,
                    elk_jar=elk_jar,
                    owlapi_home=owlapi_home,
                    java_command=elk_java_command,
                    javac_command=elk_javac_command,
                    java_options=elk_java_options,
                )
            )
        else:
            raise ValueError(f"Unsupported oracle backend: {backend}")

    print_comparison_report(
        engine_result=engine_result,
        oracle_results=oracle_results,
        target_classes=target_terms,
        query_class_by_target=query_class_by_target,
        threshold=threshold,
        show_matches=show_matches,
        show_engine_scores=show_engine_scores,
        max_diff_items=max_diff_items,
        show_timing_breakdown=show_timing_breakdown,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compare the current ConstraintDAG reasoner against owlrl, "
            "owlready2, and/or ELK on the currently supported OWL fragment."
        )
    )
    parser.add_argument("--schema", nargs="+", required=True, help="Schema / ontology RDF files.")
    parser.add_argument("--data", nargs="+", required=True, help="Instance data RDF files.")
    parser.add_argument(
        "--target-class",
        nargs="+",
        required=True,
        help=(
            "One or more target class IRIs to compare. Special selectors: "
            "'all', 'all-named-classes', 'all-defined-classes', "
            "'all-inferable-classes'."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--include-literals", action="store_true")
    parser.add_argument("--include-type-edges", action="store_true")
    parser.add_argument("--no-materialize-hierarchy", action="store_true")
    sameas_group = parser.add_mutually_exclusive_group()
    sameas_group.add_argument(
        "--materialize-sameas",
        dest="materialize_sameas",
        action="store_true",
        help="Force explicit sameAs-style preprocessing on.",
    )
    sameas_group.add_argument(
        "--no-materialize-sameas",
        dest="materialize_sameas",
        action="store_false",
        help="Disable sameAs-style preprocessing entirely.",
    )
    parser.set_defaults(materialize_sameas=None)
    haskey_group = parser.add_mutually_exclusive_group()
    haskey_group.add_argument(
        "--materialize-haskey-equality",
        dest="materialize_haskey_equality",
        action="store_true",
        help="Enable HasKey-driven equality generation.",
    )
    haskey_group.add_argument(
        "--no-materialize-haskey-equality",
        dest="materialize_haskey_equality",
        action="store_false",
        help="Disable HasKey-driven equality generation.",
    )
    parser.set_defaults(materialize_haskey_equality=None)
    roles_group = parser.add_mutually_exclusive_group()
    roles_group.add_argument(
        "--materialize-target-roles",
        dest="materialize_target_roles",
        action="store_true",
        help="Force query/filtered-query snapshots to preprocess target-role closure.",
    )
    roles_group.add_argument(
        "--no-materialize-target-roles",
        dest="materialize_target_roles",
        action="store_false",
        help="Force query/filtered-query snapshots to skip target-role preprocessing.",
    )
    parser.set_defaults(materialize_target_roles=None)
    parser.add_argument(
        "--engine-materialize-supported-types",
        action="store_true",
        help="Enable the engine's Horn-style fixpoint pass before evaluating targets.",
    )
    parser.add_argument(
        "--engine-mode",
        choices=["query", "filtered_query", "stratified"],
        default="query",
        help=(
            "query = necessary-condition admissibility; "
            "filtered_query = query candidates pruned by synchronous recheck plus stratified blockers; "
            "stratified = positive sufficient-condition closure plus negative blocker policy."
        ),
    )
    parser.add_argument(
        "--conflict-policy",
        choices=[policy.value for policy in ConflictPolicy],
        default=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
        help="Conflict policy for filtered_query/stratified engine modes.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["query", "native"],
        default="query",
        help=(
            "query = compare against a synthetic equivalentClass oracle query "
            "that mirrors DAG semantics; native = compare against inferred "
            "membership in the original target classes."
        ),
    )
    parser.add_argument(
        "--oracle-bridge-supported-definitions",
        action="store_true",
        help=(
            "Add equivalentClass bridge axioms so the oracles mirror the "
            "engine's Horn-style definitional materialization for supported "
            "named classes."
        ),
    )
    parser.add_argument(
        "--oracles",
        nargs="+",
        choices=["owlrl", "owlready2", "elk"],
        default=["owlrl", "owlready2"],
        help="Which oracle backends to run.",
    )
    parser.add_argument(
        "--owlready2-reasoner",
        choices=["hermit", "pellet"],
        default="hermit",
        help="Reasoner backend to invoke through owlready2.",
    )
    parser.add_argument(
        "--owlapi-home",
        default=str(DEFAULT_OWLAPI_HOME),
        help=(
            "Root directory for OWLAPI reasoner assets such as comparison/owlapi-5.5.1. "
            "If ELK is selected and no explicit classpath override is given, this directory "
            "is used to resolve jars from classpath.txt or maven-repo."
        ),
    )
    parser.add_argument(
        "--elk-classpath",
        help="Explicit classpath override for ELK + OWLAPI jars used by the Java helper backend.",
    )
    parser.add_argument(
        "--elk-jar",
        help="Legacy ELK jar/directory override. Prefer --owlapi-home when using the bundled Maven repo.",
    )
    parser.add_argument(
        "--elk-java-command",
        default="java",
        help="Java executable to use for the ELK backend.",
    )
    parser.add_argument(
        "--elk-javac-command",
        default="javac",
        help="javac executable to use for the ELK backend helper.",
    )
    parser.add_argument("--show-matches", action="store_true")
    parser.add_argument("--show-engine-scores", action="store_true")
    parser.add_argument(
        "--show-timing-breakdown",
        action="store_true",
        help="Print a detailed engine timing breakdown.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all skipped target warnings instead of truncating them.",
    )
    parser.add_argument("--max-diff-items", type=int, default=20)

    args = parser.parse_args()

    run_oracle_comparison(
        schema_paths=args.schema,
        data_paths=args.data,
        target_class_specs=args.target_class,
        device=args.device,
        threshold=args.threshold,
        include_literals=args.include_literals,
        include_type_edges=args.include_type_edges,
        materialize_hierarchy=not args.no_materialize_hierarchy,
        materialize_sameas=args.materialize_sameas,
        materialize_haskey_equality=args.materialize_haskey_equality,
        materialize_target_roles=args.materialize_target_roles,
        materialize_supported_types=args.engine_materialize_supported_types,
        engine_mode=args.engine_mode,
        conflict_policy=args.conflict_policy,
        query_mode=args.query_mode,
        bridge_supported_definitions=(
            args.oracle_bridge_supported_definitions or args.engine_materialize_supported_types
        ),
        oracle_backends=args.oracles,
        owlready2_reasoner=args.owlready2_reasoner,
        owlapi_home=args.owlapi_home,
        elk_classpath=args.elk_classpath,
        elk_jar=args.elk_jar,
        elk_java_command=args.elk_java_command,
        elk_javac_command=args.elk_javac_command,
        show_matches=args.show_matches,
        show_engine_scores=args.show_engine_scores,
        show_timing_breakdown=args.show_timing_breakdown,
        verbose=args.verbose,
        max_diff_items=args.max_diff_items,
    )


if __name__ == "__main__":
    main()
