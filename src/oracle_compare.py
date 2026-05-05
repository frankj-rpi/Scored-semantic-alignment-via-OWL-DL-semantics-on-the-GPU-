from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
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
    SuperDAGExecutionPlan,
    aggregate_rdflib_graphs,
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
    materialize_positive_sufficient_class_inferences,
    materialize_stratified_class_inferences,
    materialize_supported_class_inferences,
    plan_reasoning_preprocessing,
    query_target_is_obviously_supported,
    summarize_loaded_kgraph,
)
from .profiling import (
    ProfileNode,
    ProfileRecorder,
    ProfileSummary,
    ProfileValidationWarning,
    aggregate_by_category,
    render_profile_tree,
    tolerance_ms as profile_tolerance_ms,
    validate_profile_tree,
    write_profile_csv,
    write_profile_json,
)


DEFAULT_OWLAPI_HOME = Path(__file__).resolve().parents[1] / "comparison" / "owlapi-5.5.1"


ORACLE_NS = Namespace("urn:dag-oracle:")


@dataclass
class BackendQueryResult:
    backend: str
    elapsed_ms: float
    members_by_target: Dict[URIRef, Set[Identifier]]
    preprocess_elapsed_ms: float = 0.0
    postprocess_elapsed_ms: float = 0.0
    status: str = "ok"
    consistent: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class EngineQueryResult(BackendQueryResult):
    dataset: Optional[ReasoningDataset] = None
    scores_by_target: Optional[Dict[URIRef, Dict[Identifier, float]]] = None
    materialization_iterations: Optional[int] = None
    engine_mode: str = "admissibility"
    conflict_policy: Optional[str] = None
    stratified_result: Optional[StratifiedMaterializationResult] = None
    filtered_admissibility_result: Optional["FilteredAdmissibilityResult"] = None
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
    data_copy_elapsed_ms: float = 0.0
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
    result_projection_elapsed_ms: float = 0.0
    stratified_iteration_timings: Optional[List[MaterializationIterationTiming]] = None
    stratified_final_dataset_timing: Optional[PreprocessingTimings] = None
    dag_requested_device: str = "cpu"
    dag_effective_device: str = "cpu"
    torch_cuda_available: Optional[bool] = None
    dag_stats_by_target: Optional[Dict[URIRef, "DAGStats"]] = None
    super_dag_plan: Optional[SuperDAGExecutionPlan] = None
    profile_tree: Optional[ProfileNode] = None
    profile_summary: Optional[ProfileSummary] = None


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
    profile_tree: Optional[ProfileNode] = None


@dataclass
class FilteredAdmissibilityResult:
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
    profile_tree: Optional[ProfileNode] = None


FilteredQueryResult = FilteredAdmissibilityResult


@dataclass
class TargetResolutionResult:
    requested_specs: Tuple[str, ...]
    resolved_targets: List[URIRef]
    skipped_targets: List[Tuple[str, str]]


@dataclass
class EngineProfileOptions:
    materialize_hierarchy: Optional[bool]
    materialize_horn_safe_domain_range: Optional[bool]
    materialize_reflexive_properties: Optional[bool]
    materialize_sameas: Optional[bool]
    materialize_haskey_equality: Optional[bool]
    materialize_target_roles: Optional[bool]
    augment_property_domain_range: Optional[bool]
    enable_negative_verification: Optional[bool]
    native_sameas_canonicalization: bool


@dataclass
class DAGStats:
    num_nodes: int
    num_layers: int
    num_edges: int
    num_leaves: int
    max_layer_width: int
    avg_layer_width: float
    max_fan_in: int
    max_fan_out: int
    node_type_counts: Dict[str, int]


PROFILE_ALIASES = {
    "default": "default",
    "gpu-el-lite": "gpu-el-lite",
    "gpu-el": "gpu-el",
    "gpu-el-full": "gpu-el-full",
    "gpu-el-verify": "gpu-el-verify",
    "gpu-e1-lite": "gpu-el-lite",
    "gpu-e1": "gpu-el-lite",
    "gpu-e1-full": "gpu-el-full",
    "gpu-e1-verify": "gpu-el-verify",
}

ENGINE_MODE_ALIASES = {
    "query": "admissibility",
    "admissibility": "admissibility",
    "filtered_query": "filtered_admissibility",
    "filtered-query": "filtered_admissibility",
    "filtered_admissibility": "filtered_admissibility",
    "filtered-admissibility": "filtered_admissibility",
    "stratified": "stratified",
}


def _render_term(term: Identifier) -> str:
    return term.n3() if hasattr(term, "n3") else str(term)


def _compute_dag_stats(dag: ConstraintDAG) -> DAGStats:
    fan_out = [0] * len(dag.nodes)
    num_edges = 0
    max_fan_in = 0
    node_type_counts: Dict[str, int] = {}
    for node in dag.nodes:
        node_type_counts[node.ctype.name] = node_type_counts.get(node.ctype.name, 0) + 1
        child_indices = node.child_indices or []
        fan_in = len(child_indices)
        max_fan_in = max(max_fan_in, fan_in)
        num_edges += fan_in
        for child_idx in child_indices:
            if 0 <= child_idx < len(fan_out):
                fan_out[child_idx] += 1

    layer_widths = [len(layer) for layer in dag.layers]
    return DAGStats(
        num_nodes=len(dag.nodes),
        num_layers=len(dag.layers),
        num_edges=num_edges,
        num_leaves=(len(dag.layers[0]) if dag.layers else 0),
        max_layer_width=max(layer_widths) if layer_widths else 0,
        avg_layer_width=(sum(layer_widths) / len(layer_widths) if layer_widths else 0.0),
        max_fan_in=max_fan_in,
        max_fan_out=max(fan_out) if fan_out else 0,
        node_type_counts=node_type_counts,
    )


def _format_dag_stats(
    dag_stats_by_target: Optional[Dict[URIRef, DAGStats]],
    *,
    super_dag_plan: Optional[SuperDAGExecutionPlan] = None,
    verbose: bool = False,
    max_items: int = 10,
) -> List[str]:
    if not dag_stats_by_target:
        return []

    lines = ["DAG stats:"]
    stat_items = list(dag_stats_by_target.items())
    total_targets = len(stat_items)
    total_nodes = sum(stats.num_nodes for _target_term, stats in stat_items)
    total_edges = sum(stats.num_edges for _target_term, stats in stat_items)
    total_leaves = sum(stats.num_leaves for _target_term, stats in stat_items)
    max_layers = max(stats.num_layers for _target_term, stats in stat_items)
    max_layer_width = max(stats.max_layer_width for _target_term, stats in stat_items)
    max_fan_in = max(stats.max_fan_in for _target_term, stats in stat_items)
    max_fan_out = max(stats.max_fan_out for _target_term, stats in stat_items)
    avg_nodes = total_nodes / total_targets if total_targets else 0.0
    avg_edges = total_edges / total_targets if total_targets else 0.0
    lines.append(
        "  - compiled target DAGs: "
        f"count={total_targets}, total nodes={total_nodes}, total edges={total_edges}, "
        f"total leaves={total_leaves}, max layers={max_layers}, max layer width={max_layer_width}, "
        f"max fan-in={max_fan_in}, max fan-out={max_fan_out}, "
        f"avg nodes/target={avg_nodes:.2f}, avg edges/target={avg_edges:.2f}"
    )
    if super_dag_plan is not None and super_dag_plan.groups:
        total_groups = len(super_dag_plan.groups)
        cyclic_groups = sum(1 for group in super_dag_plan.groups if group.is_cyclic)
        acyclic_groups = total_groups - cyclic_groups
        max_targets_per_group = max(len(group.target_order) for group in super_dag_plan.groups)
        lines.append(
            "  - super-DAG execution plan: "
            f"groups={total_groups}, cyclic groups={cyclic_groups}, acyclic groups={acyclic_groups}, "
            f"max targets/group={max_targets_per_group}"
        )
        display_groups = super_dag_plan.groups if verbose else super_dag_plan.groups[:max_items]
        for idx, group in enumerate(display_groups, start=1):
            group_kind = "cyclic scc" if group.is_cyclic else "acyclic batch"
            preview = ", ".join(_render_term(term) for term in group.target_order[:3])
            if len(group.target_order) > 3:
                preview += f", ... ({len(group.target_order)} targets)"
            lines.append(
                f"      group {idx}: {group_kind}, targets={len(group.target_order)}"
                + (f", preview={preview}" if preview else "")
            )
        if not verbose and total_groups > len(display_groups):
            lines.append(f"      ... and {total_groups - len(display_groups)} more super-DAG groups omitted")

    ordered_items = sorted(
        stat_items,
        key=lambda item: (-item[1].num_layers, -item[1].num_nodes, -item[1].num_edges, str(item[0])),
    )
    display_items = ordered_items if verbose else ordered_items[:max_items]
    for target_term, stats in display_items:
        top_types = sorted(
            stats.node_type_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:4]
        top_types_text = ", ".join(f"{name}={count}" for name, count in top_types)
        lines.append(
            "  - "
            f"{_render_term(target_term)}: layers={stats.num_layers}, nodes={stats.num_nodes}, "
            f"edges={stats.num_edges}, leaves={stats.num_leaves}, "
            f"max layer width={stats.max_layer_width}, avg layer width={stats.avg_layer_width:.2f}, "
            f"max fan-in={stats.max_fan_in}, max fan-out={stats.max_fan_out}"
        )
        if top_types_text:
            lines.append(f"      top node types: {top_types_text}")
    if not verbose and total_targets > len(display_items):
        lines.append(f"  ... and {total_targets - len(display_items)} more target DAGs omitted")
    return lines


def _build_subclass_similarity_matrix(
    dataset: ReasoningDataset,
    compile_context: Optional[OntologyCompileContext],
) -> Optional[torch.Tensor]:
    if compile_context is None:
        return None
    num_classes = len(dataset.mapping.class_terms)
    if num_classes == 0:
        return None

    sim_class = torch.eye(num_classes, dtype=torch.float32)
    for stored_class, stored_idx in dataset.mapping.class_to_idx.items():
        for super_class in compile_context.subclass_supers.get(stored_class, ()):
            super_idx = dataset.mapping.class_to_idx.get(super_class)
            if super_idx is not None:
                sim_class[stored_idx, super_idx] = 1.0
    return sim_class


def _expand_sameas_results_for_reporting(
    dataset: Optional[ReasoningDataset],
    members_by_target: Dict[URIRef, Set[Identifier]],
    scores_by_target: Dict[URIRef, Dict[Identifier, float]],
) -> tuple[Dict[URIRef, Set[Identifier]], Dict[URIRef, Dict[Identifier, float]]]:
    if dataset is None or not dataset.sameas_members_by_canonical:
        return members_by_target, scores_by_target

    expanded_members_by_target: Dict[URIRef, Set[Identifier]] = {}
    expanded_scores_by_target: Dict[URIRef, Dict[Identifier, float]] = {}
    for target_term, members in members_by_target.items():
        expanded_members = set(members)
        expanded_scores = dict(scores_by_target.get(target_term, {}))
        for canonical, aliases in dataset.sameas_members_by_canonical.items():
            canonical_score = expanded_scores.get(canonical)
            if canonical in expanded_members:
                expanded_members.update(aliases)
            if canonical_score is not None:
                for alias in aliases:
                    expanded_scores[alias] = canonical_score
        expanded_members_by_target[target_term] = expanded_members
        expanded_scores_by_target[target_term] = expanded_scores
    return expanded_members_by_target, expanded_scores_by_target


def normalize_engine_profile_name(profile: Optional[str]) -> str:
    if profile is None:
        return "default"
    normalized = profile.strip().lower()
    resolved = PROFILE_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported engine profile: {profile}. "
            f"Supported profiles: {', '.join(sorted(PROFILE_ALIASES))}."
        )
    return resolved


def resolve_super_dag_mode(super_dag: str, profile: Optional[str]) -> str:
    normalized = super_dag.strip().lower()
    if normalized not in {"on", "off", "auto"}:
        raise ValueError("super_dag must be one of: on, off, auto")
    if normalized != "auto":
        return normalized
    resolved_profile = normalize_engine_profile_name(profile)
    if resolved_profile.startswith("gpu-el"):
        return "on"
    return "off"


def normalize_engine_mode_name(engine_mode: str) -> str:
    normalized = engine_mode.strip().lower()
    resolved = ENGINE_MODE_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported engine mode: {engine_mode}. "
            f"Supported modes: {', '.join(sorted(set(ENGINE_MODE_ALIASES.values())))}."
        )
    return resolved


def apply_engine_profile(
    *,
    profile: Optional[str],
    materialize_hierarchy: Optional[bool],
    materialize_horn_safe_domain_range: Optional[bool],
    materialize_reflexive_properties: Optional[bool],
    materialize_sameas: Optional[bool],
    materialize_haskey_equality: Optional[bool],
    materialize_target_roles: Optional[bool],
    augment_property_domain_range: Optional[bool],
    enable_negative_verification: Optional[bool],
) -> EngineProfileOptions:
    resolved_profile = normalize_engine_profile_name(profile)
    if resolved_profile == "default":
        return EngineProfileOptions(
            materialize_hierarchy=materialize_hierarchy,
            materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
            materialize_reflexive_properties=materialize_reflexive_properties,
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_target_roles=materialize_target_roles,
            augment_property_domain_range=augment_property_domain_range,
            enable_negative_verification=enable_negative_verification,
            native_sameas_canonicalization=False,
        )

    def choose(explicit: Optional[bool], profile_default: bool) -> bool:
        return explicit if explicit is not None else profile_default

    verification_default = resolved_profile == "gpu-el-verify"
    sameas_default = resolved_profile in {"gpu-el", "gpu-el-full", "gpu-el-verify"}
    haskey_default = resolved_profile == "gpu-el-full"
    return EngineProfileOptions(
        materialize_hierarchy=choose(materialize_hierarchy, False),
        materialize_horn_safe_domain_range=choose(materialize_horn_safe_domain_range, False),
        materialize_reflexive_properties=choose(materialize_reflexive_properties, False),
        materialize_sameas=choose(materialize_sameas, sameas_default),
        materialize_haskey_equality=choose(materialize_haskey_equality, haskey_default),
        materialize_target_roles=choose(materialize_target_roles, False),
        augment_property_domain_range=choose(augment_property_domain_range, True),
        enable_negative_verification=choose(enable_negative_verification, verification_default),
        native_sameas_canonicalization=sameas_default,
    )


def _sorted_terms(terms: Iterable[Identifier]) -> List[Identifier]:
    return sorted(terms, key=lambda term: _render_term(term))


def _format_elapsed_seconds(elapsed_ms: float) -> str:
    return f"{(elapsed_ms / 1000.0):.3f} s"


def _timing_reconciliation_tolerance_ms(total_elapsed_ms: float) -> float:
    return max(50.0, total_elapsed_ms * 0.01)


def _build_engine_profile_tree(engine_result: EngineQueryResult) -> tuple[ProfileNode, ProfileSummary]:
    if engine_result.profile_tree is not None:
        root = _clone_profile_node(engine_result.profile_tree)
        root.elapsed_ms_inclusive = engine_result.elapsed_ms
        root.meta["device"] = engine_result.dag_effective_device
        warnings = validate_profile_tree(root)
        disjoint_top_level_elapsed_ms = sum(child.elapsed_ms_inclusive for child in root.children)
        residual = root.elapsed_ms_inclusive - disjoint_top_level_elapsed_ms
        root_tol = profile_tolerance_ms(root.elapsed_ms_inclusive)
        if abs(residual) > root_tol:
            warnings.append(
                ProfileValidationWarning(
                    path=root.name,
                    issue="top-level children do not reconcile to engine total",
                    elapsed_ms=abs(residual),
                    tolerance_ms=root_tol,
                )
            )
        summary = ProfileSummary(
            root_elapsed_ms=root.elapsed_ms_inclusive,
            category_totals_ms=aggregate_by_category(root),
            warnings=warnings,
            reconciliation_residual_ms=residual,
        )
        return root, summary

    if engine_result.engine_mode == "stratified" and engine_result.stratified_result is not None:
        stratified_tree = engine_result.stratified_result.profile_tree
        if stratified_tree is not None:
            top_children = list(stratified_tree.children)
            if stratified_tree.self_ms > 0.0:
                top_children.append(
                    ProfileNode(
                        name="stratified_wrapper_overhead",
                        label="stratified wrapper overhead",
                        elapsed_ms_inclusive=stratified_tree.self_ms,
                        meta={"category": "host_runtime"},
                    )
                )
            if engine_result.result_projection_elapsed_ms:
                top_children.append(
                    ProfileNode(
                        name="result_projection",
                        label="result projection",
                        elapsed_ms_inclusive=engine_result.result_projection_elapsed_ms,
                        meta={"category": "host_runtime"},
                    )
                )
            root = ProfileNode(
                name="engine_total",
                label="engine_total",
                elapsed_ms_inclusive=engine_result.elapsed_ms,
                children=top_children,
                meta={"device": engine_result.dag_effective_device},
            )
            warnings = validate_profile_tree(root)
            disjoint_top_level_elapsed_ms = sum(child.elapsed_ms_inclusive for child in root.children)
            residual = root.elapsed_ms_inclusive - disjoint_top_level_elapsed_ms
            root_tol = profile_tolerance_ms(root.elapsed_ms_inclusive)
            if abs(residual) > root_tol:
                warnings.append(
                    ProfileValidationWarning(
                        path=root.name,
                        issue="top-level children do not reconcile to engine total",
                        elapsed_ms=abs(residual),
                        tolerance_ms=root_tol,
                    )
                )
            summary = ProfileSummary(
                root_elapsed_ms=root.elapsed_ms_inclusive,
                category_totals_ms=aggregate_by_category(root),
                warnings=warnings,
                reconciliation_residual_ms=residual,
            )
            return root, summary

    root = ProfileNode(
        name="engine_total",
        label="engine_total",
        elapsed_ms_inclusive=engine_result.elapsed_ms,
        meta={"device": engine_result.dag_effective_device},
    )

    schema_setup = ProfileNode(
        name="schema_setup",
        label="schema_setup",
        elapsed_ms_inclusive=(
            engine_result.schema_cache_elapsed_ms
            + engine_result.sufficient_rule_extraction_elapsed_ms
            + engine_result.sufficient_rule_index_elapsed_ms
            + engine_result.dependency_closure_elapsed_ms
        ),
        meta={"category": "tbox_cacheable"},
    )
    root.add_child(
        ProfileNode(
            name=schema_setup.name,
            label=schema_setup.label,
            elapsed_ms_inclusive=schema_setup.elapsed_ms_inclusive,
            meta=schema_setup.meta,
        )
    )
    schema_setup = root.children[-1]
    if engine_result.schema_cache_elapsed_ms:
        schema_setup.add_child(
            ProfileNode(
                name="schema_cache",
                label="schema cache / compile context",
                elapsed_ms_inclusive=engine_result.schema_cache_elapsed_ms,
                meta={"category": "tbox_cacheable"},
            )
        )
    if engine_result.sufficient_rule_extraction_elapsed_ms:
        schema_setup.add_child(
            ProfileNode(
                name="rule_extraction",
                label="rule extraction",
                elapsed_ms_inclusive=engine_result.sufficient_rule_extraction_elapsed_ms,
                meta={"category": "tbox_cacheable"},
            )
        )
    if engine_result.sufficient_rule_index_elapsed_ms:
        schema_setup.add_child(
            ProfileNode(
                name="rule_index",
                label="rule index",
                elapsed_ms_inclusive=engine_result.sufficient_rule_index_elapsed_ms,
                meta={"category": "tbox_cacheable"},
            )
        )
    if engine_result.dependency_closure_elapsed_ms:
        schema_setup.add_child(
            ProfileNode(
                name="dependency_closure",
                label="dependency closure",
                elapsed_ms_inclusive=engine_result.dependency_closure_elapsed_ms,
                meta={"category": "tbox_cacheable"},
            )
        )

    has_positive_loop = engine_result.stratified_positive_total_elapsed_ms > 0.0
    input_lowering: Optional[ProfileNode] = None
    if not has_positive_loop:
        input_lowering = root.add_child(
            ProfileNode(
                name="input_lowering",
                label="input_lowering",
                elapsed_ms_inclusive=(
                    engine_result.stratified_initial_data_copy_elapsed_ms
                    + engine_result.stratified_initial_ontology_merge_elapsed_ms
                    + engine_result.preprocessing_plan_elapsed_ms
                    + engine_result.sameas_state_init_elapsed_ms
                    + engine_result.mapping_vocab_collect_elapsed_ms
                    + engine_result.mapping_graph_scan_elapsed_ms
                    + engine_result.dataset_build_elapsed_ms
                ),
                meta={"category": "abox_once"},
            )
        )
    if input_lowering is not None and (
        engine_result.stratified_initial_data_copy_elapsed_ms or engine_result.stratified_initial_ontology_merge_elapsed_ms
    ):
        initial_graph_prep = input_lowering.add_child(
            ProfileNode(
                name="initial_graph_prep",
                label="initial_graph_prep",
                elapsed_ms_inclusive=(
                    engine_result.stratified_initial_data_copy_elapsed_ms
                    + engine_result.stratified_initial_ontology_merge_elapsed_ms
                ),
                meta={"category": "abox_once"},
            )
        )
        if engine_result.stratified_initial_data_copy_elapsed_ms:
            initial_graph_prep.add_child(
                ProfileNode(
                    name="data_copy",
                    label="data copy",
                    elapsed_ms_inclusive=engine_result.stratified_initial_data_copy_elapsed_ms,
                    meta={"category": "abox_once"},
                )
            )
        if engine_result.stratified_initial_ontology_merge_elapsed_ms:
            initial_graph_prep.add_child(
                ProfileNode(
                    name="ontology_merge",
                    label="ontology merge",
                    elapsed_ms_inclusive=engine_result.stratified_initial_ontology_merge_elapsed_ms,
                    meta={"category": "abox_once"},
                )
            )
    if input_lowering is not None and engine_result.preprocessing_plan_elapsed_ms:
        input_lowering.add_child(
            ProfileNode(
                name="preprocessing_plan",
                label="preprocessing plan",
                elapsed_ms_inclusive=engine_result.preprocessing_plan_elapsed_ms,
                meta={"category": "abox_once"},
            )
        )
    if input_lowering is not None and engine_result.sameas_state_init_elapsed_ms:
        input_lowering.add_child(
            ProfileNode(
                name="sameas_state_init",
                label="sameAs state init",
                elapsed_ms_inclusive=engine_result.sameas_state_init_elapsed_ms,
                meta={"category": "abox_once"},
            )
        )
    if input_lowering is not None:
        scan_cache = input_lowering.add_child(
        ProfileNode(
            name="scan_cache_index_prep",
            label="scan cache/index prep",
            elapsed_ms_inclusive=engine_result.mapping_vocab_collect_elapsed_ms + engine_result.mapping_graph_scan_elapsed_ms,
            meta={"category": "abox_once"},
        )
        )
        if engine_result.mapping_vocab_collect_elapsed_ms:
            scan_cache.add_child(
                ProfileNode(
                    name="mapping_vocab_collect",
                    label="vocab collect",
                    elapsed_ms_inclusive=engine_result.mapping_vocab_collect_elapsed_ms,
                    meta={"category": "abox_once"},
                )
            )
        if engine_result.mapping_graph_scan_elapsed_ms:
            scan_cache.add_child(
                ProfileNode(
                    name="mapping_graph_scan",
                    label="graph scan",
                    elapsed_ms_inclusive=engine_result.mapping_graph_scan_elapsed_ms,
                    meta={"category": "abox_once"},
                )
            )
        dataset_assembly = input_lowering.add_child(
            ProfileNode(
                name="dataset_native_assembly",
                label="dataset build/native assembly",
                elapsed_ms_inclusive=engine_result.dataset_build_elapsed_ms,
                meta={"category": "host_runtime"},
            )
        )
    else:
        dataset_assembly = None
    for name, label, elapsed, category in (
        ("data_copy", "data copy", engine_result.data_copy_elapsed_ms, "host_runtime"),
        ("ontology_merge", "ontology merge", engine_result.ontology_merge_elapsed_ms, "host_runtime"),
        ("hierarchy", "hierarchy", engine_result.hierarchy_elapsed_ms, "host_runtime"),
        ("atomic_domain_range", "atomic domain/range", engine_result.atomic_domain_range_elapsed_ms, "host_runtime"),
        ("horn_safe_domain_range", "horn-safe domain/range", engine_result.horn_safe_domain_range_elapsed_ms, "host_runtime"),
        ("sameas", "sameAs", engine_result.sameas_elapsed_ms, "host_runtime"),
        ("reflexive", "reflexive", engine_result.reflexive_elapsed_ms, "host_runtime"),
        ("target_roles", "target roles", engine_result.target_role_elapsed_ms, "host_runtime"),
    ):
        if elapsed and dataset_assembly is not None:
            dataset_assembly.add_child(
                ProfileNode(name=name, label=label, elapsed_ms_inclusive=elapsed, meta={"category": category})
            )
    if dataset_assembly is not None:
        kgraph_build = dataset_assembly.add_child(
        ProfileNode(
            name="kgraph_build",
            label="kgraph build",
            elapsed_ms_inclusive=engine_result.kgraph_build_elapsed_ms,
            meta={"category": "host_runtime"},
        )
        )
        mapping_finalize = kgraph_build.add_child(
        ProfileNode(
            name="mapping_finalize",
            label="mapping finalize",
            elapsed_ms_inclusive=engine_result.mapping_sort_elapsed_ms + engine_result.mapping_index_elapsed_ms,
            meta={"category": "host_runtime"},
        )
        )
        if engine_result.mapping_sort_elapsed_ms:
            mapping_finalize.add_child(ProfileNode("sort", "sort", engine_result.mapping_sort_elapsed_ms, meta={"category": "host_runtime"}))
        if engine_result.mapping_index_elapsed_ms:
            mapping_finalize.add_child(ProfileNode("index", "index", engine_result.mapping_index_elapsed_ms, meta={"category": "host_runtime"}))
        kgraph_internals = kgraph_build.add_child(
        ProfileNode(
            name="kgraph_internals",
            label="kgraph internals",
            elapsed_ms_inclusive=(
                engine_result.kgraph_edge_bucket_elapsed_ms
                + engine_result.kgraph_negative_helper_elapsed_ms
                + engine_result.kgraph_literal_feature_elapsed_ms
                + engine_result.kgraph_adjacency_elapsed_ms
            ),
            meta={"category": "host_runtime"},
        )
        )
        for name, label, elapsed in (
            ("edge_buckets", "edge buckets", engine_result.kgraph_edge_bucket_elapsed_ms),
            ("negative_helpers", "negative helpers", engine_result.kgraph_negative_helper_elapsed_ms),
            ("literal_features", "literal features", engine_result.kgraph_literal_feature_elapsed_ms),
            ("adjacency", "adjacency", engine_result.kgraph_adjacency_elapsed_ms),
        ):
            if elapsed:
                kgraph_internals.add_child(ProfileNode(name, label, elapsed, meta={"category": "host_runtime"}))

    positive_profile_tree: Optional[ProfileNode] = None
    if engine_result.stratified_result is not None:
        positive_profile_tree = engine_result.stratified_result.positive_result.profile_tree
    elif engine_result.filtered_admissibility_result is None and engine_result.engine_mode in {"admissibility", "filtered_admissibility"}:
        positive_profile_tree = None

    if has_positive_loop:
        if positive_profile_tree is not None:
            positive_reasoning = ProfileNode(
                name=positive_profile_tree.name,
                label=(
                    "stratified positive OWA loop"
                    if engine_result.engine_mode == "stratified"
                    else "initial forward Task M pass"
                ),
                elapsed_ms_inclusive=positive_profile_tree.elapsed_ms_inclusive,
                children=list(positive_profile_tree.children),
                meta=dict(positive_profile_tree.meta),
            )
            root.add_child(positive_reasoning)
        else:
            positive_reasoning = root.add_child(
                ProfileNode(
                    name="positive_reasoning",
                    label=(
                        "stratified positive OWA loop"
                        if engine_result.engine_mode == "stratified"
                        else "initial forward Task M pass"
                    ),
                    elapsed_ms_inclusive=engine_result.stratified_positive_total_elapsed_ms,
                    meta={"category": "host_runtime"},
                )
            )
        if schema_setup in root.children:
            root.children.remove(schema_setup)
            if all(child.name != "schema_setup" for child in positive_reasoning.children):
                positive_reasoning.children.insert(0, schema_setup)
        if positive_profile_tree is None:
            for iteration_timing in engine_result.stratified_iteration_timings or ():
                scan_total = iteration_timing.mapping_vocab_collect_elapsed_ms + iteration_timing.mapping_graph_scan_elapsed_ms
                iteration_total = (
                    scan_total
                    + iteration_timing.dataset_build_elapsed_ms
                    + iteration_timing.reasoner_setup_elapsed_ms
                    + iteration_timing.dag_compile_elapsed_ms
                    + iteration_timing.dag_eval_elapsed_ms
                    + iteration_timing.assertion_update_elapsed_ms
                )
                iteration_node = positive_reasoning.add_child(
                    ProfileNode(
                        name=f"iteration_{iteration_timing.iteration}",
                        label=f"iteration_{iteration_timing.iteration}",
                        elapsed_ms_inclusive=iteration_total,
                        meta={
                            "category": "host_runtime",
                            "iteration": iteration_timing.iteration,
                            "refresh_count": iteration_timing.dataset_refresh_count,
                        },
                    )
                )
                if iteration_timing.dataset_build_elapsed_ms:
                    dataset_refresh = iteration_node.add_child(
                        ProfileNode(
                            name="dataset_refresh",
                            label="dataset_refresh",
                            elapsed_ms_inclusive=iteration_timing.dataset_build_elapsed_ms,
                            meta={"category": "host_runtime"},
                        )
                    )
                    for name, label, elapsed in (
                        ("data_copy", "data copy", iteration_timing.data_copy_elapsed_ms),
                        ("ontology_merge", "ontology merge", iteration_timing.ontology_merge_elapsed_ms),
                        ("hierarchy", "hierarchy", iteration_timing.hierarchy_elapsed_ms),
                        ("atomic_domain_range", "atomic domain/range", iteration_timing.atomic_domain_range_elapsed_ms),
                        ("horn_safe_domain_range", "horn-safe domain/range", iteration_timing.horn_safe_domain_range_elapsed_ms),
                        ("sameas", "sameAs", iteration_timing.sameas_elapsed_ms),
                        ("reflexive", "reflexive", iteration_timing.reflexive_elapsed_ms),
                        ("target_roles", "target roles", iteration_timing.target_role_elapsed_ms),
                    ):
                        if elapsed:
                            dataset_refresh.add_child(ProfileNode(name, label, elapsed, meta={"category": "host_runtime"}))
                    kgraph_iter = dataset_refresh.add_child(
                        ProfileNode("kgraph_build", "kgraph build", iteration_timing.kgraph_build_elapsed_ms, meta={"category": "host_runtime"})
                    )
                    mapping_iter = kgraph_iter.add_child(
                        ProfileNode(
                            "mapping_finalize",
                            "mapping finalize",
                            iteration_timing.mapping_sort_elapsed_ms + iteration_timing.mapping_index_elapsed_ms,
                            meta={"category": "host_runtime"},
                        )
                    )
                    if iteration_timing.mapping_sort_elapsed_ms:
                        mapping_iter.add_child(ProfileNode("mapping_sort", "mapping sort", iteration_timing.mapping_sort_elapsed_ms, meta={"category": "host_runtime"}))
                    if iteration_timing.mapping_index_elapsed_ms:
                        mapping_iter.add_child(ProfileNode("mapping_index", "mapping index", iteration_timing.mapping_index_elapsed_ms, meta={"category": "host_runtime"}))
                    internals_iter = kgraph_iter.add_child(
                        ProfileNode(
                            "kgraph_internals",
                            "kgraph internals",
                            iteration_timing.kgraph_edge_bucket_elapsed_ms
                            + iteration_timing.kgraph_negative_helper_elapsed_ms
                            + iteration_timing.kgraph_literal_feature_elapsed_ms
                            + iteration_timing.kgraph_adjacency_elapsed_ms,
                            meta={"category": "host_runtime"},
                        )
                    )
                    for name, label, elapsed in (
                        ("edge_buckets", "edge buckets", iteration_timing.kgraph_edge_bucket_elapsed_ms),
                        ("negative_helpers", "negative helpers", iteration_timing.kgraph_negative_helper_elapsed_ms),
                        ("literal_features", "literal features", iteration_timing.kgraph_literal_feature_elapsed_ms),
                        ("adjacency", "adjacency", iteration_timing.kgraph_adjacency_elapsed_ms),
                    ):
                        if elapsed:
                            internals_iter.add_child(ProfileNode(name, label, elapsed, meta={"category": "host_runtime"}))
                if scan_total:
                    scan_iter = iteration_node.add_child(
                        ProfileNode("scan_cache_index_prep", "scan cache/index prep", scan_total, meta={"category": "abox_once"})
                    )
                    if iteration_timing.mapping_vocab_collect_elapsed_ms:
                        scan_iter.add_child(ProfileNode("mapping_vocab_collect", "vocab collect", iteration_timing.mapping_vocab_collect_elapsed_ms, meta={"category": "abox_once"}))
                    if iteration_timing.mapping_graph_scan_elapsed_ms:
                        scan_iter.add_child(ProfileNode("mapping_graph_scan", "graph scan", iteration_timing.mapping_graph_scan_elapsed_ms, meta={"category": "abox_once"}))
                if iteration_timing.reasoner_setup_elapsed_ms:
                    iteration_node.add_child(ProfileNode("reasoner_setup", "reasoner setup", iteration_timing.reasoner_setup_elapsed_ms, meta={"category": "host_runtime", "device": engine_result.dag_effective_device}))
                if iteration_timing.dag_compile_elapsed_ms:
                    iteration_node.add_child(ProfileNode("dag_compile", "dag compile", iteration_timing.dag_compile_elapsed_ms, meta={"category": "host_runtime", "device": engine_result.dag_effective_device}))
                if iteration_timing.dag_eval_elapsed_ms:
                    iteration_node.add_child(ProfileNode("dag_eval", "dag eval", iteration_timing.dag_eval_elapsed_ms, meta={"category": "device_runtime", "device": engine_result.dag_effective_device}))
                if iteration_timing.assertion_update_elapsed_ms:
                    iteration_node.add_child(ProfileNode("assertion_update", "assertion update", iteration_timing.assertion_update_elapsed_ms, meta={"category": "host_runtime"}))
    elif engine_result.dag_compile_elapsed_ms or engine_result.dag_eval_elapsed_ms:
        query_evaluation = root.add_child(
            ProfileNode(
                name="query_evaluation",
                label="query_evaluation",
                elapsed_ms_inclusive=engine_result.dag_compile_elapsed_ms + engine_result.dag_eval_elapsed_ms,
                meta={"category": "host_runtime"},
            )
        )
        if engine_result.dag_compile_elapsed_ms:
            query_evaluation.add_child(
                ProfileNode(
                    "dag_compile",
                    "dag compile",
                    engine_result.dag_compile_elapsed_ms,
                    meta={"category": "host_runtime", "device": engine_result.dag_effective_device},
                )
            )
        if engine_result.dag_eval_elapsed_ms:
            query_evaluation.add_child(
                ProfileNode(
                    "dag_eval",
                    "dag eval",
                    engine_result.dag_eval_elapsed_ms,
                    meta={"category": "device_runtime", "device": engine_result.dag_effective_device},
                )
            )

    if engine_result.engine_mode == "stratified":
        negative_reasoning_total = engine_result.stratified_negative_blocker_elapsed_ms
        if negative_reasoning_total:
            negative_reasoning = root.add_child(ProfileNode("negative_reasoning", "negative_reasoning", negative_reasoning_total, meta={"category": "host_runtime"}))
            negative_reasoning.add_child(ProfileNode("negative_blocker", "negative/blocker", engine_result.stratified_negative_blocker_elapsed_ms, meta={"category": "host_runtime"}))
        assignment_reporting_total = (
            engine_result.stratified_assignment_status_elapsed_ms
            + engine_result.stratified_conflict_policy_elapsed_ms
            + engine_result.stratified_reporting_compile_elapsed_ms
        )
        if assignment_reporting_total:
            assignment_reporting = root.add_child(ProfileNode("assignment_reporting", "assignment_reporting", assignment_reporting_total, meta={"category": "host_runtime"}))
            if engine_result.stratified_assignment_status_elapsed_ms:
                assignment_reporting.add_child(ProfileNode("assignment_status", "assignment status", engine_result.stratified_assignment_status_elapsed_ms, meta={"category": "host_runtime"}))
            if engine_result.stratified_conflict_policy_elapsed_ms:
                assignment_reporting.add_child(ProfileNode("conflict_policy", "conflict policy", engine_result.stratified_conflict_policy_elapsed_ms, meta={"category": "host_runtime"}))
            if engine_result.stratified_reporting_compile_elapsed_ms:
                assignment_reporting.add_child(ProfileNode("final_reporting_compile", "final reporting compile", engine_result.stratified_reporting_compile_elapsed_ms, meta={"category": "host_runtime"}))

    warnings = validate_profile_tree(root)
    disjoint_top_level_elapsed_ms = sum(child.elapsed_ms_inclusive for child in root.children)
    residual = root.elapsed_ms_inclusive - disjoint_top_level_elapsed_ms
    root_tol = profile_tolerance_ms(root.elapsed_ms_inclusive)
    if abs(residual) > root_tol:
        warnings.append(
            ProfileValidationWarning(
                path=root.name,
                issue="top-level children do not reconcile to engine total",
                elapsed_ms=abs(residual),
                tolerance_ms=root_tol,
            )
        )
    summary = ProfileSummary(
        root_elapsed_ms=root.elapsed_ms_inclusive,
        category_totals_ms=aggregate_by_category(root),
        warnings=warnings,
        reconciliation_residual_ms=residual,
    )
    return root, summary


@lru_cache(maxsize=8)
def _resolve_effective_torch_device(requested_device: str) -> str:
    normalized = (requested_device or "cpu").strip().lower()
    if normalized.startswith("cuda"):
        if not _torch_cuda_available():
            return "unavailable"
        try:
            return str(torch.device(requested_device))
        except Exception:
            return "cuda"
    try:
        return str(torch.device(requested_device))
    except Exception:
        return requested_device


@lru_cache(maxsize=1)
def _torch_cuda_available() -> bool:
    return torch.cuda.is_available()


def _reported_torch_cuda_available(requested_device: str) -> Optional[bool]:
    normalized = (requested_device or "cpu").strip().lower()
    if normalized.startswith("cuda"):
        return _torch_cuda_available()
    return None


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


def format_engine_timing_breakdown(
    engine_result: EngineQueryResult,
    *,
    verbose: bool = False,
) -> str:
    root = engine_result.profile_tree
    summary = engine_result.profile_summary
    if root is None or summary is None:
        root, summary = _build_engine_profile_tree(engine_result)
    lines = ["Engine timing breakdown:"]
    cuda_status = (
        "not_checked"
        if engine_result.torch_cuda_available is None
        else str(engine_result.torch_cuda_available)
    )
    lines.append(
        "  - DAG device: "
        f"requested={engine_result.dag_requested_device}, "
        f"effective={engine_result.dag_effective_device}, "
        f"torch.cuda.is_available={cuda_status}"
    )
    rendered_tree = render_profile_tree(root, warnings=summary.warnings)
    lines.extend(f"  {line}" for line in rendered_tree.splitlines())
    lines.append(
        "  - timing reconciliation: "
        f"engine={_format_elapsed_seconds(summary.root_elapsed_ms)}, "
        f"disjoint accounted={_format_elapsed_seconds(summary.root_elapsed_ms - summary.reconciliation_residual_ms)}, "
        f"residual={_format_elapsed_seconds(summary.reconciliation_residual_ms)}"
    )
    if abs(summary.reconciliation_residual_ms) > _timing_reconciliation_tolerance_ms(summary.root_elapsed_ms):
        lines.append(
            "  - timing warning: disjoint phase totals do not reconcile with engine time "
            f"within tolerance ({_format_elapsed_seconds(_timing_reconciliation_tolerance_ms(summary.root_elapsed_ms))})"
        )
    lines.extend(
        _format_dag_stats(
            engine_result.dag_stats_by_target,
            super_dag_plan=engine_result.super_dag_plan,
            verbose=verbose,
        )
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
    engine_mode: str = "admissibility",
    include_literals: bool = False,
    include_type_edges: bool = False,
    materialize_hierarchy: Optional[bool] = None,
    augment_property_domain_range: Optional[bool] = None,
) -> TargetResolutionResult:
    ontology_graph = schema_graph

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
        schema_graph,
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
                    compile_context=compile_context,
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
    *,
    source_graph: Optional[Graph] = None,
) -> Identifier:
    source = query_graph if source_graph is None else source_graph
    root_exprs = _collect_query_root_expressions(source, target_term)

    for expr in source.objects(target_term, OWL.disjointWith):
        neg_expr = BNode()
        query_graph.add((neg_expr, OWL.complementOf, expr))
        query_graph.add((neg_expr, RDF.type, OWL.Class))
        root_exprs.append(neg_expr)

    for expr in source.subjects(OWL.disjointWith, target_term):
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

    query_class_by_target: Dict[URIRef, URIRef] = {}

    if mode == "native":
        for target in target_classes:
            target_term = URIRef(target) if isinstance(target, str) else target
            query_class_by_target[target_term] = target_term
        return ontology_graph, query_class_by_target

    if bridge_supported_definitions:
        query_graph = _copy_graph(ontology_graph)
        query_graph = add_definitional_bridge_axioms(query_graph)
    else:
        query_overlay = Graph()
        query_graph = query_overlay
        query_source_graph = aggregate_rdflib_graphs((ontology_graph, query_overlay))

    for target in target_classes:
        target_term = URIRef(target) if isinstance(target, str) else target

        query_hash = sha1(str(target_term).encode("utf-8")).hexdigest()
        query_class = URIRef(ORACLE_NS[f"query/{query_hash}"])
        query_graph.add((query_class, RDF.type, OWL.Class))
        query_graph.add((query_class, RDFS.label, Literal(f"OracleQuery({target_term})")))

        query_graph.add((
            query_class,
            OWL.equivalentClass,
            _build_query_expression_term(
                query_graph,
                target_term,
                source_graph=(query_graph if bridge_supported_definitions else query_source_graph),
            ),
        ))

        query_class_by_target[target_term] = query_class

    return (
        query_graph if bridge_supported_definitions else aggregate_rdflib_graphs((ontology_graph, query_graph)),
        query_class_by_target,
    )


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

    sim_class = None
    if not dataset.preprocessing_plan or not dataset.preprocessing_plan.materialize_hierarchy.enabled:
        sim_class = _build_subclass_similarity_matrix(dataset, compile_context)

    reasoner = DAGReasoner(dataset.kg, device=device, sim_class=sim_class)
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


def _clone_profile_node(node: ProfileNode) -> ProfileNode:
    return ProfileNode(
        name=node.name,
        label=node.label,
        elapsed_ms_inclusive=node.elapsed_ms_inclusive,
        children=[_clone_profile_node(child) for child in node.children],
        meta=dict(node.meta),
    )


def _build_query_snapshot_profile_tree(
    *,
    dataset_build_elapsed_ms: float,
    data_copy_elapsed_ms: float,
    ontology_merge_elapsed_ms: float,
    hierarchy_elapsed_ms: float,
    atomic_domain_range_elapsed_ms: float,
    horn_safe_domain_range_elapsed_ms: float,
    sameas_elapsed_ms: float,
    reflexive_elapsed_ms: float,
    target_role_elapsed_ms: float,
    kgraph_build_elapsed_ms: float,
    mapping_vocab_collect_elapsed_ms: float,
    mapping_graph_scan_elapsed_ms: float,
    mapping_sort_elapsed_ms: float,
    mapping_index_elapsed_ms: float,
    kgraph_edge_bucket_elapsed_ms: float,
    kgraph_negative_helper_elapsed_ms: float,
    kgraph_literal_feature_elapsed_ms: float,
    kgraph_adjacency_elapsed_ms: float,
    dag_compile_elapsed_ms: float,
    dag_eval_elapsed_ms: float,
    device: str,
    name: str = "admissibility_snapshot",
    label: str = "admissibility snapshot",
) -> ProfileNode:
    root = ProfileNode(
        name=name,
        label=label,
        elapsed_ms_inclusive=dataset_build_elapsed_ms + dag_compile_elapsed_ms + dag_eval_elapsed_ms,
        meta={"category": "host_runtime", "device": device},
    )
    scan_total = mapping_vocab_collect_elapsed_ms + mapping_graph_scan_elapsed_ms
    if scan_total or dataset_build_elapsed_ms:
        input_lowering = root.add_child(
            ProfileNode(
                name="input_lowering",
                label="input_lowering",
                elapsed_ms_inclusive=scan_total + dataset_build_elapsed_ms,
                meta={"category": "abox_once"},
            )
        )
        if scan_total:
            scan_cache = input_lowering.add_child(
                ProfileNode(
                    name="scan_cache_index_prep",
                    label="scan cache/index prep",
                    elapsed_ms_inclusive=scan_total,
                    meta={"category": "abox_once"},
                )
            )
            if mapping_vocab_collect_elapsed_ms:
                scan_cache.add_child(
                    ProfileNode(
                        name="mapping_vocab_collect",
                        label="vocab collect",
                        elapsed_ms_inclusive=mapping_vocab_collect_elapsed_ms,
                        meta={"category": "abox_once"},
                    )
                )
            if mapping_graph_scan_elapsed_ms:
                scan_cache.add_child(
                    ProfileNode(
                        name="mapping_graph_scan",
                        label="graph scan",
                        elapsed_ms_inclusive=mapping_graph_scan_elapsed_ms,
                        meta={"category": "abox_once"},
                    )
                )
        if dataset_build_elapsed_ms:
            dataset_assembly = input_lowering.add_child(
                ProfileNode(
                    name="dataset_native_assembly",
                    label="dataset build/native assembly",
                    elapsed_ms_inclusive=dataset_build_elapsed_ms,
                    meta={"category": "host_runtime"},
                )
            )
            for child_name, child_label, child_elapsed, category in (
                ("data_copy", "data copy", data_copy_elapsed_ms, "host_runtime"),
                ("ontology_merge", "ontology merge", ontology_merge_elapsed_ms, "host_runtime"),
                ("hierarchy", "hierarchy", hierarchy_elapsed_ms, "host_runtime"),
                ("atomic_domain_range", "atomic domain/range", atomic_domain_range_elapsed_ms, "host_runtime"),
                ("horn_safe_domain_range", "horn-safe domain/range", horn_safe_domain_range_elapsed_ms, "host_runtime"),
                ("sameas", "sameAs", sameas_elapsed_ms, "host_runtime"),
                ("reflexive", "reflexive", reflexive_elapsed_ms, "host_runtime"),
                ("target_roles", "target roles", target_role_elapsed_ms, "host_runtime"),
            ):
                if child_elapsed:
                    dataset_assembly.add_child(
                        ProfileNode(
                            name=child_name,
                            label=child_label,
                            elapsed_ms_inclusive=child_elapsed,
                            meta={"category": category},
                        )
                    )
            if kgraph_build_elapsed_ms:
                kgraph_build = dataset_assembly.add_child(
                    ProfileNode(
                        name="kgraph_build",
                        label="kgraph build",
                        elapsed_ms_inclusive=kgraph_build_elapsed_ms,
                        meta={"category": "host_runtime"},
                    )
                )
                mapping_finalize_total = mapping_sort_elapsed_ms + mapping_index_elapsed_ms
                if mapping_finalize_total:
                    mapping_finalize = kgraph_build.add_child(
                        ProfileNode(
                            name="mapping_finalize",
                            label="mapping finalize",
                            elapsed_ms_inclusive=mapping_finalize_total,
                            meta={"category": "host_runtime"},
                        )
                    )
                    if mapping_sort_elapsed_ms:
                        mapping_finalize.add_child(
                            ProfileNode(
                                name="mapping_sort",
                                label="mapping sort",
                                elapsed_ms_inclusive=mapping_sort_elapsed_ms,
                                meta={"category": "host_runtime"},
                            )
                        )
                    if mapping_index_elapsed_ms:
                        mapping_finalize.add_child(
                            ProfileNode(
                                name="mapping_index",
                                label="mapping index",
                                elapsed_ms_inclusive=mapping_index_elapsed_ms,
                                meta={"category": "host_runtime"},
                            )
                        )
                kgraph_internal_total = (
                    kgraph_edge_bucket_elapsed_ms
                    + kgraph_negative_helper_elapsed_ms
                    + kgraph_literal_feature_elapsed_ms
                    + kgraph_adjacency_elapsed_ms
                )
                if kgraph_internal_total:
                    kgraph_internals = kgraph_build.add_child(
                        ProfileNode(
                            name="kgraph_internals",
                            label="kgraph internals",
                            elapsed_ms_inclusive=kgraph_internal_total,
                            meta={"category": "host_runtime"},
                        )
                    )
                    for child_name, child_label, child_elapsed in (
                        ("edge_buckets", "edge buckets", kgraph_edge_bucket_elapsed_ms),
                        ("negative_helpers", "negative helpers", kgraph_negative_helper_elapsed_ms),
                        ("literal_features", "literal features", kgraph_literal_feature_elapsed_ms),
                        ("adjacency", "adjacency", kgraph_adjacency_elapsed_ms),
                    ):
                        if child_elapsed:
                            kgraph_internals.add_child(
                                ProfileNode(
                                    name=child_name,
                                    label=child_label,
                                    elapsed_ms_inclusive=child_elapsed,
                                    meta={"category": "host_runtime"},
                                )
                            )
    if dag_compile_elapsed_ms:
        root.add_child(
            ProfileNode(
                name="dag_compile",
                label="dag compile",
                elapsed_ms_inclusive=dag_compile_elapsed_ms,
                meta={"category": "host_runtime"},
            )
        )
    if dag_eval_elapsed_ms:
        root.add_child(
            ProfileNode(
                name="dag_eval",
                label="dag eval",
                elapsed_ms_inclusive=dag_eval_elapsed_ms,
                meta={"category": "device_runtime", "device": device},
            )
        )
    return root


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
    materialize_horn_safe_domain_range: Optional[bool],
    materialize_sameas: Optional[bool],
    materialize_haskey_equality: Optional[bool],
    materialize_reflexive_properties: Optional[bool],
    materialize_target_roles: Optional[bool],
    augment_property_domain_range: Optional[bool],
    native_sameas_canonicalization: bool = False,
) -> QueryEvaluationSnapshot:
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
    dag_compile_elapsed_ms = 0.0
    dag_eval_elapsed_ms = 0.0

    initial_dataset = build_reasoning_dataset_from_graphs(
        schema_graph=schema_graph,
        data_graph=data_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_sameas=materialize_sameas,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_reflexive_properties=materialize_reflexive_properties,
        materialize_target_roles=materialize_target_roles,
        native_sameas_canonicalization=native_sameas_canonicalization,
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
                materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_reflexive_properties=materialize_reflexive_properties,
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
            materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_reflexive_properties=materialize_reflexive_properties,
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

    snapshot_tree = _build_query_snapshot_profile_tree(
        dataset_build_elapsed_ms=dataset_build_elapsed_ms,
        data_copy_elapsed_ms=(
            dataset.preprocessing_timings.data_copy_elapsed_ms
            if dataset.preprocessing_timings is not None
            else 0.0
        ),
        ontology_merge_elapsed_ms=(
            dataset.preprocessing_timings.ontology_merge_elapsed_ms
            if dataset.preprocessing_timings is not None
            else 0.0
        ),
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
        device=device,
    )

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
        profile_tree=snapshot_tree,
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
    materialize_horn_safe_domain_range: Optional[bool] = None,
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
    materialize_reflexive_properties: Optional[bool] = None,
    materialize_target_roles: Optional[bool] = None,
    materialize_supported_types: bool = False,
    augment_property_domain_range: Optional[bool] = None,
    native_sameas_canonicalization: bool = False,
    engine_mode: str = "admissibility",
    conflict_policy: str = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
    enable_negative_verification: Optional[bool] = None,
    enable_super_dag: bool = False,
) -> EngineQueryResult:
    engine_mode = normalize_engine_mode_name(engine_mode)
    dag_requested_device = device
    dag_effective_device = _resolve_effective_torch_device(device)
    torch_cuda_available = _reported_torch_cuda_available(device)
    device_to_use = "cpu" if dag_effective_device == "unavailable" else dag_effective_device

    t0 = perf_counter()
    engine_profiler: Optional[ProfileRecorder] = None
    if engine_mode in {"admissibility", "filtered_admissibility", "stratified"}:
        engine_profiler = ProfileRecorder()
        engine_profiler.start_root(
            "engine_total",
            "engine_total",
            device=dag_effective_device,
        )
    iterations: Optional[int] = None
    stratified_result: Optional[StratifiedMaterializationResult] = None
    filtered_admissibility_result: Optional[FilteredAdmissibilityResult] = None
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
    result_projection_elapsed_ms = 0.0
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
    data_copy_elapsed_ms = 0.0
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
    query_input_data_graph = data_graph

    if engine_mode in {"admissibility", "filtered_admissibility"}:
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "positive_reasoning",
                "initial forward Task M pass",
                category="host_runtime",
            ) as positive_node:
                positive_result = materialize_positive_sufficient_class_inferences(
                    schema_graph=schema_graph,
                    data_graph=data_graph,
                    include_literals=include_literals,
                    include_type_edges=include_type_edges,
                    materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
                    materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                    materialize_sameas=materialize_sameas,
                    materialize_haskey_equality=materialize_haskey_equality,
                    materialize_reflexive_properties=materialize_reflexive_properties,
                    materialize_target_roles=False,
                    native_sameas_canonicalization=native_sameas_canonicalization,
                    target_classes=target_classes,
                    threshold=threshold,
                    device=device_to_use,
                    enable_super_dag=enable_super_dag,
                )
                if positive_result.profile_tree is not None:
                    positive_node.children = [
                        _clone_profile_node(child)
                        for child in positive_result.profile_tree.children
                    ]
        else:
            positive_result = materialize_positive_sufficient_class_inferences(
                schema_graph=schema_graph,
                data_graph=data_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
                materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_reflexive_properties=materialize_reflexive_properties,
                materialize_target_roles=False,
                native_sameas_canonicalization=native_sameas_canonicalization,
                target_classes=target_classes,
                threshold=threshold,
                device=device_to_use,
                enable_super_dag=enable_super_dag,
            )
        iterations = positive_result.iterations
        positive_timings = positive_result.timings
        if positive_timings is not None:
            stratified_initial_data_copy_elapsed_ms = positive_timings.initial_data_copy_elapsed_ms
            stratified_initial_ontology_merge_elapsed_ms = positive_timings.initial_ontology_merge_elapsed_ms
            schema_cache_elapsed_ms = positive_timings.schema_cache_elapsed_ms
            preprocessing_plan_elapsed_ms = positive_timings.preprocessing_plan_elapsed_ms
            sufficient_rule_extraction_elapsed_ms = positive_timings.rule_extraction_elapsed_ms
            sufficient_rule_index_elapsed_ms = positive_timings.rule_index_elapsed_ms
            dependency_closure_elapsed_ms = positive_timings.dependency_closure_elapsed_ms
            sameas_state_init_elapsed_ms = positive_timings.sameas_state_init_elapsed_ms
            data_copy_elapsed_ms = positive_timings.data_copy_elapsed_ms
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
        if positive_result.inferred_assertions:
            seed_members_by_target: Dict[URIRef, Set[Identifier]] = defaultdict(set)
            for node_term, class_term in positive_result.inferred_assertions:
                seed_members_by_target[class_term].add(node_term)
            if engine_profiler is not None:
                with engine_profiler.scoped(
                    "seed_admissibility_input",
                    "seed admissibility input",
                    category="host_runtime",
                ):
                    query_input_data_graph = _add_type_assignments(data_graph, seed_members_by_target)
            else:
                query_input_data_graph = _add_type_assignments(data_graph, seed_members_by_target)

    if engine_mode == "stratified":
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "stratified_execution",
                "stratified execution",
                category="host_runtime",
            ) as stratified_node:
                stratified_result = materialize_stratified_class_inferences(
                    schema_graph=schema_graph,
                    data_graph=data_graph,
                    include_literals=include_literals,
                    include_type_edges=include_type_edges,
                    materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
                    materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                    materialize_sameas=materialize_sameas,
                    materialize_haskey_equality=materialize_haskey_equality,
                    materialize_reflexive_properties=materialize_reflexive_properties,
                    materialize_target_roles=False,
                    native_sameas_canonicalization=native_sameas_canonicalization,
                    target_classes=target_classes,
                    threshold=threshold,
                    device=device_to_use,
                    enable_super_dag=enable_super_dag,
                    conflict_policy=ConflictPolicy(conflict_policy),
                    enable_negative_verification=(True if enable_negative_verification is None else enable_negative_verification),
                )
                if stratified_result.profile_tree is not None:
                    stratified_node.children = [
                        _clone_profile_node(child)
                        for child in stratified_result.profile_tree.children
                    ]
        else:
            stratified_result = materialize_stratified_class_inferences(
                schema_graph=schema_graph,
                data_graph=data_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
                materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_reflexive_properties=materialize_reflexive_properties,
                materialize_target_roles=False,
                native_sameas_canonicalization=native_sameas_canonicalization,
                target_classes=target_classes,
                threshold=threshold,
                device=device_to_use,
                enable_super_dag=enable_super_dag,
                conflict_policy=ConflictPolicy(conflict_policy),
                enable_negative_verification=(True if enable_negative_verification is None else enable_negative_verification),
            )
        if engine_profiler is not None:
            engine_profiler.push(
                "stratified_result_marshalling",
                "stratified result marshalling",
                category="host_runtime",
            )
        dataset = stratified_result.positive_result.dataset
        iterations = stratified_result.positive_result.iterations
        if dataset.preprocessing_timings is not None:
            dataset_build_elapsed_ms = dataset.preprocessing_timings.dataset_build_elapsed_ms
            data_copy_elapsed_ms = dataset.preprocessing_timings.data_copy_elapsed_ms
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
    elif engine_mode == "filtered_admissibility":
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "raw_admissibility",
                "raw admissibility snapshot",
                category="host_runtime",
            ) as raw_node:
                raw_snapshot = _evaluate_query_snapshot(
                    schema_graph=schema_graph,
                    data_graph=query_input_data_graph,
                    target_classes=target_classes,
                    device=device_to_use,
                    threshold=threshold,
                    include_literals=include_literals,
                    include_type_edges=include_type_edges,
                    materialize_hierarchy=materialize_hierarchy,
                    materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                    materialize_sameas=materialize_sameas,
                    materialize_haskey_equality=materialize_haskey_equality,
                    materialize_reflexive_properties=materialize_reflexive_properties,
                    materialize_target_roles=materialize_target_roles,
                    augment_property_domain_range=augment_property_domain_range,
                    native_sameas_canonicalization=native_sameas_canonicalization,
                )
                if raw_snapshot.profile_tree is not None:
                    raw_node.children = [
                        _clone_profile_node(child)
                        for child in raw_snapshot.profile_tree.children
                    ]
        else:
            raw_snapshot = _evaluate_query_snapshot(
                schema_graph=schema_graph,
                data_graph=query_input_data_graph,
                target_classes=target_classes,
                device=device_to_use,
                threshold=threshold,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=materialize_hierarchy,
                materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_reflexive_properties=materialize_reflexive_properties,
                materialize_target_roles=materialize_target_roles,
                augment_property_domain_range=augment_property_domain_range,
                native_sameas_canonicalization=native_sameas_canonicalization,
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
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "necessary_fixpoint",
                "filtered admissibility stabilization",
                category="host_runtime",
            ):
                while True:
                    necessary_fixpoint_iterations += 1
                    with engine_profiler.scoped(
                        f"necessary_iteration_{necessary_fixpoint_iterations}",
                        f"necessary iteration {necessary_fixpoint_iterations}",
                        category="host_runtime",
                        iteration=necessary_fixpoint_iterations,
                    ) as iteration_node:
                        augmented_data_graph = _add_type_assignments(query_input_data_graph, current_members_by_target)
                        stable_snapshot = _evaluate_query_snapshot(
                            schema_graph=schema_graph,
                            data_graph=augmented_data_graph,
                            target_classes=target_classes,
                            device=device_to_use,
                            threshold=threshold,
                            include_literals=include_literals,
                            include_type_edges=include_type_edges,
                            materialize_hierarchy=materialize_hierarchy,
                            materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                            materialize_sameas=materialize_sameas,
                            materialize_haskey_equality=materialize_haskey_equality,
                            materialize_reflexive_properties=materialize_reflexive_properties,
                            materialize_target_roles=materialize_target_roles,
                            augment_property_domain_range=augment_property_domain_range,
                            native_sameas_canonicalization=native_sameas_canonicalization,
                        )
                        if stable_snapshot.profile_tree is not None:
                            iteration_node.children = [
                                _clone_profile_node(child)
                                for child in stable_snapshot.profile_tree.children
                            ]
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
        else:
            while True:
                necessary_fixpoint_iterations += 1
                augmented_data_graph = _add_type_assignments(query_input_data_graph, current_members_by_target)
                stable_snapshot = _evaluate_query_snapshot(
                    schema_graph=schema_graph,
                    data_graph=augmented_data_graph,
                    target_classes=target_classes,
                    device=device_to_use,
                    threshold=threshold,
                    include_literals=include_literals,
                    include_type_edges=include_type_edges,
                    materialize_hierarchy=materialize_hierarchy,
                    materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                    materialize_sameas=materialize_sameas,
                    materialize_haskey_equality=materialize_haskey_equality,
                    materialize_reflexive_properties=materialize_reflexive_properties,
                    materialize_target_roles=materialize_target_roles,
                    augment_property_domain_range=augment_property_domain_range,
                    native_sameas_canonicalization=native_sameas_canonicalization,
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
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "seed_closure_verification_input",
                "seed closure verification input",
                category="host_runtime",
            ):
                filtered_data_graph = _add_type_assignments(query_input_data_graph, necessary_stable_members_by_target)
        else:
            filtered_data_graph = _add_type_assignments(query_input_data_graph, necessary_stable_members_by_target)
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "closure_verification",
                "closure verification",
                category="host_runtime",
            ) as closure_node:
                stratified_result = materialize_stratified_class_inferences(
                    schema_graph=schema_graph,
                    data_graph=filtered_data_graph,
                    include_literals=include_literals,
                    include_type_edges=include_type_edges,
                    materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
                    materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                    materialize_sameas=materialize_sameas,
                    materialize_haskey_equality=materialize_haskey_equality,
                    materialize_reflexive_properties=materialize_reflexive_properties,
                    materialize_target_roles=False,
                    native_sameas_canonicalization=native_sameas_canonicalization,
                    target_classes=target_classes,
                    threshold=threshold,
                    device=device_to_use,
                    enable_super_dag=enable_super_dag,
                    conflict_policy=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED,
                    enable_negative_verification=(True if enable_negative_verification is None else enable_negative_verification),
                )
                if stratified_result.profile_tree is not None:
                    closure_node.children = [
                        _clone_profile_node(child)
                        for child in stratified_result.profile_tree.children
                    ]
        else:
            stratified_result = materialize_stratified_class_inferences(
                schema_graph=schema_graph,
                data_graph=filtered_data_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=(True if materialize_hierarchy is None else materialize_hierarchy),
                materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_reflexive_properties=materialize_reflexive_properties,
                materialize_target_roles=False,
                native_sameas_canonicalization=native_sameas_canonicalization,
                target_classes=target_classes,
                threshold=threshold,
                device=device_to_use,
                enable_super_dag=enable_super_dag,
                conflict_policy=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED,
                enable_negative_verification=(True if enable_negative_verification is None else enable_negative_verification),
            )

        if engine_profiler is not None:
            with engine_profiler.scoped(
                "filtered_result_assembly",
                "filtered result assembly",
                category="host_runtime",
            ):
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
        else:
            closure_blocked_members_by_target = {
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

            final_members_by_target = {}
            for target_term in target_classes:
                final_members_by_target[target_term] = (
                    necessary_stable_members_by_target.get(target_term, set())
                    - closure_blocked_members_by_target.get(target_term, set())
                )

        filtered_admissibility_result = FilteredAdmissibilityResult(
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
            profile_tree=None,
        )
        dataset = stable_snapshot.dataset
        members_by_target = final_members_by_target
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "filtered_score_projection",
                "filtered score projection",
                category="host_runtime",
            ):
                scores_by_target = _build_binary_scores(
                    dataset.mapping.node_terms,
                    target_classes,
                    final_members_by_target,
                )
        else:
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
            materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
            materialize_reflexive_properties=materialize_reflexive_properties,
            native_sameas_canonicalization=native_sameas_canonicalization,
            target_classes=target_classes,
            threshold=threshold,
            device=device_to_use,
        )
        dataset = materialized.dataset
        iterations = materialized.iterations
        if dataset.preprocessing_timings is not None:
            dataset_build_elapsed_ms = dataset.preprocessing_timings.dataset_build_elapsed_ms
            data_copy_elapsed_ms = dataset.preprocessing_timings.data_copy_elapsed_ms
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
        if engine_profiler is not None:
            with engine_profiler.scoped(
                "admissibility_evaluation",
                "admissibility evaluation",
                category="host_runtime",
            ) as admissibility_node:
                snapshot = _evaluate_query_snapshot(
                    schema_graph=schema_graph,
                    data_graph=query_input_data_graph,
                    target_classes=target_classes,
                    device=device_to_use,
                    threshold=threshold,
                    include_literals=include_literals,
                    include_type_edges=include_type_edges,
                    materialize_hierarchy=materialize_hierarchy,
                    materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                    materialize_sameas=materialize_sameas,
                    materialize_haskey_equality=materialize_haskey_equality,
                    materialize_reflexive_properties=materialize_reflexive_properties,
                    materialize_target_roles=materialize_target_roles,
                    augment_property_domain_range=augment_property_domain_range,
                    native_sameas_canonicalization=native_sameas_canonicalization,
                )
                if snapshot.profile_tree is not None:
                    admissibility_node.children = [
                        _clone_profile_node(child)
                        for child in snapshot.profile_tree.children
                    ]
        else:
            snapshot = _evaluate_query_snapshot(
                schema_graph=schema_graph,
                data_graph=query_input_data_graph,
                target_classes=target_classes,
                device=device_to_use,
                threshold=threshold,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=materialize_hierarchy,
                materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
                materialize_sameas=materialize_sameas,
                materialize_haskey_equality=materialize_haskey_equality,
                materialize_reflexive_properties=materialize_reflexive_properties,
                materialize_target_roles=materialize_target_roles,
                augment_property_domain_range=augment_property_domain_range,
                native_sameas_canonicalization=native_sameas_canonicalization,
            )
        dataset = snapshot.dataset
        members_by_target = snapshot.members_by_target
        scores_by_target = snapshot.scores_by_target
        dataset_build_elapsed_ms += snapshot.dataset_build_elapsed_ms
        ontology_merge_elapsed_ms += (
            snapshot.dataset.preprocessing_timings.ontology_merge_elapsed_ms
            if snapshot.dataset.preprocessing_timings is not None
            else 0.0
        )
        schema_cache_elapsed_ms += (
            snapshot.dataset.preprocessing_timings.schema_cache_elapsed_ms
            if snapshot.dataset.preprocessing_timings is not None
            else 0.0
        )
        preprocessing_plan_elapsed_ms += (
            snapshot.dataset.preprocessing_timings.preprocessing_plan_elapsed_ms
            if snapshot.dataset.preprocessing_timings is not None
            else 0.0
        )
        hierarchy_elapsed_ms += snapshot.hierarchy_elapsed_ms
        atomic_domain_range_elapsed_ms += snapshot.atomic_domain_range_elapsed_ms
        horn_safe_domain_range_elapsed_ms += snapshot.horn_safe_domain_range_elapsed_ms
        sameas_elapsed_ms += snapshot.sameas_elapsed_ms
        reflexive_elapsed_ms += snapshot.reflexive_elapsed_ms
        target_role_elapsed_ms += snapshot.target_role_elapsed_ms
        kgraph_build_elapsed_ms += snapshot.kgraph_build_elapsed_ms
        mapping_vocab_collect_elapsed_ms += snapshot.mapping_vocab_collect_elapsed_ms
        mapping_graph_scan_elapsed_ms += snapshot.mapping_graph_scan_elapsed_ms
        mapping_sort_elapsed_ms += snapshot.mapping_sort_elapsed_ms
        mapping_index_elapsed_ms += snapshot.mapping_index_elapsed_ms
        kgraph_edge_bucket_elapsed_ms += snapshot.kgraph_edge_bucket_elapsed_ms
        kgraph_negative_helper_elapsed_ms += snapshot.kgraph_negative_helper_elapsed_ms
        kgraph_literal_feature_elapsed_ms += snapshot.kgraph_literal_feature_elapsed_ms
        kgraph_adjacency_elapsed_ms += snapshot.kgraph_adjacency_elapsed_ms
        if snapshot.dataset.preprocessing_timings is not None:
            sameas_passes_elapsed_ms.extend(snapshot.dataset.preprocessing_timings.sameas_passes_elapsed_ms)
        dag_compile_elapsed_ms += snapshot.dag_compile_elapsed_ms
        dag_eval_elapsed_ms += snapshot.dag_eval_elapsed_ms

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
            data_copy_elapsed_ms = positive_timings.data_copy_elapsed_ms
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

        if engine_profiler is not None:
            with engine_profiler.scoped(
                "result_projection",
                "result projection",
                category="host_runtime",
            ):
                projection_t0 = perf_counter()
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
                result_projection_elapsed_ms = (perf_counter() - projection_t0) * 1000.0
        else:
            projection_t0 = perf_counter()
            emitted_members_by_target = {target: set() for target in target_classes}
            if stratified_result is not None:
                for status in stratified_result.policy_result.emitted_assignments:
                    if status.target_class in emitted_members_by_target:
                        emitted_members_by_target[status.target_class].add(status.node_term)

            for target_term in target_classes:
                members = emitted_members_by_target.get(target_term, set())
                scores = {}
                for node_term in dataset.mapping.node_terms:
                    score = 1.0 if node_term in members else 0.0
                    scores[node_term] = score
                members_by_target[target_term] = set(members)
                scores_by_target[target_term] = scores
            result_projection_elapsed_ms = (perf_counter() - projection_t0) * 1000.0
        if engine_profiler is not None:
            engine_profiler.pop()
    if engine_profiler is not None:
        with engine_profiler.scoped(
            "engine_postprocess",
            "engine postprocess",
            category="host_runtime",
        ):
            dag_stats_by_target: Dict[URIRef, DAGStats] = {}
            compiled_target_dags: Optional[Dict[URIRef, ConstraintDAG]] = None
            if stratified_result is not None and stratified_result.positive_result.compiled_target_dags is not None:
                compiled_target_dags = stratified_result.positive_result.compiled_target_dags
            if compiled_target_dags:
                with engine_profiler.scoped("dag_stats", "dag stats", category="host_runtime"):
                    dag_stats_by_target = {
                        target_term: _compute_dag_stats(dag)
                        for target_term, dag in compiled_target_dags.items()
                        if target_term in target_classes
                    }
    else:
        dag_stats_by_target = {}
        compiled_target_dags = None
        if stratified_result is not None and stratified_result.positive_result.compiled_target_dags is not None:
            compiled_target_dags = stratified_result.positive_result.compiled_target_dags
        if compiled_target_dags:
            dag_stats_by_target = {
                target_term: _compute_dag_stats(dag)
                for target_term, dag in compiled_target_dags.items()
                if target_term in target_classes
            }

    members_by_target, scores_by_target = _expand_sameas_results_for_reporting(
        dataset,
        members_by_target,
        scores_by_target,
    )

    elapsed_ms = (perf_counter() - t0) * 1000.0
    profile_tree: Optional[ProfileNode] = None
    profile_summary: Optional[ProfileSummary] = None
    if engine_profiler is not None:
        profile_tree = engine_profiler.build_tree()
        if profile_tree is not None:
            if filtered_admissibility_result is not None:
                filtered_admissibility_result.profile_tree = _clone_profile_node(profile_tree)
            warnings = validate_profile_tree(profile_tree)
            disjoint_top_level_elapsed_ms = sum(child.elapsed_ms_inclusive for child in profile_tree.children)
            residual = profile_tree.elapsed_ms_inclusive - disjoint_top_level_elapsed_ms
            root_tol = profile_tolerance_ms(profile_tree.elapsed_ms_inclusive)
            if abs(residual) > root_tol:
                warnings.append(
                    ProfileValidationWarning(
                        path=profile_tree.name,
                        issue="top-level children do not reconcile to engine total",
                        elapsed_ms=abs(residual),
                        tolerance_ms=root_tol,
                    )
                )
            profile_summary = ProfileSummary(
                root_elapsed_ms=profile_tree.elapsed_ms_inclusive,
                category_totals_ms=aggregate_by_category(profile_tree),
                warnings=warnings,
                reconciliation_residual_ms=residual,
            )
    if profile_tree is None or profile_summary is None:
        profile_tree, profile_summary = _build_engine_profile_tree(
            EngineQueryResult(
            backend="engine",
            elapsed_ms=elapsed_ms,
            members_by_target=members_by_target,
            dataset=dataset,
            scores_by_target=scores_by_target,
            materialization_iterations=iterations,
            engine_mode=engine_mode,
            conflict_policy=(conflict_policy if engine_mode in {"stratified", "filtered_admissibility"} else None),
            stratified_result=stratified_result,
            filtered_admissibility_result=filtered_admissibility_result,
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
            data_copy_elapsed_ms=data_copy_elapsed_ms,
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
            result_projection_elapsed_ms=result_projection_elapsed_ms,
            stratified_iteration_timings=stratified_iteration_timings,
            stratified_final_dataset_timing=stratified_final_dataset_timing,
            dag_requested_device=dag_requested_device,
            dag_effective_device=dag_effective_device,
            torch_cuda_available=torch_cuda_available,
            dag_stats_by_target=(dag_stats_by_target or None),
            super_dag_plan=(
                stratified_result.positive_result.super_dag_plan
                if stratified_result is not None
                else None
            ),
        )
        )

    return EngineQueryResult(
        backend="engine",
        elapsed_ms=elapsed_ms,
        members_by_target=members_by_target,
        dataset=dataset,
        scores_by_target=scores_by_target,
        materialization_iterations=iterations,
        engine_mode=engine_mode,
        conflict_policy=(conflict_policy if engine_mode in {"stratified", "filtered_admissibility"} else None),
        stratified_result=stratified_result,
        filtered_admissibility_result=filtered_admissibility_result,
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
        data_copy_elapsed_ms=data_copy_elapsed_ms,
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
        result_projection_elapsed_ms=result_projection_elapsed_ms,
        stratified_iteration_timings=stratified_iteration_timings,
        stratified_final_dataset_timing=stratified_final_dataset_timing,
        dag_requested_device=dag_requested_device,
        dag_effective_device=dag_effective_device,
        torch_cuda_available=torch_cuda_available,
        dag_stats_by_target=(dag_stats_by_target or None),
        super_dag_plan=(
            stratified_result.positive_result.super_dag_plan
            if stratified_result is not None
            else None
        ),
        profile_tree=profile_tree,
        profile_summary=profile_summary,
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


def _compile_owlapi_helper(
    *,
    classpath: str,
    javac_command: str = "javac",
) -> Path:
    source_path = Path(__file__).resolve().parent / "java" / "OwlapiOracleRunner.java"
    build_dir = Path(tempfile.gettempdir()) / "dag_owlapi_oracle_helper"
    class_file = build_dir / "OwlapiOracleRunner.class"

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
        preprocess_elapsed_ms = 0.0
        postprocess_elapsed_ms = 0.0
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
            elif parts[0] == "PREPROCESS_MS" and len(parts) == 2:
                preprocess_elapsed_ms = float(parts[1])
            elif parts[0] == "POSTPROCESS_MS" and len(parts) == 2:
                postprocess_elapsed_ms = float(parts[1])
            elif parts[0] == "MEMBER" and len(parts) == 3:
                target_term = URIRef(parts[1])
                member_term = URIRef(parts[2])
                members_by_target.setdefault(target_term, set()).add(member_term)

        return BackendQueryResult(
            backend="elk",
            elapsed_ms=elapsed_ms,
            members_by_target=members_by_target,
            preprocess_elapsed_ms=preprocess_elapsed_ms,
            postprocess_elapsed_ms=postprocess_elapsed_ms,
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


def validate_owlapi_reasoner_backend(
    *,
    reasoner_name: str,
    owlapi_home: Optional[str] = None,
    java_command: str = "java",
    javac_command: str = "javac",
    java_options: Optional[Sequence[str]] = None,
) -> Tuple[bool, str, Optional[str]]:
    classpath = _resolve_elk_classpath(
        elk_classpath=None,
        elk_jar=None,
        owlapi_home=owlapi_home,
    )
    if not classpath:
        return False, "No OWLAPI classpath could be resolved.", None
    try:
        helper_dir = _compile_owlapi_helper(classpath=classpath, javac_command=javac_command)
        run_classpath = os.pathsep.join([str(helper_dir), classpath])
        cmd = [java_command]
        if java_options:
            cmd.extend(java_options)
        cmd.extend(["-cp", run_classpath, "OwlapiOracleRunner", "--probe", reasoner_name])
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if "PROBE_OK" not in completed.stdout:
            return False, f"{reasoner_name} probe did not return PROBE_OK.", classpath
        return True, "", classpath
    except subprocess.CalledProcessError as exc:
        error_text = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return False, error_text, classpath
    except Exception as exc:
        return False, str(exc), classpath


def run_owlapi_reasoner_queries(
    query_graph: Graph,
    query_class_by_target: Dict[URIRef, URIRef],
    candidate_terms: Set[Identifier],
    *,
    backend_name: str,
    reasoner_name: str,
    owlapi_home: Optional[str] = None,
    java_command: str = "java",
    javac_command: str = "javac",
    java_options: Optional[Sequence[str]] = None,
) -> BackendQueryResult:
    classpath = _resolve_elk_classpath(
        elk_classpath=None,
        elk_jar=None,
        owlapi_home=owlapi_home,
    )
    if not classpath:
        return BackendQueryResult(
            backend=backend_name,
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            error="No OWLAPI classpath could be resolved.",
        )

    workspace_temp_root = Path.cwd() / ".tmp" / "owlapi-oracle"
    workspace_temp_root.mkdir(parents=True, exist_ok=True)
    temp_root = workspace_temp_root / f"dag-owlapi-oracle-{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=False)
    try:
        helper_dir = _compile_owlapi_helper(classpath=classpath, javac_command=javac_command)
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
                "OwlapiOracleRunner",
                reasoner_name,
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
            elif parts[0] == "PREPROCESS_MS" and len(parts) == 2:
                preprocess_elapsed_ms = float(parts[1])
            elif parts[0] == "POSTPROCESS_MS" and len(parts) == 2:
                postprocess_elapsed_ms = float(parts[1])
            elif parts[0] == "MEMBER" and len(parts) == 3:
                target_term = URIRef(parts[1])
                member_term = URIRef(parts[2])
                members_by_target.setdefault(target_term, set()).add(member_term)

        return BackendQueryResult(
            backend=backend_name,
            elapsed_ms=elapsed_ms,
            members_by_target=members_by_target,
            preprocess_elapsed_ms=preprocess_elapsed_ms,
            postprocess_elapsed_ms=postprocess_elapsed_ms,
            status="ok",
            consistent=True,
        )
    except subprocess.CalledProcessError as exc:
        error_text = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return BackendQueryResult(
            backend=backend_name,
            elapsed_ms=0.0,
            members_by_target={target: set() for target in query_class_by_target},
            status="error",
            error=error_text,
        )
    except Exception as exc:
        return BackendQueryResult(
            backend=backend_name,
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
    verbose: bool = False,
) -> None:
    print("=== Oracle Comparison ===")
    print(f"Engine time: {_format_elapsed_seconds(engine_result.elapsed_ms)}")
    print(f"Engine threshold: {threshold}")
    if show_timing_breakdown:
        print(format_engine_timing_breakdown(engine_result, verbose=verbose))
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
        print(f"{backend.backend} time: {_format_elapsed_seconds(backend.elapsed_ms)} ({status})")
        if backend.preprocess_elapsed_ms or backend.postprocess_elapsed_ms:
            print(
                "  timing: "
                f"preprocess={_format_elapsed_seconds(backend.preprocess_elapsed_ms)}, "
                f"postprocess={_format_elapsed_seconds(backend.postprocess_elapsed_ms)}"
            )
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
    profile: Optional[str] = None,
    materialize_hierarchy: Optional[bool] = None,
    materialize_horn_safe_domain_range: Optional[bool] = None,
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
    materialize_reflexive_properties: Optional[bool] = None,
    materialize_target_roles: Optional[bool] = None,
    augment_property_domain_range: Optional[bool] = None,
    materialize_supported_types: bool = False,
    engine_mode: str = "query",
    conflict_policy: str = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
    enable_negative_verification: Optional[bool] = None,
    query_mode: str = "query",
    bridge_supported_definitions: bool = False,
    oracle_backends: Sequence[str] = (),
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
    timing_json: Optional[str] = None,
    timing_csv: Optional[str] = None,
    graph_load_cache: str = "on",
    graph_load_cache_dir: Optional[str] = None,
    super_dag: str = "auto",
    verbose: bool = False,
) -> None:
    engine_mode = normalize_engine_mode_name(engine_mode)
    super_dag = resolve_super_dag_mode(super_dag, profile)
    profile_options = apply_engine_profile(
        profile=profile,
        materialize_hierarchy=materialize_hierarchy,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_reflexive_properties=materialize_reflexive_properties,
        materialize_sameas=materialize_sameas,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_target_roles=materialize_target_roles,
        augment_property_domain_range=augment_property_domain_range,
        enable_negative_verification=enable_negative_verification,
    )
    elk_preflight_elapsed_ms = 0.0
    resolved_elk_classpath = elk_classpath
    if "elk" in oracle_backends:
        preflight_t0 = perf_counter()
        ok, error_text, resolved_classpath = validate_elk_backend(
            elk_classpath=elk_classpath,
            elk_jar=elk_jar,
            owlapi_home=owlapi_home,
            java_command=elk_java_command,
            javac_command=elk_javac_command,
            java_options=elk_java_options,
        )
        elk_preflight_elapsed_ms = (perf_counter() - preflight_t0) * 1000.0
        if not ok:
            print("=== Oracle Comparison ===")
            print(f"ELK preflight time: {_format_elapsed_seconds(elk_preflight_elapsed_ms)}")
            print("ELK preflight failed before engine execution.")
            print(error_text)
            return
        resolved_elk_classpath = resolved_classpath
    if "openllet" in oracle_backends:
        preflight_t0 = perf_counter()
        ok, error_text, _resolved_classpath = validate_owlapi_reasoner_backend(
            reasoner_name="openllet",
            owlapi_home=owlapi_home,
            java_command=elk_java_command,
            javac_command=elk_javac_command,
        )
        _openllet_preflight_elapsed_ms = (perf_counter() - preflight_t0) * 1000.0
        if not ok:
            print("=== Oracle Comparison ===")
            print(f"Openllet preflight time: {_format_elapsed_seconds(_openllet_preflight_elapsed_ms)}")
            print("Openllet preflight failed before engine execution.")
            print(error_text)
            return

    load_schema_t0 = perf_counter()
    schema_graph = load_rdflib_graph(
        schema_paths,
        cache_mode=graph_load_cache,
        cache_dir=graph_load_cache_dir,
    )
    schema_load_elapsed_ms = (perf_counter() - load_schema_t0) * 1000.0
    load_data_t0 = perf_counter()
    data_graph = load_rdflib_graph(
        data_paths,
        cache_mode=graph_load_cache,
        cache_dir=graph_load_cache_dir,
    )
    data_load_elapsed_ms = (perf_counter() - load_data_t0) * 1000.0
    resolution_t0 = perf_counter()
    resolution = resolve_target_classes(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_class_specs=target_class_specs,
        engine_mode=engine_mode,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=profile_options.materialize_hierarchy,
        augment_property_domain_range=profile_options.augment_property_domain_range,
    )
    resolution_elapsed_ms = (perf_counter() - resolution_t0) * 1000.0
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
        materialize_hierarchy=profile_options.materialize_hierarchy,
        materialize_horn_safe_domain_range=profile_options.materialize_horn_safe_domain_range,
        materialize_sameas=profile_options.materialize_sameas,
        materialize_haskey_equality=profile_options.materialize_haskey_equality,
        materialize_reflexive_properties=profile_options.materialize_reflexive_properties,
        materialize_target_roles=profile_options.materialize_target_roles,
        materialize_supported_types=materialize_supported_types,
        augment_property_domain_range=profile_options.augment_property_domain_range,
        native_sameas_canonicalization=profile_options.native_sameas_canonicalization,
        engine_mode=engine_mode,
        conflict_policy=conflict_policy,
        enable_negative_verification=profile_options.enable_negative_verification,
        enable_super_dag=(super_dag == "on"),
    )
    if engine_result.profile_tree is None or engine_result.profile_summary is None:
        engine_result.profile_tree, engine_result.profile_summary = _build_engine_profile_tree(engine_result)
    if engine_result.profile_tree is not None and engine_result.profile_summary is not None:
        if timing_json:
            write_profile_json(timing_json, engine_result.profile_tree, engine_result.profile_summary)
        if timing_csv:
            write_profile_csv(timing_csv, engine_result.profile_tree)

    candidate_terms = set(engine_result.dataset.mapping.node_terms) if engine_result.dataset else set()
    if engine_result.dataset is not None and engine_result.dataset.sameas_members_by_canonical:
        for aliases in engine_result.dataset.sameas_members_by_canonical.values():
            candidate_terms.update(aliases)
    ontology_graph = aggregate_rdflib_graphs((schema_graph, data_graph))
    query_graph_t0 = perf_counter()
    query_graph, query_class_by_target = build_oracle_query_graph(
        ontology_graph,
        target_terms,
        mode=query_mode,
        bridge_supported_definitions=bridge_supported_definitions,
    )
    query_graph_elapsed_ms = (perf_counter() - query_graph_t0) * 1000.0

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
        elif backend == "openllet":
            oracle_results.append(
                run_owlapi_reasoner_queries(
                    query_graph=query_graph,
                    query_class_by_target=query_class_by_target,
                    candidate_terms=candidate_terms,
                    backend_name="openllet",
                    reasoner_name="openllet",
                    owlapi_home=owlapi_home,
                    java_command=elk_java_command,
                    javac_command=elk_javac_command,
                    java_options=elk_java_options,
                )
            )
        else:
            raise ValueError(f"Unsupported oracle backend: {backend}")

    print(f"Schema load time: {_format_elapsed_seconds(schema_load_elapsed_ms)}")
    print(f"Data load time: {_format_elapsed_seconds(data_load_elapsed_ms)}")
    if "elk" in oracle_backends:
        print(f"ELK preflight time: {_format_elapsed_seconds(elk_preflight_elapsed_ms)}")
    print(f"Target resolution time: {_format_elapsed_seconds(resolution_elapsed_ms)}")
    if show_timing_breakdown:
        print(f"Oracle query graph build time: {_format_elapsed_seconds(query_graph_elapsed_ms)}")
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
        verbose=verbose,
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
    parser.add_argument(
        "--graph-load-cache",
        choices=["on", "off", "refresh"],
        default="on",
        help=(
            "Persistent cache mode for rdflib dataset loading. "
            "'on' reuses cached parsed graphs when source path/mtime/size/format match, "
            "'refresh' reparses and overwrites the cache, "
            "'off' disables the cache. Default: on."
        ),
    )
    parser.add_argument(
        "--graph-load-cache-dir",
        default=None,
        help="Optional directory for persistent rdflib graph load cache files. Default: .cache/rdflib_graphs",
    )
    parser.add_argument(
        "--super-dag",
        choices=["on", "off", "auto"],
        default="auto",
        help=(
            "Opt-in merged super-DAG evaluation for multi-target positive Task M runs. "
            "'auto' enables it by default for gpu-el profiles. "
            "Currently only used when the selected target set is acyclic under the cached sufficient-rule dependency analysis."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--include-literals", action="store_true")
    parser.add_argument("--include-type-edges", action="store_true")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_ALIASES),
        default="default",
        help=(
            "Reasoning profile preset. "
            "default keeps existing behavior; "
            "gpu-el-lite disables CPU-heavy ABox closure/materialization passes by default, "
            "with no ABox sameAs reasoning; "
            "gpu-el adds native sameAs canonicalization; "
            "gpu-el-full adds HasKey-driven equality generation on top of that; "
            "gpu-el-verify matches gpu-el but keeps the negative/blocker verification pass. "
            "Aliases gpu-e1-lite, gpu-e1, gpu-e1-full, and gpu-e1-verify are accepted."
        ),
    )
    hierarchy_group = parser.add_mutually_exclusive_group()
    hierarchy_group.add_argument(
        "--materialize-hierarchy",
        dest="materialize_hierarchy",
        action="store_true",
        help="Force hierarchy materialization on, regardless of profile defaults.",
    )
    hierarchy_group.add_argument(
        "--no-materialize-hierarchy",
        dest="materialize_hierarchy",
        action="store_false",
        help="Force hierarchy materialization off, regardless of profile defaults.",
    )
    parser.set_defaults(materialize_hierarchy=None)
    horn_group = parser.add_mutually_exclusive_group()
    horn_group.add_argument(
        "--materialize-horn-safe-domain-range",
        dest="materialize_horn_safe_domain_range",
        action="store_true",
        help="Force horn-safe domain/range materialization on.",
    )
    horn_group.add_argument(
        "--no-materialize-horn-safe-domain-range",
        dest="materialize_horn_safe_domain_range",
        action="store_false",
        help="Force horn-safe domain/range materialization off.",
    )
    parser.set_defaults(materialize_horn_safe_domain_range=None)
    augment_domain_range_group = parser.add_mutually_exclusive_group()
    augment_domain_range_group.add_argument(
        "--augment-property-domain-range",
        dest="augment_property_domain_range",
        action="store_true",
        help="Enable DAG/query-time property domain/range augmentation.",
    )
    augment_domain_range_group.add_argument(
        "--no-augment-property-domain-range",
        dest="augment_property_domain_range",
        action="store_false",
        help="Disable DAG/query-time property domain/range augmentation.",
    )
    parser.set_defaults(augment_property_domain_range=None)
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
    reflexive_group = parser.add_mutually_exclusive_group()
    reflexive_group.add_argument(
        "--materialize-reflexive-properties",
        dest="materialize_reflexive_properties",
        action="store_true",
        help="Force reflexive-property preprocessing on.",
    )
    reflexive_group.add_argument(
        "--no-materialize-reflexive-properties",
        dest="materialize_reflexive_properties",
        action="store_false",
        help="Force reflexive-property preprocessing off.",
    )
    parser.set_defaults(materialize_reflexive_properties=None)
    roles_group = parser.add_mutually_exclusive_group()
    roles_group.add_argument(
        "--materialize-target-roles",
        dest="materialize_target_roles",
        action="store_true",
        help="Force admissibility/filtered-admissibility snapshots to preprocess target-role closure.",
    )
    roles_group.add_argument(
        "--no-materialize-target-roles",
        dest="materialize_target_roles",
        action="store_false",
        help="Force admissibility/filtered-admissibility snapshots to skip target-role preprocessing.",
    )
    parser.set_defaults(materialize_target_roles=None)
    verification_group = parser.add_mutually_exclusive_group()
    verification_group.add_argument(
        "--enable-negative-verification",
        dest="enable_negative_verification",
        action="store_true",
        help="Force the negative/blocker verification pass on in stratified-style flows.",
    )
    verification_group.add_argument(
        "--disable-negative-verification",
        dest="enable_negative_verification",
        action="store_false",
        help="Force the negative/blocker verification pass off in stratified-style flows.",
    )
    parser.set_defaults(enable_negative_verification=None)
    parser.add_argument(
        "--engine-materialize-supported-types",
        action="store_true",
        help="Enable the engine's Horn-style fixpoint pass before evaluating targets.",
    )
    parser.add_argument(
        "--engine-mode",
        choices=["admissibility", "filtered_admissibility", "stratified", "query", "filtered_query"],
        default="admissibility",
        help=(
            "admissibility = necessary-condition admissibility; "
            "filtered_admissibility = admissibility candidates pruned by synchronous recheck plus stratified blockers; "
            "stratified = positive sufficient-condition closure plus negative blocker policy."
        ),
    )
    parser.add_argument(
        "--conflict-policy",
        choices=[policy.value for policy in ConflictPolicy],
        default=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
        help="Conflict policy for filtered_admissibility/stratified engine modes.",
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
        choices=["owlrl", "owlready2", "elk", "openllet"],
        default=[],
        help="Which oracle backends to run. Default: none (engine-only).",
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
        "--timing-json",
        help="Write the engine timing profile tree to a JSON file.",
    )
    parser.add_argument(
        "--timing-csv",
        help="Write the engine timing profile tree to a CSV file.",
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
        profile=args.profile,
        materialize_hierarchy=args.materialize_hierarchy,
        materialize_horn_safe_domain_range=args.materialize_horn_safe_domain_range,
        augment_property_domain_range=args.augment_property_domain_range,
        materialize_sameas=args.materialize_sameas,
        materialize_haskey_equality=args.materialize_haskey_equality,
        materialize_reflexive_properties=args.materialize_reflexive_properties,
        materialize_target_roles=args.materialize_target_roles,
        materialize_supported_types=args.engine_materialize_supported_types,
        engine_mode=args.engine_mode,
        conflict_policy=args.conflict_policy,
        enable_negative_verification=args.enable_negative_verification,
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
        timing_json=args.timing_json,
        timing_csv=args.timing_csv,
        graph_load_cache=args.graph_load_cache,
        graph_load_cache_dir=args.graph_load_cache_dir,
        super_dag=args.super_dag,
        verbose=args.verbose,
        max_diff_items=args.max_diff_items,
    )


if __name__ == "__main__":
    main()
