from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set

import torch
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.term import Identifier

from .oracle_compare import (
    DEFAULT_OWLAPI_HOME,
    _build_engine_stage_summary,
    _format_elapsed_seconds,
    apply_engine_profile,
    build_oracle_query_graph,
    describe_rdflib_graph_load_source,
    format_engine_timing_breakdown,
    resolve_super_dag_mode,
    resolve_target_classes,
    run_elk_queries,
    run_engine_queries,
    run_owlapi_reasoner_queries,
    validate_elk_backend,
    validate_owlapi_reasoner_backend,
)
from .ontology_parse import (
    aggregate_rdflib_graphs,
    collect_named_class_terms,
    load_rdflib_graph,
)


DEFAULT_ENGINE_PROFILES = ("gpu-el-lite", "gpu-el", "gpu-el-full")
DEFAULT_ENGINE_DEVICES = ("cpu", "cuda")
DEFAULT_ENGINE_MODES = ("stratified", "admissibility", "filtered_admissibility")
DEFAULT_TARGET_SPECS = ("all-defined-classes",)


def _dedupe_in_order(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


@dataclass(frozen=True)
class PlannedRun:
    run_index: int
    dataset_index: int
    dataset_path: str
    kind: str  # engine|reasoner
    label: str
    profile: Optional[str] = None
    device: Optional[str] = None
    engine_mode: Optional[str] = None
    reasoner: Optional[str] = None


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _render_elapsed(total_seconds: float) -> str:
    total_seconds = max(0.0, total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress(start_t0: float, message: str) -> None:
    print(f"[{_render_elapsed(perf_counter() - start_t0)}] {message}", flush=True)


def _dataset_name(path_text: str) -> str:
    return Path(path_text).stem


def _collect_candidate_terms(data_graph: Graph) -> Set[Identifier]:
    candidates: Set[Identifier] = set()
    for subj, _pred, obj in data_graph:
        if not isinstance(subj, Literal):
            candidates.add(subj)
        if not isinstance(obj, Literal):
            candidates.add(obj)
    return candidates


def _collect_asserted_target_pairs(
    data_graph: Graph,
    target_terms: Sequence[URIRef],
) -> Set[tuple[Identifier, URIRef]]:
    target_set = set(target_terms)
    asserted: Set[tuple[Identifier, URIRef]] = set()
    for subj, _pred, obj in data_graph.triples((None, RDF.type, None)):
        if isinstance(obj, URIRef) and obj in target_set:
            asserted.add((subj, obj))
    return asserted


def _count_new_type_pairs(
    members_by_target: Dict[URIRef, Set[Identifier]],
    asserted_pairs: Set[tuple[Identifier, URIRef]],
) -> int:
    total = 0
    for target_term, members in members_by_target.items():
        for member in members:
            if (member, target_term) not in asserted_pairs:
                total += 1
    return total


def _count_property_terms(schema_graph: Graph) -> int:
    property_types = {
        RDF.Property,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.FunctionalProperty,
        OWL.TransitiveProperty,
        OWL.SymmetricProperty,
        OWL.AsymmetricProperty,
        OWL.InverseFunctionalProperty,
        OWL.ReflexiveProperty,
        OWL.IrreflexiveProperty,
    }
    props: Set[URIRef] = set()
    for subj, _pred, obj in schema_graph.triples((None, RDF.type, None)):
        if isinstance(subj, URIRef) and obj in property_types:
            props.add(subj)
    for subj, _pred, _obj in schema_graph.triples((None, RDFS.subPropertyOf, None)):
        if isinstance(subj, URIRef):
            props.add(subj)
    return len(props)


def _count_abox_individuals(data_graph: Graph) -> int:
    return len(_collect_candidate_terms(data_graph))


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _serialize_profile_summary(engine_result) -> dict:
    from .oracle_compare import _build_engine_profile_tree

    engine_result.profile_tree, engine_result.profile_summary = _build_engine_profile_tree(engine_result)
    stage_pairs = _build_engine_stage_summary(
        engine_result.profile_tree,
        engine_result,
    )
    stage_map = {label: elapsed_ms for label, elapsed_ms in stage_pairs}
    category_totals = (
        engine_result.profile_summary.category_totals_ms
        if engine_result.profile_summary is not None
        else {}
    )
    return {
        "graph_lowering_ms": stage_map.get("graph lowering", 0.0),
        "identity_normalization_ms": stage_map.get("identity normalization", 0.0),
        "engine_preprocessing_ms": stage_map.get("engine preprocessing", 0.0),
        "engine_iterations_ms": stage_map.get("engine iterations", 0.0),
        "engine_postprocessing_ms": stage_map.get("engine post-processing / reporting", 0.0),
        "tbox_cacheable_ms": float(category_totals.get("tbox_cacheable", 0.0)),
    }


def _serialize_engine_specifics(engine_result) -> dict:
    dag_stats = engine_result.dag_stats_by_target or {}
    total_dag_nodes = sum(stat.num_nodes for stat in dag_stats.values())
    total_dag_edges = sum(stat.num_edges for stat in dag_stats.values())
    max_dag_layers = max((stat.num_layers for stat in dag_stats.values()), default=0)
    return {
        "iterations": engine_result.materialization_iterations,
        "process_memory_peak_bytes": engine_result.process_memory_peak_bytes,
        "kgraph_host_bytes": engine_result.kgraph_host_bytes,
        "compiled_dag_estimated_bytes": engine_result.compiled_dag_estimated_bytes,
        "cuda_peak_allocated_bytes": engine_result.cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": engine_result.cuda_peak_reserved_bytes,
        "dag_target_count": len(dag_stats),
        "dag_total_nodes": total_dag_nodes,
        "dag_total_edges": total_dag_edges,
        "dag_max_layers": max_dag_layers,
        "timing_groups_ms": {
            "ontology_merge": engine_result.ontology_merge_elapsed_ms,
            "schema_cache": engine_result.schema_cache_elapsed_ms,
            "preprocessing_plan": engine_result.preprocessing_plan_elapsed_ms,
            "sufficient_rule_extraction": engine_result.sufficient_rule_extraction_elapsed_ms,
            "sufficient_rule_index": engine_result.sufficient_rule_index_elapsed_ms,
            "dependency_closure": engine_result.dependency_closure_elapsed_ms,
            "sameas_state_init": engine_result.sameas_state_init_elapsed_ms,
            "dataset_build": engine_result.dataset_build_elapsed_ms,
            "data_copy": engine_result.data_copy_elapsed_ms,
            "hierarchy": engine_result.hierarchy_elapsed_ms,
            "atomic_domain_range": engine_result.atomic_domain_range_elapsed_ms,
            "horn_safe_domain_range": engine_result.horn_safe_domain_range_elapsed_ms,
            "sameas": engine_result.sameas_elapsed_ms,
            "reflexive": engine_result.reflexive_elapsed_ms,
            "target_roles": engine_result.target_role_elapsed_ms,
            "kgraph_build": engine_result.kgraph_build_elapsed_ms,
            "mapping_vocab_collect": engine_result.mapping_vocab_collect_elapsed_ms,
            "mapping_graph_scan": engine_result.mapping_graph_scan_elapsed_ms,
            "mapping_sort": engine_result.mapping_sort_elapsed_ms,
            "mapping_index": engine_result.mapping_index_elapsed_ms,
            "kgraph_edge_buckets": engine_result.kgraph_edge_bucket_elapsed_ms,
            "kgraph_negative_helpers": engine_result.kgraph_negative_helper_elapsed_ms,
            "kgraph_literal_features": engine_result.kgraph_literal_feature_elapsed_ms,
            "kgraph_adjacency": engine_result.kgraph_adjacency_elapsed_ms,
            "reasoner_setup": engine_result.stratified_positive_reasoner_setup_elapsed_ms,
            "dag_compile": engine_result.dag_compile_elapsed_ms,
            "dag_eval": engine_result.dag_eval_elapsed_ms,
            "assertion_update": engine_result.stratified_positive_assertion_update_elapsed_ms,
            "negative_blocker": engine_result.stratified_negative_blocker_elapsed_ms,
            "assignment_status": engine_result.stratified_assignment_status_elapsed_ms,
            "conflict_policy": engine_result.stratified_conflict_policy_elapsed_ms,
            "reporting_compile": engine_result.stratified_reporting_compile_elapsed_ms,
            "result_projection": engine_result.result_projection_elapsed_ms,
        },
        "dag_stats_by_target": {
            str(target): {
                "nodes": stat.num_nodes,
                "edges": stat.num_edges,
                "layers": stat.num_layers,
                "leaves": stat.num_leaves,
                "max_layer_width": stat.max_layer_width,
                "avg_layer_width": stat.avg_layer_width,
                "max_fan_in": stat.max_fan_in,
                "max_fan_out": stat.max_fan_out,
            }
            for target, stat in dag_stats.items()
        },
    }


def _build_benchmark_id(
    *,
    schema_paths: Sequence[str],
    dataset_paths: Sequence[str],
    k: int,
    reasoners: Sequence[str],
    profiles: Sequence[str],
    devices: Sequence[str],
    engine_modes: Sequence[str],
    timeout_seconds: float,
    start_index: int,
) -> str:
    payload = {
        "schema_paths": list(schema_paths),
        "dataset_paths": list(dataset_paths),
        "k": k,
        "reasoners": list(reasoners),
        "profiles": list(profiles),
        "devices": list(devices),
        "engine_modes": list(engine_modes),
        "timeout_seconds": timeout_seconds,
        "start_index": start_index,
        "default_target_specs": list(DEFAULT_TARGET_SPECS),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{_now_stamp()}-{digest}"


def _plan_runs(
    dataset_paths: Sequence[str],
    reasoners: Sequence[str],
    profiles: Sequence[str],
    devices: Sequence[str],
    engine_modes: Sequence[str],
) -> List[PlannedRun]:
    runs: List[PlannedRun] = []
    run_index = 1
    for dataset_index, dataset_path in enumerate(dataset_paths, start=1):
        dataset_name = _dataset_name(dataset_path)
        for profile in profiles:
            for device in devices:
                for engine_mode in engine_modes:
                    runs.append(
                        PlannedRun(
                            run_index=run_index,
                            dataset_index=dataset_index,
                            dataset_path=dataset_path,
                            kind="engine",
                            label=f"{dataset_name} | {profile} | {device} | {engine_mode}",
                            profile=profile,
                            device=device,
                            engine_mode=engine_mode,
                        )
                    )
                    run_index += 1
        for reasoner in reasoners:
            runs.append(
                PlannedRun(
                    run_index=run_index,
                    dataset_index=dataset_index,
                    dataset_path=dataset_path,
                    kind="reasoner",
                    label=f"{dataset_name} | {reasoner}",
                    reasoner=reasoner,
                    device="cpu",
                )
            )
            run_index += 1
    return runs


def _run_engine_attempt(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    target_terms: Sequence[URIRef],
    profile: str,
    device: str,
    engine_mode: str,
) -> dict:
    profile_options = apply_engine_profile(
        profile=profile,
        materialize_hierarchy=None,
        materialize_horn_safe_domain_range=None,
        materialize_reflexive_properties=None,
        materialize_sameas=None,
        materialize_haskey_equality=None,
        materialize_target_roles=None,
        augment_property_domain_range=None,
        enable_negative_verification=None,
    )
    super_dag = resolve_super_dag_mode("auto", profile)
    engine_result = run_engine_queries(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_classes=target_terms,
        device=device,
        threshold=0.999,
        include_literals=False,
        include_type_edges=False,
        materialize_hierarchy=profile_options.materialize_hierarchy,
        materialize_horn_safe_domain_range=profile_options.materialize_horn_safe_domain_range,
        materialize_sameas=profile_options.materialize_sameas,
        materialize_haskey_equality=profile_options.materialize_haskey_equality,
        materialize_reflexive_properties=profile_options.materialize_reflexive_properties,
        materialize_target_roles=profile_options.materialize_target_roles,
        materialize_supported_types=False,
        augment_property_domain_range=profile_options.augment_property_domain_range,
        native_sameas_canonicalization=profile_options.native_sameas_canonicalization,
        engine_mode=engine_mode,
        conflict_policy="suppress_derived_keep_asserted",
        enable_negative_verification=profile_options.enable_negative_verification,
        enable_super_dag=(super_dag == "on"),
    )
    if engine_result.profile_tree is None or engine_result.profile_summary is None:
        from .oracle_compare import _build_engine_profile_tree

        engine_result.profile_tree, engine_result.profile_summary = _build_engine_profile_tree(engine_result)
    return {
        "engine_result": engine_result,
        "stage_summary": _serialize_profile_summary(engine_result),
        "engine_specifics": _serialize_engine_specifics(engine_result),
        "timing_breakdown": format_engine_timing_breakdown(engine_result, verbose=False),
    }


def _run_reasoner_attempt(
    *,
    backend: str,
    query_graph: Graph,
    query_class_by_target: Dict[URIRef, URIRef],
    candidate_terms: Set[Identifier],
    timeout_seconds: float,
    owlapi_home: Optional[str],
    elk_classpath: Optional[str],
    elk_jar: Optional[str],
    java_command: str,
    javac_command: str,
) -> object:
    if backend == "elk":
        return run_elk_queries(
            query_graph=query_graph,
            query_class_by_target=query_class_by_target,
            candidate_terms=candidate_terms,
            elk_classpath=elk_classpath,
            elk_jar=elk_jar,
            owlapi_home=owlapi_home,
            java_command=java_command,
            javac_command=javac_command,
            timeout_seconds=timeout_seconds,
        )
    if backend == "openllet":
        return run_owlapi_reasoner_queries(
            query_graph=query_graph,
            query_class_by_target=query_class_by_target,
            candidate_terms=candidate_terms,
            backend_name="openllet",
            reasoner_name="openllet",
            owlapi_home=owlapi_home,
            java_command=java_command,
            javac_command=javac_command,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported benchmark reasoner backend: {backend}")


def run_paper_benchmark(
    *,
    schema_paths: Sequence[str],
    dataset_paths: Sequence[str],
    k: int,
    reasoners: Sequence[str],
    profiles: Sequence[str] = DEFAULT_ENGINE_PROFILES,
    devices: Sequence[str] = DEFAULT_ENGINE_DEVICES,
    engine_modes: Sequence[str] = DEFAULT_ENGINE_MODES,
    timeout_seconds: float,
    start_index: int = 1,
    csv_path: str,
    log_path: Optional[str] = None,
    owlapi_home: Optional[str] = None,
    elk_classpath: Optional[str] = None,
    elk_jar: Optional[str] = None,
    java_command: str = "java",
    javac_command: str = "javac",
    graph_load_cache: str = "on",
    graph_load_cache_dir: Optional[str] = None,
) -> None:
    start_t0 = perf_counter()
    benchmark_id = _build_benchmark_id(
        schema_paths=schema_paths,
        dataset_paths=dataset_paths,
        k=k,
        reasoners=reasoners,
        profiles=profiles,
        devices=devices,
        engine_modes=engine_modes,
        timeout_seconds=timeout_seconds,
        start_index=start_index,
    )
    csv_output_path = Path(csv_path)
    log_output_path = Path(log_path) if log_path is not None else csv_output_path.with_suffix(".log.jsonl")
    planned_runs = _plan_runs(dataset_paths, reasoners, profiles, devices, engine_modes)
    total_runs = len(planned_runs)

    _progress(start_t0, f"Benchmark session {benchmark_id} starting ({total_runs} runs total, start index {start_index}).")

    resolved_elk_classpath = elk_classpath
    if "elk" in reasoners:
        _progress(start_t0, "Running one-time ELK preflight...")
        ok, error_text, resolved_classpath = validate_elk_backend(
            elk_classpath=elk_classpath,
            elk_jar=elk_jar,
            owlapi_home=owlapi_home,
            java_command=java_command,
            javac_command=javac_command,
        )
        if not ok:
            raise RuntimeError(f"ELK preflight failed: {error_text}")
        resolved_elk_classpath = resolved_classpath
        _progress(start_t0, "Finished ELK preflight.")
    if "openllet" in reasoners:
        _progress(start_t0, "Running one-time Openllet preflight...")
        ok, error_text, _resolved_classpath = validate_owlapi_reasoner_backend(
            reasoner_name="openllet",
            owlapi_home=owlapi_home,
            java_command=java_command,
            javac_command=javac_command,
        )
        if not ok:
            raise RuntimeError(f"Openllet preflight failed: {error_text}")
        _progress(start_t0, "Finished Openllet preflight.")

    schema_load_source = describe_rdflib_graph_load_source(
        schema_paths,
        cache_mode=graph_load_cache,
        cache_dir=graph_load_cache_dir,
    )
    _progress(start_t0, f"Loading schema graph from {schema_load_source}...")
    schema_graph = load_rdflib_graph(
        schema_paths,
        cache_mode=graph_load_cache,
        cache_dir=graph_load_cache_dir,
    )
    _progress(start_t0, "Finished loading schema graph.")
    tbox_class_count = len(collect_named_class_terms(schema_graph))
    tbox_property_count = _count_property_terms(schema_graph)

    runs_by_dataset: Dict[str, List[PlannedRun]] = {}
    for run in planned_runs:
        runs_by_dataset.setdefault(run.dataset_path, []).append(run)

    for dataset_path in dataset_paths:
        dataset_runs = runs_by_dataset[dataset_path]
        if all(run.run_index < start_index for run in dataset_runs):
            continue

        dataset_name = _dataset_name(dataset_path)
        data_load_source = describe_rdflib_graph_load_source(
            dataset_path,
            cache_mode=graph_load_cache,
            cache_dir=graph_load_cache_dir,
        )
        _progress(start_t0, f"Loading dataset {dataset_name} from {data_load_source}...")
        data_graph = load_rdflib_graph(
            dataset_path,
            cache_mode=graph_load_cache,
            cache_dir=graph_load_cache_dir,
        )
        _progress(start_t0, f"Finished loading dataset {dataset_name}.")

        resolution = resolve_target_classes(
            schema_graph=schema_graph,
            data_graph=data_graph,
            target_class_specs=DEFAULT_TARGET_SPECS,
            engine_mode="stratified",
            include_literals=False,
            include_type_edges=False,
            materialize_hierarchy=False,
            augment_property_domain_range=True,
        )
        target_terms = resolution.resolved_targets
        if not target_terms:
            raise RuntimeError(f"No targets resolved for dataset {dataset_name}.")

        asserted_target_pairs = _collect_asserted_target_pairs(data_graph, target_terms)
        candidate_terms = _collect_candidate_terms(data_graph)
        dataset_triple_count = len(data_graph)
        abox_individual_count = _count_abox_individuals(data_graph)
        query_graph, query_class_by_target = build_oracle_query_graph(
            aggregate_rdflib_graphs((schema_graph, data_graph)),
            target_terms,
            mode="native",
        )

        for run in dataset_runs:
            if run.run_index < start_index:
                continue

            row_base = {
                "benchmark_id": benchmark_id,
                "run_index": run.run_index,
                "dataset_name": dataset_name,
                "target_class_count": len(target_terms),
                "tbox_class_count": tbox_class_count,
                "tbox_property_count": tbox_property_count,
                "node_count": abox_individual_count,
                "triple_count": dataset_triple_count,
                "kind": run.kind,
                "name": run.profile if run.kind == "engine" else run.reasoner,
                "device": run.device or "cpu",
                "engine_mode": run.engine_mode or "",
                "k": k,
                "timeout_seconds": timeout_seconds,
            }

            status = "ok"
            error_code = ""
            attempts_completed = 0
            total_elapsed_values: List[float] = []
            setup_values: List[float] = []
            reasoning_values: List[float] = []
            retrieval_values: List[float] = []
            inference_values: List[float] = []
            engine_stage_values: Dict[str, List[float]] = {
                "graph_lowering_ms": [],
                "identity_normalization_ms": [],
                "engine_preprocessing_ms": [],
                "engine_iterations_ms": [],
                "engine_postprocessing_ms": [],
                "tbox_cacheable_ms": [],
            }
            engine_extra_values: Dict[str, List[float]] = {
                "iterations": [],
                "process_memory_peak_bytes": [],
                "kgraph_host_bytes": [],
                "compiled_dag_estimated_bytes": [],
                "cuda_peak_allocated_bytes": [],
                "cuda_peak_reserved_bytes": [],
                "dag_target_count": [],
                "dag_total_nodes": [],
                "dag_total_edges": [],
                "dag_max_layers": [],
                "dag_eval_ms": [],
            }
            last_timing_groups: Optional[dict] = None
            last_dag_stats_by_target: Optional[dict] = None
            last_timing_breakdown: Optional[str] = None

            for attempt in range(1, k + 1):
                attempt_t0 = perf_counter()
                try:
                    if run.kind == "engine":
                        attempt_result = _run_engine_attempt(
                            schema_graph=schema_graph,
                            data_graph=data_graph,
                            target_terms=target_terms,
                            profile=run.profile or "gpu-el",
                            device=run.device or "cpu",
                            engine_mode=run.engine_mode or "stratified",
                        )
                        engine_result = attempt_result["engine_result"]
                        stage_summary = attempt_result["stage_summary"]
                        engine_specifics = attempt_result["engine_specifics"]
                        last_timing_breakdown = attempt_result["timing_breakdown"]

                        setup_ms = stage_summary["graph_lowering_ms"]
                        reasoning_ms = (
                            stage_summary["identity_normalization_ms"]
                            + stage_summary["engine_preprocessing_ms"]
                            + stage_summary["engine_iterations_ms"]
                            - stage_summary["tbox_cacheable_ms"]
                        )
                        retrieval_ms = stage_summary["engine_postprocessing_ms"]
                        total_elapsed_ms = engine_result.elapsed_ms
                        new_type_inferences = _count_new_type_pairs(
                            engine_result.members_by_target,
                            asserted_target_pairs,
                        )

                        total_elapsed_values.append(total_elapsed_ms)
                        setup_values.append(setup_ms)
                        reasoning_values.append(reasoning_ms)
                        retrieval_values.append(retrieval_ms)
                        inference_values.append(float(new_type_inferences))
                        for key in engine_stage_values:
                            engine_stage_values[key].append(float(stage_summary[key]))
                        for key in engine_extra_values:
                            if key == "dag_eval_ms":
                                value = engine_specifics["timing_groups_ms"].get("dag_eval")
                            else:
                                value = engine_specifics.get(key)
                            if value is not None:
                                engine_extra_values[key].append(float(value))
                        last_timing_groups = engine_specifics["timing_groups_ms"]
                        last_dag_stats_by_target = engine_specifics["dag_stats_by_target"]

                        if timeout_seconds > 0 and (total_elapsed_ms / 1000.0) > timeout_seconds:
                            status = "timeout"
                            error_code = f"ENGINE_TIMEOUT>{timeout_seconds}s"
                    else:
                        backend_result = _run_reasoner_attempt(
                            backend=run.reasoner or "elk",
                            query_graph=query_graph,
                            query_class_by_target=query_class_by_target,
                            candidate_terms=candidate_terms,
                            timeout_seconds=timeout_seconds,
                            owlapi_home=owlapi_home,
                            elk_classpath=resolved_elk_classpath,
                            elk_jar=elk_jar,
                            java_command=java_command,
                            javac_command=javac_command,
                        )
                        total_elapsed_ms = backend_result.elapsed_ms
                        setup_ms = backend_result.setup_elapsed_ms
                        reasoning_ms = backend_result.preprocess_elapsed_ms
                        retrieval_ms = backend_result.postprocess_elapsed_ms
                        new_type_inferences = _count_new_type_pairs(
                            backend_result.members_by_target,
                            asserted_target_pairs,
                        )

                        if backend_result.status != "ok":
                            status = "timeout" if backend_result.status == "timeout" else "error"
                            error_code = backend_result.error or backend_result.status.upper()
                        elif timeout_seconds > 0 and (total_elapsed_ms / 1000.0) > timeout_seconds:
                            status = "timeout"
                            error_code = f"REASONER_TIMEOUT>{timeout_seconds}s"
                        else:
                            total_elapsed_values.append(total_elapsed_ms)
                            setup_values.append(setup_ms)
                            reasoning_values.append(reasoning_ms)
                            retrieval_values.append(retrieval_ms)
                            inference_values.append(float(new_type_inferences))

                    attempts_completed += 1
                    _append_jsonl(
                        log_output_path,
                        {
                            "benchmark_id": benchmark_id,
                            "run_index": run.run_index,
                            "attempt": attempt,
                            "dataset_name": dataset_name,
                            "label": run.label,
                            "status": status,
                            "error_code": error_code,
                            "attempt_wall_seconds": perf_counter() - attempt_t0,
                            "engine_timing_breakdown": last_timing_breakdown,
                            "engine_timing_groups_ms": last_timing_groups,
                            "engine_dag_stats_by_target": last_dag_stats_by_target,
                        },
                    )
                    if status != "ok":
                        break
                except torch.cuda.OutOfMemoryError as exc:
                    status = "error"
                    error_code = "CUDA_OOM"
                    _append_jsonl(
                        log_output_path,
                        {
                            "benchmark_id": benchmark_id,
                            "run_index": run.run_index,
                            "attempt": attempt,
                            "dataset_name": dataset_name,
                            "label": run.label,
                            "status": status,
                            "error_code": error_code,
                            "exception": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                except Exception as exc:
                    status = "error"
                    error_code = exc.__class__.__name__
                    _append_jsonl(
                        log_output_path,
                        {
                            "benchmark_id": benchmark_id,
                            "run_index": run.run_index,
                            "attempt": attempt,
                            "dataset_name": dataset_name,
                            "label": run.label,
                            "status": status,
                            "error_code": error_code,
                            "exception": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break

            row = dict(row_base)
            row.update(
                {
                    "status": status.upper(),
                    "error_code": error_code,
                    "attempts_completed": attempts_completed,
                    "setup_ms_avg": _safe_mean(setup_values),
                    "reasoning_ms_avg": _safe_mean(reasoning_values),
                    "retrieval_ms_avg": _safe_mean(retrieval_values),
                    "total_elapsed_ms_avg": _safe_mean(total_elapsed_values),
                    "total_type_inferences_avg": _safe_mean(inference_values),
                    "graph_lowering_ms_avg": _safe_mean(engine_stage_values["graph_lowering_ms"]),
                    "identity_normalization_ms_avg": _safe_mean(engine_stage_values["identity_normalization_ms"]),
                    "engine_preprocessing_ms_avg": _safe_mean(engine_stage_values["engine_preprocessing_ms"]),
                    "engine_iterations_ms_avg": _safe_mean(engine_stage_values["engine_iterations_ms"]),
                    "engine_postprocessing_ms_avg": _safe_mean(engine_stage_values["engine_postprocessing_ms"]),
                    "tbox_cacheable_ms_avg": _safe_mean(engine_stage_values["tbox_cacheable_ms"]),
                    "iterations_avg": _safe_mean(engine_extra_values["iterations"]),
                    "process_memory_peak_bytes_avg": _safe_mean(engine_extra_values["process_memory_peak_bytes"]),
                    "kgraph_host_bytes_avg": _safe_mean(engine_extra_values["kgraph_host_bytes"]),
                    "compiled_dag_estimated_bytes_avg": _safe_mean(engine_extra_values["compiled_dag_estimated_bytes"]),
                    "cuda_peak_allocated_bytes_avg": _safe_mean(engine_extra_values["cuda_peak_allocated_bytes"]),
                    "cuda_peak_reserved_bytes_avg": _safe_mean(engine_extra_values["cuda_peak_reserved_bytes"]),
                    "dag_target_count_avg": _safe_mean(engine_extra_values["dag_target_count"]),
                    "dag_total_nodes_avg": _safe_mean(engine_extra_values["dag_total_nodes"]),
                    "dag_total_edges_avg": _safe_mean(engine_extra_values["dag_total_edges"]),
                    "dag_max_layers_avg": _safe_mean(engine_extra_values["dag_max_layers"]),
                    "dag_eval_ms_avg": _safe_mean(engine_extra_values["dag_eval_ms"]),
                }
            )
            if row["status"] != "OK":
                for key in (
                    "setup_ms_avg",
                    "reasoning_ms_avg",
                    "retrieval_ms_avg",
                    "total_elapsed_ms_avg",
                    "total_type_inferences_avg",
                    "graph_lowering_ms_avg",
                    "identity_normalization_ms_avg",
                    "engine_preprocessing_ms_avg",
                    "engine_iterations_ms_avg",
                    "engine_postprocessing_ms_avg",
                    "tbox_cacheable_ms_avg",
                    "iterations_avg",
                    "process_memory_peak_bytes_avg",
                    "kgraph_host_bytes_avg",
                    "compiled_dag_estimated_bytes_avg",
                    "cuda_peak_allocated_bytes_avg",
                    "cuda_peak_reserved_bytes_avg",
                    "dag_target_count_avg",
                    "dag_total_nodes_avg",
                    "dag_total_edges_avg",
                    "dag_max_layers_avg",
                    "dag_eval_ms_avg",
                ):
                    row[key] = None
            _append_csv_row(csv_output_path, row)

            summary_bits = [f"{run.run_index}/{total_runs}", dataset_name, run.kind]
            if run.kind == "engine":
                summary_bits.extend([run.profile or "", run.device or "", run.engine_mode or ""])
            else:
                summary_bits.append(run.reasoner or "")
            summary_bits.append(f"status={row['status']}")
            if row["total_elapsed_ms_avg"] is not None:
                summary_bits.append(f"avg_total={_format_elapsed_seconds(float(row['total_elapsed_ms_avg']))}")
            _progress(start_t0, "Completed run " + " | ".join(summary_bits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-running benchmark harness for paper-quality measurements.")
    parser.add_argument("--schema", nargs="+", required=True, help="Schema / ontology RDF files.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset RDF files to benchmark.")
    parser.add_argument("-k", type=int, required=True, help="Number of repetitions per configuration.")
    parser.add_argument("--reasoners", nargs="*", default=["elk", "openllet"], choices=["elk", "openllet"], help="Reasoners to benchmark.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_ENGINE_PROFILES),
        choices=list(DEFAULT_ENGINE_PROFILES),
        help="Subset of engine profiles to benchmark.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=list(DEFAULT_ENGINE_DEVICES),
        choices=list(DEFAULT_ENGINE_DEVICES),
        help="Subset of engine devices to benchmark.",
    )
    parser.add_argument(
        "--engine-modes",
        nargs="+",
        default=list(DEFAULT_ENGINE_MODES),
        choices=list(DEFAULT_ENGINE_MODES),
        help="Subset of engine modes to benchmark.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="Timeout for each reasoner run; engine uses soft timeout only.")
    parser.add_argument("--start-index", type=int, default=1, help="Resume benchmark plan from this 1-based run index.")
    parser.add_argument("--csv-path", required=True, help="Append-only CSV output path.")
    parser.add_argument("--log-path", help="JSONL log output path. Defaults next to CSV.")
    parser.add_argument("--owlapi-home", default=str(DEFAULT_OWLAPI_HOME), help="OWLAPI/Openllet/ELK home directory.")
    parser.add_argument("--elk-classpath", help="Explicit ELK classpath override.")
    parser.add_argument("--elk-jar", help="Explicit ELK jar/path override.")
    parser.add_argument("--java-command", default="java")
    parser.add_argument("--javac-command", default="javac")
    parser.add_argument("--graph-load-cache", choices=["off", "on", "refresh"], default="on")
    parser.add_argument("--graph-load-cache-dir")
    args = parser.parse_args()

    profiles = _dedupe_in_order(args.profiles)
    devices = _dedupe_in_order(args.devices)
    engine_modes = _dedupe_in_order(args.engine_modes)

    run_paper_benchmark(
        schema_paths=args.schema,
        dataset_paths=args.datasets,
        k=args.k,
        reasoners=args.reasoners,
        profiles=profiles,
        devices=devices,
        engine_modes=engine_modes,
        timeout_seconds=args.timeout_seconds,
        start_index=args.start_index,
        csv_path=args.csv_path,
        log_path=args.log_path,
        owlapi_home=args.owlapi_home,
        elk_classpath=args.elk_classpath,
        elk_jar=args.elk_jar,
        java_command=args.java_command,
        javac_command=args.javac_command,
        graph_load_cache=args.graph_load_cache,
        graph_load_cache_dir=args.graph_load_cache_dir,
    )


if __name__ == "__main__":
    main()
