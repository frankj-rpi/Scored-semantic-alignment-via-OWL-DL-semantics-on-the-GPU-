from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from rdflib.term import Identifier, Literal

from .consistency_harness import (
    BucketStats,
    FragmentGeneratorConfig,
    GeneratedFragmentCase,
    _case_requires_literals,
    _check_owlready2_consistency,
    _construct_bucket_key,
    _copy_graph,
    _dag_construct_tags,
    _ensure_dir,
    _format_bucket_key,
    _merge_graphs,
    _sanitize_bucket_key,
    _save_case_graphs,
    expand_construct_specs,
    generate_random_fragment_case,
)
from .ontology_parse import ConflictPolicy, compile_class_to_dag, compile_sufficient_condition_dag
from .oracle_compare import (
    DEFAULT_OWLAPI_HOME,
    _query_mode_compilation_kwargs,
    _render_term,
    aggregate_rdflib_graphs,
    apply_engine_profile,
    build_oracle_query_graph,
    normalize_engine_profile_name,
    resolve_oracle_query_mode,
    resolve_super_dag_mode,
    run_engine_queries,
    run_owlapi_reasoner_queries,
)


DEFAULT_ENGINE_PROFILE = "gpu-dl"


@dataclass
class CoverageExample:
    seed: int
    target_class: URIRef
    node_term: Identifier
    constructs: Tuple[str, ...]
    kind: str  # fp | fn
    engine_mode: str
    profile: str


@dataclass
class CoverageBucketStats:
    engine_pairs: int = 0
    oracle_pairs: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    fp_examples: List[CoverageExample] = field(default_factory=list)
    fn_examples: List[CoverageExample] = field(default_factory=list)


@dataclass
class CoverageSummary:
    requested_cases: int
    generated_cases: int
    attempts: int
    base_consistent_cases: int
    oracle_completed_cases: int
    oracle_timeout_cases: int
    oracle_error_cases: int
    engine_mode: str
    engine_profile: str
    oracle_backend: str
    timeout_seconds: float
    total_engine_pairs: int
    total_oracle_pairs: int
    total_true_positives: int
    total_false_positives: int
    total_false_negatives: int
    generation_elapsed_ms: float
    engine_elapsed_ms: float
    oracle_elapsed_ms: float
    base_consistency_check_elapsed_ms: float
    bucket_stats: Dict[Tuple[str, ...], CoverageBucketStats]
    run_settings: Dict[str, object]
    save_root: Optional[str] = None


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


def _member_pairs_without_asserted(
    members_by_target: Dict[URIRef, Set[Identifier]],
    asserted_pairs: Set[tuple[Identifier, URIRef]],
    *,
    allowed_members: Optional[Set[Identifier]] = None,
) -> Set[tuple[Identifier, URIRef]]:
    pairs: Set[tuple[Identifier, URIRef]] = set()
    for target_term, members in members_by_target.items():
        for member in members:
            if allowed_members is not None and member not in allowed_members:
                continue
            pair = (member, target_term)
            if pair not in asserted_pairs:
                pairs.add(pair)
    return pairs


def _case_target_bucket_key(
    *,
    case: GeneratedFragmentCase,
    target_class: URIRef,
    dataset_graph: Graph,
    mapping,
    engine_mode: str,
    augment_property_domain_range: bool,
) -> Tuple[str, ...]:
    if engine_mode == "stratified":
        dag = compile_sufficient_condition_dag(dataset_graph, mapping, target_class)
    else:
        dag = compile_class_to_dag(
            dataset_graph,
            mapping,
            target_class,
            augment_property_domain_range=augment_property_domain_range,
            **_query_mode_compilation_kwargs(engine_mode),
        )
    return _construct_bucket_key(
        set(case.target_constructs.get(target_class, ())) | _dag_construct_tags(dag)
    )


def _safe_div(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def _write_case_examples(
    *,
    case_dir: str,
    fp_examples: Sequence[CoverageExample],
    fn_examples: Sequence[CoverageExample],
) -> None:
    payload = {
        "false_positives": [
            {
                "seed": example.seed,
                "target_class": str(example.target_class),
                "node_term": _render_term(example.node_term),
                "constructs": list(example.constructs),
                "engine_mode": example.engine_mode,
                "profile": example.profile,
            }
            for example in fp_examples
        ],
        "false_negatives": [
            {
                "seed": example.seed,
                "target_class": str(example.target_class),
                "node_term": _render_term(example.node_term),
                "constructs": list(example.constructs),
                "engine_mode": example.engine_mode,
                "profile": example.profile,
            }
            for example in fn_examples
        ],
    }
    with open(os.path.join(case_dir, "coverage-examples.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_coverage_harness(
    *,
    num_cases: int,
    max_attempts: int,
    start_seed: int,
    config: FragmentGeneratorConfig,
    engine_mode: str,
    engine_profile: str = DEFAULT_ENGINE_PROFILE,
    threshold: float = 0.999,
    device: str = "cpu",
    materialize_hierarchy: Optional[bool] = None,
    augment_property_domain_range: Optional[bool] = None,
    owlready2_reasoner: str = "hermit",
    timeout_seconds: float = 30.0,
    max_examples_per_bucket: int = 3,
    owlapi_home: Optional[str] = None,
    save_cases: bool = True,
    save_dir: str = os.path.join("data", "runs", "coverage-harness"),
) -> CoverageSummary:
    engine_mode = engine_mode.strip().lower()
    normalized_profile = normalize_engine_profile_name(engine_profile)
    profile_options = apply_engine_profile(
        profile=normalized_profile,
        materialize_hierarchy=None,
        materialize_horn_safe_domain_range=None,
        materialize_reflexive_properties=None,
        materialize_sameas=None,
        materialize_haskey_equality=None,
        materialize_target_roles=None,
        augment_property_domain_range=None,
        enable_negative_verification=None,
    )
    enable_super_dag = resolve_super_dag_mode("auto", normalized_profile) == "on"
    requested_query_mode = resolve_oracle_query_mode("auto", engine_mode)

    generated_cases = 0
    attempts = 0
    seed = start_seed
    base_consistent_cases = 0
    oracle_completed_cases = 0
    oracle_timeout_cases = 0
    oracle_error_cases = 0
    total_engine_pairs = 0
    total_oracle_pairs = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    generation_elapsed_ms = 0.0
    engine_elapsed_ms = 0.0
    oracle_elapsed_ms = 0.0
    base_consistency_check_elapsed_ms = 0.0
    merged_bucket_stats: Dict[Tuple[str, ...], CoverageBucketStats] = {}

    run_save_root: Optional[str] = None
    if save_cases:
        run_save_root = os.path.join(save_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        _ensure_dir(run_save_root)

    while generated_cases < num_cases and attempts < max_attempts:
        generation_t0 = perf_counter()
        case = generate_random_fragment_case(config, seed=seed)
        generation_elapsed_ms += (perf_counter() - generation_t0) * 1000.0
        attempts += 1
        seed += 1

        merged_graph = _merge_graphs(case.schema_graph, case.data_graph)
        consistent, elapsed_ms, error = _check_owlready2_consistency(
            merged_graph,
            reasoner_name=owlready2_reasoner,
        )
        base_consistency_check_elapsed_ms += elapsed_ms
        if not consistent:
            if error:
                raise RuntimeError(
                    "owlready2 consistency check failed before coverage harness execution: "
                    + error
                )
            continue

        generated_cases += 1
        base_consistent_cases += 1

        case_dir = (
            _save_case_graphs(
                case,
                config=config,
                save_root=run_save_root,
                threshold=threshold,
                materialize_hierarchy=(
                    materialize_hierarchy if materialize_hierarchy is not None else profile_options.materialize_hierarchy
                ),
                augment_property_domain_range=(
                    augment_property_domain_range
                    if augment_property_domain_range is not None
                    else bool(profile_options.augment_property_domain_range)
                ),
                engine_mode=engine_mode,
                conflict_policy=(
                    ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value
                    if engine_mode == "stratified"
                    else None
                ),
                base_consistent=True,
            )
            if run_save_root is not None
            else None
        )

        include_literals = _case_requires_literals(case)
        engine_t0 = perf_counter()
        engine_result = run_engine_queries(
            schema_graph=case.schema_graph,
            data_graph=case.data_graph,
            target_classes=case.target_classes,
            device=device,
            threshold=threshold,
            include_literals=include_literals,
            include_type_edges=False,
            materialize_hierarchy=(
                profile_options.materialize_hierarchy
                if materialize_hierarchy is None
                else materialize_hierarchy
            ),
            materialize_horn_safe_domain_range=profile_options.materialize_horn_safe_domain_range,
            materialize_sameas=profile_options.materialize_sameas,
            materialize_haskey_equality=profile_options.materialize_haskey_equality,
            materialize_reflexive_properties=profile_options.materialize_reflexive_properties,
            materialize_target_roles=profile_options.materialize_target_roles,
            augment_property_domain_range=(
                profile_options.augment_property_domain_range
                if augment_property_domain_range is None
                else augment_property_domain_range
            ),
            enable_negative_verification=profile_options.enable_negative_verification,
            enable_negative_materialization=profile_options.enable_negative_materialization,
            native_sameas_canonicalization=profile_options.native_sameas_canonicalization,
            engine_mode=engine_mode,
            conflict_policy=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
            enable_super_dag=enable_super_dag,
        )
        engine_elapsed_ms += (perf_counter() - engine_t0) * 1000.0

        ontology_graph = aggregate_rdflib_graphs((case.schema_graph, case.data_graph))
        query_graph, query_class_by_target = build_oracle_query_graph(
            ontology_graph,
            case.target_classes,
            mode=requested_query_mode,
        )
        candidate_terms = _collect_candidate_terms(case.data_graph)
        oracle_t0 = perf_counter()
        oracle_result = run_owlapi_reasoner_queries(
            query_graph=query_graph,
            query_class_by_target=query_class_by_target,
            candidate_terms=candidate_terms,
            backend_name="openllet",
            reasoner_name="openllet",
            owlapi_home=owlapi_home,
            timeout_seconds=timeout_seconds,
        )
        oracle_elapsed_ms += (perf_counter() - oracle_t0) * 1000.0
        if oracle_result.status != "ok":
            if oracle_result.status == "timeout":
                oracle_timeout_cases += 1
            else:
                oracle_error_cases += 1
            if case_dir is not None:
                with open(os.path.join(case_dir, "oracle-error.json"), "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "backend": oracle_result.backend,
                            "status": oracle_result.status,
                            "error": oracle_result.error,
                        },
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
            continue
        oracle_completed_cases += 1

        asserted_pairs = _collect_asserted_target_pairs(case.data_graph, case.target_classes)
        engine_pairs = _member_pairs_without_asserted(
            engine_result.members_by_target,
            asserted_pairs,
            allowed_members=candidate_terms,
        )
        oracle_pairs = _member_pairs_without_asserted(
            oracle_result.members_by_target,
            asserted_pairs,
            allowed_members=candidate_terms,
        )
        true_positives = engine_pairs & oracle_pairs
        false_positives = engine_pairs - oracle_pairs
        false_negatives = oracle_pairs - engine_pairs

        total_engine_pairs += len(engine_pairs)
        total_oracle_pairs += len(oracle_pairs)
        total_true_positives += len(true_positives)
        total_false_positives += len(false_positives)
        total_false_negatives += len(false_negatives)

        fp_examples_for_case: List[CoverageExample] = []
        fn_examples_for_case: List[CoverageExample] = []

        dataset_graph = engine_result.dataset.ontology_graph if engine_result.dataset is not None else ontology_graph
        dataset_mapping = engine_result.dataset.mapping if engine_result.dataset is not None else None

        for target_class in case.target_classes:
            if dataset_mapping is None:
                continue
            bucket_key = _case_target_bucket_key(
                case=case,
                target_class=target_class,
                dataset_graph=dataset_graph,
                mapping=dataset_mapping,
                engine_mode=engine_mode,
                augment_property_domain_range=(
                    profile_options.augment_property_domain_range
                    if augment_property_domain_range is None
                    else augment_property_domain_range
                ) or False,
            )
            bucket = merged_bucket_stats.setdefault(bucket_key, CoverageBucketStats())

            engine_target_pairs = {
                pair for pair in engine_pairs
                if pair[1] == target_class
            }
            oracle_target_pairs = {
                pair for pair in oracle_pairs
                if pair[1] == target_class
            }
            tp_target_pairs = engine_target_pairs & oracle_target_pairs
            fp_target_pairs = engine_target_pairs - oracle_target_pairs
            fn_target_pairs = oracle_target_pairs - engine_target_pairs

            bucket.engine_pairs += len(engine_target_pairs)
            bucket.oracle_pairs += len(oracle_target_pairs)
            bucket.true_positives += len(tp_target_pairs)
            bucket.false_positives += len(fp_target_pairs)
            bucket.false_negatives += len(fn_target_pairs)

            for node_term, _target in sorted(fp_target_pairs, key=lambda item: (str(item[1]), str(item[0]))):
                example = CoverageExample(
                    seed=case.seed,
                    target_class=target_class,
                    node_term=node_term,
                    constructs=bucket_key,
                    kind="fp",
                    engine_mode=engine_mode,
                    profile=normalized_profile,
                )
                fp_examples_for_case.append(example)
                if len(bucket.fp_examples) < max_examples_per_bucket:
                    bucket.fp_examples.append(example)
            for node_term, _target in sorted(fn_target_pairs, key=lambda item: (str(item[1]), str(item[0]))):
                example = CoverageExample(
                    seed=case.seed,
                    target_class=target_class,
                    node_term=node_term,
                    constructs=bucket_key,
                    kind="fn",
                    engine_mode=engine_mode,
                    profile=normalized_profile,
                )
                fn_examples_for_case.append(example)
                if len(bucket.fn_examples) < max_examples_per_bucket:
                    bucket.fn_examples.append(example)

        if case_dir is not None and (fp_examples_for_case or fn_examples_for_case):
            _write_case_examples(
                case_dir=case_dir,
                fp_examples=fp_examples_for_case,
                fn_examples=fn_examples_for_case,
            )

    return CoverageSummary(
        requested_cases=num_cases,
        generated_cases=generated_cases,
        attempts=attempts,
        base_consistent_cases=base_consistent_cases,
        oracle_completed_cases=oracle_completed_cases,
        oracle_timeout_cases=oracle_timeout_cases,
        oracle_error_cases=oracle_error_cases,
        engine_mode=engine_mode,
        engine_profile=normalized_profile,
        oracle_backend="openllet",
        timeout_seconds=timeout_seconds,
        total_engine_pairs=total_engine_pairs,
        total_oracle_pairs=total_oracle_pairs,
        total_true_positives=total_true_positives,
        total_false_positives=total_false_positives,
        total_false_negatives=total_false_negatives,
        generation_elapsed_ms=generation_elapsed_ms,
        engine_elapsed_ms=engine_elapsed_ms,
        oracle_elapsed_ms=oracle_elapsed_ms,
        base_consistency_check_elapsed_ms=base_consistency_check_elapsed_ms,
        bucket_stats=merged_bucket_stats,
        run_settings={
            "num_cases": num_cases,
            "max_attempts": max_attempts,
            "start_seed": start_seed,
            "threshold": threshold,
            "device": device,
            "engine_mode": engine_mode,
            "engine_profile": normalized_profile,
            "materialize_hierarchy": materialize_hierarchy,
            "augment_property_domain_range": augment_property_domain_range,
            "owlready2_reasoner": owlready2_reasoner,
            "timeout_seconds": timeout_seconds,
            "max_examples_per_bucket": max_examples_per_bucket,
            "save_cases": save_cases,
            "save_dir": save_dir,
            "generator_config": dict(config.__dict__),
        },
        save_root=run_save_root,
    )


def format_coverage_summary(summary: CoverageSummary) -> str:
    lines: List[str] = []
    lines.append("=== Coverage Harness ===")
    lines.append(f"Engine mode: {summary.engine_mode}")
    lines.append(f"Engine profile: {summary.engine_profile}")
    lines.append(f"Oracle backend: {summary.oracle_backend}")
    lines.append(f"Oracle timeout: {summary.timeout_seconds:.1f} s")
    lines.append(f"Requested cases: {summary.requested_cases}")
    lines.append(f"Generated base-consistent cases: {summary.generated_cases}")
    lines.append(f"Generation attempts: {summary.attempts}")
    lines.append(f"Oracle-completed cases: {summary.oracle_completed_cases}")
    lines.append(f"Oracle timeouts: {summary.oracle_timeout_cases}")
    lines.append(f"Oracle errors: {summary.oracle_error_cases}")
    lines.append(f"Engine novel pairs: {summary.total_engine_pairs}")
    lines.append(f"Oracle novel pairs: {summary.total_oracle_pairs}")
    lines.append(f"True positives: {summary.total_true_positives}")
    lines.append(f"False positives: {summary.total_false_positives}")
    lines.append(f"False negatives: {summary.total_false_negatives}")
    lines.append(
        f"Precision: {_safe_div(summary.total_true_positives, summary.total_true_positives + summary.total_false_positives):.4f}"
    )
    lines.append(
        f"Recall: {_safe_div(summary.total_true_positives, summary.total_true_positives + summary.total_false_negatives):.4f}"
    )
    lines.append(
        f"False-positive rate over engine claims: {_safe_div(summary.total_false_positives, summary.total_engine_pairs):.4f}"
    )
    if summary.save_root:
        lines.append(f"Saved cases: {summary.save_root}")
    lines.append("")
    lines.append("Stage timings:")
    lines.append(f"  - Random case generation: {summary.generation_elapsed_ms:.3f} ms")
    lines.append(f"  - Base consistency filtering: {summary.base_consistency_check_elapsed_ms:.3f} ms")
    lines.append(f"  - Engine runs: {summary.engine_elapsed_ms:.3f} ms")
    lines.append(f"  - Openllet runs: {summary.oracle_elapsed_ms:.3f} ms")
    lines.append("")
    lines.append("Coverage buckets by construct set:")
    exercised_buckets = {
        bucket_key: bucket
        for bucket_key, bucket in summary.bucket_stats.items()
        if bucket.engine_pairs > 0 or bucket.oracle_pairs > 0
    }
    if not exercised_buckets:
        lines.append("  (no exercised buckets)")
        return "\n".join(lines)

    ordered_buckets = sorted(
        exercised_buckets.items(),
        key=lambda item: (-item[1].false_positives, -item[1].false_negatives, -item[1].oracle_pairs, item[0]),
    )
    for bucket_key, bucket in ordered_buckets:
        precision = _safe_div(bucket.true_positives, bucket.true_positives + bucket.false_positives)
        recall = _safe_div(bucket.true_positives, bucket.true_positives + bucket.false_negatives)
        lines.append(
            f"  - [{_format_bucket_key(bucket_key)}] "
            f"engine={bucket.engine_pairs}, oracle={bucket.oracle_pairs}, "
            f"tp={bucket.true_positives}, fp={bucket.false_positives}, fn={bucket.false_negatives}, "
            f"precision={precision:.3f}, recall={recall:.3f}"
        )
        for example in bucket.fp_examples:
            lines.append(
                f"      fp example: seed={example.seed}, node={_render_term(example.node_term)}, "
                f"target={_render_term(example.target_class)}"
            )
        for example in bucket.fn_examples:
            lines.append(
                f"      fn example: seed={example.seed}, node={_render_term(example.node_term)}, "
                f"target={_render_term(example.target_class)}"
            )
    return "\n".join(lines)


def _write_run_summary(summary: CoverageSummary) -> None:
    if not summary.save_root:
        return
    payload = {
        "requested_cases": summary.requested_cases,
        "generated_cases": summary.generated_cases,
        "attempts": summary.attempts,
        "base_consistent_cases": summary.base_consistent_cases,
        "oracle_completed_cases": summary.oracle_completed_cases,
        "oracle_timeout_cases": summary.oracle_timeout_cases,
        "oracle_error_cases": summary.oracle_error_cases,
        "engine_mode": summary.engine_mode,
        "engine_profile": summary.engine_profile,
        "oracle_backend": summary.oracle_backend,
        "timeout_seconds": summary.timeout_seconds,
        "totals": {
            "engine_pairs": summary.total_engine_pairs,
            "oracle_pairs": summary.total_oracle_pairs,
            "true_positives": summary.total_true_positives,
            "false_positives": summary.total_false_positives,
            "false_negatives": summary.total_false_negatives,
            "precision": _safe_div(summary.total_true_positives, summary.total_true_positives + summary.total_false_positives),
            "recall": _safe_div(summary.total_true_positives, summary.total_true_positives + summary.total_false_negatives),
            "false_positive_rate_over_engine_claims": _safe_div(summary.total_false_positives, summary.total_engine_pairs),
        },
        "timings_ms": {
            "generation": summary.generation_elapsed_ms,
            "base_consistency_check": summary.base_consistency_check_elapsed_ms,
            "engine": summary.engine_elapsed_ms,
            "oracle": summary.oracle_elapsed_ms,
        },
        "run_settings": summary.run_settings,
        "buckets": {
            _format_bucket_key(bucket_key): {
                "engine_pairs": bucket.engine_pairs,
                "oracle_pairs": bucket.oracle_pairs,
                "true_positives": bucket.true_positives,
                "false_positives": bucket.false_positives,
                "false_negatives": bucket.false_negatives,
                "precision": _safe_div(bucket.true_positives, bucket.true_positives + bucket.false_positives),
                "recall": _safe_div(bucket.true_positives, bucket.true_positives + bucket.false_negatives),
                "false_positive_examples": [
                    {
                        "seed": example.seed,
                        "target_class": str(example.target_class),
                        "node_term": _render_term(example.node_term),
                        "constructs": list(example.constructs),
                    }
                    for example in bucket.fp_examples
                ],
                "false_negative_examples": [
                    {
                        "seed": example.seed,
                        "target_class": str(example.target_class),
                        "node_term": _render_term(example.node_term),
                        "constructs": list(example.constructs),
                    }
                    for example in bucket.fn_examples
                ],
            }
            for bucket_key, bucket in summary.bucket_stats.items()
        },
    }
    with open(os.path.join(summary.save_root, "run-summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with open(os.path.join(summary.save_root, "run-summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(format_coverage_summary(summary))


def print_coverage_summary(summary: CoverageSummary) -> None:
    print(format_coverage_summary(summary))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate small random OWL-like cases and compare engine outputs "
            "against Openllet coverage on novel (node, target) pairs."
        )
    )
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-primitive-classes", type=int, default=5)
    parser.add_argument("--num-target-classes", type=int, default=4)
    parser.add_argument("--num-properties", type=int, default=3)
    parser.add_argument("--num-datatype-properties", type=int, default=2)
    parser.add_argument("--num-individuals", type=int, default=8)
    parser.add_argument("--min-axioms-per-target", type=int, default=1)
    parser.add_argument("--max-axioms-per-target", type=int, default=3)
    parser.add_argument("--type-probability", type=float, default=0.35)
    parser.add_argument("--edge-probability", type=float, default=0.18)
    parser.add_argument("--datatype-edge-probability", type=float, default=0.35)
    parser.add_argument("--disjoint-pair-probability", type=float, default=0.10)
    parser.add_argument("--domain-range-axiom-probability", type=float, default=0.35)
    parser.add_argument(
        "--constructs",
        nargs="+",
        choices=[
            "subclass",
            "intersection",
            "union",
            "exists",
            "forall",
            "datatype",
            "has-value",
            "data-oneof",
            "nominal",
            "has-self",
            "reflexive",
            "geq-cardinality",
            "functional-data-property",
            "negative-object-property",
            "negative-data-property",
            "has-key",
            "disjoint",
            "domain",
            "range",
            "OWL-EL",
            "OWL_EL",
            "EL++",
        ],
        default=["OWL-EL"],
        help="Construct names and/or profile names to include in random generation.",
    )
    parser.add_argument("--engine-mode", choices=["stratified", "admissibility", "filtered_admissibility"], default="stratified")
    parser.add_argument("--engine-profile", default=DEFAULT_ENGINE_PROFILE)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-examples-per-bucket", type=int, default=3)
    parser.add_argument("--owlapi-home", default=str(DEFAULT_OWLAPI_HOME))
    parser.add_argument("--owlready2-reasoner", choices=["hermit", "pellet"], default="hermit")
    parser.add_argument("--no-save-cases", action="store_true")
    parser.add_argument(
        "--save-dir",
        default=os.path.join("data", "runs", "coverage-harness"),
        help="Directory under which successful harness runs are persisted.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = FragmentGeneratorConfig(
        num_primitive_classes=args.num_primitive_classes,
        num_target_classes=args.num_target_classes,
        num_properties=args.num_properties,
        num_datatype_properties=args.num_datatype_properties,
        num_individuals=args.num_individuals,
        min_axioms_per_target=args.min_axioms_per_target,
        max_axioms_per_target=args.max_axioms_per_target,
        type_probability=args.type_probability,
        edge_probability=args.edge_probability,
        datatype_edge_probability=args.datatype_edge_probability,
        disjoint_pair_probability=args.disjoint_pair_probability,
        domain_range_axiom_probability=args.domain_range_axiom_probability,
        allowed_constructs=expand_construct_specs(tuple(args.constructs)),
    )
    summary = run_coverage_harness(
        num_cases=args.num_cases,
        max_attempts=args.max_attempts,
        start_seed=args.start_seed,
        config=config,
        engine_mode=args.engine_mode,
        engine_profile=args.engine_profile,
        threshold=args.threshold,
        device=args.device,
        owlready2_reasoner=args.owlready2_reasoner,
        timeout_seconds=args.timeout_seconds,
        max_examples_per_bucket=args.max_examples_per_bucket,
        owlapi_home=args.owlapi_home,
        save_cases=not args.no_save_cases,
        save_dir=args.save_dir,
    )
    _write_run_summary(summary)
    print_coverage_summary(summary)


if __name__ == "__main__":
    main()
