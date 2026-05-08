from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set

from rdflib import Graph
from rdflib.term import Identifier, URIRef

from .oracle_compare import (
    DEFAULT_OWLAPI_HOME,
    _expand_sameas_results_for_reporting,
    _render_term,
    apply_engine_profile,
    build_oracle_query_graph,
    normalize_engine_profile_name,
    resolve_oracle_query_mode,
    resolve_super_dag_mode,
    resolve_target_classes,
    run_engine_queries,
    run_owlapi_reasoner_queries,
    validate_owlapi_reasoner_backend,
)
from .ontology_parse import aggregate_rdflib_graphs, load_rdflib_graph
from .ontology_parse import collect_negative_fragment_support_status


DEFAULT_MODES: tuple[str, ...] = (
    "stratified",
    "admissibility",
    "filtered_admissibility",
)


@dataclass(frozen=True)
class ToyCase:
    name: str
    schema: str
    data: str
    skip_reason: Optional[str] = None
    mode_skip_reasons: Dict[str, str] | None = None
    expected_fail_reasons: Dict[str, str] | None = None


TOY_CASES: tuple[ToyCase, ...] = (
    ToyCase(
        name="data_some_oneof_people",
        schema="data/toys/data_some_oneof_people.owl.ttl",
        data="data/toys/data_some_oneof_people_data.ttl",
    ),
    ToyCase(
        name="functional_data_people",
        schema="data/toys/functional_data_people.owl.ttl",
        data="data/toys/functional_data_people_data.ttl",
    ),
    ToyCase(
        name="toy_people",
        schema="data/toys/toy_people.owl.ttl",
        data="data/toys/toy_people_data.ttl",
    ),
    ToyCase(
        name="toy_sameas_native",
        schema="data/toys/toy_sameas_native_schema.ttl",
        data="data/toys/toy_sameas_native_data.ttl",
    ),
    ToyCase(
        name="toy_hobby_chain",
        schema="data/toys/toy_hobby_chain_schema.ttl",
        data="data/toys/toy_hobby_chain_data.ttl",
    ),
    ToyCase(
        name="toy_superdag_acyclic",
        schema="data/toys/toy_superdag_acyclic_schema.ttl",
        data="data/toys/toy_superdag_acyclic_data.ttl",
    ),
    ToyCase(
        name="toy_superdag_scc",
        schema="data/toys/toy_superdag_scc_schema.ttl",
        data="data/toys/toy_superdag_scc_data.ttl",
    ),
    ToyCase(
        name="toy_negative_assertion",
        schema="data/toys/toy_negative_assertion_schema.ttl",
        data="data/toys/toy_negative_assertion_data.ttl",
        mode_skip_reasons={
            "admissibility": "Negative class reasoning is only implemented for stratified mode so far.",
            "filtered_admissibility": "Negative class reasoning is only implemented for stratified mode so far.",
        },
    ),
    ToyCase(
        name="toy_negative_conclusion",
        schema="data/toys/toy_negative_conclusion_schema.ttl",
        data="data/toys/toy_negative_conclusion_data.ttl",
        mode_skip_reasons={
            "admissibility": "Negative class reasoning is only implemented for stratified mode so far.",
            "filtered_admissibility": "Negative class reasoning is only implemented for stratified mode so far.",
        },
    ),
    ToyCase(
        name="toy_disjunction_chain",
        schema="data/toys/toy_disjunction_chain_schema.ttl",
        data="data/toys/toy_disjunction_chain_data.ttl",
        expected_fail_reasons={
            "admissibility": "Disjunctive negative-side certification is not yet supported natively.",
            "filtered_admissibility": "Disjunctive negative-side certification is not yet supported natively.",
        },
    ),
    ToyCase(
        name="toy_disjunction_branching",
        schema="data/toys/toy_disjunction_branching_schema.ttl",
        data="data/toys/toy_disjunction_branching_data.ttl",
        expected_fail_reasons={
            "admissibility": "Tableau-style branching over disjunction is not yet supported natively.",
            "filtered_admissibility": "Tableau-style branching over disjunction is not yet supported natively.",
        },
    ),
    ToyCase(
        name="toy_superdag_multiscc",
        schema="data/toys/toy_superdag_multiscc_schema.ttl",
        data="data/toys/toy_superdag_multiscc_data.ttl",
        skip_reason="Skipped because Openllet is known to struggle badly on this hard SCC toy.",
    ),
)


def _sorted_terms(terms: Sequence[Identifier]) -> List[Identifier]:
    return sorted(terms, key=lambda term: _render_term(term))


def _compare_members(
    engine_members_by_target: Dict[URIRef, Set[Identifier]],
    oracle_members_by_target: Dict[URIRef, Set[Identifier]],
    target_terms: Sequence[URIRef],
) -> List[str]:
    mismatches: List[str] = []
    for target_term in target_terms:
        engine_members = engine_members_by_target.get(target_term, set())
        oracle_members = oracle_members_by_target.get(target_term, set())
        if engine_members == oracle_members:
            continue
        only_engine = _sorted_terms(tuple(engine_members - oracle_members))
        only_oracle = _sorted_terms(tuple(oracle_members - engine_members))
        details: List[str] = [f"target={_render_term(target_term)}"]
        if only_engine:
            details.append(
                "only_engine=" + ", ".join(_render_term(term) for term in only_engine)
            )
        if only_oracle:
            details.append(
                "only_openllet=" + ", ".join(_render_term(term) for term in only_oracle)
            )
        mismatches.append(" | ".join(details))
    return mismatches


def _run_case(
    *,
    toy_case: ToyCase,
    mode: str,
    device: str,
    owlapi_home: Optional[str],
) -> tuple[str, List[str]]:
    schema_graph = load_rdflib_graph(toy_case.schema, cache_mode="off")
    data_graph = load_rdflib_graph(toy_case.data, cache_mode="off")

    profile = "gpu-dl"
    resolved_profile = normalize_engine_profile_name(profile)
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
    include_literals = resolved_profile == "gpu-dl"

    resolution = resolve_target_classes(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_class_specs=("all-named-classes",),
        engine_mode=mode,
        include_literals=include_literals,
        include_type_edges=False,
        materialize_hierarchy=profile_options.materialize_hierarchy,
        augment_property_domain_range=profile_options.augment_property_domain_range,
    )
    if resolution.skipped_targets:
        skipped_text = "; ".join(
            f"{target}: {reason}" for target, reason in resolution.skipped_targets
        )
        return "FAIL", [f"unexpected skipped targets: {skipped_text}"]
    if not resolution.resolved_targets:
        return "FAIL", ["no target classes resolved"]

    super_dag = resolve_super_dag_mode("auto", profile)
    engine_result = run_engine_queries(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_classes=resolution.resolved_targets,
        device=device,
        threshold=0.999,
        include_literals=include_literals,
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
        engine_mode=mode,
        conflict_policy="suppress_derived_keep_asserted",
        enable_negative_verification=profile_options.enable_negative_verification,
        enable_negative_materialization=profile_options.enable_negative_materialization,
        enable_super_dag=(super_dag == "on"),
    )

    effective_query_mode = resolve_oracle_query_mode("auto", mode)
    ontology_graph = aggregate_rdflib_graphs((schema_graph, data_graph))
    negative_support_status = collect_negative_fragment_support_status(
        schema_graph,
        resolution.resolved_targets,
    )
    negative_support_details = [
        (
            f"negative_support target={_render_term(target_term)} "
            f"complete={status.is_complete} "
            f"reasons={', '.join(status.reasons) if status.reasons else 'none'}"
        )
        for target_term, status in sorted(negative_support_status.items(), key=lambda item: str(item[0]))
    ]
    query_graph, query_class_by_target = build_oracle_query_graph(
        ontology_graph,
        resolution.resolved_targets,
        mode=effective_query_mode,
    )

    candidate_terms = set(engine_result.dataset.mapping.node_terms) if engine_result.dataset else set()
    if engine_result.dataset is not None and engine_result.dataset.sameas_members_by_canonical:
        for aliases in engine_result.dataset.sameas_members_by_canonical.values():
            candidate_terms.update(aliases)

    oracle_result = run_owlapi_reasoner_queries(
        query_graph=query_graph,
        query_class_by_target=query_class_by_target,
        candidate_terms=candidate_terms,
        backend_name="openllet",
        reasoner_name="openllet",
        owlapi_home=owlapi_home,
    )
    if oracle_result.status != "ok":
        return "FAIL", [
            f"openllet status={oracle_result.status}: {oracle_result.error or 'unknown error'}",
            *negative_support_details,
        ]

    engine_members = engine_result.members_by_target
    engine_scores = engine_result.scores_by_target or {}
    engine_members, _engine_scores = _expand_sameas_results_for_reporting(
        engine_result.dataset,
        engine_members,
        engine_scores,
    )
    mismatches = _compare_members(
        engine_members,
        oracle_result.members_by_target,
        resolution.resolved_targets,
    )
    if mismatches:
        return "FAIL", [*mismatches, *negative_support_details]
    return "PASS", negative_support_details


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run gpu-dl toy regressions against Openllet for stratified, "
            "admissibility, and filtered_admissibility."
        )
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Engine device to use for gpu-dl toy checks. Default: cpu.",
    )
    parser.add_argument(
        "--owlapi-home",
        default=str(DEFAULT_OWLAPI_HOME),
        help="Path to OWLAPI/Openllet helper home. Default: comparison/owlapi-5.5.1",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(DEFAULT_MODES),
        default=list(DEFAULT_MODES),
        help="Subset of engine modes to test. Default: all three oracle-supported modes.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[],
        help="Optional subset of toy case names to run.",
    )
    args = parser.parse_args()

    ok, error_text, _classpath = validate_owlapi_reasoner_backend(
        reasoner_name="openllet",
        owlapi_home=args.owlapi_home,
    )
    if not ok:
        print(f"Openllet preflight failed: {error_text}", file=sys.stderr)
        raise SystemExit(2)

    selected_case_names = set(args.cases)
    active_cases = [
        toy_case
        for toy_case in TOY_CASES
        if not selected_case_names or toy_case.name in selected_case_names
    ]
    if not active_cases:
        print("No toy cases selected.", file=sys.stderr)
        raise SystemExit(2)

    t0 = perf_counter()
    total = 0
    passed = 0
    skipped = 0
    failed = 0
    xfailed = 0
    xpassed = 0

    for toy_case in active_cases:
        if toy_case.skip_reason is not None:
            print(f"SKIP {toy_case.name}: {toy_case.skip_reason}")
            skipped += len(args.modes)
            continue
        for mode in args.modes:
            mode_skip_reason = (
                None
                if toy_case.mode_skip_reasons is None
                else toy_case.mode_skip_reasons.get(mode)
            )
            if mode_skip_reason is not None:
                print(f"SKIP {toy_case.name} [{mode}]: {mode_skip_reason}")
                skipped += 1
                continue
            expected_fail_reason = (
                None
                if toy_case.expected_fail_reasons is None
                else toy_case.expected_fail_reasons.get(mode)
            )
            total += 1
            try:
                status, details = _run_case(
                    toy_case=toy_case,
                    mode=mode,
                    device=args.device,
                    owlapi_home=args.owlapi_home,
                )
            except Exception as exc:
                status, details = "FAIL", [f"exception: {exc}"]
            if status == "PASS" and expected_fail_reason is None:
                passed += 1
                print(f"PASS {toy_case.name} [{mode}]")
            elif status == "PASS" and expected_fail_reason is not None:
                xpassed += 1
                print(f"XPASS {toy_case.name} [{mode}]")
                print(f"  expected-fail reason: {expected_fail_reason}")
                for detail in details:
                    print(f"  {detail}")
            else:
                if expected_fail_reason is not None:
                    xfailed += 1
                    print(f"XFAIL {toy_case.name} [{mode}]")
                    print(f"  expected-fail reason: {expected_fail_reason}")
                else:
                    failed += 1
                    print(f"FAIL {toy_case.name} [{mode}]")
                for detail in details:
                    print(f"  {detail}")

    elapsed = perf_counter() - t0
    print(
        f"Summary: passed={passed}, failed={failed}, skipped={skipped}, "
        f"xfailed={xfailed}, xpassed={xpassed}, "
        f"elapsed={elapsed:.2f}s"
    )
    raise SystemExit(1 if failed or xpassed else 0)


if __name__ == "__main__":
    main()
