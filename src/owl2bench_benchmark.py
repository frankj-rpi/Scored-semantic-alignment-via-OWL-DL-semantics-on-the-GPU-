from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set

from rdflib import Graph, RDF, URIRef
from rdflib.term import Identifier

from .oracle_compare import (
    DEFAULT_OWLAPI_HOME,
    PROFILE_ALIASES,
    _format_elapsed_seconds,
    apply_engine_profile,
    BackendQueryResult,
    build_oracle_query_graph,
    format_engine_timing_breakdown,
    format_skipped_target_warnings,
    resolve_target_classes,
    resolve_super_dag_mode,
    run_elk_queries,
    run_engine_queries,
    run_owlready2_queries,
    run_owlrl_queries,
    validate_elk_backend,
)
from .ontology_parse import load_rdflib_graph


def _merge_graphs(*graphs: Graph) -> Graph:
    merged = Graph()
    for graph in graphs:
        for triple in graph:
            merged.add(triple)
    return merged


def _count_total_matches(members_by_target: Dict[URIRef, Set[Identifier]]) -> int:
    return sum(len(members) for members in members_by_target.values())


def _summarize_backend_agreement(
    *,
    targets: Sequence[URIRef],
    engine_members_by_target: Dict[URIRef, Set[Identifier]],
    backend: BackendQueryResult,
) -> tuple[int, int]:
    agreed = 0
    compared = 0
    if backend.status != "ok":
        return agreed, compared

    for target_term in targets:
        compared += 1
        if engine_members_by_target.get(target_term, set()) == backend.members_by_target.get(target_term, set()):
            agreed += 1
    return agreed, compared


def run_owl2bench_benchmark(
    *,
    schema_paths: Sequence[str],
    data_paths: Sequence[str],
    target_class_specs: Sequence[str],
    modes: Sequence[str],
    device: str = "cpu",
    threshold: float = 0.999,
    include_literals: bool = True,
    include_type_edges: bool = False,
    profile: Optional[str] = None,
    materialize_hierarchy: Optional[bool] = None,
    materialize_horn_safe_domain_range: Optional[bool] = None,
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
    materialize_reflexive_properties: Optional[bool] = None,
    materialize_target_roles: Optional[bool] = None,
    augment_property_domain_range: Optional[bool] = None,
    conflict_policy: str = "suppress_derived_keep_asserted",
    enable_negative_verification: Optional[bool] = None,
    query_mode: str = "query",
    stratified_query_mode: str = "native",
    oracle_backends: Sequence[str] = (),
    owlready2_reasoner: str = "hermit",
    owlapi_home: Optional[str] = None,
    elk_classpath: Optional[str] = None,
    elk_jar: Optional[str] = None,
    elk_java_command: str = "java",
    elk_javac_command: str = "javac",
    show_timing_breakdown: bool = False,
    graph_load_cache: str = "on",
    graph_load_cache_dir: Optional[str] = None,
    super_dag: str = "auto",
    verbose: bool = False,
) -> None:
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
    super_dag = resolve_super_dag_mode(super_dag, profile)
    schema_graph = load_rdflib_graph(
        schema_paths,
        cache_mode=graph_load_cache,
        cache_dir=graph_load_cache_dir,
    )
    data_graph = load_rdflib_graph(
        data_paths,
        cache_mode=graph_load_cache,
        cache_dir=graph_load_cache_dir,
    )
    ontology_graph = _merge_graphs(schema_graph, data_graph)

    print("=== OWL2Bench Benchmark ===")
    print(f"Schema files: {len(schema_paths)}")
    print(f"Data files: {len(data_paths)}")
    print(f"Requested target selector(s): {', '.join(target_class_specs)}")
    print(f"Oracles: {', '.join(oracle_backends) if oracle_backends else 'none'}")

    resolved_elk_classpath = elk_classpath
    if "elk" in oracle_backends:
        ok, error_text, resolved_classpath = validate_elk_backend(
            elk_classpath=elk_classpath,
            elk_jar=elk_jar,
            owlapi_home=owlapi_home,
            java_command=elk_java_command,
            javac_command=elk_javac_command,
        )
        if not ok:
            print("")
            print("ELK preflight failed before engine execution.")
            print(error_text)
            return
        resolved_elk_classpath = resolved_classpath
    if "openllet" in oracle_backends:
        ok, error_text, _resolved_classpath = validate_owlapi_reasoner_backend(
            reasoner_name="openllet",
            owlapi_home=owlapi_home,
            java_command=elk_java_command,
            javac_command=elk_javac_command,
        )
        if not ok:
            print("")
            print("Openllet preflight failed before engine execution.")
            print(error_text)
            return

    resolution_cache: Dict[str, tuple[float, object]] = {}
    for engine_mode in modes:
        print("")
        print(f"--- Engine mode: {engine_mode} ---")
        resolution_key = "admissibility-like" if engine_mode in {"admissibility", "filtered_admissibility", "scored_semantic_alignment", "query", "filtered_query"} else engine_mode
        cached_resolution = resolution_cache.get(resolution_key)
        if cached_resolution is None:
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
            resolution_cache[resolution_key] = (resolution_elapsed_ms, resolution)
        else:
            resolution_elapsed_ms, resolution = cached_resolution
        warning_text = format_skipped_target_warnings(resolution, verbose=verbose)
        if warning_text:
            print(warning_text)
        print(f"Target resolution time: {_format_elapsed_seconds(resolution_elapsed_ms)}")
        print(f"Resolved target classes: {len(resolution.resolved_targets)}")
        if not resolution.resolved_targets:
            print("No target classes resolved for this mode.")
            continue

        engine_result = run_engine_queries(
            schema_graph=schema_graph,
            data_graph=data_graph,
            target_classes=resolution.resolved_targets,
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
            augment_property_domain_range=profile_options.augment_property_domain_range,
            engine_mode=engine_mode,
            conflict_policy=conflict_policy,
                enable_negative_verification=profile_options.enable_negative_verification,
                enable_super_dag=(super_dag == "on"),
            )

        effective_query_mode = query_mode if engine_mode != "stratified" else stratified_query_mode
        query_graph, query_class_by_target = build_oracle_query_graph(
            ontology_graph,
            resolution.resolved_targets,
            mode=effective_query_mode,
            bridge_supported_definitions=(engine_mode == "stratified"),
        )
        candidate_terms = set(engine_result.dataset.mapping.node_terms) if engine_result.dataset else set()

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
                    )
                )
            else:
                raise ValueError(f"Unsupported oracle backend: {backend}")

        print(f"Engine threshold: {threshold}")
        print(f"Engine time: {_format_elapsed_seconds(engine_result.elapsed_ms)}")
        print(f"Total engine matches: {_count_total_matches(engine_result.members_by_target)}")
        if show_timing_breakdown:
            print(format_engine_timing_breakdown(engine_result))

        for backend in oracle_results:
            status = backend.status
            if backend.consistent is True:
                status += ", consistent"
            elif backend.consistent is False:
                status += ", inconsistent"
            agreed, compared = _summarize_backend_agreement(
                targets=resolution.resolved_targets,
                engine_members_by_target=engine_result.members_by_target,
                backend=backend,
            )
            print(
                f"{backend.backend}: time={_format_elapsed_seconds(backend.elapsed_ms)}, "
                f"status={status}, target agreement={agreed}/{compared}"
            )
            if backend.error:
                print(f"  error: {backend.error}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a small benchmark over OWL2Bench using the existing engine/oracle comparison pipeline."
    )
    parser.add_argument(
        "--schema",
        nargs="+",
        default=["data\\owl2bench\\UNIV-BENCH-OWL2EL.owl"],
        help="Schema / ontology RDF files.",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data\\owl2bench\\OWL2EL-1.owl"],
        help="Instance data RDF files.",
    )
    parser.add_argument(
        "--target-class",
        nargs="+",
        default=["all-defined-classes"],
        help=(
            "Target classes or selectors. Special selectors: all, all-named-classes, "
            "all-defined-classes, all-inferable-classes."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["admissibility", "filtered_admissibility", "scored_semantic_alignment", "stratified", "query", "filtered_query"],
        default=["admissibility", "filtered_admissibility", "scored_semantic_alignment", "stratified"],
        help="Engine modes to benchmark.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
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
    literal_group = parser.add_mutually_exclusive_group()
    literal_group.add_argument(
        "--include-literals",
        dest="include_literals",
        action="store_true",
        help="Lift literals into the KGraph during evaluation.",
    )
    literal_group.add_argument(
        "--no-include-literals",
        dest="include_literals",
        action="store_false",
        help="Disable literal lifting explicitly.",
    )
    parser.set_defaults(include_literals=True)
    parser.add_argument("--include-type-edges", action="store_true")
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
    parser.add_argument("--conflict-policy", default="suppress_derived_keep_asserted")
    parser.add_argument(
        "--query-mode",
        choices=["query", "native"],
        default="query",
        help="Oracle query mode for admissibility and filtered_admissibility engine modes.",
    )
    parser.add_argument(
        "--stratified-query-mode",
        choices=["query", "native"],
        default="native",
        help="Oracle query mode for stratified engine mode.",
    )
    parser.add_argument(
        "--oracles",
        nargs="+",
        choices=["owlrl", "owlready2", "elk", "openllet"],
        default=[],
        help="Which oracle backends to run. Default: none (engine-only).",
    )
    parser.add_argument("--owlready2-reasoner", choices=["hermit", "pellet"], default="hermit")
    parser.add_argument(
        "--owlapi-home",
        default=str(DEFAULT_OWLAPI_HOME),
        help=(
            "Root directory for OWLAPI reasoner assets such as comparison/owlapi-5.5.1. "
            "If ELK is selected and no explicit classpath override is given, this directory "
            "is used to resolve jars from classpath.txt or maven-repo."
        ),
    )
    parser.add_argument("--elk-classpath")
    parser.add_argument("--elk-jar")
    parser.add_argument("--elk-java-command", default="java")
    parser.add_argument("--elk-javac-command", default="javac")
    parser.add_argument("--show-timing-breakdown", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_owl2bench_benchmark(
        schema_paths=args.schema,
        data_paths=args.data,
        target_class_specs=args.target_class,
        modes=args.modes,
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
        conflict_policy=args.conflict_policy,
        enable_negative_verification=args.enable_negative_verification,
        query_mode=args.query_mode,
        stratified_query_mode=args.stratified_query_mode,
        oracle_backends=args.oracles,
        owlready2_reasoner=args.owlready2_reasoner,
        owlapi_home=args.owlapi_home,
        elk_classpath=args.elk_classpath,
        elk_jar=args.elk_jar,
        elk_java_command=args.elk_java_command,
        elk_javac_command=args.elk_javac_command,
        show_timing_breakdown=args.show_timing_breakdown,
        graph_load_cache=args.graph_load_cache,
        graph_load_cache_dir=args.graph_load_cache_dir,
        super_dag=args.super_dag,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
