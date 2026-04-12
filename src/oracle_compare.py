from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import contextlib
import io
import os
import tempfile

import torch
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.term import Identifier

from .dag_reasoner import DAGReasoner
from .ontology_parse import (
    ReasoningDataset,
    build_reasoning_dataset_from_graphs,
    collect_inferable_named_classes,
    compile_class_to_dag,
    load_rdflib_graph,
    materialize_supported_class_inferences,
    plan_reasoning_preprocessing,
    summarize_loaded_kgraph,
)


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
    dataset_build_elapsed_ms: float = 0.0
    hierarchy_elapsed_ms: float = 0.0
    atomic_domain_range_elapsed_ms: float = 0.0
    horn_safe_domain_range_elapsed_ms: float = 0.0
    target_role_elapsed_ms: float = 0.0
    kgraph_build_elapsed_ms: float = 0.0
    dag_compile_elapsed_ms: float = 0.0
    dag_eval_elapsed_ms: float = 0.0


def _render_term(term: Identifier) -> str:
    return term.n3() if hasattr(term, "n3") else str(term)


def _sorted_terms(terms: Iterable[Identifier]) -> List[Identifier]:
    return sorted(terms, key=lambda term: _render_term(term))


def _copy_graph(graph: Graph) -> Graph:
    copied = Graph()
    for triple in graph:
        copied.add(triple)
    return copied


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
    materialize_supported_types: bool = False,
    augment_property_domain_range: Optional[bool] = None,
) -> EngineQueryResult:
    device_to_use = device
    if device_to_use == "cuda" and not torch.cuda.is_available():
        device_to_use = "cpu"

    t0 = perf_counter()
    iterations: Optional[int] = None

    dataset_t0 = perf_counter()
    if materialize_supported_types:
        materialized = materialize_supported_class_inferences(
            schema_graph=schema_graph,
            data_graph=data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            target_classes=target_classes,
            threshold=threshold,
            device=device_to_use,
        )
        dataset = materialized.dataset
        iterations = materialized.iterations
    else:
        dataset = build_reasoning_dataset_from_graphs(
            schema_graph=schema_graph,
            data_graph=data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            target_classes=target_classes,
        )
    dataset_build_elapsed_ms = (perf_counter() - dataset_t0) * 1000.0
    hierarchy_elapsed_ms = 0.0
    atomic_domain_range_elapsed_ms = 0.0
    horn_safe_domain_range_elapsed_ms = 0.0
    target_role_elapsed_ms = 0.0
    kgraph_build_elapsed_ms = 0.0
    if dataset.preprocessing_timings is not None:
        dataset_build_elapsed_ms = dataset.preprocessing_timings.dataset_build_elapsed_ms
        hierarchy_elapsed_ms = dataset.preprocessing_timings.hierarchy_elapsed_ms
        atomic_domain_range_elapsed_ms = dataset.preprocessing_timings.atomic_domain_range_elapsed_ms
        horn_safe_domain_range_elapsed_ms = dataset.preprocessing_timings.horn_safe_domain_range_elapsed_ms
        target_role_elapsed_ms = dataset.preprocessing_timings.target_role_elapsed_ms
        kgraph_build_elapsed_ms = dataset.preprocessing_timings.kgraph_build_elapsed_ms

    query_plan = plan_reasoning_preprocessing(
        dataset.ontology_graph,
        target_classes=target_classes,
        augment_property_domain_range=augment_property_domain_range,
    )

    reasoner = DAGReasoner(dataset.kg, device=device_to_use)
    compile_t0 = perf_counter()
    for target_term in target_classes:
        dag = compile_class_to_dag(
            dataset.ontology_graph,
            dataset.mapping,
            target_term,
            augment_property_domain_range=query_plan.augment_property_domain_range.enabled,
        )
        reasoner.add_concept(str(target_term), dag)
    dag_compile_elapsed_ms = (perf_counter() - compile_t0) * 1000.0

    eval_t0 = perf_counter()
    score_matrix = reasoner.evaluate_all().detach().cpu()
    dag_eval_elapsed_ms = (perf_counter() - eval_t0) * 1000.0
    elapsed_ms = (perf_counter() - t0) * 1000.0

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

    return EngineQueryResult(
        backend="engine",
        elapsed_ms=elapsed_ms,
        members_by_target=members_by_target,
        dataset=dataset,
        scores_by_target=scores_by_target,
        materialization_iterations=iterations,
        consistent=None,
        dataset_build_elapsed_ms=dataset_build_elapsed_ms,
        hierarchy_elapsed_ms=hierarchy_elapsed_ms,
        atomic_domain_range_elapsed_ms=atomic_domain_range_elapsed_ms,
        horn_safe_domain_range_elapsed_ms=horn_safe_domain_range_elapsed_ms,
        target_role_elapsed_ms=target_role_elapsed_ms,
        kgraph_build_elapsed_ms=kgraph_build_elapsed_ms,
        dag_compile_elapsed_ms=dag_compile_elapsed_ms,
        dag_eval_elapsed_ms=dag_eval_elapsed_ms,
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
) -> None:
    print("=== Oracle Comparison ===")
    print(f"Engine threshold: {threshold}")
    print(f"Engine time: {engine_result.elapsed_ms:.3f} ms")
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
    target_classes: Sequence[str],
    device: str = "cuda",
    threshold: float = 0.999,
    include_literals: bool = False,
    include_type_edges: bool = False,
    materialize_hierarchy: bool = True,
    materialize_supported_types: bool = False,
    query_mode: str = "query",
    bridge_supported_definitions: bool = False,
    oracle_backends: Sequence[str] = ("owlrl", "owlready2"),
    owlready2_reasoner: str = "hermit",
    show_matches: bool = False,
    show_engine_scores: bool = False,
    max_diff_items: int = 20,
) -> None:
    schema_graph = load_rdflib_graph(schema_paths)
    data_graph = load_rdflib_graph(data_paths)
    target_terms = [URIRef(target_class) for target_class in target_classes]

    engine_result = run_engine_queries(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_classes=target_terms,
        device=device,
        threshold=threshold,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_supported_types=materialize_supported_types,
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
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compare the current ConstraintDAG reasoner against owlrl and/or "
            "owlready2 on the currently supported OWL fragment."
        )
    )
    parser.add_argument("--schema", nargs="+", required=True, help="Schema / ontology RDF files.")
    parser.add_argument("--data", nargs="+", required=True, help="Instance data RDF files.")
    parser.add_argument(
        "--target-class",
        nargs="+",
        required=True,
        help="One or more target class IRIs to compare.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--include-literals", action="store_true")
    parser.add_argument("--include-type-edges", action="store_true")
    parser.add_argument("--no-materialize-hierarchy", action="store_true")
    parser.add_argument(
        "--engine-materialize-supported-types",
        action="store_true",
        help="Enable the engine's Horn-style fixpoint pass before evaluating targets.",
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
        choices=["owlrl", "owlready2"],
        default=["owlrl", "owlready2"],
        help="Which oracle backends to run.",
    )
    parser.add_argument(
        "--owlready2-reasoner",
        choices=["hermit", "pellet"],
        default="hermit",
        help="Reasoner backend to invoke through owlready2.",
    )
    parser.add_argument("--show-matches", action="store_true")
    parser.add_argument("--show-engine-scores", action="store_true")
    parser.add_argument("--max-diff-items", type=int, default=20)

    args = parser.parse_args()

    run_oracle_comparison(
        schema_paths=args.schema,
        data_paths=args.data,
        target_classes=args.target_class,
        device=args.device,
        threshold=args.threshold,
        include_literals=args.include_literals,
        include_type_edges=args.include_type_edges,
        materialize_hierarchy=not args.no_materialize_hierarchy,
        materialize_supported_types=args.engine_materialize_supported_types,
        query_mode=args.query_mode,
        bridge_supported_definitions=(
            args.oracle_bridge_supported_definitions or args.engine_materialize_supported_types
        ),
        oracle_backends=args.oracles,
        owlready2_reasoner=args.owlready2_reasoner,
        show_matches=args.show_matches,
        show_engine_scores=args.show_engine_scores,
        max_diff_items=args.max_diff_items,
    )


if __name__ == "__main__":
    main()
