from __future__ import annotations

from typing import Dict, Iterable, List, Set

from rdflib import URIRef
from rdflib.term import Identifier

from .ontology_parse import describe_preprocessing_plan, load_rdflib_graph
from .oracle_compare import _render_term, run_engine_queries


def _format_members_by_target(
    target_classes: List[URIRef],
    members_by_target: Dict[URIRef, Set[Identifier]],
) -> str:
    lines: List[str] = []
    for target_term in target_classes:
        lines.append(f"{_render_term(target_term)}:")
        members = sorted((_render_term(term) for term in members_by_target.get(target_term, set())))
        if not members:
            lines.append("  (none)")
            continue
        for member in members:
            lines.append(f"  - {member}")
    return "\n".join(lines)


def run_filtered_query_example(
    *,
    schema_paths: List[str],
    data_paths: List[str],
    target_classes: List[str],
    device: str = "cuda",
    threshold: float = 0.999,
) -> None:
    schema_graph = load_rdflib_graph(schema_paths)
    data_graph = load_rdflib_graph(data_paths)
    target_terms = [URIRef(target_class) for target_class in target_classes]

    result = run_engine_queries(
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_classes=target_terms,
        device=device,
        threshold=threshold,
        include_literals=False,
        include_type_edges=False,
        materialize_hierarchy=None,
        augment_property_domain_range=None,
        engine_mode="filtered_query",
    )
    filtered = result.filtered_query_result
    if result.dataset is None or filtered is None:
        raise RuntimeError("filtered_query did not return a dataset/result payload.")

    print("=== Filtered Query Example ===")
    print(f"Threshold: {threshold}")
    print("")
    print(describe_preprocessing_plan(result.dataset.preprocessing_plan))
    print("")
    print("Raw pointwise candidates:")
    print(_format_members_by_target(target_terms, filtered.raw_members_by_target))
    print("")
    print("After synchronous necessary-condition recheck:")
    print(_format_members_by_target(target_terms, filtered.necessary_stable_members_by_target))
    print("")
    print("Blocked by closure / stratified pass:")
    print(_format_members_by_target(target_terms, filtered.closure_blocked_members_by_target))
    print("")
    print("Final emitted candidates:")
    print(_format_members_by_target(target_terms, filtered.final_members_by_target))
    print("")
    print("Counts:")
    print(f"  - raw_candidate_assignments: {filtered.raw_candidate_count}")
    print(f"  - necessary_condition_retractions: {filtered.necessary_retraction_count}")
    print(f"  - closure_blocked_retractions: {filtered.closure_blocked_retraction_count}")
    print(f"  - final_emitted_assignments: {filtered.final_emitted_count}")
    print(f"  - necessary_fixpoint_iterations: {filtered.necessary_fixpoint_iterations}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the filtered_query mode on a hand-built example fixture."
    )
    parser.add_argument("--schema", nargs="+", required=True, help="Schema / ontology RDF files.")
    parser.add_argument("--data", nargs="+", required=True, help="Instance data RDF files.")
    parser.add_argument("--target-class", nargs="+", required=True, help="Target class IRIs.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.999)

    args = parser.parse_args()

    run_filtered_query_example(
        schema_paths=args.schema,
        data_paths=args.data,
        target_classes=args.target_class,
        device=args.device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
