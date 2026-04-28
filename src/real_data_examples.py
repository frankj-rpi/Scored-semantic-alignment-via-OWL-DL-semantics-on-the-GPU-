from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from rdflib import URIRef

from .dag_reasoner import DAGReasoner
from .ontology_parse import (
    build_dag_dependency_report,
    compile_class_to_dag,
    describe_preprocessing_plan,
    describe_dag_dependency_report,
    describe_constraint_dag,
    load_reasoning_dataset,
    plan_reasoning_preprocessing,
    summarize_loaded_kgraph,
)


def _format_matches(rows: List[Tuple[str, float]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(f"  - {term}: {score:.4f}" for term, score in rows)


def run_reasoning_example(
    *,
    schema_paths: List[str],
    data_paths: List[str],
    target_class: str,
    device: str = "cuda",
    include_literals: bool = False,
    include_type_edges: bool = False,
    materialize_hierarchy: Optional[bool] = None,
    materialize_atomic_domain_range: Optional[bool] = None,
    materialize_horn_safe_domain_range: Optional[bool] = None,
    materialize_reflexive_properties: Optional[bool] = None,
    materialize_target_roles: Optional[bool] = None,
    augment_property_domain_range: Optional[bool] = None,
    threshold: float = 0.999,
    show_nonzero: bool = False,
    show_dependencies: bool = False,
) -> None:
    dataset = load_reasoning_dataset(
        schema_paths=schema_paths,
        data_paths=data_paths,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_atomic_domain_range=materialize_atomic_domain_range,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_reflexive_properties=materialize_reflexive_properties,
        materialize_target_roles=materialize_target_roles,
        target_classes=[target_class],
    )
    query_plan = plan_reasoning_preprocessing(
        dataset.ontology_graph,
        target_classes=[target_class],
        materialize_hierarchy=materialize_hierarchy,
        materialize_atomic_domain_range=materialize_atomic_domain_range,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_reflexive_properties=materialize_reflexive_properties,
        materialize_target_roles=materialize_target_roles,
        augment_property_domain_range=augment_property_domain_range,
    )

    dag = compile_class_to_dag(
        dataset.ontology_graph,
        dataset.mapping,
        target_class,
        augment_property_domain_range=query_plan.augment_property_domain_range.enabled,
    )
    dependency_report = (
        build_dag_dependency_report(
            dataset.ontology_graph,
            dataset.mapping,
            target_class,
            augment_property_domain_range=query_plan.augment_property_domain_range.enabled,
        )
        if show_dependencies
        else None
    )

    device_to_use = device
    if device_to_use == "cuda" and not torch.cuda.is_available():
        device_to_use = "cpu"

    reasoner = DAGReasoner(dataset.kg, device=device_to_use)
    concept_name = URIRef(target_class).n3()
    reasoner.add_concept(concept_name, dag)
    scores = reasoner.evaluate_all()[:, 0].detach().cpu()

    matching_rows: List[Tuple[str, float]] = []
    nonzero_rows: List[Tuple[str, float]] = []
    for idx, term in enumerate(dataset.mapping.node_terms):
        score = float(scores[idx].item())
        if score <= 0.0:
            continue

        rendered = term.n3() if hasattr(term, "n3") else str(term)
        nonzero_rows.append((rendered, score))
        if score >= threshold:
            matching_rows.append((rendered, score))

    nonzero_rows.sort(key=lambda row: (-row[1], row[0]))
    matching_rows.sort(key=lambda row: row[0])

    print("=== Real-Data Reasoning Example ===")
    print(f"Target class: {target_class}")
    print(f"Using device: {device_to_use}")
    print("")
    print(describe_preprocessing_plan(dataset.preprocessing_plan))
    print(
        f"Query augmentation: {'on' if query_plan.augment_property_domain_range.enabled else 'off'} "
        f"(policy={query_plan.augment_property_domain_range.policy}; "
        f"{query_plan.augment_property_domain_range.reason})"
    )
    print("")
    print(summarize_loaded_kgraph(dataset.kg, dataset.mapping, max_items=10))
    print("")
    print("Compiled DAG:")
    print(describe_constraint_dag(dag, dataset.mapping))
    if dependency_report is not None:
        print("")
        print("Dependency report:")
        print(describe_dag_dependency_report(dependency_report))
    print("")
    print(f"Matches with score >= {threshold}:")
    print(_format_matches(matching_rows))

    if show_nonzero:
        print("")
        print("All non-zero instance scores:")
        print(_format_matches(nonzero_rows))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end example: schema/data load -> class compile -> DAG evaluation."
    )
    parser.add_argument(
        "--schema",
        nargs="+",
        required=True,
        help="Schema / ontology RDF files.",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Instance data RDF files.",
    )
    parser.add_argument(
        "--target-class",
        required=True,
        help="Class IRI to compile and evaluate.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Execution device for DAG evaluation.",
    )
    parser.add_argument(
        "--include-literals",
        action="store_true",
        help="Lift literals into the KGraph instead of dropping them from instance edges.",
    )
    parser.add_argument(
        "--include-type-edges",
        action="store_true",
        help="Also materialize rdf:type as a property edge.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.999,
        help="Score threshold for reporting matches.",
    )
    hierarchy_group = parser.add_mutually_exclusive_group()
    hierarchy_group.add_argument("--materialize-hierarchy", dest="materialize_hierarchy", action="store_true")
    hierarchy_group.add_argument(
        "--no-materialize-hierarchy",
        dest="materialize_hierarchy",
        action="store_false",
        help="Disable subclass/subproperty materialization into the instance graph.",
    )
    roles_group = parser.add_mutually_exclusive_group()
    roles_group.add_argument(
        "--materialize-target-roles",
        dest="materialize_target_roles",
        action="store_true",
        help=(
            "Enable targeted role saturation for the queried class: subProperty, "
            "inverseOf, property chains, and transitive properties only within "
            "the relevant role closure of the target DAG."
        ),
    )
    roles_group.add_argument(
        "--no-materialize-target-roles",
        dest="materialize_target_roles",
        action="store_false",
        help="Disable targeted role saturation explicitly.",
    )
    atomic_domain_group = parser.add_mutually_exclusive_group()
    atomic_domain_group.add_argument(
        "--materialize-domain-range-atomic",
        dest="materialize_domain_range_atomic",
        action="store_true",
        help=(
            "Materialize only Horn-safe atomic rdfs:domain / rdfs:range type "
            "consequences into the instance graph before evaluation."
        ),
    )
    atomic_domain_group.add_argument(
        "--no-materialize-domain-range-atomic",
        dest="materialize_domain_range_atomic",
        action="store_false",
        help="Disable atomic-only domain/range materialization explicitly.",
    )
    horn_domain_group = parser.add_mutually_exclusive_group()
    horn_domain_group.add_argument(
        "--materialize-domain-range-horn",
        dest="materialize_domain_range_horn",
        action="store_true",
        help=(
            "Materialize Horn-safe rdfs:domain / rdfs:range type consequences "
            "(atomic classes and intersections of atomic classes) into the instance graph."
        ),
    )
    horn_domain_group.add_argument(
        "--no-materialize-domain-range-horn",
        dest="materialize_domain_range_horn",
        action="store_false",
        help="Disable Horn-safe domain/range materialization explicitly.",
    )
    reflexive_group = parser.add_mutually_exclusive_group()
    reflexive_group.add_argument(
        "--materialize-reflexive-properties",
        dest="materialize_reflexive_properties",
        action="store_true",
        help="Materialize self-edges for owl:ReflexiveProperty terms before evaluation.",
    )
    reflexive_group.add_argument(
        "--no-materialize-reflexive-properties",
        dest="materialize_reflexive_properties",
        action="store_false",
        help="Disable reflexive-property materialization explicitly.",
    )
    augment_group = parser.add_mutually_exclusive_group()
    augment_group.add_argument(
        "--augment-domain-range",
        dest="augment_domain_range",
        action="store_true",
        help=(
            "Augment the queried class DAG with query-time rdfs:domain / "
            "rdfs:range witness and universal-consistency branches."
        ),
    )
    augment_group.add_argument(
        "--no-augment-domain-range",
        dest="augment_domain_range",
        action="store_false",
        help="Disable query-time domain/range augmentation explicitly.",
    )
    parser.set_defaults(
        materialize_hierarchy=None,
        materialize_target_roles=None,
        materialize_domain_range_atomic=None,
        materialize_domain_range_horn=None,
        materialize_reflexive_properties=None,
        augment_domain_range=None,
    )
    parser.add_argument(
        "--show-nonzero",
        action="store_true",
        help="Print all non-zero scores, not just threshold matches.",
    )
    parser.add_argument(
        "--show-dependencies",
        action="store_true",
        help="Print the DAG-node-level dependency closure report for the target class.",
    )

    args = parser.parse_args()

    run_reasoning_example(
        schema_paths=args.schema,
        data_paths=args.data,
        target_class=args.target_class,
        device=args.device,
        include_literals=args.include_literals,
        include_type_edges=args.include_type_edges,
        materialize_hierarchy=args.materialize_hierarchy,
        materialize_atomic_domain_range=args.materialize_domain_range_atomic,
        materialize_horn_safe_domain_range=args.materialize_domain_range_horn,
        materialize_reflexive_properties=args.materialize_reflexive_properties,
        materialize_target_roles=args.materialize_target_roles,
        augment_property_domain_range=args.augment_domain_range,
        threshold=args.threshold,
        show_nonzero=args.show_nonzero,
        show_dependencies=args.show_dependencies,
    )


if __name__ == "__main__":
    main()
