from __future__ import annotations

from typing import List, Tuple

import torch

from .dag_reasoner import DAGReasoner
from .ontology_parse import (
    build_dag_dependency_report,
    ClassMaterializationResult,
    compile_class_to_dag,
    compile_sufficient_condition_dag,
    ConflictPolicy,
    describe_dag_dependency_report,
    materialize_stratified_class_inferences,
    load_rdflib_graph,
    materialize_positive_sufficient_class_inferences,
    materialize_supported_class_inferences,
    summarize_loaded_kgraph,
)


def _format_assertions(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(f"  - {node} :: {class_iri}" for node, class_iri in rows)


def _format_blockers(rows: List[Tuple[str, str, str]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(f"  - {node} :: blocked {class_iri} via {blocker_iri}" for node, class_iri, blocker_iri in rows)


def _format_assignment_statuses(rows: List[Tuple[str, str, str]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(f"  - {node} :: {class_iri} [{status}]" for node, class_iri, status in rows)


def _format_policy_assertions(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(f"  - {node} :: {class_iri}" for node, class_iri in rows)


def run_rule_materialization_example(
    *,
    schema_paths: List[str],
    data_paths: List[str],
    device: str = "cuda",
    threshold: float = 0.999,
    include_literals: bool = False,
    materialize_hierarchy: bool = True,
    materialize_target_roles: bool = False,
    target_class: str | None = None,
    show_dependencies: bool = False,
    inference_mode: str = "definitional",
    conflict_policy: str = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
) -> None:
    schema_graph = load_rdflib_graph(schema_paths)
    data_graph = load_rdflib_graph(data_paths)

    device_to_use = device
    if device_to_use == "cuda" and not torch.cuda.is_available():
        device_to_use = "cpu"

    negative_result = None
    assignment_statuses = None
    if inference_mode == "stratified":
        stratified = materialize_stratified_class_inferences(
            schema_graph=schema_graph,
            data_graph=data_graph,
            include_literals=include_literals,
            materialize_hierarchy=materialize_hierarchy,
            materialize_target_roles=materialize_target_roles,
            target_classes=[target_class] if target_class is not None else None,
            threshold=threshold,
            device=device_to_use,
            conflict_policy=ConflictPolicy(conflict_policy),
        )
        result = stratified.positive_result
        negative_result = stratified.negative_result
        assignment_statuses = stratified.assignment_statuses
        policy_result = stratified.policy_result
    elif inference_mode == "sufficient":
        policy_result = None
        result = materialize_positive_sufficient_class_inferences(
            schema_graph=schema_graph,
            data_graph=data_graph,
            include_literals=include_literals,
            materialize_hierarchy=materialize_hierarchy,
            materialize_target_roles=materialize_target_roles,
            target_classes=[target_class] if target_class is not None else None,
            threshold=threshold,
            device=device_to_use,
        )
    else:
        policy_result = None
        result = materialize_supported_class_inferences(
            schema_graph=schema_graph,
            data_graph=data_graph,
            include_literals=include_literals,
            materialize_hierarchy=materialize_hierarchy,
            materialize_target_roles=materialize_target_roles,
            target_classes=[target_class] if target_class is not None else None,
            threshold=threshold,
            device=device_to_use,
        )

    assertions = [
        (
            node_term.n3() if hasattr(node_term, "n3") else str(node_term),
            class_term.n3() if hasattr(class_term, "n3") else str(class_term),
        )
        for node_term, class_term in result.inferred_assertions
    ]
    assertions.sort()

    print("=== Rule Materialization Example ===")
    print(f"Using device: {device_to_use}")
    print(f"Inference mode: {inference_mode}")
    if policy_result is not None:
        print(f"Conflict policy: {policy_result.policy.value}")
    print(f"Iterations: {result.iterations}")
    print("")
    print(summarize_loaded_kgraph(result.dataset.kg, result.dataset.mapping, max_items=10))
    print("")
    print("Inferred rdf:type assertions:")
    print(_format_assertions(assertions))
    if negative_result is not None:
        blocker_rows = [
            (
                blocked.node_term.n3() if hasattr(blocked.node_term, "n3") else str(blocked.node_term),
                blocked.target_class.n3() if hasattr(blocked.target_class, "n3") else str(blocked.target_class),
                blocked.blocker_class.n3() if hasattr(blocked.blocker_class, "n3") else str(blocked.blocker_class),
            )
            for blocked in negative_result.blocked_assertions
        ]
        blocker_rows.sort()
        conflict_rows = [
            (
                blocked.node_term.n3() if hasattr(blocked.node_term, "n3") else str(blocked.node_term),
                blocked.target_class.n3() if hasattr(blocked.target_class, "n3") else str(blocked.target_class),
                blocked.blocker_class.n3() if hasattr(blocked.blocker_class, "n3") else str(blocked.blocker_class),
            )
            for blocked in negative_result.conflicting_positive_assertions
        ]
        conflict_rows.sort()
        print("")
        print("Blocked class assignments:")
        print(_format_blockers(blocker_rows))
        print("")
        print("Conflicts with positive closure:")
        print(_format_blockers(conflict_rows))
        status_rows = []
        for status in assignment_statuses or []:
            bits = []
            if status.asserted:
                bits.append("asserted")
            if status.positively_derived:
                bits.append("positively-derived")
            if status.blocked:
                bits.append("blocked")
            if status.conflicted:
                bits.append("conflicted")
            if status.blocker_classes:
                bits.append(
                    "via="
                    + ",".join(
                        blocker.n3() if hasattr(blocker, "n3") else str(blocker)
                        for blocker in status.blocker_classes
                    )
                )
            status_rows.append(
                (
                    status.node_term.n3() if hasattr(status.node_term, "n3") else str(status.node_term),
                    status.target_class.n3() if hasattr(status.target_class, "n3") else str(status.target_class),
                    ", ".join(bits) if bits else "unmarked",
                )
            )
        status_rows.sort()
        print("")
        print("Assignment statuses:")
        print(_format_assignment_statuses(status_rows))
        emitted_rows = [
            (
                node_term.n3() if hasattr(node_term, "n3") else str(node_term),
                class_term.n3() if hasattr(class_term, "n3") else str(class_term),
            )
            for node_term, class_term in policy_result.emitted_derived_assertions
        ]
        suppressed_rows = [
            (
                status.node_term.n3() if hasattr(status.node_term, "n3") else str(status.node_term),
                status.target_class.n3() if hasattr(status.target_class, "n3") else str(status.target_class),
            )
            for status in policy_result.suppressed_derived_assignments
        ]
        asserted_conflict_rows = [
            (
                status.node_term.n3() if hasattr(status.node_term, "n3") else str(status.node_term),
                status.target_class.n3() if hasattr(status.target_class, "n3") else str(status.target_class),
            )
            for status in policy_result.asserted_conflicts
        ]
        print("")
        print("Policy-emitted derived assertions:")
        print(_format_policy_assertions(sorted(emitted_rows)))
        print("")
        print("Policy-suppressed derived assertions:")
        print(_format_policy_assertions(sorted(suppressed_rows)))
        print("")
        print("Asserted conflicts:")
        print(_format_policy_assertions(sorted(asserted_conflict_rows)))
        if policy_result.failed:
            print("")
            print(f"Policy failure: {policy_result.failure_reason}")

    if target_class is not None:
        target_term = target_class
        if inference_mode in {"sufficient", "stratified"}:
            dag = compile_sufficient_condition_dag(
                result.dataset.ontology_graph,
                result.dataset.mapping,
                target_class,
            )
        else:
            dag = compile_class_to_dag(result.dataset.ontology_graph, result.dataset.mapping, target_class)
        dependency_report = (
            build_dag_dependency_report(result.dataset.ontology_graph, result.dataset.mapping, target_class)
            if show_dependencies and inference_mode == "definitional"
            else None
        )
        reasoner = DAGReasoner(result.dataset.kg, device=device_to_use)
        reasoner.add_concept(target_class, dag)
        scores = reasoner.evaluate_all()[:, 0].detach().cpu()

        print("")
        if dependency_report is not None:
            print("Dependency report:")
            print(describe_dag_dependency_report(dependency_report))
            print("")
        elif show_dependencies and inference_mode != "definitional":
            print("Dependency report:")
            print("  (currently only available for the definitional / necessary-condition compiler)")
            print("")
        print(f"Final matches for {target_class}:")
        blocked_node_terms = set()
        emitted_node_terms = None
        if negative_result is not None:
            blocked_node_terms = {
                blocked.node_term
                for blocked in negative_result.blocked_assertions
                if str(blocked.target_class) == str(target_term)
            }
            emitted_node_terms = {
                status.node_term
                for status in (policy_result.emitted_assignments if policy_result is not None else [])
                if str(status.target_class) == str(target_term)
            }
        printed = False
        for idx, node_term in enumerate(result.dataset.mapping.node_terms):
            score = float(scores[idx].item())
            if score >= threshold:
                rendered = node_term.n3() if hasattr(node_term, "n3") else str(node_term)
                suffix_parts = []
                if node_term in blocked_node_terms:
                    suffix_parts.append("blocked")
                if emitted_node_terms is not None and node_term not in emitted_node_terms:
                    suffix_parts.append("not-emitted")
                suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
                print(f"  - {rendered}: {score:.4f}{suffix}")
                printed = True
        if not printed:
            print("  (none)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end Horn-style type materialization over the supported OWL fragment."
    )
    parser.add_argument("--schema", nargs="+", required=True, help="Schema / ontology RDF files.")
    parser.add_argument("--data", nargs="+", required=True, help="Instance data RDF files.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--include-literals", action="store_true")
    parser.add_argument("--no-materialize-hierarchy", action="store_true")
    parser.add_argument(
        "--materialize-target-roles",
        action="store_true",
        help=(
            "Enable targeted role saturation for the target class dependency "
            "closure during the Horn-style materialization pass."
        ),
    )
    parser.add_argument(
        "--target-class",
        default=None,
        help=(
            "Optional class IRI to print final matches for after rule "
            "materialization. When provided, helper materialization is "
            "restricted to the target's relevance closure."
        ),
    )
    parser.add_argument(
        "--show-dependencies",
        action="store_true",
        help="Print the DAG-node-level dependency closure report for the target class.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=["definitional", "sufficient", "stratified"],
        default="definitional",
        help=(
            "Choose between the older definitional fixpoint materializer, the "
            "positive OWA sufficient-condition materializer, and the initial "
            "stratified positive+negative blocker pass."
        ),
    )
    parser.add_argument(
        "--conflict-policy",
        choices=[policy.value for policy in ConflictPolicy],
        default=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
        help=(
            "Conflict-handling policy for stratified mode. Ignored for other inference modes."
        ),
    )

    args = parser.parse_args()

    run_rule_materialization_example(
        schema_paths=args.schema,
        data_paths=args.data,
        device=args.device,
        threshold=args.threshold,
        include_literals=args.include_literals,
        materialize_hierarchy=not args.no_materialize_hierarchy,
        materialize_target_roles=args.materialize_target_roles,
        target_class=args.target_class,
        show_dependencies=args.show_dependencies,
        inference_mode=args.inference_mode,
        conflict_policy=args.conflict_policy,
    )


if __name__ == "__main__":
    main()
