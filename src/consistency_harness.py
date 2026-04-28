from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD
from rdflib.term import Identifier

from .constraints import ConstraintDAG, ConstraintType
from .explanations import explain_dataset_query
from .ontology_parse import (
    ConflictPolicy,
    compile_class_to_dag,
    compile_sufficient_condition_dag,
    describe_preprocessing_plan,
    plan_reasoning_preprocessing,
)
from .oracle_compare import _render_term, run_engine_queries


DEFAULT_CONSTRUCTS: Tuple[str, ...] = (
    "subclass",
    "intersection",
    "union",
    "exists",
    "forall",
    "datatype",
    "nominal",
    "geq-cardinality",
    "disjoint",
    "domain",
    "range",
)
CONSTRUCT_PROFILES: Dict[str, Tuple[str, ...]] = {
    "OWL-EL": (
        "subclass",
        "intersection",
        "exists",
        "datatype",
        "has-value",
        "data-oneof",
        "nominal",
        "has-self",
        "reflexive",
        "disjoint",
        "domain",
        "range",
        "functional-data-property",
        "negative-object-property",
        "negative-data-property",
        "has-key",
    ),
    "OWL_EL": (
        "subclass",
        "intersection",
        "exists",
        "datatype",
        "has-value",
        "data-oneof",
        "nominal",
        "has-self",
        "reflexive",
        "disjoint",
        "domain",
        "range",
        "functional-data-property",
        "negative-object-property",
        "negative-data-property",
        "has-key",
    ),
    "EL++": (
        "subclass",
        "intersection",
        "exists",
        "datatype",
        "has-value",
        "data-oneof",
        "nominal",
        "has-self",
        "reflexive",
        "disjoint",
        "domain",
        "range",
        "functional-data-property",
        "negative-object-property",
        "negative-data-property",
        "has-key",
    ),
}
HARNESS_META = Namespace("urn:dag-consistency-meta:")


@dataclass(frozen=True)
class FragmentGeneratorConfig:
    num_primitive_classes: int = 5
    num_target_classes: int = 4
    num_properties: int = 3
    num_datatype_properties: int = 2
    num_individuals: int = 8
    min_axioms_per_target: int = 1
    max_axioms_per_target: int = 3
    type_probability: float = 0.35
    edge_probability: float = 0.18
    datatype_edge_probability: float = 0.35
    disjoint_pair_probability: float = 0.10
    domain_range_axiom_probability: float = 0.35
    allowed_constructs: Tuple[str, ...] = DEFAULT_CONSTRUCTS


@dataclass
class GeneratedFragmentCase:
    seed: int
    schema_graph: Graph
    data_graph: Graph
    target_classes: List[URIRef]
    target_constructs: Dict[URIRef, Tuple[str, ...]]
    individuals: List[URIRef]


@dataclass
class ConsistencyFailure:
    seed: int
    target_class: URIRef
    node_term: Identifier
    score: float
    constructs: Tuple[str, ...]
    reasoner: str
    error: Optional[str] = None


@dataclass
class BucketStats:
    tested: int = 0
    failures: int = 0
    examples: List[ConsistencyFailure] = None

    def __post_init__(self) -> None:
        if self.examples is None:
            self.examples = []


@dataclass
class HarnessSummary:
    requested_cases: int
    generated_cases: int
    attempts: int
    base_consistent_cases: int
    total_perfect_scores: int
    total_checked_assertions: int
    total_failures: int
    bucket_stats: Dict[Tuple[str, ...], BucketStats]
    generation_elapsed_ms: float = 0.0
    preprocessing_elapsed_ms: float = 0.0
    dataset_build_elapsed_ms: float = 0.0
    hierarchy_elapsed_ms: float = 0.0
    atomic_domain_range_elapsed_ms: float = 0.0
    horn_safe_domain_range_elapsed_ms: float = 0.0
    sameas_elapsed_ms: float = 0.0
    reflexive_elapsed_ms: float = 0.0
    target_role_elapsed_ms: float = 0.0
    kgraph_build_elapsed_ms: float = 0.0
    dag_compile_elapsed_ms: float = 0.0
    dag_eval_elapsed_ms: float = 0.0
    base_consistency_check_elapsed_ms: float = 0.0
    assertion_consistency_check_elapsed_ms: float = 0.0
    engine_mode: str = "query"
    conflict_policy: Optional[str] = None
    raw_positive_assignment_count: int = 0
    policy_emitted_assignment_count: int = 0
    policy_suppressed_assignment_count: int = 0
    raw_candidate_assignment_count: int = 0
    necessary_condition_retraction_count: int = 0
    closure_blocked_retraction_count: int = 0
    final_emitted_assignment_count: int = 0
    run_settings: Optional[Dict[str, object]] = None
    case_preprocessing_summaries: Optional[Dict[str, str]] = None
    save_root: Optional[str] = None


def expand_construct_specs(construct_specs: Sequence[str]) -> Tuple[str, ...]:
    expanded: List[str] = []
    seen: Set[str] = set()
    for spec in construct_specs:
        profile_members = CONSTRUCT_PROFILES.get(spec)
        if profile_members is not None:
            for member in profile_members:
                if member not in seen:
                    expanded.append(member)
                    seen.add(member)
            continue
        if spec not in seen:
            expanded.append(spec)
            seen.add(spec)
    return tuple(expanded)


def _copy_graph(graph: Graph) -> Graph:
    copied = Graph()
    for triple in graph:
        copied.add(triple)
    return copied


def _merge_graphs(*graphs: Graph) -> Graph:
    merged = Graph()
    for graph in graphs:
        for triple in graph:
            merged.add(triple)
    return merged


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_bucket_key(bucket_key: Tuple[str, ...]) -> str:
    return "-".join(bucket_key).replace(" ", "_")


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _inject_case_metadata(
    case: GeneratedFragmentCase,
    *,
    config: FragmentGeneratorConfig,
    generated_at: str,
    threshold: float,
    materialize_hierarchy: bool,
    augment_property_domain_range: bool,
    base_consistent: bool,
) -> Graph:
    schema_graph = _copy_graph(case.schema_graph)
    ontology_term = URIRef(f"urn:dag-consistency-case:{case.seed}:ontology")
    schema_graph.bind("meta", HARNESS_META)
    schema_graph.add((ontology_term, RDF.type, OWL.Ontology))
    schema_graph.add((ontology_term, HARNESS_META.seed, Literal(case.seed)))
    schema_graph.add((ontology_term, HARNESS_META.generatedAt, Literal(generated_at, datatype=XSD.dateTime)))
    schema_graph.add((ontology_term, HARNESS_META.baseConsistent, Literal(base_consistent)))
    schema_graph.add((ontology_term, HARNESS_META.threshold, Literal(threshold)))
    schema_graph.add((ontology_term, HARNESS_META.materializeHierarchy, Literal(materialize_hierarchy)))
    schema_graph.add((ontology_term, HARNESS_META.augmentDomainRange, Literal(augment_property_domain_range)))
    schema_graph.add(
        (
            ontology_term,
            HARNESS_META.generatorConfigJson,
            Literal(json.dumps(config.__dict__, sort_keys=True)),
        )
    )
    for target_class in case.target_classes:
        schema_graph.add((ontology_term, HARNESS_META.targetClass, target_class))
        schema_graph.add(
            (
                target_class,
                HARNESS_META.generatedConstructs,
                Literal(json.dumps(list(case.target_constructs.get(target_class, ())), sort_keys=True)),
            )
        )
    return schema_graph


def _save_case_graphs(
    case: GeneratedFragmentCase,
    *,
    config: FragmentGeneratorConfig,
    save_root: str,
    threshold: float,
    materialize_hierarchy: bool,
    augment_property_domain_range: bool,
    engine_mode: str,
    conflict_policy: Optional[str],
    base_consistent: bool,
) -> str:
    case_dir = os.path.join(save_root, f"seed-{case.seed:06d}")
    _ensure_dir(case_dir)
    generated_at = _now_utc_iso()
    schema_with_meta = _inject_case_metadata(
        case,
        config=config,
        generated_at=generated_at,
        threshold=threshold,
        materialize_hierarchy=materialize_hierarchy,
        augment_property_domain_range=augment_property_domain_range,
        base_consistent=base_consistent,
    )
    schema_with_meta.serialize(destination=os.path.join(case_dir, "schema.ttl"), format="turtle")
    case.data_graph.serialize(destination=os.path.join(case_dir, "data.ttl"), format="turtle")

    manifest = {
        "seed": case.seed,
        "generated_at": generated_at,
        "threshold": threshold,
        "materialize_hierarchy": materialize_hierarchy,
        "augment_property_domain_range": augment_property_domain_range,
        "engine_mode": engine_mode,
        "conflict_policy": conflict_policy,
        "base_consistent": base_consistent,
        "target_classes": [str(term) for term in case.target_classes],
        "target_constructs": {str(k): list(v) for k, v in case.target_constructs.items()},
        "config": dict(config.__dict__),
    }
    with open(os.path.join(case_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return case_dir


def _write_run_summary(summary: HarnessSummary) -> None:
    if not summary.save_root:
        return
    payload = {
        "requested_cases": summary.requested_cases,
        "generated_cases": summary.generated_cases,
        "attempts": summary.attempts,
        "base_consistent_cases": summary.base_consistent_cases,
        "total_perfect_scores": summary.total_perfect_scores,
        "total_checked_assertions": summary.total_checked_assertions,
        "total_failures": summary.total_failures,
        "engine": {
            "mode": summary.engine_mode,
            "conflict_policy": summary.conflict_policy,
        },
        "policy_counts": {
            "raw_positive_assignments": summary.raw_positive_assignment_count,
            "policy_emitted_assignments": summary.policy_emitted_assignment_count,
            "policy_suppressed_assignments": summary.policy_suppressed_assignment_count,
        },
        "filtered_query_counts": {
            "raw_candidate_assignments": summary.raw_candidate_assignment_count,
            "necessary_condition_retractions": summary.necessary_condition_retraction_count,
            "closure_blocked_retractions": summary.closure_blocked_retraction_count,
            "final_emitted_assignments": summary.final_emitted_assignment_count,
        },
        "run_settings": summary.run_settings,
        "case_preprocessing_summaries": summary.case_preprocessing_summaries,
        "timings_ms": {
            "generation": summary.generation_elapsed_ms,
            "preprocessing_total": summary.preprocessing_elapsed_ms,
            "dataset_build": summary.dataset_build_elapsed_ms,
            "hierarchy_materialization": summary.hierarchy_elapsed_ms,
            "atomic_domain_range_materialization": summary.atomic_domain_range_elapsed_ms,
            "horn_safe_domain_range_materialization": summary.horn_safe_domain_range_elapsed_ms,
            "sameas_materialization": summary.sameas_elapsed_ms,
            "reflexive_property_materialization": summary.reflexive_elapsed_ms,
            "target_role_materialization": summary.target_role_elapsed_ms,
            "kgraph_build": summary.kgraph_build_elapsed_ms,
            "dag_compile": summary.dag_compile_elapsed_ms,
            "dag_evaluation": summary.dag_eval_elapsed_ms,
            "base_consistency_check": summary.base_consistency_check_elapsed_ms,
            "assertion_consistency_check": summary.assertion_consistency_check_elapsed_ms,
            "error_checking_total": (
                summary.base_consistency_check_elapsed_ms
                + summary.assertion_consistency_check_elapsed_ms
            ),
        },
        "buckets": {
            _format_bucket_key(bucket_key): {
                "tested": bucket.tested,
                "failures": bucket.failures,
                "examples": [
                    {
                        "seed": example.seed,
                        "node_term": _render_term(example.node_term),
                        "target_class": str(example.target_class),
                        "score": example.score,
                        "constructs": list(example.constructs),
                        "reasoner": example.reasoner,
                        "error": example.error,
                    }
                    for example in bucket.examples
                ],
            }
            for bucket_key, bucket in summary.bucket_stats.items()
        },
    }
    with open(os.path.join(summary.save_root, "run-summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with open(os.path.join(summary.save_root, "run-summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(format_harness_summary(summary))


def _list_expression(graph: Graph, predicate: URIRef, members: Sequence[Identifier]) -> Identifier:
    expr = BNode()
    head = BNode()
    graph.add((expr, RDF.type, OWL.Class))
    graph.add((expr, predicate, head))
    Collection(graph, head, list(members))
    return expr


def _restriction_expression(
    graph: Graph,
    prop: URIRef,
    filler: Identifier,
    *,
    universal: bool,
) -> Identifier:
    expr = BNode()
    graph.add((expr, RDF.type, OWL.Restriction))
    graph.add((expr, OWL.onProperty, prop))
    graph.add((expr, OWL.allValuesFrom if universal else OWL.someValuesFrom, filler))
    return expr


def _has_value_restriction_expression(
    graph: Graph,
    *,
    prop: URIRef,
    value: Identifier,
) -> Identifier:
    expr = BNode()
    graph.add((expr, RDF.type, OWL.Restriction))
    graph.add((expr, OWL.onProperty, prop))
    graph.add((expr, OWL.hasValue, value))
    return expr


def _has_self_restriction_expression(
    graph: Graph,
    *,
    prop: URIRef,
) -> Identifier:
    expr = BNode()
    graph.add((expr, RDF.type, OWL.Restriction))
    graph.add((expr, OWL.onProperty, prop))
    graph.add((expr, OWL.hasSelf, Literal(True, datatype=XSD.boolean)))
    return expr


def _min_qualified_cardinality_expression(
    graph: Graph,
    *,
    prop: URIRef,
    filler: Identifier,
    target: int,
) -> Identifier:
    expr = BNode()
    graph.add((expr, RDF.type, OWL.Restriction))
    graph.add((expr, OWL.onProperty, prop))
    graph.add((expr, OWL.minQualifiedCardinality, Literal(target, datatype=XSD.nonNegativeInteger)))
    graph.add((expr, OWL.onClass, filler))
    return expr


def _datatype_restriction_expression(
    graph: Graph,
    *,
    datatype: URIRef,
    min_inclusive: Optional[int] = None,
    max_inclusive: Optional[int] = None,
) -> Identifier:
    expr = BNode()
    graph.add((expr, RDF.type, RDFS.Datatype))
    graph.add((expr, OWL.onDatatype, datatype))
    restrictions: List[Identifier] = []
    if min_inclusive is not None:
        facet = BNode()
        graph.add((facet, XSD.minInclusive, Literal(min_inclusive, datatype=datatype)))
        restrictions.append(facet)
    if max_inclusive is not None:
        facet = BNode()
        graph.add((facet, XSD.maxInclusive, Literal(max_inclusive, datatype=datatype)))
        restrictions.append(facet)
    if restrictions:
        head = BNode()
        graph.add((expr, OWL.withRestrictions, head))
        Collection(graph, head, restrictions)
    return expr


def _nominal_expression(
    graph: Graph,
    members: Sequence[Identifier],
) -> Identifier:
    expr = BNode()
    head = BNode()
    graph.add((expr, RDF.type, OWL.Class))
    graph.add((expr, OWL.oneOf, head))
    Collection(graph, head, list(members))
    return expr


def _datatype_oneof_expression(
    graph: Graph,
    members: Sequence[Literal],
) -> Identifier:
    expr = BNode()
    head = BNode()
    graph.add((expr, RDF.type, RDFS.Datatype))
    graph.add((expr, OWL.oneOf, head))
    Collection(graph, head, list(members))
    return expr


def _has_key_axiom(
    graph: Graph,
    *,
    class_term: URIRef,
    props: Sequence[URIRef],
) -> None:
    head = BNode()
    graph.add((class_term, OWL.hasKey, head))
    Collection(graph, head, list(props))


def _negative_property_assertion(
    graph: Graph,
    *,
    source: Identifier,
    prop: URIRef,
    target: Identifier,
) -> None:
    assertion = BNode()
    graph.add((assertion, RDF.type, OWL.NegativePropertyAssertion))
    graph.add((assertion, OWL.sourceIndividual, source))
    graph.add((assertion, OWL.assertionProperty, prop))
    if isinstance(target, Literal):
        graph.add((assertion, OWL.targetValue, target))
    else:
        graph.add((assertion, OWL.targetIndividual, target))


def _choose_named_class(rng: random.Random, choices: Sequence[URIRef], *, exclude: Optional[URIRef] = None) -> URIRef:
    eligible = [term for term in choices if term != exclude]
    if not eligible:
        raise ValueError("No eligible named class choices remain.")
    return rng.choice(eligible)


def _build_boolean_filler(
    graph: Graph,
    rng: random.Random,
    *,
    available_named_classes: Sequence[URIRef],
    allowed_constructs: Set[str],
) -> Tuple[Identifier, Set[str]]:
    if not available_named_classes:
        raise ValueError("At least one available named class is required.")

    options = ["named"]
    if "intersection" in allowed_constructs and len(available_named_classes) >= 2:
        options.append("intersection")
    if "union" in allowed_constructs and len(available_named_classes) >= 2:
        options.append("union")

    choice = rng.choice(options)
    if choice == "named":
        return _choose_named_class(rng, available_named_classes), {"atomic"}

    members = rng.sample(list(available_named_classes), k=2)
    if choice == "intersection":
        return _list_expression(graph, OWL.intersectionOf, members), {"atomic", "intersection"}
    return _list_expression(graph, OWL.unionOf, members), {"atomic", "union"}


def _build_named_boolean_expression(
    graph: Graph,
    rng: random.Random,
    *,
    available_named_classes: Sequence[URIRef],
    predicate: URIRef,
) -> Tuple[Identifier, Set[str]]:
    members = rng.sample(list(available_named_classes), k=2)
    if predicate == OWL.intersectionOf:
        return _list_expression(graph, predicate, members), {"atomic", "intersection"}
    return _list_expression(graph, predicate, members), {"atomic", "union"}


def _build_horn_safe_domain_range_expression(
    graph: Graph,
    rng: random.Random,
    *,
    anchor_class: URIRef,
    available_named_classes: Sequence[URIRef],
    allowed_constructs: Set[str],
) -> Tuple[Identifier, Set[str]]:
    if "intersection" in allowed_constructs and len(available_named_classes) >= 2 and rng.random() < 0.5:
        other = _choose_named_class(rng, available_named_classes, exclude=anchor_class)
        return _list_expression(graph, OWL.intersectionOf, [anchor_class, other]), {"atomic", "intersection"}
    return anchor_class, {"atomic"}


def _case_requires_literals(case: GeneratedFragmentCase) -> bool:
    return any(isinstance(obj, Literal) for _subj, _pred, obj in case.data_graph)


def generate_random_fragment_case(
    config: FragmentGeneratorConfig,
    *,
    seed: int,
) -> GeneratedFragmentCase:
    rng = random.Random(seed)
    ns = Namespace(f"urn:dag-consistency:{seed}#")
    schema_graph = Graph()
    data_graph = Graph()
    allowed_constructs = set(config.allowed_constructs)

    primitive_classes = [URIRef(ns[f"BaseClass{i}"]) for i in range(config.num_primitive_classes)]
    target_classes = [URIRef(ns[f"TargetClass{i}"]) for i in range(config.num_target_classes)]
    all_classes = primitive_classes + target_classes
    object_properties = [URIRef(ns[f"prop{i}"]) for i in range(config.num_properties)]
    datatype_properties = (
        [URIRef(ns[f"dataProp{i}"]) for i in range(config.num_datatype_properties)]
        if {
            "datatype",
            "has-value",
            "data-oneof",
            "functional-data-property",
            "negative-data-property",
            "has-key",
        } & allowed_constructs
        else []
    )
    individuals = [URIRef(ns[f"node{i}"]) for i in range(config.num_individuals)]
    sample_string_literals = [
        Literal("alpha", datatype=XSD.string),
        Literal("beta", datatype=XSD.string),
        Literal("gamma", datatype=XSD.string),
        Literal("delta", datatype=XSD.string),
    ]

    def random_integer_literal() -> Literal:
        return Literal(rng.choice([2, 7, 12, 16, 18, 21, 27]), datatype=XSD.integer)

    def random_string_literal(*, exclude: Optional[Literal] = None) -> Literal:
        eligible = [lit for lit in sample_string_literals if lit != exclude]
        if not eligible:
            eligible = sample_string_literals
        return rng.choice(eligible)

    for class_term in all_classes:
        schema_graph.add((class_term, RDF.type, OWL.Class))
    for prop_term in object_properties:
        schema_graph.add((prop_term, RDF.type, OWL.ObjectProperty))
    for prop_term in datatype_properties:
        schema_graph.add((prop_term, RDF.type, OWL.DatatypeProperty))
    for individual in individuals:
        data_graph.add((individual, RDF.type, OWL.NamedIndividual))

    target_constructs: Dict[URIRef, Tuple[str, ...]] = {}
    available_named_classes: List[URIRef] = list(primitive_classes)
    target_definition_constructs = sorted(
        construct for construct in allowed_constructs if construct not in {"domain", "range"}
    )

    for target_class in target_classes:
        constructs_used: Set[str] = set()
        num_axioms = rng.randint(config.min_axioms_per_target, config.max_axioms_per_target)

        for _ in range(num_axioms):
            if not target_definition_constructs:
                choice = "subclass"
            else:
                choice = rng.choice(target_definition_constructs)
            if choice == "subclass":
                parent = _choose_named_class(rng, available_named_classes, exclude=target_class)
                schema_graph.add((target_class, RDFS.subClassOf, parent))
                constructs_used.update({"subclass", "atomic"})
            elif choice == "intersection":
                expr, tags = _build_named_boolean_expression(
                    schema_graph,
                    rng,
                    available_named_classes=available_named_classes,
                    predicate=OWL.intersectionOf,
                )
                schema_graph.add((target_class, RDFS.subClassOf, expr))
                constructs_used.update({"subclass", "intersection"})
                constructs_used.update(tags)
            elif choice == "union":
                expr, tags = _build_named_boolean_expression(
                    schema_graph,
                    rng,
                    available_named_classes=available_named_classes,
                    predicate=OWL.unionOf,
                )
                schema_graph.add((target_class, RDFS.subClassOf, expr))
                constructs_used.update({"subclass", "union"})
                constructs_used.update(tags)
            elif choice == "exists":
                filler, tags = _build_boolean_filler(
                    schema_graph,
                    rng,
                    available_named_classes=available_named_classes,
                    allowed_constructs=allowed_constructs,
                )
                restriction = _restriction_expression(
                    schema_graph,
                    rng.choice(object_properties),
                    filler,
                    universal=False,
                )
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "exists"})
                constructs_used.update(tags)
            elif choice == "forall":
                filler, tags = _build_boolean_filler(
                    schema_graph,
                    rng,
                    available_named_classes=available_named_classes,
                    allowed_constructs=allowed_constructs,
                )
                restriction = _restriction_expression(
                    schema_graph,
                    rng.choice(object_properties),
                    filler,
                    universal=True,
                )
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "forall"})
                constructs_used.update(tags)
            elif choice == "datatype":
                if not datatype_properties:
                    continue
                min_inclusive = rng.choice([None, 5, 10, 18])
                max_inclusive = None if min_inclusive is not None else rng.choice([None, 19, 30])
                filler = _datatype_restriction_expression(
                    schema_graph,
                    datatype=XSD.integer,
                    min_inclusive=min_inclusive,
                    max_inclusive=max_inclusive,
                )
                restriction = _restriction_expression(
                    schema_graph,
                    rng.choice(datatype_properties),
                    filler,
                    universal=False,
                )
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "datatype"})
            elif choice == "has-value":
                use_datatype = bool(datatype_properties and rng.random() < 0.5)
                witness = rng.choice(individuals)
                if use_datatype:
                    prop_term = rng.choice(datatype_properties)
                    value = random_string_literal() if rng.random() < 0.5 else random_integer_literal()
                    restriction = _has_value_restriction_expression(
                        schema_graph,
                        prop=prop_term,
                        value=value,
                    )
                    data_graph.add((witness, prop_term, value))
                    constructs_used.update({"subclass", "has-value", "datatype"})
                else:
                    prop_term = rng.choice(object_properties)
                    value = rng.choice(individuals)
                    restriction = _has_value_restriction_expression(
                        schema_graph,
                        prop=prop_term,
                        value=value,
                    )
                    data_graph.add((witness, prop_term, value))
                    constructs_used.update({"subclass", "has-value", "nominal"})
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
            elif choice == "data-oneof":
                if not datatype_properties:
                    continue
                prop_term = rng.choice(datatype_properties)
                value = random_string_literal()
                filler = _datatype_oneof_expression(schema_graph, [value])
                restriction = _restriction_expression(
                    schema_graph,
                    prop_term,
                    filler,
                    universal=False,
                )
                witness = rng.choice(individuals)
                data_graph.add((witness, prop_term, value))
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "datatype", "data-oneof"})
            elif choice == "nominal":
                nominal_members = rng.sample(list(individuals), k=1 if len(individuals) == 1 else rng.choice([1, 2]))
                nominal_expr = _nominal_expression(schema_graph, nominal_members)
                schema_graph.add((target_class, RDFS.subClassOf, nominal_expr))
                constructs_used.update({"subclass", "nominal"})
            elif choice == "has-self":
                prop_term = rng.choice(object_properties)
                restriction = _has_self_restriction_expression(
                    schema_graph,
                    prop=prop_term,
                )
                witness = rng.choice(individuals)
                data_graph.add((witness, prop_term, witness))
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "has-self"})
            elif choice == "reflexive":
                prop_term = rng.choice(object_properties)
                schema_graph.add((prop_term, RDF.type, OWL.ReflexiveProperty))
                restriction = _has_self_restriction_expression(
                    schema_graph,
                    prop=prop_term,
                )
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "has-self", "reflexive"})
            elif choice == "geq-cardinality":
                filler, tags = _build_boolean_filler(
                    schema_graph,
                    rng,
                    available_named_classes=available_named_classes,
                    allowed_constructs={"intersection"} if "intersection" in allowed_constructs else set(),
                )
                restriction = _min_qualified_cardinality_expression(
                    schema_graph,
                    prop=rng.choice(object_properties),
                    filler=filler,
                    target=rng.choice([1, 2, 3]),
                )
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "geq-cardinality"})
                constructs_used.update(tags)
            elif choice == "functional-data-property":
                if not datatype_properties:
                    continue
                prop_term = rng.choice(datatype_properties)
                schema_graph.add((prop_term, RDF.type, OWL.FunctionalProperty))
                value = random_string_literal()
                restriction = _has_value_restriction_expression(
                    schema_graph,
                    prop=prop_term,
                    value=value,
                )
                ok_subject = rng.choice(individuals)
                other_subject = rng.choice(individuals)
                data_graph.add((ok_subject, prop_term, value))
                data_graph.add((other_subject, prop_term, random_string_literal(exclude=value)))
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "has-value", "datatype", "functional-data-property"})
            elif choice == "negative-object-property":
                prop_term = rng.choice(object_properties)
                value = rng.choice(individuals)
                restriction = _has_value_restriction_expression(
                    schema_graph,
                    prop=prop_term,
                    value=value,
                )
                blocked_subject = rng.choice(individuals)
                allowed_subject = rng.choice(individuals)
                data_graph.add((allowed_subject, prop_term, value))
                _negative_property_assertion(
                    data_graph,
                    source=blocked_subject,
                    prop=prop_term,
                    target=value,
                )
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "has-value", "negative-object-property", "nominal"})
            elif choice == "negative-data-property":
                if not datatype_properties:
                    continue
                prop_term = rng.choice(datatype_properties)
                value = random_string_literal()
                restriction = _has_value_restriction_expression(
                    schema_graph,
                    prop=prop_term,
                    value=value,
                )
                blocked_subject = rng.choice(individuals)
                allowed_subject = rng.choice(individuals)
                data_graph.add((allowed_subject, prop_term, value))
                _negative_property_assertion(
                    data_graph,
                    source=blocked_subject,
                    prop=prop_term,
                    target=value,
                )
                schema_graph.add((target_class, RDFS.subClassOf, restriction))
                constructs_used.update({"subclass", "has-value", "datatype", "negative-data-property"})
            elif choice == "has-key":
                if not datatype_properties or len(individuals) < 3 or not primitive_classes:
                    continue
                helper_class = rng.choice(primitive_classes)
                key_prop = rng.choice(datatype_properties)
                _has_key_axiom(schema_graph, class_term=helper_class, props=[key_prop])
                anchor, alias, outsider = rng.sample(list(individuals), k=3)
                shared_value = random_string_literal()
                outsider_value = random_string_literal(exclude=shared_value)
                data_graph.add((anchor, RDF.type, helper_class))
                data_graph.add((alias, RDF.type, helper_class))
                data_graph.add((outsider, RDF.type, helper_class))
                data_graph.add((anchor, key_prop, shared_value))
                data_graph.add((alias, key_prop, shared_value))
                data_graph.add((outsider, key_prop, outsider_value))
                nominal_expr = _nominal_expression(schema_graph, [anchor])
                schema_graph.add((target_class, OWL.equivalentClass, nominal_expr))
                constructs_used.update({"has-key", "nominal"})
            elif choice == "disjoint":
                other = _choose_named_class(rng, available_named_classes, exclude=target_class)
                schema_graph.add((target_class, OWL.disjointWith, other))
                constructs_used.update({"disjoint", "atomic"})
            else:
                raise ValueError(f"Unsupported construct choice: {choice}")

        if not constructs_used:
            parent = _choose_named_class(rng, available_named_classes, exclude=target_class)
            schema_graph.add((target_class, RDFS.subClassOf, parent))
            constructs_used.update({"subclass", "atomic"})

        target_constructs[target_class] = tuple(sorted(constructs_used))
        available_named_classes.append(target_class)

    if "domain" in allowed_constructs or "range" in allowed_constructs:
        for target_class in target_classes:
            if "domain" in allowed_constructs and rng.random() < config.domain_range_axiom_probability:
                expr, tags = _build_horn_safe_domain_range_expression(
                    schema_graph,
                    rng,
                    anchor_class=target_class,
                    available_named_classes=available_named_classes,
                    allowed_constructs=allowed_constructs,
                )
                schema_graph.add((rng.choice(object_properties), RDFS.domain, expr))
                target_constructs[target_class] = tuple(
                    sorted(set(target_constructs[target_class]) | {"domain"} | tags)
                )

            if "range" in allowed_constructs and rng.random() < config.domain_range_axiom_probability:
                expr, tags = _build_horn_safe_domain_range_expression(
                    schema_graph,
                    rng,
                    anchor_class=target_class,
                    available_named_classes=available_named_classes,
                    allowed_constructs=allowed_constructs,
                )
                schema_graph.add((rng.choice(object_properties), RDFS.range, expr))
                target_constructs[target_class] = tuple(
                    sorted(set(target_constructs[target_class]) | {"range"} | tags)
                )

    for idx, left in enumerate(primitive_classes):
        for right in primitive_classes[idx + 1:]:
            if rng.random() < config.disjoint_pair_probability:
                schema_graph.add((left, OWL.disjointWith, right))

    for individual in individuals:
        for class_term in primitive_classes:
            if rng.random() < config.type_probability:
                data_graph.add((individual, RDF.type, class_term))

    for prop_term in object_properties:
        for src in individuals:
            for dst in individuals:
                if src == dst:
                    continue
                if rng.random() < config.edge_probability:
                    data_graph.add((src, prop_term, dst))

    for prop_term in datatype_properties:
        for src in individuals:
            if rng.random() < config.datatype_edge_probability:
                value = Literal(rng.choice([2, 7, 12, 16, 18, 21, 27]), datatype=XSD.integer)
                data_graph.add((src, prop_term, value))

    return GeneratedFragmentCase(
        seed=seed,
        schema_graph=schema_graph,
        data_graph=data_graph,
        target_classes=target_classes,
        target_constructs=target_constructs,
        individuals=individuals,
    )


def _dag_construct_tags(dag: ConstraintDAG) -> Set[str]:
    tags: Set[str] = set()
    for node in dag.nodes:
        if node.ctype == ConstraintType.ATOMIC_CLASS:
            tags.add("atomic")
        elif node.ctype in (ConstraintType.EXISTS_RESTRICTION, ConstraintType.EXISTS_TRANSITIVE_RESTRICTION):
            tags.add("exists")
        elif node.ctype == ConstraintType.HAS_SELF_RESTRICTION:
            tags.add("has-self")
        elif node.ctype == ConstraintType.FORALL_RESTRICTION:
            tags.add("forall")
        elif node.ctype == ConstraintType.INTERSECTION:
            tags.add("intersection")
        elif node.ctype == ConstraintType.UNION:
            tags.add("union")
        elif node.ctype == ConstraintType.NEGATION:
            tags.add("negation")
        elif node.ctype == ConstraintType.NOMINAL:
            tags.add("nominal")
        elif node.ctype == ConstraintType.DATATYPE_CONSTRAINT:
            tags.add("datatype")
        elif node.ctype == ConstraintType.MIN_CARDINALITY_RESTRICTION:
            tags.add("geq-cardinality")
        elif node.ctype == ConstraintType.MAX_CARDINALITY_RESTRICTION:
            tags.add("max-cardinality")
        elif node.ctype == ConstraintType.EXACT_CARDINALITY_RESTRICTION:
            tags.add("exact-cardinality")
    return tags


def _construct_bucket_key(tags: Iterable[str]) -> Tuple[str, ...]:
    cleaned = tuple(sorted({tag for tag in tags if tag}))
    return cleaned if cleaned else ("atomic",)


def _check_owlready2_consistency(
    graph: Graph,
    *,
    reasoner_name: str = "hermit",
) -> Tuple[bool, float, Optional[str]]:
    try:
        from owlready2 import World, sync_reasoner, sync_reasoner_pellet
        from owlready2.base import OwlReadyInconsistentOntologyError
    except ImportError as exc:
        return False, 0.0, str(exc)

    temp_path: Optional[str] = None
    try:
        handle = tempfile.NamedTemporaryFile(suffix=".owl", delete=False)
        temp_path = handle.name
        handle.close()
        graph.serialize(destination=temp_path, format="xml")

        stderr_buffer = io.StringIO()
        world = World()
        t0 = perf_counter()
        with contextlib.redirect_stderr(stderr_buffer):
            ontology = world.get_ontology(temp_path).load()
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
        return True, elapsed_ms, None
    except OwlReadyInconsistentOntologyError:
        elapsed_ms = (perf_counter() - t0) * 1000.0 if "t0" in locals() else 0.0
        return False, elapsed_ms, None
    except Exception as exc:
        return False, 0.0, str(exc)
    finally:
        if temp_path and os.path.exists(temp_path):
            with contextlib.suppress(PermissionError):
                os.unlink(temp_path)


def _format_bucket_key(bucket_key: Tuple[str, ...]) -> str:
    return ", ".join(bucket_key)


def _collect_consistency_buckets(
    *,
    case: GeneratedFragmentCase,
    threshold: float,
    device: str,
    materialize_hierarchy: bool,
    owlready2_reasoner: str,
    max_examples_per_bucket: int,
    augment_property_domain_range: bool,
    engine_mode: str,
    conflict_policy: str,
    case_dir: Optional[str],
) -> Tuple[
    int,
    int,
    List[ConsistencyFailure],
    Dict[Tuple[str, ...], BucketStats],
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    str,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    include_literals = _case_requires_literals(case)
    engine_result = run_engine_queries(
        schema_graph=case.schema_graph,
        data_graph=case.data_graph,
        target_classes=case.target_classes,
        device=device,
        threshold=threshold,
        include_literals=include_literals,
        include_type_edges=False,
        materialize_hierarchy=materialize_hierarchy,
        materialize_supported_types=False,
        augment_property_domain_range=augment_property_domain_range,
        engine_mode=engine_mode,
        conflict_policy=conflict_policy,
    )

    if engine_result.dataset is None or engine_result.scores_by_target is None:
        raise RuntimeError("Engine query run did not return a dataset or scores.")

    query_plan = None
    if engine_mode != "stratified":
        query_plan = plan_reasoning_preprocessing(
            engine_result.dataset.ontology_graph,
            target_classes=case.target_classes,
            materialize_hierarchy=materialize_hierarchy,
            augment_property_domain_range=augment_property_domain_range,
        )
    if case_dir is not None:
        preprocessing_payload = {
            "engine": {
                "mode": engine_mode,
                "conflict_policy": conflict_policy if engine_mode in {"stratified", "filtered_query"} else None,
            },
            "loader": {
                "materialize_hierarchy": vars(engine_result.dataset.preprocessing_plan.materialize_hierarchy),
                "materialize_atomic_domain_range": vars(engine_result.dataset.preprocessing_plan.materialize_atomic_domain_range),
                "materialize_horn_safe_domain_range": vars(engine_result.dataset.preprocessing_plan.materialize_horn_safe_domain_range),
                "materialize_sameas": vars(engine_result.dataset.preprocessing_plan.materialize_sameas),
                "materialize_reflexive_properties": vars(engine_result.dataset.preprocessing_plan.materialize_reflexive_properties),
                "materialize_target_roles": vars(engine_result.dataset.preprocessing_plan.materialize_target_roles),
                "augment_property_domain_range": vars(engine_result.dataset.preprocessing_plan.augment_property_domain_range),
            },
        }
        if query_plan is not None:
            preprocessing_payload["query"] = {
                "augment_property_domain_range": vars(query_plan.augment_property_domain_range),
            }
        with open(os.path.join(case_dir, "preprocessing-plan.json"), "w", encoding="utf-8") as handle:
            json.dump(preprocessing_payload, handle, indent=2, sort_keys=True)
    preprocessing_text = describe_preprocessing_plan(engine_result.dataset.preprocessing_plan)
    preprocessing_lines = [preprocessing_text, f"Engine mode: {engine_mode}"]
    if engine_mode in {"stratified", "filtered_query"}:
        preprocessing_lines.append(f"Conflict policy: {conflict_policy}")
    elif query_plan is not None:
        preprocessing_lines.append(
            f"Query augmentation: {'on' if query_plan.augment_property_domain_range.enabled else 'off'} "
            f"(policy={query_plan.augment_property_domain_range.policy}; "
            f"{query_plan.augment_property_domain_range.reason})"
        )
    preprocessing_summary_text = "\n".join(preprocessing_lines)

    merged_graph = _merge_graphs(case.schema_graph, case.data_graph)
    bucket_stats: Dict[Tuple[str, ...], BucketStats] = {}
    failures: List[ConsistencyFailure] = []
    perfect_scores = 0
    checked_assertions = 0
    preprocessing_elapsed_ms = (
        engine_result.dataset_build_elapsed_ms + engine_result.dag_compile_elapsed_ms
    )
    assertion_consistency_check_elapsed_ms = 0.0
    raw_positive_assignment_count = 0
    policy_emitted_assignment_count = 0
    policy_suppressed_assignment_count = 0
    raw_candidate_assignment_count = 0
    necessary_condition_retraction_count = 0
    closure_blocked_retraction_count = 0
    final_emitted_assignment_count = 0
    raw_candidate_assignment_count = 0
    necessary_condition_retraction_count = 0
    closure_blocked_retraction_count = 0
    final_emitted_assignment_count = 0

    if engine_mode == "stratified" and engine_result.stratified_result is not None:
        raw_positive_assignment_count = sum(
            1
            for status in engine_result.stratified_result.assignment_statuses
            if status.asserted or status.positively_derived
        )
        policy_emitted_assignment_count = len(
            engine_result.stratified_result.policy_result.emitted_assignments
        )
        policy_suppressed_assignment_count = (
            raw_positive_assignment_count - policy_emitted_assignment_count
        )
    elif engine_mode == "filtered_query" and engine_result.filtered_query_result is not None:
        raw_candidate_assignment_count = engine_result.filtered_query_result.raw_candidate_count
        necessary_condition_retraction_count = (
            engine_result.filtered_query_result.necessary_retraction_count
        )
        closure_blocked_retraction_count = (
            engine_result.filtered_query_result.closure_blocked_retraction_count
        )
        final_emitted_assignment_count = engine_result.filtered_query_result.final_emitted_count

    for target_class in case.target_classes:
        if engine_mode == "stratified":
            dag = compile_sufficient_condition_dag(
                engine_result.dataset.ontology_graph,
                engine_result.dataset.mapping,
                target_class,
            )
        else:
            dag = compile_class_to_dag(
                engine_result.dataset.ontology_graph,
                engine_result.dataset.mapping,
                target_class,
                augment_property_domain_range=augment_property_domain_range,
            )
        construct_key = _construct_bucket_key(
            set(case.target_constructs.get(target_class, ())) | _dag_construct_tags(dag)
        )
        bucket = bucket_stats.setdefault(construct_key, BucketStats())

        scores = engine_result.scores_by_target[target_class]
        for node_term, score in scores.items():
            if score < threshold:
                continue
            perfect_scores += 1

            if not isinstance(node_term, URIRef):
                continue
            if (node_term, RDF.type, target_class) in case.data_graph:
                continue

            checked_assertions += 1
            bucket.tested += 1

            augmented_graph = _copy_graph(merged_graph)
            augmented_graph.add((node_term, RDF.type, target_class))
            consistent, elapsed_ms, error = _check_owlready2_consistency(
                augmented_graph,
                reasoner_name=owlready2_reasoner,
            )
            assertion_consistency_check_elapsed_ms += elapsed_ms
            if consistent:
                continue

            bucket.failures += 1
            failure = ConsistencyFailure(
                seed=case.seed,
                target_class=target_class,
                node_term=node_term,
                score=score,
                constructs=construct_key,
                reasoner=owlready2_reasoner,
                error=error,
            )
            failures.append(failure)
            if len(bucket.examples) < max_examples_per_bucket:
                bucket.examples.append(failure)
            if case_dir is not None:
                explanations_dir = os.path.join(case_dir, "explanations")
                _ensure_dir(explanations_dir)
                example_index = len(failures) - 1
                explanation_path = os.path.join(
                    explanations_dir,
                    f"failure-{example_index:03d}-{_sanitize_bucket_key(construct_key)}.txt",
                )
                with open(explanation_path, "w", encoding="utf-8") as handle:
                    if engine_mode == "stratified" and engine_result.stratified_result is not None:
                        matching_statuses = [
                            status
                            for status in engine_result.stratified_result.assignment_statuses
                            if status.node_term == node_term and status.target_class == target_class
                        ]
                        handle.write("=== Stratified Failure Summary ===\n")
                        handle.write(f"target={target_class.n3()}\n")
                        handle.write(f"node={_render_term(node_term)}\n")
                        handle.write(f"policy={engine_result.conflict_policy}\n")
                        handle.write("statuses:\n")
                        if not matching_statuses:
                            handle.write("  (none)\n")
                        else:
                            for status in matching_statuses:
                                handle.write(
                                    f"  asserted={status.asserted}, positively_derived={status.positively_derived}, "
                                    f"blocked={status.blocked}, conflicted={status.conflicted}, "
                                    f"blockers={[str(term) for term in status.blocker_classes]}\n"
                                )
                    elif engine_mode == "filtered_query" and engine_result.filtered_query_result is not None:
                        filtered = engine_result.filtered_query_result
                        raw_match = node_term in filtered.raw_members_by_target.get(target_class, set())
                        necessary_match = node_term in filtered.necessary_stable_members_by_target.get(target_class, set())
                        closure_blocked = node_term in filtered.closure_blocked_members_by_target.get(target_class, set())
                        final_match = node_term in filtered.final_members_by_target.get(target_class, set())
                        handle.write("=== Filtered Query Failure Summary ===\n")
                        handle.write(f"target={target_class.n3()}\n")
                        handle.write(f"node={_render_term(node_term)}\n")
                        handle.write(f"raw_candidate={raw_match}\n")
                        handle.write(f"survived_necessary_fixpoint={necessary_match}\n")
                        handle.write(f"closure_blocked={closure_blocked}\n")
                        handle.write(f"final_emitted={final_match}\n")
                        handle.write(
                            f"necessary_fixpoint_iterations={filtered.necessary_fixpoint_iterations}\n"
                        )
                        blockers = sorted(
                            str(blocked.blocker_class)
                            for blocked in filtered.stratified_result.negative_result.blocked_assertions
                            if blocked.node_term == node_term and blocked.target_class == target_class
                        )
                        handle.write(f"blockers={blockers}\n")
                    else:
                        explanation = explain_dataset_query(
                            engine_result.dataset,
                            target_class=target_class,
                            node_term=node_term,
                            augment_property_domain_range=augment_property_domain_range,
                            device=device,
                        )
                        handle.write(explanation.text)

        if case_dir is not None:
            case_scores_dir = os.path.join(case_dir, "scores")
            _ensure_dir(case_scores_dir)
            score_payload = {
                _render_term(node_term): float(score)
                for node_term, score in scores.items()
            }
            with open(
                os.path.join(
                    case_scores_dir,
                    _safe_filename(
                        target_class.split("#")[-1]
                        if "#" in str(target_class)
                        else str(target_class).rsplit("/", 1)[-1]
                    ) + ".json",
                ),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(score_payload, handle, indent=2, sort_keys=True)

    if case_dir is not None and failures:
        failure_payload = [
            {
                "seed": failure.seed,
                "target_class": str(failure.target_class),
                "node_term": _render_term(failure.node_term),
                "score": failure.score,
                "constructs": list(failure.constructs),
                "reasoner": failure.reasoner,
                "error": failure.error,
            }
            for failure in failures
        ]
        with open(os.path.join(case_dir, "failures.json"), "w", encoding="utf-8") as handle:
            json.dump(failure_payload, handle, indent=2, sort_keys=True)

    return (
        perfect_scores,
        checked_assertions,
        failures,
        bucket_stats,
        preprocessing_elapsed_ms,
        engine_result.dataset_build_elapsed_ms,
        engine_result.hierarchy_elapsed_ms,
        engine_result.atomic_domain_range_elapsed_ms,
        engine_result.horn_safe_domain_range_elapsed_ms,
        engine_result.sameas_elapsed_ms,
        engine_result.reflexive_elapsed_ms,
        engine_result.target_role_elapsed_ms,
        engine_result.kgraph_build_elapsed_ms,
        engine_result.dag_compile_elapsed_ms,
        engine_result.dag_eval_elapsed_ms,
        assertion_consistency_check_elapsed_ms,
        preprocessing_summary_text,
        raw_positive_assignment_count,
        policy_emitted_assignment_count,
        policy_suppressed_assignment_count,
        raw_candidate_assignment_count,
        necessary_condition_retraction_count,
        closure_blocked_retraction_count,
        final_emitted_assignment_count,
    )


def run_consistency_harness(
    *,
    num_cases: int,
    max_attempts: int,
    start_seed: int,
    config: FragmentGeneratorConfig,
    threshold: float = 0.999,
    device: str = "cuda",
    materialize_hierarchy: Optional[bool] = None,
    owlready2_reasoner: str = "hermit",
    max_examples_per_bucket: int = 3,
    augment_property_domain_range: Optional[bool] = None,
    engine_mode: str = "query",
    conflict_policy: str = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
    save_cases: bool = True,
    save_dir: str = os.path.join("data", "runs", "consistency-harness"),
) -> HarnessSummary:
    generated_cases = 0
    attempts = 0
    seed = start_seed
    base_consistent_cases = 0
    total_perfect_scores = 0
    total_checked_assertions = 0
    total_failures = 0
    generation_elapsed_ms = 0.0
    preprocessing_elapsed_ms = 0.0
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
    base_consistency_check_elapsed_ms = 0.0
    assertion_consistency_check_elapsed_ms = 0.0
    raw_positive_assignment_count = 0
    policy_emitted_assignment_count = 0
    policy_suppressed_assignment_count = 0
    raw_candidate_assignment_count = 0
    necessary_condition_retraction_count = 0
    closure_blocked_retraction_count = 0
    final_emitted_assignment_count = 0
    merged_bucket_stats: Dict[Tuple[str, ...], BucketStats] = {}
    case_preprocessing_summaries: Dict[str, str] = {}
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
                    "owlready2 consistency check failed before harness execution: "
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
                materialize_hierarchy=materialize_hierarchy,
                augment_property_domain_range=augment_property_domain_range,
                engine_mode=engine_mode,
                conflict_policy=(conflict_policy if engine_mode in {"stratified", "filtered_query"} else None),
                base_consistent=True,
            )
            if run_save_root is not None
            else None
        )

        (
            perfect_scores,
            checked_assertions,
            failures,
            bucket_stats,
            case_preprocessing_elapsed_ms,
            case_dataset_build_elapsed_ms,
            case_hierarchy_elapsed_ms,
            case_atomic_domain_range_elapsed_ms,
            case_horn_safe_domain_range_elapsed_ms,
            case_sameas_elapsed_ms,
            case_reflexive_elapsed_ms,
            case_target_role_elapsed_ms,
            case_kgraph_build_elapsed_ms,
            case_dag_compile_elapsed_ms,
            case_dag_eval_elapsed_ms,
            case_assertion_consistency_check_elapsed_ms,
            case_preprocessing_summary_text,
            case_raw_positive_assignment_count,
            case_policy_emitted_assignment_count,
            case_policy_suppressed_assignment_count,
            case_raw_candidate_assignment_count,
            case_necessary_condition_retraction_count,
            case_closure_blocked_retraction_count,
            case_final_emitted_assignment_count,
        ) = _collect_consistency_buckets(
            case=case,
            threshold=threshold,
            device=device,
            materialize_hierarchy=materialize_hierarchy,
            owlready2_reasoner=owlready2_reasoner,
            max_examples_per_bucket=max_examples_per_bucket,
            augment_property_domain_range=augment_property_domain_range,
            engine_mode=engine_mode,
            conflict_policy=conflict_policy,
            case_dir=case_dir,
        )

        total_perfect_scores += perfect_scores
        total_checked_assertions += checked_assertions
        total_failures += len(failures)
        preprocessing_elapsed_ms += case_preprocessing_elapsed_ms
        dataset_build_elapsed_ms += case_dataset_build_elapsed_ms
        hierarchy_elapsed_ms += case_hierarchy_elapsed_ms
        atomic_domain_range_elapsed_ms += case_atomic_domain_range_elapsed_ms
        horn_safe_domain_range_elapsed_ms += case_horn_safe_domain_range_elapsed_ms
        sameas_elapsed_ms += case_sameas_elapsed_ms
        reflexive_elapsed_ms += case_reflexive_elapsed_ms
        target_role_elapsed_ms += case_target_role_elapsed_ms
        kgraph_build_elapsed_ms += case_kgraph_build_elapsed_ms
        dag_compile_elapsed_ms += case_dag_compile_elapsed_ms
        dag_eval_elapsed_ms += case_dag_eval_elapsed_ms
        assertion_consistency_check_elapsed_ms += case_assertion_consistency_check_elapsed_ms
        raw_positive_assignment_count += case_raw_positive_assignment_count
        policy_emitted_assignment_count += case_policy_emitted_assignment_count
        policy_suppressed_assignment_count += case_policy_suppressed_assignment_count
        raw_candidate_assignment_count += case_raw_candidate_assignment_count
        necessary_condition_retraction_count += case_necessary_condition_retraction_count
        closure_blocked_retraction_count += case_closure_blocked_retraction_count
        final_emitted_assignment_count += case_final_emitted_assignment_count
        case_preprocessing_summaries[f"seed-{case.seed:06d}"] = case_preprocessing_summary_text

        for bucket_key, bucket in bucket_stats.items():
            merged = merged_bucket_stats.setdefault(bucket_key, BucketStats())
            merged.tested += bucket.tested
            merged.failures += bucket.failures
            remaining = max_examples_per_bucket - len(merged.examples)
            if remaining > 0:
                merged.examples.extend(bucket.examples[:remaining])

    return HarnessSummary(
        requested_cases=num_cases,
        generated_cases=generated_cases,
        attempts=attempts,
        base_consistent_cases=base_consistent_cases,
        total_perfect_scores=total_perfect_scores,
        total_checked_assertions=total_checked_assertions,
        total_failures=total_failures,
        bucket_stats=merged_bucket_stats,
        generation_elapsed_ms=generation_elapsed_ms,
        preprocessing_elapsed_ms=preprocessing_elapsed_ms,
        dataset_build_elapsed_ms=dataset_build_elapsed_ms,
        hierarchy_elapsed_ms=hierarchy_elapsed_ms,
        atomic_domain_range_elapsed_ms=atomic_domain_range_elapsed_ms,
        horn_safe_domain_range_elapsed_ms=horn_safe_domain_range_elapsed_ms,
        sameas_elapsed_ms=sameas_elapsed_ms,
        reflexive_elapsed_ms=reflexive_elapsed_ms,
        target_role_elapsed_ms=target_role_elapsed_ms,
        kgraph_build_elapsed_ms=kgraph_build_elapsed_ms,
        dag_compile_elapsed_ms=dag_compile_elapsed_ms,
        dag_eval_elapsed_ms=dag_eval_elapsed_ms,
        base_consistency_check_elapsed_ms=base_consistency_check_elapsed_ms,
        assertion_consistency_check_elapsed_ms=assertion_consistency_check_elapsed_ms,
        engine_mode=engine_mode,
        conflict_policy=(conflict_policy if engine_mode in {"stratified", "filtered_query"} else None),
        raw_positive_assignment_count=raw_positive_assignment_count,
        policy_emitted_assignment_count=policy_emitted_assignment_count,
        policy_suppressed_assignment_count=policy_suppressed_assignment_count,
        raw_candidate_assignment_count=raw_candidate_assignment_count,
        necessary_condition_retraction_count=necessary_condition_retraction_count,
        closure_blocked_retraction_count=closure_blocked_retraction_count,
        final_emitted_assignment_count=final_emitted_assignment_count,
        run_settings={
            "num_cases": num_cases,
            "max_attempts": max_attempts,
            "start_seed": start_seed,
            "threshold": threshold,
            "device": device,
            "materialize_hierarchy": materialize_hierarchy,
            "augment_property_domain_range": augment_property_domain_range,
            "owlready2_reasoner": owlready2_reasoner,
            "max_examples_per_bucket": max_examples_per_bucket,
            "save_cases": save_cases,
            "save_dir": save_dir,
            "generator_config": dict(config.__dict__),
        },
        case_preprocessing_summaries=case_preprocessing_summaries,
        save_root=run_save_root,
    )


def format_harness_summary(summary: HarnessSummary) -> str:
    lines: List[str] = []
    lines.append("=== Consistency Guarantee Harness ===")
    lines.append(f"Engine mode: {summary.engine_mode}")
    if summary.engine_mode in {"stratified", "filtered_query"} and summary.conflict_policy is not None:
        lines.append(f"Conflict policy: {summary.conflict_policy}")
    if summary.run_settings:
        lines.append("Run settings:")
        ordered_keys = [
            "threshold",
            "device",
            "materialize_hierarchy",
            "augment_property_domain_range",
            "owlready2_reasoner",
            "max_examples_per_bucket",
            "num_cases",
            "max_attempts",
            "start_seed",
            "save_cases",
            "save_dir",
            "generator_config",
        ]
        seen: Set[str] = set()
        for key in ordered_keys:
            if key not in summary.run_settings:
                continue
            seen.add(key)
            lines.append(f"  - {key}: {summary.run_settings[key]}")
        for key in sorted(summary.run_settings.keys()):
            if key in seen:
                continue
            lines.append(f"  - {key}: {summary.run_settings[key]}")
    if summary.engine_mode == "stratified":
        lines.append("Policy assignment counts:")
        lines.append(f"  - raw_positive_assignments: {summary.raw_positive_assignment_count}")
        lines.append(f"  - policy_emitted_assignments: {summary.policy_emitted_assignment_count}")
        lines.append(f"  - policy_suppressed_assignments: {summary.policy_suppressed_assignment_count}")
    elif summary.engine_mode == "filtered_query":
        lines.append("Filtered-query assignment counts:")
        lines.append(f"  - raw_candidate_assignments: {summary.raw_candidate_assignment_count}")
        lines.append(
            f"  - necessary_condition_retractions: {summary.necessary_condition_retraction_count}"
        )
        lines.append(
            f"  - closure_blocked_retractions: {summary.closure_blocked_retraction_count}"
        )
        lines.append(f"  - final_emitted_assignments: {summary.final_emitted_assignment_count}")
    lines.append(f"Requested cases: {summary.requested_cases}")
    lines.append(f"Generated base-consistent cases: {summary.generated_cases}")
    lines.append(f"Generation attempts: {summary.attempts}")
    lines.append(f"Perfect scores found: {summary.total_perfect_scores}")
    lines.append(f"Assertions checked: {summary.total_checked_assertions}")
    lines.append(f"Consistency failures: {summary.total_failures}")
    if summary.save_root:
        lines.append(f"Saved cases: {summary.save_root}")
    lines.append("")
    lines.append("Stage timings:")
    lines.append(f"  - Random graph generation: {summary.generation_elapsed_ms:.3f} ms")
    lines.append(
        f"  - Preprocessing total: {summary.preprocessing_elapsed_ms:.3f} ms "
        f"(loader/dataset build={summary.dataset_build_elapsed_ms:.3f} ms, "
        f"DAG compile={summary.dag_compile_elapsed_ms:.3f} ms)"
    )
    lines.append(
        "      preprocessing breakdown: "
        f"hierarchy={summary.hierarchy_elapsed_ms:.3f} ms, "
        f"atomic domain/range={summary.atomic_domain_range_elapsed_ms:.3f} ms, "
        f"horn-safe domain/range={summary.horn_safe_domain_range_elapsed_ms:.3f} ms, "
        f"sameAs={summary.sameas_elapsed_ms:.3f} ms, "
        f"reflexive properties={summary.reflexive_elapsed_ms:.3f} ms, "
        f"target roles={summary.target_role_elapsed_ms:.3f} ms, "
        f"kgraph build={summary.kgraph_build_elapsed_ms:.3f} ms"
    )
    lines.append(f"  - DAG evaluation: {summary.dag_eval_elapsed_ms:.3f} ms")
    lines.append(
        f"  - Error checking total: "
        f"{summary.base_consistency_check_elapsed_ms + summary.assertion_consistency_check_elapsed_ms:.3f} ms "
        f"(base consistency={summary.base_consistency_check_elapsed_ms:.3f} ms, "
        f"assertion checks={summary.assertion_consistency_check_elapsed_ms:.3f} ms)"
    )

    if summary.generated_cases < summary.requested_cases:
        lines.append("Warning: fewer consistent cases were generated than requested.")

    if summary.case_preprocessing_summaries:
        lines.append("")
        lines.append("Per-Case Preprocessing Plans:")
        for case_key in sorted(summary.case_preprocessing_summaries.keys()):
            lines.append(f"  [{case_key}]")
            for raw_line in summary.case_preprocessing_summaries[case_key].splitlines():
                lines.append(f"    {raw_line}")

    lines.append("")
    lines.append("Failure buckets by construct set:")
    exercised_buckets = {
        bucket_key: bucket
        for bucket_key, bucket in summary.bucket_stats.items()
        if bucket.tested > 0
    }
    if not exercised_buckets:
        lines.append("  (no buckets; no assertions were checked)")
        return "\n".join(lines)

    ordered_buckets = sorted(
        exercised_buckets.items(),
        key=lambda item: (-item[1].failures, -item[1].tested, item[0]),
    )
    for bucket_key, bucket in ordered_buckets:
        failure_rate = (bucket.failures / bucket.tested) if bucket.tested else 0.0
        lines.append(
            f"  - [{_format_bucket_key(bucket_key)}] "
            f"tested={bucket.tested}, failures={bucket.failures}, "
            f"failure_rate={failure_rate:.3f}"
        )
        for example in bucket.examples:
            lines.append(
                "      example: "
                f"seed={example.seed}, node={_render_term(example.node_term)}, "
                f"target={_render_term(example.target_class)}, score={example.score:.4f}"
            )
            if example.error:
                lines.append(f"      error: {example.error}")
    return "\n".join(lines)


def print_harness_summary(summary: HarnessSummary) -> None:
    print(format_harness_summary(summary))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fragment-bounded random ontology/KG pairs and test the "
            "selected engine mode's local consistency guarantee against owlready2."
        )
    )
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=25)
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
        default=list(DEFAULT_CONSTRUCTS),
        help="Construct names and/or profile names to include in random generation.",
    )
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--engine-mode",
        choices=["query", "filtered_query", "stratified"],
        default="query",
        help=(
            "query = necessary-condition admissibility path; "
            "filtered_query = query candidates pruned by synchronous recheck plus stratified blockers; "
            "stratified = positive sufficient-condition closure plus negative blocker policy."
        ),
    )
    parser.add_argument(
        "--conflict-policy",
        choices=[policy.value for policy in ConflictPolicy],
        default=ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED.value,
        help="Conflict policy used in stratified mode.",
    )
    hierarchy_group = parser.add_mutually_exclusive_group()
    hierarchy_group.add_argument("--materialize-hierarchy", dest="materialize_hierarchy", action="store_true")
    hierarchy_group.add_argument("--no-materialize-hierarchy", dest="materialize_hierarchy", action="store_false")
    parser.set_defaults(materialize_hierarchy=None)
    parser.add_argument(
        "--owlready2-reasoner",
        choices=["hermit", "pellet"],
        default="hermit",
    )
    parser.add_argument("--max-examples-per-bucket", type=int, default=3)
    parser.add_argument(
        "--no-save-cases",
        action="store_true",
        help="Do not persist base-consistent generated cases to disk.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join("data", "runs", "consistency-harness"),
        help="Directory under which successful harness runs are persisted.",
    )
    augment_group = parser.add_mutually_exclusive_group()
    augment_group.add_argument(
        "--augment-domain-range",
        dest="augment_domain_range",
        action="store_true",
        help=(
            "Evaluate targets with query-time rdfs:domain / rdfs:range augmentation enabled. "
            "This is also enabled automatically whenever the generated construct set includes "
            "`domain` or `range`."
        ),
    )
    augment_group.add_argument(
        "--no-augment-domain-range",
        dest="augment_domain_range",
        action="store_false",
        help="Explicitly disable query-time domain/range augmentation.",
    )
    parser.set_defaults(augment_domain_range=None)
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
    auto_augment_domain_range = ("domain" in args.constructs or "range" in args.constructs)
    summary = run_consistency_harness(
        num_cases=args.num_cases,
        max_attempts=args.max_attempts,
        start_seed=args.start_seed,
        config=config,
        threshold=args.threshold,
        device=args.device,
        materialize_hierarchy=args.materialize_hierarchy,
        owlready2_reasoner=args.owlready2_reasoner,
        max_examples_per_bucket=args.max_examples_per_bucket,
        augment_property_domain_range=(
            auto_augment_domain_range if args.augment_domain_range is None else args.augment_domain_range
        ),
        engine_mode=args.engine_mode,
        conflict_policy=args.conflict_policy,
        save_cases=not args.no_save_cases,
        save_dir=args.save_dir,
    )
    _write_run_summary(summary)
    print_harness_summary(summary)


if __name__ == "__main__":
    main()
