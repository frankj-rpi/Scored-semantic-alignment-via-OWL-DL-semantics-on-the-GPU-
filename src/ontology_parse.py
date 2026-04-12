from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections import defaultdict
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD
from rdflib.term import Identifier

from .constraints import (
    CardinalityAgg,
    ConstraintDAG,
    ConstraintNode,
    ConstraintType,
    IntersectionAgg,
    TraversalDirection,
)
from .graph import KGraph


@dataclass
class RDFKGraphMapping:
    """
    Stable mappings between RDF terms and KGraph indices.

    - node_terms[i] is the RDF term for node i.
    - prop_terms[p] is the RDF predicate for property p.
    - class_terms[c] is the RDF class term for class c.
    """

    node_terms: List[Identifier]
    prop_terms: List[URIRef]
    class_terms: List[Identifier]
    datatype_terms: List[URIRef]
    node_to_idx: Dict[Identifier, int]
    prop_to_idx: Dict[URIRef, int]
    class_to_idx: Dict[Identifier, int]
    datatype_to_idx: Dict[URIRef, int]


@dataclass
class ReasoningDataset:
    """
    Bundles a clean TBox/ABox loading workflow.

    - schema_graph: ontology / vocabulary triples
    - data_graph: instance triples that become KGraph nodes and edges
    - ontology_graph: union of schema_graph and data_graph, used for compilation
    - kg: instance-level KGraph
    - mapping: shared vocabulary mapping for the KGraph and compiler
    """

    schema_graph: Graph
    data_graph: Graph
    ontology_graph: Graph
    kg: KGraph
    mapping: RDFKGraphMapping
    preprocessing_plan: Optional["PreprocessingPlan"] = None
    preprocessing_timings: Optional["PreprocessingTimings"] = None


@dataclass
class ClassMaterializationResult:
    dataset: ReasoningDataset
    inferred_assertions: List[Tuple[Identifier, URIRef]]
    iterations: int


@dataclass
class RoleSaturationResult:
    data_graph: Graph
    seed_properties: List[URIRef]
    relevant_properties: List[URIRef]
    inferred_edges: List[Tuple[Identifier, URIRef, Identifier]]
    iterations: int


@dataclass
class TargetDependencyClosure:
    target_classes: List[URIRef]
    referenced_named_classes: List[URIRef]
    inferable_named_classes: List[URIRef]
    referenced_properties: List[URIRef]


@dataclass
class DAGNodeDependency:
    idx: int
    ctype: ConstraintType
    direct_class_terms: List[Identifier]
    direct_property_terms: List[URIRef]
    subtree_class_terms: List[Identifier]
    subtree_property_terms: List[URIRef]
    subtree_inferable_named_classes: List[URIRef]


@dataclass
class DAGDependencyReport:
    target_class: URIRef
    dag: ConstraintDAG
    node_dependencies: Dict[int, DAGNodeDependency]
    referenced_named_classes: List[URIRef]
    inferable_named_classes: List[URIRef]
    referenced_properties: List[URIRef]


@dataclass
class PropertyExpressionAxioms:
    property_term: URIRef
    domain_expressions: List[Identifier]
    range_expressions: List[Identifier]


class SufficientConditionKind(Enum):
    TOP = "top"
    ATOMIC_CLASS = "atomic_class"
    NOMINAL = "nominal"
    DATATYPE_CONSTRAINT = "datatype_constraint"
    EXISTS = "exists"
    INTERSECTION = "intersection"
    MIN_CARDINALITY = "min_cardinality"


@dataclass(frozen=True)
class NormalizedSufficientCondition:
    kind: SufficientConditionKind
    class_term: Optional[URIRef] = None
    node_term: Optional[Identifier] = None
    datatype_term: Optional[URIRef] = None
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    min_inclusive: bool = True
    max_inclusive: bool = True
    prop_term: Optional[URIRef] = None
    prop_direction: TraversalDirection = TraversalDirection.FORWARD
    cardinality_target: Optional[int] = None
    children: Tuple["NormalizedSufficientCondition", ...] = ()


@dataclass(frozen=True)
class NormalizedSufficientRule:
    consequent_class: URIRef
    antecedent: NormalizedSufficientCondition
    source_kind: str
    source_term: Optional[Identifier] = None
    source_rendered: Optional[str] = None
    tags: Tuple[str, ...] = ()


@dataclass
class NormalizedSufficientRuleSet:
    rules: List[NormalizedSufficientRule]
    skipped_axioms: List[str]


@dataclass
class PreprocessingPassDecision:
    name: str
    policy: str
    enabled: bool
    reason: str


@dataclass
class PreprocessingPlan:
    materialize_hierarchy: PreprocessingPassDecision
    materialize_atomic_domain_range: PreprocessingPassDecision
    materialize_horn_safe_domain_range: PreprocessingPassDecision
    materialize_target_roles: PreprocessingPassDecision
    augment_property_domain_range: PreprocessingPassDecision


@dataclass
class PreprocessingTimings:
    hierarchy_elapsed_ms: float = 0.0
    atomic_domain_range_elapsed_ms: float = 0.0
    horn_safe_domain_range_elapsed_ms: float = 0.0
    target_role_elapsed_ms: float = 0.0
    kgraph_build_elapsed_ms: float = 0.0

    @property
    def preprocessing_elapsed_ms(self) -> float:
        return (
            self.hierarchy_elapsed_ms
            + self.atomic_domain_range_elapsed_ms
            + self.horn_safe_domain_range_elapsed_ms
            + self.target_role_elapsed_ms
        )

    @property
    def dataset_build_elapsed_ms(self) -> float:
        return self.preprocessing_elapsed_ms + self.kgraph_build_elapsed_ms


def _ensure_sequence(paths: str | Path | Sequence[str | Path]) -> List[str]:
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(p) for p in paths]


def _normalize_policy(explicit_flag: Optional[bool], policy: str, *, default: str) -> str:
    if explicit_flag is True:
        return "on"
    if explicit_flag is False:
        return "off"
    normalized = (policy or default).lower()
    if normalized not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported preprocessing policy: {policy}")
    return normalized


def _has_any_triple(graph: Graph, predicate: URIRef, obj: Optional[Identifier] = None) -> bool:
    return any(True for _ in graph.triples((None, predicate, obj)))


def _has_negative_constructs(graph: Graph) -> bool:
    return _has_any_triple(graph, OWL.disjointWith) or _has_any_triple(graph, OWL.complementOf)


def _has_hierarchy_axioms(graph: Graph) -> bool:
    return _has_any_triple(graph, RDFS.subClassOf) or _has_any_triple(graph, RDFS.subPropertyOf)


def _has_domain_or_range_axioms(graph: Graph) -> bool:
    return _has_any_triple(graph, RDFS.domain) or _has_any_triple(graph, RDFS.range)


def _has_role_axioms(graph: Graph) -> bool:
    return (
        _has_any_triple(graph, RDFS.subPropertyOf)
        or _has_any_triple(graph, OWL.inverseOf)
        or _has_any_triple(graph, OWL.propertyChainAxiom)
        or _has_any_triple(graph, RDF.type, OWL.TransitiveProperty)
    )


def plan_reasoning_preprocessing(
    ontology_graph: Graph,
    *,
    target_classes: Optional[Sequence[str | URIRef]] = None,
    materialize_hierarchy: Optional[bool] = None,
    materialize_hierarchy_policy: str = "auto",
    materialize_atomic_domain_range: Optional[bool] = None,
    materialize_atomic_domain_range_policy: str = "off",
    materialize_horn_safe_domain_range: Optional[bool] = None,
    materialize_horn_safe_domain_range_policy: str = "auto",
    materialize_target_roles: Optional[bool] = None,
    materialize_target_roles_policy: str = "auto",
    augment_property_domain_range: Optional[bool] = None,
    augment_property_domain_range_policy: str = "auto",
) -> PreprocessingPlan:
    has_hierarchy = _has_hierarchy_axioms(ontology_graph)
    has_domain_range = _has_domain_or_range_axioms(ontology_graph)
    has_negative = _has_negative_constructs(ontology_graph)
    has_role_axioms = _has_role_axioms(ontology_graph)
    has_targets = bool(target_classes)

    hierarchy_policy = _normalize_policy(materialize_hierarchy, materialize_hierarchy_policy, default="auto")
    atomic_domain_range_policy = _normalize_policy(
        materialize_atomic_domain_range,
        materialize_atomic_domain_range_policy,
        default="off",
    )
    horn_domain_range_policy = _normalize_policy(
        materialize_horn_safe_domain_range,
        materialize_horn_safe_domain_range_policy,
        default="auto",
    )
    target_roles_policy = _normalize_policy(
        materialize_target_roles,
        materialize_target_roles_policy,
        default="auto",
    )
    augment_domain_range_policy = _normalize_policy(
        augment_property_domain_range,
        augment_property_domain_range_policy,
        default="auto",
    )

    if hierarchy_policy == "on":
        hierarchy_decision = PreprocessingPassDecision(
            name="materialize_hierarchy",
            policy=hierarchy_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif hierarchy_policy == "off":
        hierarchy_decision = PreprocessingPassDecision(
            name="materialize_hierarchy",
            policy=hierarchy_policy,
            enabled=False,
            reason="explicitly disabled",
        )
    else:
        hierarchy_decision = PreprocessingPassDecision(
            name="materialize_hierarchy",
            policy=hierarchy_policy,
            enabled=has_hierarchy,
            reason=(
                "auto: hierarchy axioms detected"
                if has_hierarchy
                else "auto: no subclass/subproperty axioms detected"
            ),
        )

    if horn_domain_range_policy == "on":
        horn_domain_range_decision = PreprocessingPassDecision(
            name="materialize_horn_safe_domain_range",
            policy=horn_domain_range_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif horn_domain_range_policy == "off":
        horn_domain_range_decision = PreprocessingPassDecision(
            name="materialize_horn_safe_domain_range",
            policy=horn_domain_range_policy,
            enabled=False,
            reason="explicitly disabled",
        )
    else:
        enable_horn_domain_range = has_domain_range
        reason = "auto: no domain/range axioms detected"
        if enable_horn_domain_range:
            if has_negative:
                reason = "auto: domain/range axioms plus negative constraints detected"
            else:
                reason = "auto: domain/range axioms detected"
        horn_domain_range_decision = PreprocessingPassDecision(
            name="materialize_horn_safe_domain_range",
            policy=horn_domain_range_policy,
            enabled=enable_horn_domain_range,
            reason=reason,
        )

    if horn_domain_range_decision.enabled:
        atomic_domain_range_decision = PreprocessingPassDecision(
            name="materialize_atomic_domain_range",
            policy=atomic_domain_range_policy,
            enabled=False,
            reason="suppressed because Horn-safe domain/range materialization is enabled",
        )
    elif atomic_domain_range_policy == "on":
        atomic_domain_range_decision = PreprocessingPassDecision(
            name="materialize_atomic_domain_range",
            policy=atomic_domain_range_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif atomic_domain_range_policy == "off":
        atomic_domain_range_decision = PreprocessingPassDecision(
            name="materialize_atomic_domain_range",
            policy=atomic_domain_range_policy,
            enabled=False,
            reason="explicitly disabled",
        )
    else:
        atomic_domain_range_decision = PreprocessingPassDecision(
            name="materialize_atomic_domain_range",
            policy=atomic_domain_range_policy,
            enabled=False,
            reason="auto: prefer Horn-safe domain/range materialization when domain/range axioms are present",
        )

    if target_roles_policy == "on":
        target_roles_decision = PreprocessingPassDecision(
            name="materialize_target_roles",
            policy=target_roles_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif target_roles_policy == "off":
        target_roles_decision = PreprocessingPassDecision(
            name="materialize_target_roles",
            policy=target_roles_policy,
            enabled=False,
            reason="explicitly disabled",
        )
    else:
        enabled = bool(has_targets and has_role_axioms)
        if not has_targets:
            reason = "auto: no target classes supplied"
        elif has_role_axioms:
            reason = "auto: relevant role axioms may affect target queries"
        else:
            reason = "auto: no role axioms detected"
        target_roles_decision = PreprocessingPassDecision(
            name="materialize_target_roles",
            policy=target_roles_policy,
            enabled=enabled,
            reason=reason,
        )

    if augment_domain_range_policy == "on":
        augment_domain_range_decision = PreprocessingPassDecision(
            name="augment_property_domain_range",
            policy=augment_domain_range_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif augment_domain_range_policy == "off":
        augment_domain_range_decision = PreprocessingPassDecision(
            name="augment_property_domain_range",
            policy=augment_domain_range_policy,
            enabled=False,
            reason="explicitly disabled",
        )
    else:
        enabled = has_domain_range and has_targets
        if not has_targets:
            reason = "auto: no target classes supplied"
        elif has_domain_range:
            reason = "auto: domain/range axioms detected"
        else:
            reason = "auto: no domain/range axioms detected"
        augment_domain_range_decision = PreprocessingPassDecision(
            name="augment_property_domain_range",
            policy=augment_domain_range_policy,
            enabled=enabled,
            reason=reason,
        )

    return PreprocessingPlan(
        materialize_hierarchy=hierarchy_decision,
        materialize_atomic_domain_range=atomic_domain_range_decision,
        materialize_horn_safe_domain_range=horn_domain_range_decision,
        materialize_target_roles=target_roles_decision,
        augment_property_domain_range=augment_domain_range_decision,
    )


def describe_preprocessing_plan(plan: Optional[PreprocessingPlan]) -> str:
    if plan is None:
        return "Preprocessing plan: (none)"

    decisions = [
        plan.materialize_hierarchy,
        plan.materialize_atomic_domain_range,
        plan.materialize_horn_safe_domain_range,
        plan.materialize_target_roles,
        plan.augment_property_domain_range,
    ]
    lines = ["Preprocessing plan:"]
    for decision in decisions:
        lines.append(
            f"  - {decision.name}: {'on' if decision.enabled else 'off'} "
            f"(policy={decision.policy}; {decision.reason})"
        )
    return "\n".join(lines)


def _term_sort_key(term: Identifier) -> Tuple[str, str]:
    if isinstance(term, URIRef):
        return ("U", str(term))
    if isinstance(term, BNode):
        return ("B", str(term))
    if isinstance(term, Literal):
        datatype = "" if term.datatype is None else str(term.datatype)
        language = "" if term.language is None else str(term.language)
        return ("L", f"{term}|{datatype}|{language}")
    return ("Z", str(term))


_NUMERIC_XSD_TYPES: set[URIRef] = {
    XSD.integer,
    XSD.int,
    XSD.long,
    XSD.short,
    XSD.byte,
    XSD.nonNegativeInteger,
    XSD.nonPositiveInteger,
    XSD.positiveInteger,
    XSD.negativeInteger,
    XSD.unsignedLong,
    XSD.unsignedInt,
    XSD.unsignedShort,
    XSD.unsignedByte,
    XSD.decimal,
    XSD.float,
    XSD.double,
}


def _is_datatype_term(term: Identifier) -> bool:
    if not isinstance(term, URIRef):
        return False
    if str(term).startswith(str(XSD)):
        return True
    return False


def _literal_to_float(value: Identifier | None) -> Optional[float]:
    if not isinstance(value, Literal):
        return None
    try:
        return float(value.toPython())
    except (TypeError, ValueError):
        return None


def guess_rdf_format(path: str | Path) -> Optional[str]:
    """
    Guess an rdflib parser format from the file extension.

    Returns None when rdflib should infer it.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".ttl":
        return "turtle"
    if suffix == ".nt":
        return "nt"
    if suffix in {".owl", ".rdf", ".xml"}:
        return "xml"
    if suffix == ".n3":
        return "n3"
    if suffix == ".jsonld":
        return "json-ld"
    if suffix == ".trig":
        return "trig"
    return None


def load_rdflib_graph(
    paths: str | Path | Sequence[str | Path],
    formats: Optional[Sequence[Optional[str]]] = None,
) -> Graph:
    """
    Load one or more RDF files into a single rdflib.Graph.
    """
    path_list = _ensure_sequence(paths)
    if formats is not None and len(formats) != len(path_list):
        raise ValueError("formats must match the number of input paths.")

    graph = Graph()
    for i, path in enumerate(path_list):
        fmt = None if formats is None else formats[i]
        graph.parse(path, format=fmt or guess_rdf_format(path))
    return graph


def merge_rdflib_graphs(graphs: Sequence[Graph]) -> Graph:
    merged = Graph()
    for graph in graphs:
        for triple in graph:
            merged.add(triple)
    return merged


def _copy_graph(graph: Graph) -> Graph:
    copied = Graph()
    for triple in graph:
        copied.add(triple)
    return copied


def _compute_transitive_super_map(graph: Graph, predicate: URIRef) -> Dict[URIRef, set[URIRef]]:
    direct_supers: Dict[URIRef, set[URIRef]] = defaultdict(set)
    nodes: set[URIRef] = set()

    for child, _pred, parent in graph.triples((None, predicate, None)):
        if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
            continue
        direct_supers[child].add(parent)
        nodes.add(child)
        nodes.add(parent)

    closure: Dict[URIRef, set[URIRef]] = {}
    for node in nodes:
        seen = {node}
        supers: set[URIRef] = set()
        stack = list(direct_supers.get(node, ()))
        while stack:
            parent = stack.pop()
            if parent in seen:
                continue
            seen.add(parent)
            supers.add(parent)
            stack.extend(direct_supers.get(parent, ()))
        closure[node] = supers

    return closure


def materialize_hierarchy_closure(
    ontology_graph: Graph,
    data_graph: Graph,
) -> Graph:
    """
    Materialize simple hierarchy entailments into the instance graph:
    - rdf:type propagation over rdfs:subClassOf
    - property propagation over rdfs:subPropertyOf
    """

    subclass_supers = _compute_transitive_super_map(ontology_graph, RDFS.subClassOf)
    subproperty_supers = _compute_transitive_super_map(ontology_graph, RDFS.subPropertyOf)

    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    for subj, pred, obj in data_graph:
        if pred == RDF.type and isinstance(obj, URIRef):
            for super_class in subclass_supers.get(obj, ()):
                materialized.add((subj, RDF.type, super_class))
        elif isinstance(pred, URIRef):
            for super_prop in subproperty_supers.get(pred, ()):
                materialized.add((subj, super_prop, obj))

    return materialized


def materialize_class_hierarchy_closure(
    ontology_graph: Graph,
    data_graph: Graph,
) -> Graph:
    """
    Materialize only rdf:type propagation over rdfs:subClassOf.
    """

    subclass_supers = _compute_transitive_super_map(ontology_graph, RDFS.subClassOf)

    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    for subj, pred, obj in data_graph:
        if pred != RDF.type or not isinstance(obj, URIRef):
            continue
        for super_class in subclass_supers.get(obj, ()):
            materialized.add((subj, RDF.type, super_class))

    return materialized


def materialize_atomic_domain_range_closure(
    ontology_graph: Graph,
    data_graph: Graph,
) -> Graph:
    """
    Materialize only Horn-safe atomic domain/range consequences:
    - (s P o) and domain(P, C) -> (s rdf:type C)
    - (s P o) and range(P, C) -> (o rdf:type C)

    Complex expressions such as unions are intentionally left to the query-time
    augmentation path.
    """

    property_axioms = collect_property_expression_axioms(ontology_graph)
    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    for subj, pred, obj in data_graph:
        if pred == RDF.type or not isinstance(pred, URIRef):
            continue

        axioms = property_axioms.get(pred)
        if axioms is None:
            continue

        for domain_expr in axioms.domain_expressions:
            if isinstance(domain_expr, URIRef) and not _is_datatype_term(domain_expr):
                materialized.add((subj, RDF.type, domain_expr))

        if isinstance(obj, Literal):
            continue

        for range_expr in axioms.range_expressions:
            if isinstance(range_expr, URIRef) and not _is_datatype_term(range_expr):
                materialized.add((obj, RDF.type, range_expr))

    return materialized


def _extract_horn_safe_named_class_consequents(
    ontology_graph: Graph,
    expr: Identifier,
) -> Optional[List[URIRef]]:
    if isinstance(expr, URIRef):
        if _is_datatype_term(expr):
            return None
        return [expr]

    if not isinstance(expr, BNode):
        return None

    members: List[URIRef] = []

    intersection_head = ontology_graph.value(expr, OWL.intersectionOf)
    if isinstance(intersection_head, BNode):
        for member in Collection(ontology_graph, intersection_head):
            sub_terms = _extract_horn_safe_named_class_consequents(ontology_graph, member)
            if sub_terms is None:
                return None
            members.extend(sub_terms)
        deduped = list(dict.fromkeys(members))
        return deduped if deduped else None

    return None


def materialize_horn_safe_domain_range_closure(
    ontology_graph: Graph,
    data_graph: Graph,
) -> Graph:
    """
    Materialize Horn-safe domain/range consequences into rdf:type assertions.

    Supported consequents:
    - atomic named classes
    - intersections of atomic named classes

    Unsupported consequents such as unions, complements, restrictions, and
    datatype expressions are intentionally ignored here and left to the
    query-time augmentation path.
    """

    property_axioms = collect_property_expression_axioms(ontology_graph)
    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    for subj, pred, obj in data_graph:
        if pred == RDF.type or not isinstance(pred, URIRef):
            continue

        axioms = property_axioms.get(pred)
        if axioms is None:
            continue

        for domain_expr in axioms.domain_expressions:
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, domain_expr)
            if consequents is None:
                continue
            for class_term in consequents:
                materialized.add((subj, RDF.type, class_term))

        if isinstance(obj, Literal):
            continue

        for range_expr in axioms.range_expressions:
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, range_expr)
            if consequents is None:
                continue
            for class_term in consequents:
                materialized.add((obj, RDF.type, class_term))

    return materialized


def _collect_direct_subproperties(graph: Graph) -> Dict[URIRef, set[URIRef]]:
    super_to_subs: Dict[URIRef, set[URIRef]] = defaultdict(set)
    for child, _pred, parent in graph.triples((None, RDFS.subPropertyOf, None)):
        if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
            continue
        super_to_subs[parent].add(child)
    return super_to_subs


def _collect_inverse_properties(graph: Graph) -> Dict[URIRef, set[URIRef]]:
    inverses: Dict[URIRef, set[URIRef]] = defaultdict(set)
    for left, _pred, right in graph.triples((None, OWL.inverseOf, None)):
        if not isinstance(left, URIRef) or not isinstance(right, URIRef):
            continue
        inverses[left].add(right)
        inverses[right].add(left)
    return inverses


def _collect_property_chains(graph: Graph) -> Dict[URIRef, List[Tuple[URIRef, ...]]]:
    chains: Dict[URIRef, List[Tuple[URIRef, ...]]] = defaultdict(list)
    for conclusion, _pred, list_head in graph.triples((None, OWL.propertyChainAxiom, None)):
        if not isinstance(conclusion, URIRef) or not isinstance(list_head, BNode):
            continue
        members: List[URIRef] = []
        for member in Collection(graph, list_head):
            if not isinstance(member, URIRef):
                members = []
                break
            members.append(member)
        if len(members) >= 2:
            chains[conclusion].append(tuple(members))
    return chains


def _collect_transitive_properties(graph: Graph) -> set[URIRef]:
    transitive_props: set[URIRef] = set()
    for prop, _pred, obj in graph.triples((None, RDF.type, OWL.TransitiveProperty)):
        if isinstance(prop, URIRef):
            transitive_props.add(prop)
    return transitive_props


def _collect_target_root_expressions(
    ontology_graph: Graph,
    target_term: URIRef,
) -> List[Identifier]:
    root_exprs = list(ontology_graph.objects(target_term, RDFS.subClassOf))
    root_exprs.extend(expr for expr in ontology_graph.objects(target_term, OWL.equivalentClass) if expr != target_term)
    root_exprs.extend(ontology_graph.objects(target_term, OWL.intersectionOf))
    root_exprs.extend(ontology_graph.objects(target_term, OWL.disjointWith))
    root_exprs.extend(
        expr for expr in ontology_graph.subjects(OWL.disjointWith, target_term) if expr != target_term
    )
    return root_exprs


def collect_referenced_named_classes_for_class(
    ontology_graph: Graph,
    target_class: str | URIRef,
) -> List[URIRef]:
    """
    Collect named classes reachable from the target class's query surface.

    This follows the same compiler-visible surfaces as the current DAG path and
    recursively expands through referenced named-class definitions. The result
    is useful for targeted helper materialization.
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    visited: set[Identifier] = set()
    class_terms: set[URIRef] = set()

    def visit_expr(expr: Identifier) -> None:
        if expr in visited:
            return
        visited.add(expr)

        if isinstance(expr, URIRef) and not _is_datatype_term(expr):
            class_terms.add(expr)

        for head in ontology_graph.objects(expr, OWL.intersectionOf):
            if not isinstance(head, BNode):
                continue
            for member in Collection(ontology_graph, head):
                visit_expr(member)

        for head in ontology_graph.objects(expr, OWL.unionOf):
            if not isinstance(head, BNode):
                continue
            for member in Collection(ontology_graph, head):
                visit_expr(member)

        for filler in ontology_graph.objects(expr, OWL.complementOf):
            visit_expr(filler)

        restriction_types = set(ontology_graph.objects(expr, RDF.type))
        if OWL.Restriction in restriction_types:
            some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
            all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
            on_class = ontology_graph.value(expr, OWL.onClass)
            if some_filler is not None:
                visit_expr(some_filler)
            if all_filler is not None:
                visit_expr(all_filler)
            if on_class is not None:
                visit_expr(on_class)

        if isinstance(expr, (URIRef, BNode)):
            for sub_expr in ontology_graph.objects(expr, RDFS.subClassOf):
                visit_expr(sub_expr)
            for eq_expr in ontology_graph.objects(expr, OWL.equivalentClass):
                if eq_expr != expr:
                    visit_expr(eq_expr)
            for head in ontology_graph.objects(expr, OWL.intersectionOf):
                if not isinstance(head, BNode):
                    continue
                for member in Collection(ontology_graph, head):
                    visit_expr(member)
            for head in ontology_graph.objects(expr, OWL.unionOf):
                if not isinstance(head, BNode):
                    continue
                for member in Collection(ontology_graph, head):
                    visit_expr(member)
            for disjoint_expr in ontology_graph.objects(expr, OWL.disjointWith):
                visit_expr(disjoint_expr)
            for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, expr):
                if disjoint_expr != expr:
                    visit_expr(disjoint_expr)

    root_exprs = _collect_target_root_expressions(ontology_graph, target_term)
    if not root_exprs:
        visit_expr(target_term)
    else:
        class_terms.add(target_term)
        for expr in root_exprs:
            visit_expr(expr)

    return sorted(class_terms, key=str)


def collect_referenced_named_classes_for_classes(
    ontology_graph: Graph,
    target_classes: Sequence[str | URIRef],
) -> List[URIRef]:
    class_terms: set[URIRef] = set()
    for target_class in target_classes:
        class_terms.update(collect_referenced_named_classes_for_class(ontology_graph, target_class))
    return sorted(class_terms, key=str)


def collect_referenced_properties_for_class(
    ontology_graph: Graph,
    target_class: str | URIRef,
) -> List[URIRef]:
    """
    Collect the properties directly referenced by a target class query.

    This follows the same definitional surfaces the current compiler uses:
    subclass/equivalent/intersection/complement/disjoint expressions and
    property restrictions reachable from the target class.
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    visited: set[Identifier] = set()
    props: set[URIRef] = set()

    def visit_expr(expr: Identifier) -> None:
        if expr in visited:
            return
        visited.add(expr)

        for head in ontology_graph.objects(expr, OWL.intersectionOf):
            if not isinstance(head, BNode):
                continue
            for member in Collection(ontology_graph, head):
                visit_expr(member)

        for head in ontology_graph.objects(expr, OWL.unionOf):
            if not isinstance(head, BNode):
                continue
            for member in Collection(ontology_graph, head):
                visit_expr(member)

        for filler in ontology_graph.objects(expr, OWL.complementOf):
            visit_expr(filler)

        restriction_types = set(ontology_graph.objects(expr, RDF.type))
        if OWL.Restriction in restriction_types:
            prop_expr = ontology_graph.value(expr, OWL.onProperty)
            prop, prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
            if isinstance(prop, URIRef):
                props.add(prop)
            some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
            all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
            if some_filler is not None:
                visit_expr(some_filler)
            if all_filler is not None:
                visit_expr(all_filler)

        if isinstance(expr, (URIRef, BNode)):
            for sub_expr in ontology_graph.objects(expr, RDFS.subClassOf):
                visit_expr(sub_expr)
            for eq_expr in ontology_graph.objects(expr, OWL.equivalentClass):
                if eq_expr != expr:
                    visit_expr(eq_expr)
            for head in ontology_graph.objects(expr, OWL.intersectionOf):
                if not isinstance(head, BNode):
                    continue
                for member in Collection(ontology_graph, head):
                    visit_expr(member)
            for head in ontology_graph.objects(expr, OWL.unionOf):
                if not isinstance(head, BNode):
                    continue
                for member in Collection(ontology_graph, head):
                    visit_expr(member)
            for disjoint_expr in ontology_graph.objects(expr, OWL.disjointWith):
                visit_expr(disjoint_expr)
            for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, expr):
                if disjoint_expr != expr:
                    visit_expr(disjoint_expr)

    root_exprs = _collect_target_root_expressions(ontology_graph, target_term)

    if not root_exprs:
        visit_expr(target_term)
    else:
        for expr in root_exprs:
            visit_expr(expr)

    return sorted(props, key=str)


def collect_referenced_properties_for_classes(
    ontology_graph: Graph,
    target_classes: Sequence[str | URIRef],
) -> List[URIRef]:
    props: set[URIRef] = set()
    for target_class in target_classes:
        props.update(collect_referenced_properties_for_class(ontology_graph, target_class))
    return sorted(props, key=str)


def compute_target_dependency_closure(
    ontology_graph: Graph,
    mapping: RDFKGraphMapping,
    target_classes: Sequence[str | URIRef],
) -> TargetDependencyClosure:
    reports = [
        build_dag_dependency_report(ontology_graph, mapping, target_class)
        for target_class in target_classes
    ]
    target_terms = sorted({report.target_class for report in reports}, key=str)
    referenced_named_classes = sorted(
        {
            class_term
            for report in reports
            for class_term in report.referenced_named_classes
        },
        key=str,
    )
    inferable_named_classes = sorted(
        {
            class_term
            for report in reports
            for class_term in report.inferable_named_classes
        },
        key=str,
    )
    referenced_properties = sorted(
        {
            prop
            for report in reports
            for prop in report.referenced_properties
        },
        key=str,
    )
    return TargetDependencyClosure(
        target_classes=target_terms,
        referenced_named_classes=referenced_named_classes,
        inferable_named_classes=inferable_named_classes,
        referenced_properties=referenced_properties,
    )


def compute_relevant_role_property_closure(
    ontology_graph: Graph,
    seed_properties: Sequence[URIRef],
) -> List[URIRef]:
    """
    Compute the backwards dependency closure of properties needed to derive the
    queried role constraints.

    If the query references property R, then the closure includes:
    - subproperties P with P ⊑ R
    - inverse-linked properties that can derive R
    - chain antecedents for any chain concluding in R
    - recursively, the dependencies of all of the above
    """

    super_to_subs = _collect_direct_subproperties(ontology_graph)
    inverses = _collect_inverse_properties(ontology_graph)
    chains_by_conclusion = _collect_property_chains(ontology_graph)
    transitive_props = _collect_transitive_properties(ontology_graph)

    relevant: set[URIRef] = {prop for prop in seed_properties if isinstance(prop, URIRef)}
    queue: List[URIRef] = list(relevant)

    while queue:
        prop = queue.pop()

        for child in super_to_subs.get(prop, ()):
            if child not in relevant:
                relevant.add(child)
                queue.append(child)

        for inverse_prop in inverses.get(prop, ()):
            if inverse_prop not in relevant:
                relevant.add(inverse_prop)
                queue.append(inverse_prop)

        for chain in chains_by_conclusion.get(prop, ()):
            for chain_prop in chain:
                if chain_prop not in relevant:
                    relevant.add(chain_prop)
                    queue.append(chain_prop)

        if prop in transitive_props and prop not in relevant:
            relevant.add(prop)
            queue.append(prop)

    return sorted(relevant, key=str)


def _edges_to_forward_index(
    edges: Iterable[Tuple[Identifier, Identifier]],
) -> Dict[Identifier, set[Identifier]]:
    forward: Dict[Identifier, set[Identifier]] = defaultdict(set)
    for subj, obj in edges:
        forward[subj].add(obj)
    return forward


def _compose_path_edge_maps(
    edge_maps: Sequence[Dict[Identifier, set[Identifier]]],
) -> set[Tuple[Identifier, Identifier]]:
    if not edge_maps:
        return set()

    current_pairs: set[Tuple[Identifier, Identifier]] = set()
    for subj, dsts in edge_maps[0].items():
        for dst in dsts:
            current_pairs.add((subj, dst))

    for edge_map in edge_maps[1:]:
        if not current_pairs:
            return set()
        next_pairs: set[Tuple[Identifier, Identifier]] = set()
        for start, mid in current_pairs:
            for dst in edge_map.get(mid, ()):
                next_pairs.add((start, dst))
        current_pairs = next_pairs

    return current_pairs


def materialize_target_role_closure(
    ontology_graph: Graph,
    data_graph: Graph,
    *,
    seed_properties: Optional[Sequence[URIRef]] = None,
    target_classes: Sequence[str | URIRef],
    max_iterations: int = 20,
) -> RoleSaturationResult:
    """
    Materialize only the role closure relevant to the queried class set.

    Supported role axioms:
    - `rdfs:subPropertyOf`
    - `owl:inverseOf`
    - `owl:propertyChainAxiom`
    - `owl:TransitiveProperty`
    """

    if seed_properties is None:
        seed_properties = collect_referenced_properties_for_classes(ontology_graph, target_classes)
    relevant_properties = compute_relevant_role_property_closure(ontology_graph, seed_properties)

    materialized_graph = _copy_graph(data_graph)
    if not relevant_properties:
        return RoleSaturationResult(
            data_graph=materialized_graph,
            seed_properties=[],
            relevant_properties=[],
            inferred_edges=[],
            iterations=0,
        )

    relevant_set = set(relevant_properties)
    direct_supers: Dict[URIRef, set[URIRef]] = defaultdict(set)
    for child, _pred, parent in ontology_graph.triples((None, RDFS.subPropertyOf, None)):
        if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
            continue
        if child in relevant_set and parent in relevant_set:
            direct_supers[child].add(parent)

    inverses = _collect_inverse_properties(ontology_graph)
    filtered_inverses = {
        prop: {inv for inv in inverse_props if inv in relevant_set}
        for prop, inverse_props in inverses.items()
        if prop in relevant_set
    }

    chains_by_conclusion = _collect_property_chains(ontology_graph)
    transitive_props = _collect_transitive_properties(ontology_graph)
    filtered_chains: Dict[URIRef, List[Tuple[URIRef, ...]]] = defaultdict(list)

    for conclusion, chains in chains_by_conclusion.items():
        if conclusion not in relevant_set:
            continue
        for chain in chains:
            if all(prop in relevant_set for prop in chain):
                filtered_chains[conclusion].append(chain)

    for prop in transitive_props:
        if prop in relevant_set:
            filtered_chains[prop].append((prop, prop))

    known_edges: Dict[URIRef, set[Tuple[Identifier, Identifier]]] = {
        prop: set() for prop in relevant_properties
    }
    for subj, pred, obj in data_graph:
        if pred == RDF.type or not isinstance(pred, URIRef):
            continue
        if pred not in relevant_set or isinstance(obj, Literal):
            continue
        known_edges[pred].add((subj, obj))

    delta_edges: Dict[URIRef, set[Tuple[Identifier, Identifier]]] = {
        prop: set(edges) for prop, edges in known_edges.items()
    }
    inferred_edges: List[Tuple[Identifier, URIRef, Identifier]] = []
    iterations = 0

    while iterations < max_iterations and any(delta_edges[prop] for prop in relevant_properties):
        iterations += 1
        known_forward = {
            prop: _edges_to_forward_index(edges)
            for prop, edges in known_edges.items()
            if edges
        }
        delta_forward = {
            prop: _edges_to_forward_index(edges)
            for prop, edges in delta_edges.items()
            if edges
        }
        additions: Dict[URIRef, set[Tuple[Identifier, Identifier]]] = {
            prop: set() for prop in relevant_properties
        }

        for child, supers in direct_supers.items():
            child_delta = delta_edges.get(child, set())
            if not child_delta:
                continue
            for super_prop in supers:
                additions[super_prop].update(child_delta)

        for prop, inverse_props in filtered_inverses.items():
            prop_delta = delta_edges.get(prop, set())
            if not prop_delta:
                continue
            reversed_edges = {(obj, subj) for subj, obj in prop_delta}
            for inverse_prop in inverse_props:
                additions[inverse_prop].update(reversed_edges)

        for conclusion, chains in filtered_chains.items():
            for chain in chains:
                for idx, chain_prop in enumerate(chain):
                    if not delta_edges.get(chain_prop):
                        continue
                    edge_maps: List[Dict[Identifier, set[Identifier]]] = []
                    valid = True
                    for chain_idx, prop in enumerate(chain):
                        forward = (
                            delta_forward.get(prop)
                            if chain_idx == idx
                            else known_forward.get(prop)
                        )
                        if not forward:
                            valid = False
                            break
                        edge_maps.append(forward)
                    if not valid:
                        continue
                    additions[conclusion].update(_compose_path_edge_maps(edge_maps))

        new_delta: Dict[URIRef, set[Tuple[Identifier, Identifier]]] = {
            prop: set() for prop in relevant_properties
        }
        for prop, candidates in additions.items():
            truly_new = candidates - known_edges[prop]
            if not truly_new:
                continue
            known_edges[prop].update(truly_new)
            new_delta[prop] = truly_new
            for subj, obj in sorted(truly_new, key=lambda pair: (str(pair[0]), str(pair[1]))):
                materialized_graph.add((subj, prop, obj))
                inferred_edges.append((subj, prop, obj))

        delta_edges = new_delta

    return RoleSaturationResult(
        data_graph=materialized_graph,
        seed_properties=sorted(seed_properties, key=str),
        relevant_properties=relevant_properties,
        inferred_edges=inferred_edges,
        iterations=iterations,
    )


def _collect_named_class_terms(graph: Graph) -> set[Identifier]:
    class_terms: set[Identifier] = set()

    for subj, _pred, obj in graph.triples((None, RDF.type, None)):
        if isinstance(obj, URIRef):
            class_terms.add(obj)
        if obj in {OWL.Class, RDFS.Class}:
            if isinstance(subj, URIRef):
                class_terms.add(subj)

    for subj, _pred, obj in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(subj, URIRef):
            class_terms.add(subj)
        if isinstance(obj, URIRef) and not _is_datatype_term(obj):
            class_terms.add(obj)

    for subj, _pred, obj in graph.triples((None, OWL.equivalentClass, None)):
        if isinstance(subj, URIRef):
            class_terms.add(subj)
        if isinstance(obj, URIRef) and not _is_datatype_term(obj):
            class_terms.add(obj)

    for subj, _pred, obj in graph.triples((None, OWL.complementOf, None)):
        if isinstance(subj, URIRef):
            class_terms.add(subj)
        if isinstance(obj, URIRef) and not _is_datatype_term(obj):
            class_terms.add(obj)

    for subj, _pred, obj in graph.triples((None, OWL.disjointWith, None)):
        if isinstance(subj, URIRef):
            class_terms.add(subj)
        if isinstance(obj, URIRef) and not _is_datatype_term(obj):
            class_terms.add(obj)

    for subj, _pred, obj in graph.triples((None, OWL.someValuesFrom, None)):
        if isinstance(subj, URIRef):
            class_terms.add(subj)
        if isinstance(obj, URIRef) and not _is_datatype_term(obj):
            class_terms.add(obj)

    for subj, _pred, obj in graph.triples((None, OWL.allValuesFrom, None)):
        if isinstance(subj, URIRef):
            class_terms.add(subj)
        if isinstance(obj, URIRef) and not _is_datatype_term(obj):
            class_terms.add(obj)

    for subj, _pred, obj in graph.triples((None, OWL.onClass, None)):
        if isinstance(subj, URIRef):
            class_terms.add(subj)
        if isinstance(obj, URIRef) and not _is_datatype_term(obj):
            class_terms.add(obj)

    for _subj, _pred, obj in graph.triples((None, OWL.intersectionOf, None)):
        if isinstance(obj, BNode):
            for member in Collection(graph, obj):
                if isinstance(member, URIRef):
                    class_terms.add(member)

    for _subj, _pred, obj in graph.triples((None, OWL.unionOf, None)):
        if isinstance(obj, BNode):
            for member in Collection(graph, obj):
                if isinstance(member, URIRef) and not _is_datatype_term(member):
                    class_terms.add(member)

    return class_terms


def _collect_datatype_terms(graph: Graph) -> set[URIRef]:
    datatype_terms: set[URIRef] = set()

    for _subj, _pred, obj in graph.triples((None, RDF.type, RDFS.Datatype)):
        if isinstance(_subj, URIRef):
            datatype_terms.add(_subj)

    for _subj, _pred, obj in graph.triples((None, OWL.onDatatype, None)):
        if isinstance(obj, URIRef):
            datatype_terms.add(obj)

    for _subj, _pred, obj in graph.triples((None, OWL.someValuesFrom, None)):
        if _is_datatype_term(obj):
            datatype_terms.add(obj)

    for _subj, _pred, obj in graph.triples((None, OWL.allValuesFrom, None)):
        if _is_datatype_term(obj):
            datatype_terms.add(obj)

    for _subj, _pred, obj in graph:
        if isinstance(obj, Literal) and isinstance(obj.datatype, URIRef):
            datatype_terms.add(obj.datatype)

    return datatype_terms


def _collect_property_terms(graph: Graph) -> set[URIRef]:
    prop_terms: set[URIRef] = set()

    for _subj, pred, _obj in graph:
        if isinstance(pred, URIRef) and pred != RDF.type:
            prop_terms.add(pred)

    for subj, _pred, obj in graph.triples((None, RDF.type, None)):
        if obj in {OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property} and isinstance(
            subj, URIRef
        ):
            prop_terms.add(subj)

    for subj, _pred, obj in graph.triples((None, RDFS.subPropertyOf, None)):
        if isinstance(subj, URIRef):
            prop_terms.add(subj)
        if isinstance(obj, URIRef):
            prop_terms.add(obj)

    for _subj, _pred, obj in graph.triples((None, OWL.onProperty, None)):
        if isinstance(obj, URIRef):
            prop_terms.add(obj)

    return prop_terms


def collect_property_expression_axioms(
    ontology_graph: Graph,
) -> Dict[URIRef, PropertyExpressionAxioms]:
    property_terms = _collect_property_terms(ontology_graph)
    axioms: Dict[URIRef, PropertyExpressionAxioms] = {}

    for prop in sorted(property_terms, key=str):
        domain_exprs = sorted(
            {expr for expr in ontology_graph.objects(prop, RDFS.domain)},
            key=_term_sort_key,
        )
        range_exprs = sorted(
            {expr for expr in ontology_graph.objects(prop, RDFS.range)},
            key=_term_sort_key,
        )
        if not domain_exprs and not range_exprs:
            continue
        axioms[prop] = PropertyExpressionAxioms(
            property_term=prop,
            domain_expressions=domain_exprs,
            range_expressions=range_exprs,
        )

    return axioms


def describe_property_expression_axioms(
    ontology_graph: Graph,
    axioms: PropertyExpressionAxioms,
) -> str:
    lines = [f"property={axioms.property_term.n3()}"]
    if axioms.domain_expressions:
        lines.append(
            "domains="
            + "["
            + ", ".join(describe_owl_expression(ontology_graph, expr) for expr in axioms.domain_expressions)
            + "]"
        )
    else:
        lines.append("domains=[]")
    if axioms.range_expressions:
        lines.append(
            "ranges="
            + "["
            + ", ".join(describe_owl_expression(ontology_graph, expr) for expr in axioms.range_expressions)
            + "]"
        )
    else:
        lines.append("ranges=[]")
    return "\n".join(lines)


def rdflib_graph_to_kgraph(
    graph: Graph,
    *,
    vocab_graph: Optional[Graph] = None,
    include_literals: bool = True,
    include_type_edges: bool = False,
) -> tuple[KGraph, RDFKGraphMapping]:
    """
    Convert an rdflib.Graph into a KGraph.

    Conventions:
    - `rdf:type` triples populate the node type matrix.
    - Other predicates become KGraph properties.
    - Literals can optionally be lifted into graph nodes so datatype-property
      edges are preserved structurally.
    - `rdf:type` edges are excluded from the property adjacency by default,
      because the engine already models them in `node_types`.
    """
    node_terms_set: set[Identifier] = set()
    vocab_source = graph if vocab_graph is None else merge_rdflib_graphs((vocab_graph, graph))
    prop_terms_set: set[URIRef] = _collect_property_terms(vocab_source)
    class_terms_set: set[Identifier] = _collect_named_class_terms(vocab_source)
    datatype_terms_set: set[URIRef] = _collect_datatype_terms(vocab_source)

    class_assertions: Dict[Identifier, set[Identifier]] = {}
    edge_buckets: Dict[URIRef, List[Tuple[Identifier, Identifier]]] = {}

    for subj, pred, obj in graph:
        node_terms_set.add(subj)

        if pred == RDF.type:
            class_terms_set.add(obj)
            class_assertions.setdefault(subj, set()).add(obj)
            if include_type_edges and not isinstance(obj, Literal):
                node_terms_set.add(obj)
                edge_buckets.setdefault(pred, []).append((subj, obj))
                prop_terms_set.add(pred)
            continue

        prop_terms_set.add(pred)

        if isinstance(obj, Literal) and not include_literals:
            continue

        node_terms_set.add(obj)
        edge_buckets.setdefault(pred, []).append((subj, obj))

    node_terms = sorted(node_terms_set, key=_term_sort_key)
    prop_terms = sorted(prop_terms_set, key=_term_sort_key)
    class_terms = sorted(class_terms_set, key=_term_sort_key)
    datatype_terms = sorted(datatype_terms_set, key=str)

    node_to_idx = {term: idx for idx, term in enumerate(node_terms)}
    prop_to_idx = {term: idx for idx, term in enumerate(prop_terms)}
    class_to_idx = {term: idx for idx, term in enumerate(class_terms)}
    datatype_to_idx = {term: idx for idx, term in enumerate(datatype_terms)}

    num_nodes = len(node_terms)
    num_classes = len(class_terms)
    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    literal_datatype_idx = torch.full((num_nodes,), -1, dtype=torch.int64)
    literal_numeric_value = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
    for subj, asserted_classes in class_assertions.items():
        subj_idx = node_to_idx.get(subj)
        if subj_idx is None:
            continue
        for class_term in asserted_classes:
            class_idx = class_to_idx[class_term]
            node_types[subj_idx, class_idx] = 1.0

    for node_idx, term in enumerate(node_terms):
        if not isinstance(term, Literal):
            continue
        if isinstance(term.datatype, URIRef) and term.datatype in datatype_to_idx:
            literal_datatype_idx[node_idx] = datatype_to_idx[term.datatype]
        numeric_value = None
        if isinstance(term.datatype, URIRef) and term.datatype in _NUMERIC_XSD_TYPES:
            numeric_value = _literal_to_float(term)
        elif term.datatype is None:
            numeric_value = _literal_to_float(term)
        if numeric_value is not None:
            literal_numeric_value[node_idx] = float(numeric_value)

    offsets_p: List[torch.Tensor] = []
    neighbors_p: List[torch.Tensor] = []
    for prop_term in prop_terms:
        src_to_dst: Dict[int, List[int]] = {}
        for subj, obj in edge_buckets.get(prop_term, []):
            src_idx = node_to_idx[subj]
            dst_idx = node_to_idx[obj]
            src_to_dst.setdefault(src_idx, []).append(dst_idx)

        offsets = torch.zeros(num_nodes + 1, dtype=torch.int32)
        all_neighbors: List[int] = []

        for src_idx in range(num_nodes):
            dsts = src_to_dst.get(src_idx, [])
            if dsts:
                dsts = sorted(set(dsts))
                all_neighbors.extend(dsts)
            offsets[src_idx + 1] = len(all_neighbors)

        neighbors = torch.tensor(all_neighbors, dtype=torch.int32)
        offsets_p.append(offsets)
        neighbors_p.append(neighbors)

    mapping = RDFKGraphMapping(
        node_terms=node_terms,
        prop_terms=prop_terms,
        class_terms=class_terms,
        datatype_terms=datatype_terms,
        node_to_idx=node_to_idx,
        prop_to_idx=prop_to_idx,
        class_to_idx=class_to_idx,
        datatype_to_idx=datatype_to_idx,
    )

    return (
        KGraph(
            num_nodes=num_nodes,
            offsets_p=offsets_p,
            neighbors_p=neighbors_p,
            node_types=node_types,
            literal_datatype_idx=literal_datatype_idx,
            literal_numeric_value=literal_numeric_value,
        ),
        mapping,
    )


def load_kgraph_from_rdf(
    paths: str | Path | Sequence[str | Path],
    *,
    formats: Optional[Sequence[Optional[str]]] = None,
    include_literals: bool = True,
    include_type_edges: bool = False,
) -> tuple[KGraph, RDFKGraphMapping, Graph]:
    """
    Convenience wrapper:
    - load one or more RDF files into an rdflib.Graph
    - convert that graph into a KGraph plus stable mappings
    """
    graph = load_rdflib_graph(paths, formats=formats)
    kg, mapping = rdflib_graph_to_kgraph(
        graph,
        vocab_graph=graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
    )
    return kg, mapping, graph


def load_reasoning_dataset(
    *,
    schema_paths: str | Path | Sequence[str | Path] | None = None,
    data_paths: str | Path | Sequence[str | Path] | None = None,
    schema_formats: Optional[Sequence[Optional[str]]] = None,
    data_formats: Optional[Sequence[Optional[str]]] = None,
    include_literals: bool = True,
    include_type_edges: bool = False,
    materialize_hierarchy: Optional[bool] = None,
    materialize_hierarchy_policy: str = "auto",
    materialize_atomic_domain_range: Optional[bool] = None,
    materialize_atomic_domain_range_policy: str = "off",
    materialize_horn_safe_domain_range: Optional[bool] = None,
    materialize_horn_safe_domain_range_policy: str = "auto",
    materialize_target_roles: Optional[bool] = None,
    materialize_target_roles_policy: str = "auto",
    target_classes: Optional[Sequence[str | URIRef]] = None,
    dependency_closure: Optional[TargetDependencyClosure] = None,
) -> ReasoningDataset:
    """
    Clean TBox/ABox workflow:
    - schema paths contribute ontology vocabulary and compilation axioms
    - data paths contribute the instance graph that becomes the KGraph
    """

    schema_graph = (
        load_rdflib_graph(schema_paths, formats=schema_formats)
        if schema_paths is not None
        else Graph()
    )
    data_graph = (
        load_rdflib_graph(data_paths, formats=data_formats)
        if data_paths is not None
        else Graph()
    )
    return build_reasoning_dataset_from_graphs(
        schema_graph=schema_graph,
        data_graph=data_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_hierarchy_policy=materialize_hierarchy_policy,
        materialize_atomic_domain_range=materialize_atomic_domain_range,
        materialize_atomic_domain_range_policy=materialize_atomic_domain_range_policy,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_horn_safe_domain_range_policy=materialize_horn_safe_domain_range_policy,
        materialize_target_roles=materialize_target_roles,
        materialize_target_roles_policy=materialize_target_roles_policy,
        target_classes=target_classes,
        dependency_closure=dependency_closure,
    )


def build_reasoning_dataset_from_graphs(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    include_literals: bool = True,
    include_type_edges: bool = False,
    materialize_hierarchy: Optional[bool] = None,
    materialize_hierarchy_policy: str = "auto",
    materialize_atomic_domain_range: Optional[bool] = None,
    materialize_atomic_domain_range_policy: str = "off",
    materialize_horn_safe_domain_range: Optional[bool] = None,
    materialize_horn_safe_domain_range_policy: str = "auto",
    materialize_target_roles: Optional[bool] = None,
    materialize_target_roles_policy: str = "auto",
    target_classes: Optional[Sequence[str | URIRef]] = None,
    dependency_closure: Optional[TargetDependencyClosure] = None,
) -> ReasoningDataset:
    ontology_graph = merge_rdflib_graphs((schema_graph, data_graph))
    preprocessing_timings = PreprocessingTimings()
    preprocessing_plan = plan_reasoning_preprocessing(
        ontology_graph,
        target_classes=target_classes,
        materialize_hierarchy=materialize_hierarchy,
        materialize_hierarchy_policy=materialize_hierarchy_policy,
        materialize_atomic_domain_range=materialize_atomic_domain_range,
        materialize_atomic_domain_range_policy=materialize_atomic_domain_range_policy,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_horn_safe_domain_range_policy=materialize_horn_safe_domain_range_policy,
        materialize_target_roles=materialize_target_roles,
        materialize_target_roles_policy=materialize_target_roles_policy,
    )

    effective_data_graph = _copy_graph(data_graph)
    if preprocessing_plan.materialize_hierarchy.enabled:
        t0 = perf_counter()
        if preprocessing_plan.materialize_target_roles.enabled and target_classes:
            effective_data_graph = materialize_class_hierarchy_closure(ontology_graph, effective_data_graph)
        else:
            effective_data_graph = materialize_hierarchy_closure(ontology_graph, effective_data_graph)
        preprocessing_timings.hierarchy_elapsed_ms += (perf_counter() - t0) * 1000.0
        ontology_graph = merge_rdflib_graphs((schema_graph, effective_data_graph))

    if (
        preprocessing_plan.materialize_atomic_domain_range.enabled
        or preprocessing_plan.materialize_horn_safe_domain_range.enabled
    ):
        if preprocessing_plan.materialize_horn_safe_domain_range.enabled:
            t0 = perf_counter()
            effective_data_graph = materialize_horn_safe_domain_range_closure(
                ontology_graph,
                effective_data_graph,
            )
            preprocessing_timings.horn_safe_domain_range_elapsed_ms += (perf_counter() - t0) * 1000.0
        else:
            t0 = perf_counter()
            effective_data_graph = materialize_atomic_domain_range_closure(
                ontology_graph,
                effective_data_graph,
            )
            preprocessing_timings.atomic_domain_range_elapsed_ms += (perf_counter() - t0) * 1000.0
        if preprocessing_plan.materialize_hierarchy.enabled:
            t0 = perf_counter()
            effective_data_graph = materialize_class_hierarchy_closure(
                ontology_graph,
                effective_data_graph,
            )
            preprocessing_timings.hierarchy_elapsed_ms += (perf_counter() - t0) * 1000.0
        ontology_graph = merge_rdflib_graphs((schema_graph, effective_data_graph))

    if preprocessing_plan.materialize_target_roles.enabled and target_classes:
        if dependency_closure is None:
            analysis_source = effective_data_graph if len(effective_data_graph) > 0 else ontology_graph
            _kg, analysis_mapping = rdflib_graph_to_kgraph(
                analysis_source,
                vocab_graph=ontology_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
            )
            dependency_closure = compute_target_dependency_closure(
                ontology_graph,
                analysis_mapping,
                target_classes,
            )
        t0 = perf_counter()
        role_result = materialize_target_role_closure(
            ontology_graph,
            effective_data_graph,
            seed_properties=dependency_closure.referenced_properties,
            target_classes=target_classes,
        )
        preprocessing_timings.target_role_elapsed_ms += (perf_counter() - t0) * 1000.0
        effective_data_graph = role_result.data_graph
        ontology_graph = merge_rdflib_graphs((schema_graph, effective_data_graph))

    kg_source = effective_data_graph if len(effective_data_graph) > 0 else ontology_graph
    t0 = perf_counter()
    kg, mapping = rdflib_graph_to_kgraph(
        kg_source,
        vocab_graph=ontology_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
    )
    preprocessing_timings.kgraph_build_elapsed_ms = (perf_counter() - t0) * 1000.0

    return ReasoningDataset(
        schema_graph=schema_graph,
        data_graph=effective_data_graph,
        ontology_graph=ontology_graph,
        kg=kg,
        mapping=mapping,
        preprocessing_plan=preprocessing_plan,
        preprocessing_timings=preprocessing_timings,
    )


def summarize_loaded_kgraph(
    kg: KGraph,
    mapping: RDFKGraphMapping,
    *,
    max_items: int = 10,
) -> str:
    """
    Produce a compact human-readable summary for CLI/demo use.
    """
    num_edges = sum(int(neighbors.numel()) for neighbors in kg.neighbors_p)
    active_props = sum(1 for neighbors in kg.neighbors_p if int(neighbors.numel()) > 0)
    typed_nodes = int((kg.node_types.sum(dim=1) > 0).sum().item()) if kg.node_types.numel() else 0

    lines = [
        f"num_nodes={kg.num_nodes}",
        f"num_props={len(mapping.prop_terms)}",
        f"num_classes={len(mapping.class_terms)}",
        f"num_edges={num_edges}",
        f"active_props={active_props}",
        f"typed_nodes={typed_nodes}",
        "",
        "sample nodes:",
    ]
    for term in mapping.node_terms[:max_items]:
        lines.append(f"  - {term.n3()}")

    lines.append("")
    lines.append("sample properties:")
    for term in mapping.prop_terms[:max_items]:
        lines.append(f"  - {term.n3()}")

    lines.append("")
    lines.append("sample classes:")
    for term in mapping.class_terms[:max_items]:
        lines.append(f"  - {term.n3()}")

    return "\n".join(lines)


def _rdf_list_members(graph: Graph, head: Identifier) -> List[Identifier]:
    if not isinstance(head, BNode):
        return []
    return list(Collection(graph, head))


def _resolve_property_expression(
    ontology_graph: Graph,
    prop_expr: Identifier | None,
) -> Tuple[URIRef, TraversalDirection]:
    if isinstance(prop_expr, URIRef):
        return prop_expr, TraversalDirection.FORWARD

    if isinstance(prop_expr, BNode):
        inverse_target = ontology_graph.value(prop_expr, OWL.inverseOf)
        if isinstance(inverse_target, URIRef):
            return inverse_target, TraversalDirection.BACKWARD

    raise ValueError(f"Unsupported property expression: {prop_expr!r}")


def _literal_to_nonnegative_int(value: Identifier | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, Literal):
        try:
            as_int = int(value.toPython())
        except (TypeError, ValueError):
            return None
        if as_int < 0:
            return None
    return as_int


def describe_owl_expression(
    ontology_graph: Graph,
    expr: Identifier,
) -> str:
    if isinstance(expr, URIRef):
        return expr.n3()
    if isinstance(expr, Literal):
        return expr.n3()
    if not isinstance(expr, BNode):
        return repr(expr)

    one_of_head = ontology_graph.value(expr, OWL.oneOf)
    if isinstance(one_of_head, BNode):
        members = [describe_owl_expression(ontology_graph, member) for member in Collection(ontology_graph, one_of_head)]
        return "oneOf(" + ", ".join(members) + ")"

    union_head = ontology_graph.value(expr, OWL.unionOf)
    if isinstance(union_head, BNode):
        members = [describe_owl_expression(ontology_graph, member) for member in Collection(ontology_graph, union_head)]
        return "union(" + ", ".join(members) + ")"

    intersection_head = ontology_graph.value(expr, OWL.intersectionOf)
    if isinstance(intersection_head, BNode):
        members = [describe_owl_expression(ontology_graph, member) for member in Collection(ontology_graph, intersection_head)]
        return "intersection(" + ", ".join(members) + ")"

    complement_target = ontology_graph.value(expr, OWL.complementOf)
    if complement_target is not None:
        return "not(" + describe_owl_expression(ontology_graph, complement_target) + ")"

    datatype_spec = _parse_datatype_restriction(ontology_graph, expr, None)
    if datatype_spec is not None:
        datatype_term = datatype_spec["datatype_term"]
        parts = [datatype_term.n3()]
        if datatype_spec["numeric_min"] is not None:
            op = ">=" if datatype_spec["min_inclusive"] else ">"
            parts.append(f"{op}{datatype_spec['numeric_min']}")
        if datatype_spec["numeric_max"] is not None:
            op = "<=" if datatype_spec["max_inclusive"] else "<"
            parts.append(f"{op}{datatype_spec['numeric_max']}")
        return "datatype(" + ", ".join(parts) + ")"

    restriction_types = set(ontology_graph.objects(expr, RDF.type))
    if OWL.Restriction in restriction_types:
        prop_expr = ontology_graph.value(expr, OWL.onProperty)
        try:
            prop_term, prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
            prop_rendered = prop_term.n3() if prop_direction == TraversalDirection.FORWARD else f"{prop_term.n3()}^-1"
        except ValueError:
            prop_rendered = repr(prop_expr)

        some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
        if some_filler is not None:
            return f"exists({prop_rendered}, {describe_owl_expression(ontology_graph, some_filler)})"

        all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
        if all_filler is not None:
            return f"forall({prop_rendered}, {describe_owl_expression(ontology_graph, all_filler)})"

        on_class = ontology_graph.value(expr, OWL.onClass)
        cardinality_fields = (
            ("minCardinality", ontology_graph.value(expr, OWL.minCardinality)),
            ("maxCardinality", ontology_graph.value(expr, OWL.maxCardinality)),
            ("cardinality", ontology_graph.value(expr, OWL.cardinality)),
            ("minQualifiedCardinality", ontology_graph.value(expr, OWL.minQualifiedCardinality)),
            ("maxQualifiedCardinality", ontology_graph.value(expr, OWL.maxQualifiedCardinality)),
            ("qualifiedCardinality", ontology_graph.value(expr, OWL.qualifiedCardinality)),
        )
        for label, value in cardinality_fields:
            cardinality = _literal_to_nonnegative_int(value)
            if cardinality is None:
                continue
            if on_class is None:
                return f"{label}({prop_rendered}, {cardinality})"
            return f"{label}({prop_rendered}, {cardinality}, {describe_owl_expression(ontology_graph, on_class)})"

    return expr.n3()


def _parse_datatype_restriction(
    ontology_graph: Graph,
    expr: Identifier,
    mapping: Optional[RDFKGraphMapping],
) -> Optional[Dict[str, object]]:
    base_datatype: Optional[URIRef] = None
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    min_inclusive = True
    max_inclusive = True

    if isinstance(expr, URIRef) and (mapping is None or expr in mapping.datatype_to_idx):
        base_datatype = expr
    else:
        on_datatype = ontology_graph.value(expr, OWL.onDatatype)
        if isinstance(on_datatype, URIRef) and (mapping is None or on_datatype in mapping.datatype_to_idx):
            base_datatype = on_datatype
            restrictions_head = ontology_graph.value(expr, OWL.withRestrictions)
            if isinstance(restrictions_head, BNode):
                for restriction in Collection(ontology_graph, restrictions_head):
                    if not isinstance(restriction, BNode):
                        continue
                    lower_inclusive = _literal_to_float(
                        ontology_graph.value(restriction, XSD.minInclusive)
                    )
                    lower_exclusive = _literal_to_float(
                        ontology_graph.value(restriction, XSD.minExclusive)
                    )
                    upper_inclusive = _literal_to_float(
                        ontology_graph.value(restriction, XSD.maxInclusive)
                    )
                    upper_exclusive = _literal_to_float(
                        ontology_graph.value(restriction, XSD.maxExclusive)
                    )
                    if lower_inclusive is not None:
                        numeric_min = lower_inclusive
                        min_inclusive = True
                    if lower_exclusive is not None:
                        numeric_min = lower_exclusive
                        min_inclusive = False
                    if upper_inclusive is not None:
                        numeric_max = upper_inclusive
                        max_inclusive = True
                    if upper_exclusive is not None:
                        numeric_max = upper_exclusive
                        max_inclusive = False

    if base_datatype is None:
        return None

    result = {
        "datatype_term": base_datatype,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "min_inclusive": min_inclusive,
        "max_inclusive": max_inclusive,
    }
    if mapping is not None:
        result["datatype_idx"] = mapping.datatype_to_idx[base_datatype]
    return result


def _top_sufficient_condition() -> NormalizedSufficientCondition:
    return NormalizedSufficientCondition(kind=SufficientConditionKind.TOP)


def _flatten_intersection_conditions(
    children: Sequence[NormalizedSufficientCondition],
) -> Tuple[NormalizedSufficientCondition, ...]:
    flattened: List[NormalizedSufficientCondition] = []
    for child in children:
        if child.kind == SufficientConditionKind.INTERSECTION:
            flattened.extend(child.children)
        else:
            flattened.append(child)
    return tuple(flattened)


def _normalize_positive_sufficient_conditions(
    ontology_graph: Graph,
    expr: Identifier,
) -> Optional[List[NormalizedSufficientCondition]]:
    if isinstance(expr, URIRef):
        if _is_datatype_term(expr):
            return [
                NormalizedSufficientCondition(
                    kind=SufficientConditionKind.DATATYPE_CONSTRAINT,
                    datatype_term=expr,
                )
            ]
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.ATOMIC_CLASS,
                class_term=expr,
            )
        ]

    if isinstance(expr, Literal):
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.NOMINAL,
                node_term=expr,
            )
        ]

    if not isinstance(expr, BNode):
        return None

    one_of_head = ontology_graph.value(expr, OWL.oneOf)
    if isinstance(one_of_head, BNode):
        conditions: List[NormalizedSufficientCondition] = []
        for member in Collection(ontology_graph, one_of_head):
            conditions.append(
                NormalizedSufficientCondition(
                    kind=SufficientConditionKind.NOMINAL,
                    node_term=member,
                )
            )
        return conditions or None

    datatype_spec = _parse_datatype_restriction(ontology_graph, expr, None)
    if datatype_spec is not None:
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.DATATYPE_CONSTRAINT,
                datatype_term=datatype_spec["datatype_term"],
                numeric_min=datatype_spec["numeric_min"],
                numeric_max=datatype_spec["numeric_max"],
                min_inclusive=bool(datatype_spec["min_inclusive"]),
                max_inclusive=bool(datatype_spec["max_inclusive"]),
            )
        ]

    union_head = ontology_graph.value(expr, OWL.unionOf)
    if isinstance(union_head, BNode):
        return None

    intersection_head = ontology_graph.value(expr, OWL.intersectionOf)
    if isinstance(intersection_head, BNode):
        members: List[NormalizedSufficientCondition] = []
        for member in Collection(ontology_graph, intersection_head):
            normalized_member_conditions = _normalize_positive_sufficient_conditions(ontology_graph, member)
            if normalized_member_conditions is None or len(normalized_member_conditions) != 1:
                return None
            members.append(normalized_member_conditions[0])
        flattened = _flatten_intersection_conditions(members)
        if not flattened:
            return [_top_sufficient_condition()]
        if len(flattened) == 1:
            return [flattened[0]]
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.INTERSECTION,
                children=flattened,
            )
        ]

    if ontology_graph.value(expr, OWL.complementOf) is not None:
        return None

    restriction_types = set(ontology_graph.objects(expr, RDF.type))
    if OWL.Restriction not in restriction_types:
        return None

    prop_expr = ontology_graph.value(expr, OWL.onProperty)
    try:
        prop_term, prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
    except ValueError:
        return None

    some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
    if some_filler is not None:
        child_conditions = _normalize_positive_sufficient_conditions(ontology_graph, some_filler)
        if child_conditions is None or len(child_conditions) != 1:
            return None
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.EXISTS,
                prop_term=prop_term,
                prop_direction=prop_direction,
                children=(child_conditions[0],),
            )
        ]

    min_cardinality = _literal_to_nonnegative_int(ontology_graph.value(expr, OWL.minCardinality))
    min_qualified = _literal_to_nonnegative_int(ontology_graph.value(expr, OWL.minQualifiedCardinality))
    target = min_qualified if min_qualified is not None else min_cardinality
    if target is not None:
        on_class = ontology_graph.value(expr, OWL.onClass)
        if on_class is None:
            child_condition = _top_sufficient_condition()
        else:
            child_conditions = _normalize_positive_sufficient_conditions(ontology_graph, on_class)
            if child_conditions is None or len(child_conditions) != 1:
                return None
            child_condition = child_conditions[0]
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.MIN_CARDINALITY,
                prop_term=prop_term,
                prop_direction=prop_direction,
                cardinality_target=target,
                children=(child_condition,),
            )
        ]

    return None


def describe_normalized_sufficient_condition(
    condition: NormalizedSufficientCondition,
) -> str:
    if condition.kind == SufficientConditionKind.TOP:
        return "top"
    if condition.kind == SufficientConditionKind.ATOMIC_CLASS and condition.class_term is not None:
        return condition.class_term.n3()
    if condition.kind == SufficientConditionKind.NOMINAL and condition.node_term is not None:
        rendered = condition.node_term.n3() if hasattr(condition.node_term, "n3") else str(condition.node_term)
        return "{" + rendered + "}"
    if condition.kind == SufficientConditionKind.DATATYPE_CONSTRAINT and condition.datatype_term is not None:
        parts = [condition.datatype_term.n3()]
        if condition.numeric_min is not None:
            op = ">=" if condition.min_inclusive else ">"
            parts.append(f"{op}{condition.numeric_min}")
        if condition.numeric_max is not None:
            op = "<=" if condition.max_inclusive else "<"
            parts.append(f"{op}{condition.numeric_max}")
        return "datatype(" + ", ".join(parts) + ")"
    if condition.kind == SufficientConditionKind.EXISTS and condition.prop_term is not None:
        prop_rendered = (
            condition.prop_term.n3()
            if condition.prop_direction == TraversalDirection.FORWARD
            else f"{condition.prop_term.n3()}^-1"
        )
        child = condition.children[0] if condition.children else _top_sufficient_condition()
        return f"exists({prop_rendered}, {describe_normalized_sufficient_condition(child)})"
    if condition.kind == SufficientConditionKind.INTERSECTION:
        return "intersection(" + ", ".join(describe_normalized_sufficient_condition(child) for child in condition.children) + ")"
    if condition.kind == SufficientConditionKind.MIN_CARDINALITY and condition.prop_term is not None:
        prop_rendered = (
            condition.prop_term.n3()
            if condition.prop_direction == TraversalDirection.FORWARD
            else f"{condition.prop_term.n3()}^-1"
        )
        child = condition.children[0] if condition.children else _top_sufficient_condition()
        return (
            f"minCardinality({prop_rendered}, {condition.cardinality_target}, "
            f"{describe_normalized_sufficient_condition(child)})"
        )
    return condition.kind.value


def describe_normalized_sufficient_rule(rule: NormalizedSufficientRule) -> str:
    tags = f" tags={list(rule.tags)}" if rule.tags else ""
    source = rule.source_kind
    rendered_source = rule.source_rendered
    if rendered_source is None and rule.source_term is not None:
        rendered_source = (
            rule.source_term.n3()
            if hasattr(rule.source_term, "n3")
            else str(rule.source_term)
        )
    if rendered_source is not None:
        source += "=" + rendered_source
    return (
        f"{describe_normalized_sufficient_condition(rule.antecedent)}"
        f" -> {rule.consequent_class.n3()} [{source}]{tags}"
    )


def collect_normalized_sufficient_condition_rules(
    ontology_graph: Graph,
) -> NormalizedSufficientRuleSet:
    rules: List[NormalizedSufficientRule] = []
    skipped_axioms: List[str] = []
    seen_rule_keys: set[Tuple[str, str, str, Tuple[str, ...]]] = set()

    def add_rule(
        *,
        consequent_class: URIRef,
        antecedent: NormalizedSufficientCondition,
        source_kind: str,
        source_term: Optional[Identifier],
        tags: Sequence[str],
    ) -> None:
        antecedent_rendered = describe_normalized_sufficient_condition(antecedent)
        key = (str(consequent_class), antecedent_rendered, source_kind, tuple(sorted(set(tags))))
        if key in seen_rule_keys:
            return
        seen_rule_keys.add(key)
        rules.append(
            NormalizedSufficientRule(
                consequent_class=consequent_class,
                antecedent=antecedent,
                source_kind=source_kind,
                source_term=source_term,
                source_rendered=(
                    describe_owl_expression(ontology_graph, source_term)
                    if source_term is not None
                    else None
                ),
                tags=tuple(sorted(set(tags))),
            )
        )

    for consequent_class in sorted(_collect_named_class_terms(ontology_graph), key=str):
        if not isinstance(consequent_class, URIRef) or _is_datatype_term(consequent_class):
            continue

        for antecedent_expr in ontology_graph.subjects(RDFS.subClassOf, consequent_class):
            normalized = _normalize_positive_sufficient_conditions(ontology_graph, antecedent_expr)
            if normalized is None:
                skipped_axioms.append(
                    f"subClassOf({describe_owl_expression(ontology_graph, antecedent_expr)} -> {consequent_class.n3()})"
                )
                continue
            for antecedent in normalized:
                add_rule(
                    consequent_class=consequent_class,
                    antecedent=antecedent,
                    source_kind="subClassOf",
                    source_term=antecedent_expr,
                    tags=[antecedent.kind.value],
                )

        equivalent_sources = set(ontology_graph.objects(consequent_class, OWL.equivalentClass))
        equivalent_sources.update(ontology_graph.subjects(OWL.equivalentClass, consequent_class))
        for antecedent_expr in sorted(equivalent_sources, key=_term_sort_key):
            if antecedent_expr == consequent_class:
                continue
            normalized = _normalize_positive_sufficient_conditions(ontology_graph, antecedent_expr)
            if normalized is None:
                skipped_axioms.append(
                    f"equivalentClass({describe_owl_expression(ontology_graph, antecedent_expr)} -> {consequent_class.n3()})"
                )
                continue
            for antecedent in normalized:
                add_rule(
                    consequent_class=consequent_class,
                    antecedent=antecedent,
                    source_kind="equivalentClass",
                    source_term=antecedent_expr,
                    tags=[antecedent.kind.value],
                )

    property_axioms = collect_property_expression_axioms(ontology_graph)
    top_condition = _top_sufficient_condition()
    for prop_term, axioms in property_axioms.items():
        for domain_expr in axioms.domain_expressions:
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, domain_expr)
            if consequents is None:
                skipped_axioms.append(
                    f"domain({prop_term.n3()}, {describe_owl_expression(ontology_graph, domain_expr)})"
                )
                continue
            antecedent = NormalizedSufficientCondition(
                kind=SufficientConditionKind.EXISTS,
                prop_term=prop_term,
                prop_direction=TraversalDirection.FORWARD,
                children=(top_condition,),
            )
            for consequent_class in consequents:
                add_rule(
                    consequent_class=consequent_class,
                    antecedent=antecedent,
                    source_kind="domain",
                    source_term=prop_term,
                    tags=["domain", "exists"],
                )

        for range_expr in axioms.range_expressions:
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, range_expr)
            if consequents is None:
                skipped_axioms.append(
                    f"range({prop_term.n3()}, {describe_owl_expression(ontology_graph, range_expr)})"
                )
                continue
            antecedent = NormalizedSufficientCondition(
                kind=SufficientConditionKind.EXISTS,
                prop_term=prop_term,
                prop_direction=TraversalDirection.BACKWARD,
                children=(top_condition,),
            )
            for consequent_class in consequents:
                add_rule(
                    consequent_class=consequent_class,
                    antecedent=antecedent,
                    source_kind="range",
                    source_term=prop_term,
                    tags=["exists", "range"],
                )

    rules.sort(
        key=lambda rule: (
            str(rule.consequent_class),
            describe_normalized_sufficient_condition(rule.antecedent),
            rule.source_kind,
        )
    )
    skipped_axioms = sorted(set(skipped_axioms))
    return NormalizedSufficientRuleSet(rules=rules, skipped_axioms=skipped_axioms)


def describe_normalized_sufficient_rule_set(
    rule_set: NormalizedSufficientRuleSet,
    *,
    max_rules: int = 50,
    max_skipped_axioms: int = 20,
) -> str:
    lines = [f"normalized_rules={len(rule_set.rules)}"]
    for rule in rule_set.rules[:max_rules]:
        lines.append("  - " + describe_normalized_sufficient_rule(rule))
    if len(rule_set.rules) > max_rules:
        lines.append(f"  ... ({len(rule_set.rules) - max_rules} more rules)")

    lines.append(f"skipped_axioms={len(rule_set.skipped_axioms)}")
    for skipped in rule_set.skipped_axioms[:max_skipped_axioms]:
        lines.append("  - " + skipped)
    if len(rule_set.skipped_axioms) > max_skipped_axioms:
        lines.append(f"  ... ({len(rule_set.skipped_axioms) - max_skipped_axioms} more skipped)")
    return "\n".join(lines)


def collect_inferable_sufficient_rule_classes(
    ontology_graph: Graph,
) -> List[URIRef]:
    rule_set = collect_normalized_sufficient_condition_rules(ontology_graph)
    consequents = {rule.consequent_class for rule in rule_set.rules}
    return sorted(consequents, key=str)


def _collect_condition_named_classes(
    condition: NormalizedSufficientCondition,
) -> set[URIRef]:
    classes: set[URIRef] = set()
    if condition.kind == SufficientConditionKind.ATOMIC_CLASS and condition.class_term is not None:
        classes.add(condition.class_term)
    for child in condition.children:
        classes.update(_collect_condition_named_classes(child))
    return classes


def _collect_condition_properties(
    condition: NormalizedSufficientCondition,
) -> set[URIRef]:
    props: set[URIRef] = set()
    if condition.prop_term is not None:
        props.add(condition.prop_term)
    for child in condition.children:
        props.update(_collect_condition_properties(child))
    return props


def compute_sufficient_rule_dependency_closure(
    rule_set: NormalizedSufficientRuleSet,
    target_classes: Sequence[str | URIRef],
) -> TargetDependencyClosure:
    target_terms = [URIRef(term) if isinstance(term, str) else term for term in target_classes]
    rules_by_consequent: Dict[URIRef, List[NormalizedSufficientRule]] = defaultdict(list)
    for rule in rule_set.rules:
        rules_by_consequent[rule.consequent_class].append(rule)

    closure_classes: set[URIRef] = set()
    referenced_named_classes: set[URIRef] = set()
    referenced_properties: set[URIRef] = set()
    stack = list(target_terms)

    while stack:
        class_term = stack.pop()
        if class_term in closure_classes:
            continue
        closure_classes.add(class_term)
        referenced_named_classes.add(class_term)
        for rule in rules_by_consequent.get(class_term, ()):
            antecedent_classes = _collect_condition_named_classes(rule.antecedent)
            antecedent_props = _collect_condition_properties(rule.antecedent)
            referenced_named_classes.update(antecedent_classes)
            referenced_properties.update(antecedent_props)
            for antecedent_class in antecedent_classes:
                if antecedent_class not in closure_classes and antecedent_class in rules_by_consequent:
                    stack.append(antecedent_class)

    return TargetDependencyClosure(
        target_classes=sorted(target_terms, key=str),
        referenced_named_classes=sorted(referenced_named_classes, key=str),
        inferable_named_classes=sorted(closure_classes, key=str),
        referenced_properties=sorted(referenced_properties, key=str),
    )


def compile_normalized_sufficient_condition_to_dag(
    mapping: RDFKGraphMapping,
    condition: NormalizedSufficientCondition,
    *,
    intersection_agg: IntersectionAgg = IntersectionAgg.MIN,
    cardinality_agg: CardinalityAgg = CardinalityAgg.STRICT,
) -> ConstraintDAG:
    nodes: List[ConstraintNode] = []
    memo: Dict[NormalizedSufficientCondition, int] = {}

    def new_node(**kwargs) -> int:
        idx = len(nodes)
        node = ConstraintNode(idx=idx, **kwargs)
        nodes.append(node)
        return idx

    def compile_condition(term: NormalizedSufficientCondition) -> int:
        cached = memo.get(term)
        if cached is not None:
            return cached

        if term.kind == SufficientConditionKind.TOP:
            idx = new_node(ctype=ConstraintType.CONST)
        elif term.kind == SufficientConditionKind.ATOMIC_CLASS:
            if term.class_term is None or term.class_term not in mapping.class_to_idx:
                raise KeyError(f"Class {term.class_term} not present in class mapping.")
            idx = new_node(
                ctype=ConstraintType.ATOMIC_CLASS,
                class_idx=mapping.class_to_idx[term.class_term],
            )
        elif term.kind == SufficientConditionKind.NOMINAL:
            if term.node_term is None or term.node_term not in mapping.node_to_idx:
                raise KeyError(f"Node {term.node_term} not present in node mapping.")
            idx = new_node(
                ctype=ConstraintType.NOMINAL,
                node_idx=mapping.node_to_idx[term.node_term],
            )
        elif term.kind == SufficientConditionKind.DATATYPE_CONSTRAINT:
            datatype_idx = None
            if term.datatype_term is not None:
                if term.datatype_term not in mapping.datatype_to_idx:
                    raise KeyError(f"Datatype {term.datatype_term} not present in datatype mapping.")
                datatype_idx = mapping.datatype_to_idx[term.datatype_term]
            idx = new_node(
                ctype=ConstraintType.DATATYPE_CONSTRAINT,
                datatype_idx=datatype_idx,
                numeric_min=term.numeric_min,
                numeric_max=term.numeric_max,
                min_inclusive=term.min_inclusive,
                max_inclusive=term.max_inclusive,
            )
        elif term.kind == SufficientConditionKind.EXISTS:
            if term.prop_term is None or term.prop_term not in mapping.prop_to_idx:
                raise KeyError(f"Property {term.prop_term} not present in property mapping.")
            if len(term.children) != 1:
                raise ValueError("EXISTS sufficient conditions must have exactly one child.")
            child_idx = compile_condition(term.children[0])
            idx = new_node(
                ctype=ConstraintType.EXISTS_RESTRICTION,
                prop_idx=mapping.prop_to_idx[term.prop_term],
                prop_direction=term.prop_direction,
                child_indices=[child_idx],
            )
        elif term.kind == SufficientConditionKind.INTERSECTION:
            child_indices = [compile_condition(child) for child in term.children]
            idx = new_node(
                ctype=ConstraintType.INTERSECTION,
                child_indices=child_indices,
                intersection_agg=intersection_agg,
            )
        elif term.kind == SufficientConditionKind.MIN_CARDINALITY:
            if term.prop_term is None or term.prop_term not in mapping.prop_to_idx:
                raise KeyError(f"Property {term.prop_term} not present in property mapping.")
            if term.cardinality_target is None:
                raise ValueError("MIN_CARDINALITY sufficient conditions must provide a target.")
            if len(term.children) != 1:
                raise ValueError("MIN_CARDINALITY sufficient conditions must have exactly one child.")
            child_idx = compile_condition(term.children[0])
            idx = new_node(
                ctype=ConstraintType.MIN_CARDINALITY_RESTRICTION,
                prop_idx=mapping.prop_to_idx[term.prop_term],
                prop_direction=term.prop_direction,
                cardinality_target=term.cardinality_target,
                cardinality_agg=cardinality_agg,
                child_indices=[child_idx],
            )
        else:
            raise ValueError(f"Unsupported sufficient condition kind: {term.kind}")

        memo[term] = idx
        return idx

    root_idx = compile_condition(condition)
    layers = _compute_layers(nodes)
    return ConstraintDAG(nodes=nodes, root_idx=root_idx, layers=layers)


def compile_sufficient_condition_dag(
    ontology_graph: Graph,
    mapping: RDFKGraphMapping,
    target_class: str | URIRef,
    *,
    rule_set: Optional[NormalizedSufficientRuleSet] = None,
    include_atomic_seed: bool = True,
    intersection_agg: IntersectionAgg = IntersectionAgg.MIN,
    cardinality_agg: CardinalityAgg = CardinalityAgg.STRICT,
) -> ConstraintDAG:
    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    rules = rule_set or collect_normalized_sufficient_condition_rules(ontology_graph)
    antecedents = [
        rule.antecedent
        for rule in rules.rules
        if rule.consequent_class == target_term
    ]

    if include_atomic_seed:
        antecedents = [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.ATOMIC_CLASS,
                class_term=target_term,
            )
        ] + antecedents

    if not antecedents:
        raise ValueError(f"No sufficient-condition antecedents found for {target_term.n3()}.")

    nodes: List[ConstraintNode] = []
    memo: Dict[NormalizedSufficientCondition, int] = {}

    def new_node(**kwargs) -> int:
        idx = len(nodes)
        node = ConstraintNode(idx=idx, **kwargs)
        nodes.append(node)
        return idx

    def compile_condition(term: NormalizedSufficientCondition) -> int:
        cached = memo.get(term)
        if cached is not None:
            return cached

        compiled = compile_normalized_sufficient_condition_to_dag(
            mapping,
            term,
            intersection_agg=intersection_agg,
            cardinality_agg=cardinality_agg,
        )
        base_idx = len(nodes)
        for node in compiled.nodes:
            adjusted_children = (
                [base_idx + child_idx for child_idx in (node.child_indices or [])]
                if node.child_indices
                else None
            )
            new_node(
                ctype=node.ctype,
                class_idx=node.class_idx,
                node_idx=node.node_idx,
                datatype_idx=node.datatype_idx,
                numeric_min=node.numeric_min,
                numeric_max=node.numeric_max,
                min_inclusive=node.min_inclusive,
                max_inclusive=node.max_inclusive,
                prop_idx=node.prop_idx,
                prop_direction=node.prop_direction,
                cardinality_target=node.cardinality_target,
                cardinality_delta=node.cardinality_delta,
                cardinality_agg=node.cardinality_agg,
                child_indices=adjusted_children,
                intersection_agg=node.intersection_agg,
                scale_factor=node.scale_factor,
            )
        root_idx = base_idx + compiled.root_idx
        memo[term] = root_idx
        return root_idx

    antecedent_roots = [compile_condition(antecedent) for antecedent in antecedents]
    if len(antecedent_roots) == 1:
        root_idx = antecedent_roots[0]
    else:
        root_idx = new_node(
            ctype=ConstraintType.UNION,
            child_indices=antecedent_roots,
        )

    layers = _compute_layers(nodes)
    return ConstraintDAG(nodes=nodes, root_idx=root_idx, layers=layers)


def _expression_entails_target_class(
    ontology_graph: Graph,
    expr: Identifier,
    target_term: URIRef,
    subclass_supers: Dict[URIRef, set[URIRef]],
    memo: Optional[Dict[Tuple[str, str], bool]] = None,
    active_pairs: Optional[set[Tuple[str, str]]] = None,
) -> bool:
    """
    Conservative entailment check for the supported positive fragment.

    This is used only for query-time domain/range augmentation, so it is
    intentionally incomplete but should stay sound for the cases it returns
    True on.
    """

    if memo is None:
        memo = {}
    if active_pairs is None:
        active_pairs = set()

    lhs_key = describe_owl_expression(ontology_graph, expr)
    rhs_key = describe_owl_expression(ontology_graph, target_term)
    key = (lhs_key, rhs_key)
    if key in memo:
        return memo[key]
    if key in active_pairs:
        return False

    active_pairs.add(key)
    result = False

    if expr == target_term or lhs_key == rhs_key:
        result = True
    else:
        target_roots = _collect_target_root_expressions(ontology_graph, target_term)
        if target_roots and all(
            _expression_entails_expression(
                ontology_graph,
                expr,
                root,
                subclass_supers,
                memo,
                active_pairs,
            )
            for root in target_roots
        ):
            result = True
        elif isinstance(expr, URIRef):
            result = target_term in subclass_supers.get(expr, set())
        elif isinstance(expr, BNode):
            expr_union = ontology_graph.value(expr, OWL.unionOf)
            expr_intersection = ontology_graph.value(expr, OWL.intersectionOf)
            if isinstance(expr_union, BNode):
                result = all(
                    _expression_entails_target_class(
                        ontology_graph,
                        member,
                        target_term,
                        subclass_supers,
                        memo,
                        active_pairs,
                    )
                    for member in Collection(ontology_graph, expr_union)
                )
            elif isinstance(expr_intersection, BNode):
                result = any(
                    _expression_entails_target_class(
                        ontology_graph,
                        member,
                        target_term,
                        subclass_supers,
                        memo,
                        active_pairs,
                    )
                    for member in Collection(ontology_graph, expr_intersection)
                )

    active_pairs.remove(key)
    memo[key] = result
    return result


def _expression_entails_expression(
    ontology_graph: Graph,
    lhs: Identifier,
    rhs: Identifier,
    subclass_supers: Dict[URIRef, set[URIRef]],
    memo: Optional[Dict[Tuple[str, str], bool]] = None,
    active_pairs: Optional[set[Tuple[str, str]]] = None,
) -> bool:
    if memo is None:
        memo = {}
    if active_pairs is None:
        active_pairs = set()

    lhs_key = describe_owl_expression(ontology_graph, lhs)
    rhs_key = describe_owl_expression(ontology_graph, rhs)
    key = (lhs_key, rhs_key)
    if key in memo:
        return memo[key]
    if key in active_pairs:
        return False

    active_pairs.add(key)
    result = False

    if lhs == rhs or lhs_key == rhs_key:
        result = True
    elif isinstance(rhs, URIRef):
        result = _expression_entails_target_class(
            ontology_graph,
            lhs,
            rhs,
            subclass_supers,
            memo,
            active_pairs,
        )
    else:
        rhs_union = ontology_graph.value(rhs, OWL.unionOf)
        rhs_intersection = ontology_graph.value(rhs, OWL.intersectionOf)

        if isinstance(rhs_union, BNode):
            rhs_members = list(Collection(ontology_graph, rhs_union))
            if isinstance(lhs, BNode):
                lhs_union = ontology_graph.value(lhs, OWL.unionOf)
                if isinstance(lhs_union, BNode):
                    lhs_members = list(Collection(ontology_graph, lhs_union))
                    result = all(
                        _expression_entails_expression(
                            ontology_graph,
                            member,
                            rhs,
                            subclass_supers,
                            memo,
                            active_pairs,
                        )
                        for member in lhs_members
                    )
                else:
                    result = any(
                        _expression_entails_expression(
                            ontology_graph,
                            lhs,
                            member,
                            subclass_supers,
                            memo,
                            active_pairs,
                        )
                        for member in rhs_members
                    )
            else:
                result = any(
                    _expression_entails_expression(
                        ontology_graph,
                        lhs,
                        member,
                        subclass_supers,
                        memo,
                        active_pairs,
                    )
                    for member in rhs_members
                )
        elif isinstance(rhs_intersection, BNode):
            rhs_members = list(Collection(ontology_graph, rhs_intersection))
            result = all(
                _expression_entails_expression(
                    ontology_graph,
                    lhs,
                    member,
                    subclass_supers,
                    memo,
                    active_pairs,
                )
                for member in rhs_members
            )
        elif isinstance(lhs, BNode):
            lhs_union = ontology_graph.value(lhs, OWL.unionOf)
            lhs_intersection = ontology_graph.value(lhs, OWL.intersectionOf)

            if isinstance(lhs_union, BNode):
                lhs_members = list(Collection(ontology_graph, lhs_union))
                result = all(
                    _expression_entails_expression(
                        ontology_graph,
                        member,
                        rhs,
                        subclass_supers,
                        memo,
                        active_pairs,
                    )
                    for member in lhs_members
                )
            elif isinstance(lhs_intersection, BNode):
                lhs_members = list(Collection(ontology_graph, lhs_intersection))
                result = any(
                    _expression_entails_expression(
                        ontology_graph,
                        member,
                        rhs,
                        subclass_supers,
                        memo,
                        active_pairs,
                    )
                    for member in lhs_members
                )

    active_pairs.remove(key)
    memo[key] = result
    return result
    return None


def _compute_layers(nodes: List[ConstraintNode]) -> List[List[int]]:
    depth_cache: Dict[int, int] = {}

    def depth(node_idx: int) -> int:
        if node_idx in depth_cache:
            return depth_cache[node_idx]

        child_indices = nodes[node_idx].child_indices or []
        if not child_indices:
            result = 0
        else:
            result = 1 + max(depth(child_idx) for child_idx in child_indices)
        depth_cache[node_idx] = result
        return result

    max_depth = 0
    for idx in range(len(nodes)):
        max_depth = max(max_depth, depth(idx))

    layers: List[List[int]] = [[] for _ in range(max_depth + 1)]
    for idx in range(len(nodes)):
        layers[depth(idx)].append(idx)
    return layers


def describe_constraint_dag(dag: ConstraintDAG, mapping: Optional[RDFKGraphMapping] = None) -> str:
    lines = [f"root_idx={dag.root_idx}", f"layers={dag.layers}", "nodes:"]
    for node in dag.nodes:
        parts = [f"idx={node.idx}", f"ctype={node.ctype.name}"]
        if node.class_idx is not None:
            if mapping is not None and 0 <= node.class_idx < len(mapping.class_terms):
                parts.append(f"class={mapping.class_terms[node.class_idx].n3()}")
            else:
                parts.append(f"class_idx={node.class_idx}")
        if node.node_idx is not None:
            if mapping is not None and 0 <= node.node_idx < len(mapping.node_terms):
                term = mapping.node_terms[node.node_idx]
                parts.append(f"node={term.n3() if hasattr(term, 'n3') else term}")
            else:
                parts.append(f"node_idx={node.node_idx}")
        if node.datatype_idx is not None:
            if mapping is not None and 0 <= node.datatype_idx < len(mapping.datatype_terms):
                parts.append(f"datatype={mapping.datatype_terms[node.datatype_idx].n3()}")
            else:
                parts.append(f"datatype_idx={node.datatype_idx}")
        if node.numeric_min is not None:
            op = ">=" if node.min_inclusive else ">"
            parts.append(f"{op}{node.numeric_min}")
        if node.numeric_max is not None:
            op = "<=" if node.max_inclusive else "<"
            parts.append(f"{op}{node.numeric_max}")
        if node.prop_idx is not None:
            if mapping is not None and 0 <= node.prop_idx < len(mapping.prop_terms):
                parts.append(f"prop={mapping.prop_terms[node.prop_idx].n3()}")
            else:
                parts.append(f"prop_idx={node.prop_idx}")
            if node.prop_direction != TraversalDirection.FORWARD:
                parts.append(f"dir={node.prop_direction.value}")
        if node.cardinality_target is not None:
            parts.append(f"target={node.cardinality_target}")
        if node.cardinality_delta is not None:
            parts.append(f"delta={node.cardinality_delta}")
        if node.cardinality_agg != CardinalityAgg.STRICT:
            parts.append(f"card_agg={node.cardinality_agg.value}")
        if node.child_indices:
            parts.append(f"children={node.child_indices}")
        if node.intersection_agg is not None:
            parts.append(f"agg={node.intersection_agg.value}")
        if node.scale_factor is not None:
            parts.append(f"scale={node.scale_factor}")
        lines.append("  - " + ", ".join(parts))
    return "\n".join(lines)


def build_dag_dependency_report(
    ontology_graph: Graph,
    mapping: RDFKGraphMapping,
    target_class: str | URIRef,
    *,
    intersection_agg: IntersectionAgg = IntersectionAgg.MIN,
    cardinality_agg: CardinalityAgg = CardinalityAgg.STRICT,
    augment_property_domain_range: bool = False,
) -> DAGDependencyReport:
    """
    Compile a target class and derive per-DAG-node dependency closures.

    This makes the query dependency structure explicit at the DAG-node level so
    helper materialization passes can share one target-aware dependency source.
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    dag = compile_class_to_dag(
        ontology_graph,
        mapping,
        target_term,
        intersection_agg=intersection_agg,
        cardinality_agg=cardinality_agg,
        augment_property_domain_range=augment_property_domain_range,
    )

    node_dependencies: Dict[int, DAGNodeDependency] = {}
    inferable_cache: Dict[URIRef, bool] = {}

    for layer in dag.layers:
        for node_idx in layer:
            node = dag.nodes[node_idx]
            direct_class_terms: List[Identifier] = []
            direct_property_terms: List[URIRef] = []

            if node.class_idx is not None and 0 <= node.class_idx < len(mapping.class_terms):
                direct_class_terms.append(mapping.class_terms[node.class_idx])
            if node.prop_idx is not None and 0 <= node.prop_idx < len(mapping.prop_terms):
                direct_property_terms.append(mapping.prop_terms[node.prop_idx])

            subtree_class_terms: set[Identifier] = set(direct_class_terms)
            subtree_property_terms: set[URIRef] = set(direct_property_terms)

            for child_idx in node.child_indices or []:
                child_dep = node_dependencies[child_idx]
                subtree_class_terms.update(child_dep.subtree_class_terms)
                subtree_property_terms.update(child_dep.subtree_property_terms)

            subtree_inferable_named_classes: set[URIRef] = set()
            for class_term in subtree_class_terms:
                if not isinstance(class_term, URIRef):
                    continue
                if class_term not in inferable_cache:
                    inferable_cache[class_term] = _has_nontrivial_definition(ontology_graph, class_term)
                if inferable_cache[class_term]:
                    subtree_inferable_named_classes.add(class_term)

            node_dependencies[node_idx] = DAGNodeDependency(
                idx=node_idx,
                ctype=node.ctype,
                direct_class_terms=sorted(direct_class_terms, key=_term_sort_key),
                direct_property_terms=sorted(direct_property_terms, key=str),
                subtree_class_terms=sorted(subtree_class_terms, key=_term_sort_key),
                subtree_property_terms=sorted(subtree_property_terms, key=str),
                subtree_inferable_named_classes=sorted(subtree_inferable_named_classes, key=str),
            )

    root_dep = node_dependencies[dag.root_idx]
    referenced_named_classes = sorted(
        {
            class_term
            for class_term in root_dep.subtree_class_terms
            if isinstance(class_term, URIRef)
        }
        | {target_term},
        key=str,
    )

    inferable_named_classes = sorted(
        {
            class_term
            for class_term in root_dep.subtree_inferable_named_classes
            if isinstance(class_term, URIRef)
        },
        key=str,
    )
    if _has_nontrivial_definition(ontology_graph, target_term) and target_term not in inferable_named_classes:
        inferable_named_classes.append(target_term)
        inferable_named_classes.sort(key=str)

    return DAGDependencyReport(
        target_class=target_term,
        dag=dag,
        node_dependencies=node_dependencies,
        referenced_named_classes=referenced_named_classes,
        inferable_named_classes=inferable_named_classes,
        referenced_properties=root_dep.subtree_property_terms,
    )


def describe_dag_dependency_report(report: DAGDependencyReport) -> str:
    lines = [
        f"target_class={report.target_class.n3()}",
        f"referenced_named_classes={[term.n3() for term in report.referenced_named_classes]}",
        f"inferable_named_classes={[term.n3() for term in report.inferable_named_classes]}",
        f"referenced_properties={[term.n3() for term in report.referenced_properties]}",
        "node_dependencies:",
    ]
    for node_idx in sorted(report.node_dependencies):
        node_dep = report.node_dependencies[node_idx]
        parts = [
            f"idx={node_dep.idx}",
            f"ctype={node_dep.ctype.name}",
        ]
        if node_dep.direct_class_terms:
            parts.append(
                "direct_classes=" + "[" + ", ".join(_render.n3() if hasattr(_render, "n3") else str(_render) for _render in node_dep.direct_class_terms) + "]"
            )
        if node_dep.direct_property_terms:
            parts.append(
                "direct_props="
                + "["
                + ", ".join(
                    (
                        term.n3()
                        if report.dag.nodes[node_dep.idx].prop_direction == TraversalDirection.FORWARD
                        else f"{term.n3()}^-1"
                    )
                    for term in node_dep.direct_property_terms
                )
                + "]"
            )
        if node_dep.subtree_inferable_named_classes:
            parts.append(
                "subtree_inferable="
                + "["
                + ", ".join(term.n3() for term in node_dep.subtree_inferable_named_classes)
                + "]"
            )
        lines.append("  - " + ", ".join(parts))
    return "\n".join(lines)


def compile_class_to_dag(
    ontology_graph: Graph,
    mapping: RDFKGraphMapping,
    target_class: str | URIRef,
    *,
    intersection_agg: IntersectionAgg = IntersectionAgg.MIN,
    cardinality_agg: CardinalityAgg = CardinalityAgg.STRICT,
    augment_property_domain_range: bool = False,
) -> ConstraintDAG:
    """
    Compile a small OWL fragment into a ConstraintDAG.

    Supported fragments:
    - named atomic classes
    - `owl:oneOf` nominals
    - `owl:unionOf`
    - direct datatype fillers such as `xsd:integer`
    - `owl:onDatatype` + `owl:withRestrictions` with basic numeric facets
    - `rdfs:subClassOf`
    - `owl:equivalentClass`
    - `owl:intersectionOf`
    - `owl:complementOf`
    - `owl:disjointWith` for direct class disjointness
    - `owl:Restriction` with `owl:onProperty` + `owl:someValuesFrom`
    - `owl:Restriction` with `owl:onProperty` + `owl:allValuesFrom`
    - `owl:minCardinality`, `owl:maxCardinality`, `owl:cardinality`
    - `owl:minQualifiedCardinality`, `owl:maxQualifiedCardinality`, `owl:qualifiedCardinality`
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    nodes: List[ConstraintNode] = []
    memo: Dict[Identifier, int] = {}
    direct_subs = _collect_direct_subproperties(ontology_graph)
    inverse_props = _collect_inverse_properties(ontology_graph)
    chains_by_conclusion = _collect_property_chains(ontology_graph)
    transitive_props = _collect_transitive_properties(ontology_graph)
    subproperty_closure_cache: Dict[URIRef, Tuple[URIRef, ...]] = {}
    restriction_memo: Dict[Tuple[str, URIRef, str, int], int] = {}

    def new_node(**kwargs) -> int:
        idx = len(nodes)
        node = ConstraintNode(idx=idx, **kwargs)
        nodes.append(node)
        return idx

    top_idx: Optional[int] = None

    def top_node() -> int:
        nonlocal top_idx
        if top_idx is None:
            top_idx = new_node(ctype=ConstraintType.CONST)
        return top_idx

    def direct_restriction_node(
        *,
        prop: URIRef,
        prop_direction: TraversalDirection,
        child_idx: int,
        universal: bool,
    ) -> int:
        if universal:
            ctype = ConstraintType.FORALL_RESTRICTION
        elif prop in transitive_props:
            ctype = ConstraintType.EXISTS_TRANSITIVE_RESTRICTION
        else:
            ctype = ConstraintType.EXISTS_RESTRICTION
        return new_node(
            ctype=ctype,
            prop_idx=mapping.prop_to_idx[prop],
            prop_direction=prop_direction,
            child_indices=[child_idx],
        )

    def transitive_subproperties(prop: URIRef) -> Tuple[URIRef, ...]:
        cached = subproperty_closure_cache.get(prop)
        if cached is not None:
            return cached

        seen: set[URIRef] = set()
        stack = list(direct_subs.get(prop, ()))
        while stack:
            child = stack.pop()
            if child in seen:
                continue
            seen.add(child)
            stack.extend(direct_subs.get(child, ()))

        result = tuple(sorted(seen, key=str))
        subproperty_closure_cache[prop] = result
        return result

    def compile_property_restriction(
        *,
        prop: URIRef,
        prop_direction: TraversalDirection,
        child_idx: int,
        universal: bool,
        stack: Optional[set[Tuple[str, URIRef, str, int]]] = None,
    ) -> int:
        key = (
            "forall" if universal else "exists",
            prop,
            prop_direction.value,
            child_idx,
        )
        if key in restriction_memo:
            return restriction_memo[key]

        active_stack = set() if stack is None else set(stack)
        if key in active_stack:
            idx = direct_restriction_node(
                prop=prop,
                prop_direction=prop_direction,
                child_idx=child_idx,
                universal=universal,
            )
            restriction_memo[key] = idx
            return idx

        active_stack.add(key)
        branch_indices: List[int] = [
            direct_restriction_node(
                prop=prop,
                prop_direction=prop_direction,
                child_idx=child_idx,
                universal=universal,
            )
        ]

        for subprop in transitive_subproperties(prop):
            if subprop not in mapping.prop_to_idx:
                continue
            branch_indices.append(
                compile_property_restriction(
                    prop=subprop,
                    prop_direction=prop_direction,
                    child_idx=child_idx,
                    universal=universal,
                    stack=active_stack,
                )
            )

        for inverse_prop in sorted(inverse_props.get(prop, ()), key=str):
            if inverse_prop not in mapping.prop_to_idx:
                continue
            inverse_direction = (
                TraversalDirection.BACKWARD
                if prop_direction == TraversalDirection.FORWARD
                else TraversalDirection.FORWARD
            )
            branch_indices.append(
                compile_property_restriction(
                    prop=inverse_prop,
                    prop_direction=inverse_direction,
                    child_idx=child_idx,
                    universal=universal,
                    stack=active_stack,
                )
            )

        for chain in chains_by_conclusion.get(prop, ()):
            if not all(chain_prop in mapping.prop_to_idx for chain_prop in chain):
                continue
            ordered_chain = chain if prop_direction == TraversalDirection.FORWARD else tuple(reversed(chain))
            nested_child_idx = child_idx
            for chain_prop in reversed(ordered_chain):
                nested_child_idx = compile_property_restriction(
                    prop=chain_prop,
                    prop_direction=prop_direction,
                    child_idx=nested_child_idx,
                    universal=universal,
                    stack=active_stack,
                )
            branch_indices.append(nested_child_idx)

        deduped_branches = list(dict.fromkeys(branch_indices))
        if len(deduped_branches) == 1:
            idx = deduped_branches[0]
        else:
            idx = new_node(
                ctype=ConstraintType.INTERSECTION if universal else ConstraintType.UNION,
                child_indices=deduped_branches,
                intersection_agg=intersection_agg if universal else None,
            )
        restriction_memo[key] = idx
        return idx

    def compile_expr(expr: Identifier) -> int:
        if expr in memo:
            return memo[expr]

        datatype_spec = _parse_datatype_restriction(ontology_graph, expr, mapping)
        if datatype_spec is not None:
            idx = new_node(
                ctype=ConstraintType.DATATYPE_CONSTRAINT,
                datatype_idx=datatype_spec["datatype_idx"],
                numeric_min=datatype_spec["numeric_min"],
                numeric_max=datatype_spec["numeric_max"],
                min_inclusive=bool(datatype_spec["min_inclusive"]),
                max_inclusive=bool(datatype_spec["max_inclusive"]),
            )
            memo[expr] = idx
            return idx

        one_of_heads = list(ontology_graph.objects(expr, OWL.oneOf))
        if one_of_heads:
            members: List[int] = []
            for head in one_of_heads:
                for member in _rdf_list_members(ontology_graph, head):
                    if member not in mapping.node_to_idx:
                        raise KeyError(f"Nominal member {member!r} not present in KGraph node mapping.")
                    members.append(
                        new_node(
                            ctype=ConstraintType.NOMINAL,
                            node_idx=mapping.node_to_idx[member],
                        )
                    )
            if len(members) < 1:
                raise ValueError(f"oneOf for {expr.n3()} had no members.")
            idx = members[0] if len(members) == 1 else new_node(
                ctype=ConstraintType.UNION,
                child_indices=members,
            )
            memo[expr] = idx
            return idx

        union_heads = list(ontology_graph.objects(expr, OWL.unionOf))
        if union_heads:
            members: List[int] = []
            for head in union_heads:
                for member in _rdf_list_members(ontology_graph, head):
                    members.append(compile_expr(member))
            if len(members) < 2:
                if len(members) == 1:
                    memo[expr] = members[0]
                    return members[0]
                raise ValueError(f"unionOf for {expr.n3()} had no members.")
            idx = new_node(
                ctype=ConstraintType.UNION,
                child_indices=members,
            )
            memo[expr] = idx
            return idx

        intersection_heads = list(ontology_graph.objects(expr, OWL.intersectionOf))
        if intersection_heads:
            members: List[int] = []
            for head in intersection_heads:
                for member in _rdf_list_members(ontology_graph, head):
                    members.append(compile_expr(member))
            if len(members) < 2:
                if len(members) == 1:
                    memo[expr] = members[0]
                    return members[0]
                raise ValueError(f"intersectionOf for {expr.n3()} had no members.")
            idx = new_node(
                ctype=ConstraintType.INTERSECTION,
                child_indices=members,
                intersection_agg=intersection_agg,
            )
            memo[expr] = idx
            return idx

        complement_targets = list(ontology_graph.objects(expr, OWL.complementOf))
        if complement_targets:
            if len(complement_targets) != 1:
                raise ValueError(f"Unsupported complementOf shape for {expr.n3()}.")
            child_idx = compile_expr(complement_targets[0])
            idx = new_node(
                ctype=ConstraintType.NEGATION,
                child_indices=[child_idx],
            )
            memo[expr] = idx
            return idx

        restriction_types = set(ontology_graph.objects(expr, RDF.type))
        if OWL.Restriction in restriction_types:
            prop_expr = ontology_graph.value(expr, OWL.onProperty)
            some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
            all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
            min_card = _literal_to_nonnegative_int(ontology_graph.value(expr, OWL.minCardinality))
            max_card = _literal_to_nonnegative_int(ontology_graph.value(expr, OWL.maxCardinality))
            exact_card = _literal_to_nonnegative_int(ontology_graph.value(expr, OWL.cardinality))
            min_qualified = _literal_to_nonnegative_int(
                ontology_graph.value(expr, OWL.minQualifiedCardinality)
            )
            max_qualified = _literal_to_nonnegative_int(
                ontology_graph.value(expr, OWL.maxQualifiedCardinality)
            )
            exact_qualified = _literal_to_nonnegative_int(
                ontology_graph.value(expr, OWL.qualifiedCardinality)
            )
            on_class = ontology_graph.value(expr, OWL.onClass)
            try:
                prop, prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
            except ValueError as exc:
                raise ValueError(f"Unsupported restriction shape for {expr.n3()}.") from exc
            if (
                some_filler is None
                and all_filler is None
                and min_card is None
                and max_card is None
                and exact_card is None
                and min_qualified is None
                and max_qualified is None
                and exact_qualified is None
            ):
                raise ValueError(f"Unsupported restriction shape for {expr.n3()}.")
            if prop not in mapping.prop_to_idx:
                raise KeyError(f"Property {prop} not present in KGraph property mapping.")

            if some_filler is not None:
                child_idx = compile_expr(some_filler)
                idx = compile_property_restriction(
                    prop=prop,
                    prop_direction=prop_direction,
                    child_idx=child_idx,
                    universal=False,
                )
            elif all_filler is not None:
                child_idx = compile_expr(all_filler)
                idx = compile_property_restriction(
                    prop=prop,
                    prop_direction=prop_direction,
                    child_idx=child_idx,
                    universal=True,
                )
            elif min_card is not None or max_card is not None or exact_card is not None:
                cardinality_terms = [term for term in (min_card, max_card, exact_card) if term is not None]
                if len(cardinality_terms) != 1:
                    raise ValueError(f"Unsupported cardinality shape for {expr.n3()}.")

                if min_card is not None:
                    ctype = ConstraintType.MIN_CARDINALITY_RESTRICTION
                    target = min_card
                elif max_card is not None:
                    ctype = ConstraintType.MAX_CARDINALITY_RESTRICTION
                    target = max_card
                else:
                    ctype = ConstraintType.EXACT_CARDINALITY_RESTRICTION
                    target = exact_card

                idx = new_node(
                    ctype=ctype,
                    prop_idx=mapping.prop_to_idx[prop],
                    prop_direction=prop_direction,
                    cardinality_target=target,
                    cardinality_agg=cardinality_agg,
                    child_indices=None,
                )
            else:
                cardinality_terms = [
                    term for term in (min_qualified, max_qualified, exact_qualified) if term is not None
                ]
                if len(cardinality_terms) != 1 or on_class is None:
                    raise ValueError(f"Unsupported qualified cardinality shape for {expr.n3()}.")

                child_idx = compile_expr(on_class)
                if min_qualified is not None:
                    ctype = ConstraintType.MIN_CARDINALITY_RESTRICTION
                    target = min_qualified
                elif max_qualified is not None:
                    ctype = ConstraintType.MAX_CARDINALITY_RESTRICTION
                    target = max_qualified
                else:
                    ctype = ConstraintType.EXACT_CARDINALITY_RESTRICTION
                    target = exact_qualified

                idx = new_node(
                    ctype=ctype,
                    prop_idx=mapping.prop_to_idx[prop],
                    prop_direction=prop_direction,
                    cardinality_target=target,
                    cardinality_agg=cardinality_agg,
                    child_indices=[child_idx],
                )
            memo[expr] = idx
            return idx

        if isinstance(expr, (URIRef, BNode)):
            if expr not in mapping.class_to_idx:
                raise KeyError(f"Class expression {expr} not present in KGraph class mapping.")
            idx = new_node(
                ctype=ConstraintType.ATOMIC_CLASS,
                class_idx=mapping.class_to_idx[expr],
            )
            memo[expr] = idx
            return idx

        raise ValueError(f"Unsupported class expression: {expr!r}")

    root_children: List[int] = []

    for expr in ontology_graph.objects(target_term, RDFS.subClassOf):
        root_children.append(compile_expr(expr))

    for expr in ontology_graph.objects(target_term, OWL.equivalentClass):
        if expr != target_term:
            root_children.append(compile_expr(expr))

    for expr in ontology_graph.objects(target_term, OWL.disjointWith):
        root_children.append(
            new_node(
                ctype=ConstraintType.NEGATION,
                child_indices=[compile_expr(expr)],
            )
        )

    for expr in ontology_graph.subjects(OWL.disjointWith, target_term):
        if expr != target_term:
            root_children.append(
                new_node(
                    ctype=ConstraintType.NEGATION,
                    child_indices=[compile_expr(expr)],
                )
            )

    for expr in ontology_graph.objects(target_term, OWL.intersectionOf):
        for member in _rdf_list_members(ontology_graph, expr):
            root_children.append(compile_expr(member))

    if not root_children:
        base_idx = compile_expr(target_term)
    elif len(root_children) == 1:
        base_idx = root_children[0]
    else:
        base_idx = new_node(
            ctype=ConstraintType.INTERSECTION,
            child_indices=root_children,
            intersection_agg=intersection_agg,
        )

    if not augment_property_domain_range:
        root_idx = base_idx
        return ConstraintDAG(nodes=nodes, root_idx=root_idx, layers=_compute_layers(nodes))

    witness_children: List[int] = []
    constraint_children: List[int] = []
    property_axioms = collect_property_expression_axioms(ontology_graph)
    subclass_supers = _compute_transitive_super_map(ontology_graph, RDFS.subClassOf)
    entailment_memo: Dict[Tuple[str, str], bool] = {}

    for prop, axioms in property_axioms.items():
        if any(
            _expression_entails_target_class(
                ontology_graph,
                expr,
                target_term,
                subclass_supers,
                entailment_memo,
            )
            for expr in axioms.domain_expressions
        ):
            witness_children.append(
                compile_property_restriction(
                    prop=prop,
                    prop_direction=TraversalDirection.FORWARD,
                    child_idx=top_node(),
                    universal=False,
                )
            )
            for range_expr in axioms.range_expressions:
                constraint_children.append(
                    compile_property_restriction(
                        prop=prop,
                        prop_direction=TraversalDirection.FORWARD,
                        child_idx=compile_expr(range_expr),
                        universal=True,
                )
            )

        if any(
            _expression_entails_target_class(
                ontology_graph,
                expr,
                target_term,
                subclass_supers,
                entailment_memo,
            )
            for expr in axioms.range_expressions
        ):
            witness_children.append(
                compile_property_restriction(
                    prop=prop,
                    prop_direction=TraversalDirection.BACKWARD,
                    child_idx=top_node(),
                    universal=False,
                )
            )
            for domain_expr in axioms.domain_expressions:
                constraint_children.append(
                    compile_property_restriction(
                        prop=prop,
                        prop_direction=TraversalDirection.BACKWARD,
                        child_idx=compile_expr(domain_expr),
                        universal=True,
                    )
                )

    evidence_children = [base_idx] + witness_children
    evidence_idx = (
        evidence_children[0]
        if len(evidence_children) == 1
        else new_node(
            ctype=ConstraintType.UNION,
            child_indices=list(dict.fromkeys(evidence_children)),
        )
    )

    final_children = [evidence_idx] + list(dict.fromkeys(constraint_children))
    root_idx = (
        final_children[0]
        if len(final_children) == 1
        else new_node(
            ctype=ConstraintType.INTERSECTION,
            child_indices=final_children,
            intersection_agg=intersection_agg,
        )
    )

    return ConstraintDAG(nodes=nodes, root_idx=root_idx, layers=_compute_layers(nodes))


def _has_nontrivial_definition(ontology_graph: Graph, class_term: URIRef) -> bool:
    for expr in ontology_graph.objects(class_term, RDFS.subClassOf):
        if isinstance(expr, BNode):
            return True
    for expr in ontology_graph.objects(class_term, OWL.equivalentClass):
        if isinstance(expr, BNode):
            return True
    if any(True for _ in ontology_graph.objects(class_term, OWL.intersectionOf)):
        return True
    if any(True for _ in ontology_graph.objects(class_term, OWL.complementOf)):
        return True
    if any(True for _ in ontology_graph.objects(class_term, OWL.disjointWith)):
        return True
    if any(True for _ in ontology_graph.subjects(OWL.disjointWith, class_term)):
        return True
    return False


def collect_inferable_named_classes(ontology_graph: Graph) -> List[URIRef]:
    targets: set[URIRef] = set()
    for class_term in _collect_named_class_terms(ontology_graph):
        if isinstance(class_term, URIRef) and _has_nontrivial_definition(ontology_graph, class_term):
            targets.add(class_term)
    return sorted(targets, key=str)


def materialize_supported_class_inferences(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    include_literals: bool = True,
    include_type_edges: bool = False,
    materialize_hierarchy: bool = True,
    materialize_target_roles: bool = False,
    target_classes: Optional[Sequence[str | URIRef]] = None,
    threshold: float = 0.999,
    max_iterations: int = 10,
    device: str = "cpu",
) -> ClassMaterializationResult:
    """
    Materialize named-class assertions for the supported OWL fragment by:
    - building a reasoning dataset
    - compiling supported named class definitions into DAGs
    - evaluating them over the current ABox
    - adding new rdf:type assertions
    - iterating to a fixpoint

    This uses the current compiler's definitional interpretation for supported
    named classes. It is a practical Horn-style materialization step, not a full
    OWL reasoner.
    """

    current_data_graph = Graph()
    for triple in data_graph:
        current_data_graph.add(triple)

    inferred_assertions: List[Tuple[Identifier, URIRef]] = []
    iterations = 0
    closure: Optional[TargetDependencyClosure] = None
    if target_classes:
        analysis_dataset = build_reasoning_dataset_from_graphs(
            schema_graph=schema_graph,
            data_graph=current_data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            materialize_target_roles=False,
        )
        closure = compute_target_dependency_closure(
            analysis_dataset.ontology_graph,
            analysis_dataset.mapping,
            target_classes,
        )

    while iterations < max_iterations:
        iterations += 1
        dataset = build_reasoning_dataset_from_graphs(
            schema_graph=schema_graph,
            data_graph=current_data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            materialize_target_roles=materialize_target_roles,
            target_classes=target_classes,
            dependency_closure=closure,
        )

        target_classes_to_materialize = (
            closure.inferable_named_classes if closure is not None else collect_inferable_named_classes(dataset.ontology_graph)
        )
        if not target_classes_to_materialize:
            return ClassMaterializationResult(
                dataset=dataset,
                inferred_assertions=inferred_assertions,
                iterations=iterations,
            )

        from .dag_reasoner import DAGReasoner

        reasoner = DAGReasoner(dataset.kg, device=device)
        for class_term in target_classes_to_materialize:
            dag = compile_class_to_dag(dataset.ontology_graph, dataset.mapping, class_term)
            reasoner.add_concept(str(class_term), dag)

        scores = reasoner.evaluate_all().detach().cpu()

        additions_this_round: List[Tuple[Identifier, URIRef]] = []
        for class_col, class_term in enumerate(target_classes_to_materialize):
            for node_idx, node_term in enumerate(dataset.mapping.node_terms):
                if float(scores[node_idx, class_col].item()) < threshold:
                    continue
                triple = (node_term, RDF.type, class_term)
                if triple in current_data_graph:
                    continue
                current_data_graph.add(triple)
                additions_this_round.append((node_term, class_term))

        if not additions_this_round:
            return ClassMaterializationResult(
                dataset=dataset,
                inferred_assertions=inferred_assertions,
                iterations=iterations,
            )

        inferred_assertions.extend(additions_this_round)

    dataset = build_reasoning_dataset_from_graphs(
        schema_graph=schema_graph,
        data_graph=current_data_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_target_roles=materialize_target_roles,
        target_classes=target_classes,
        dependency_closure=closure,
    )
    return ClassMaterializationResult(
        dataset=dataset,
        inferred_assertions=inferred_assertions,
        iterations=iterations,
    )


def materialize_positive_sufficient_class_inferences(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    include_literals: bool = True,
    include_type_edges: bool = False,
    materialize_hierarchy: bool = True,
    materialize_target_roles: bool = False,
    target_classes: Optional[Sequence[str | URIRef]] = None,
    threshold: float = 0.999,
    max_iterations: int = 10,
    device: str = "cpu",
) -> ClassMaterializationResult:
    """
    Materialize positive OWA-style class assertions from normalized sufficient
    conditions over the supported Horn-friendly fragment.

    This pass treats extracted antecedents as sufficient evidence for the target
    class and iterates to a fixpoint. It is intentionally separate from the
    existing definitional / necessary-condition materializer.
    """

    current_data_graph = Graph()
    for triple in data_graph:
        current_data_graph.add(triple)

    inferred_assertions: List[Tuple[Identifier, URIRef]] = []
    iterations = 0
    closure: Optional[TargetDependencyClosure] = None
    target_terms = [URIRef(term) if isinstance(term, str) else term for term in target_classes] if target_classes else None
    base_rule_set = collect_normalized_sufficient_condition_rules(merge_rdflib_graphs((schema_graph, data_graph)))
    if target_terms:
        closure = compute_sufficient_rule_dependency_closure(base_rule_set, target_terms)

    while iterations < max_iterations:
        iterations += 1
        dataset = build_reasoning_dataset_from_graphs(
            schema_graph=schema_graph,
            data_graph=current_data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=materialize_hierarchy,
            materialize_target_roles=materialize_target_roles,
            target_classes=target_terms,
            dependency_closure=closure,
        )

        dataset_rule_set = collect_normalized_sufficient_condition_rules(dataset.ontology_graph)
        target_classes_to_materialize = (
            [
                class_term
                for class_term in closure.inferable_named_classes
                if any(rule.consequent_class == class_term for rule in dataset_rule_set.rules)
            ]
            if closure is not None
            else sorted({rule.consequent_class for rule in dataset_rule_set.rules}, key=str)
        )
        if not target_classes_to_materialize:
            return ClassMaterializationResult(
                dataset=dataset,
                inferred_assertions=inferred_assertions,
                iterations=iterations,
            )

        from .dag_reasoner import DAGReasoner

        reasoner = DAGReasoner(dataset.kg, device=device)
        for class_term in target_classes_to_materialize:
            dag = compile_sufficient_condition_dag(
                dataset.ontology_graph,
                dataset.mapping,
                class_term,
                rule_set=dataset_rule_set,
            )
            reasoner.add_concept(str(class_term), dag)

        scores = reasoner.evaluate_all().detach().cpu()

        additions_this_round: List[Tuple[Identifier, URIRef]] = []
        for class_col, class_term in enumerate(target_classes_to_materialize):
            for node_idx, node_term in enumerate(dataset.mapping.node_terms):
                if float(scores[node_idx, class_col].item()) < threshold:
                    continue
                triple = (node_term, RDF.type, class_term)
                if triple in current_data_graph:
                    continue
                current_data_graph.add(triple)
                additions_this_round.append((node_term, class_term))

        if not additions_this_round:
            return ClassMaterializationResult(
                dataset=dataset,
                inferred_assertions=inferred_assertions,
                iterations=iterations,
            )

        inferred_assertions.extend(additions_this_round)

    dataset = build_reasoning_dataset_from_graphs(
        schema_graph=schema_graph,
        data_graph=current_data_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_target_roles=materialize_target_roles,
        target_classes=target_terms,
        dependency_closure=closure,
    )
    return ClassMaterializationResult(
        dataset=dataset,
        inferred_assertions=inferred_assertions,
        iterations=iterations,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Load RDF/OWL files into a KGraph.")
    parser.add_argument("paths", nargs="*", help="RDF/OWL files to merge and load")
    parser.add_argument(
        "--schema",
        nargs="+",
        default=None,
        help="Schema / ontology files for a clean TBox load.",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=None,
        help="Instance data files for a clean ABox load.",
    )
    parser.add_argument(
        "--exclude-literals",
        action="store_true",
        help="Drop triples whose object is a literal instead of lifting literals to nodes.",
    )
    parser.add_argument(
        "--include-type-edges",
        action="store_true",
        help="Also materialize rdf:type as a normal property edge in addition to node_types.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="How many sample nodes/properties/classes to print.",
    )
    parser.add_argument(
        "--compile-class",
        type=str,
        default=None,
        help="Optional class IRI to compile into a ConstraintDAG after loading.",
    )
    parser.add_argument(
        "--compile-sufficient-class",
        type=str,
        default=None,
        help="Optional class IRI to compile using the normalized sufficient-condition rule view.",
    )
    parser.add_argument(
        "--describe-property",
        type=str,
        default=None,
        help="Optional property IRI to print parsed rdfs:domain / rdfs:range expressions for.",
    )
    parser.add_argument(
        "--describe-sufficient-rules",
        action="store_true",
        help="Print the normalized sufficient-condition rule schema extracted from the ontology.",
    )
    parser.add_argument(
        "--no-materialize-hierarchy",
        action="store_true",
        help="Disable subclass/subproperty materialization into the instance graph.",
    )

    args = parser.parse_args()

    if (args.schema is not None or args.data is not None) and args.paths:
        parser.error("Use either positional paths or --schema/--data, not both.")

    if args.schema is not None or args.data is not None:
        dataset = load_reasoning_dataset(
            schema_paths=args.schema,
            data_paths=args.data,
            include_literals=not args.exclude_literals,
            include_type_edges=args.include_type_edges,
            materialize_hierarchy=not args.no_materialize_hierarchy,
        )
        kg = dataset.kg
        mapping = dataset.mapping
        ontology_graph = dataset.ontology_graph
    else:
        if not args.paths:
            parser.error("Provide RDF paths, or use --schema/--data.")
        kg, mapping, ontology_graph = load_kgraph_from_rdf(
            args.paths,
            include_literals=not args.exclude_literals,
            include_type_edges=args.include_type_edges,
        )

    print(summarize_loaded_kgraph(kg, mapping, max_items=args.max_items))
    if args.compile_class is not None:
        dag = compile_class_to_dag(ontology_graph, mapping, args.compile_class)
        print("")
        print(describe_constraint_dag(dag, mapping))
    if args.compile_sufficient_class is not None:
        dag = compile_sufficient_condition_dag(ontology_graph, mapping, args.compile_sufficient_class)
        print("")
        print(describe_constraint_dag(dag, mapping))
    if args.describe_property is not None:
        property_axioms = collect_property_expression_axioms(ontology_graph)
        property_term = URIRef(args.describe_property)
        axioms = property_axioms.get(property_term)
        print("")
        if axioms is None:
            print(f"property={property_term.n3()}")
            print("domains=[]")
            print("ranges=[]")
        else:
            print(describe_property_expression_axioms(ontology_graph, axioms))
    if args.describe_sufficient_rules:
        print("")
        print(describe_normalized_sufficient_rule_set(collect_normalized_sufficient_condition_rules(ontology_graph)))


if __name__ == "__main__":
    main()
