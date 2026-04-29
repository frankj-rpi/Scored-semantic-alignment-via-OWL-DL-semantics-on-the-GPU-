from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.graph import ReadOnlyGraphAggregate
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
class GraphBuildScanCache:
    node_terms_set: set[Identifier]
    prop_terms_set: set[URIRef]
    class_terms_set: set[Identifier]
    datatype_terms_set: set[URIRef]
    type_pairs: List[Tuple[Identifier, Identifier]]
    type_edge_triples: List[Tuple[Identifier, URIRef, Identifier]]
    edge_triples: List[Tuple[Identifier, URIRef, Identifier]]
    edge_triples_by_pred: Dict[URIRef, List[Tuple[Identifier, Identifier]]]
    negative_helper_edges: List[Tuple[URIRef, Identifier, Identifier]]


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
class NamedClassDependencyAnalysis:
    named_classes: List[URIRef]
    equivalence_members: Dict[URIRef, List[URIRef]]
    canonical_map: Dict[URIRef, URIRef]
    direct_dependencies: Dict[URIRef, List[URIRef]]
    cycle_components: List[List[URIRef]]
    cycle_component_by_class: Dict[URIRef, List[URIRef]]
    reaches_cycle_by_class: Dict[URIRef, bool]


@dataclass
class OntologyCompileContext:
    dependency_analysis: NamedClassDependencyAnalysis
    direct_subs: Dict[URIRef, set[URIRef]]
    inverse_props: Dict[URIRef, set[URIRef]]
    chains_by_conclusion: Dict[URIRef, List[Tuple[URIRef, ...]]]
    transitive_props: set[URIRef]
    functional_props: set[URIRef]
    sameas_equivalence_map: Dict[Identifier, List[Identifier]]
    property_axioms: Dict[URIRef, PropertyExpressionAxioms]
    subclass_supers: Dict[URIRef, set[URIRef]]
    target_root_expressions_cache: Dict[URIRef, List[Identifier]] = field(default_factory=dict)
    expression_support_cache: Dict[Identifier, bool] = field(default_factory=dict)
    target_support_cache: Dict[Tuple[URIRef, bool], bool] = field(default_factory=dict)
    entailment_cache: Dict[Tuple[str, str], bool] = field(default_factory=dict)


@dataclass
class ClassMaterializationResult:
    dataset: ReasoningDataset
    inferred_assertions: List[Tuple[Identifier, URIRef]]
    iterations: int
    timings: Optional["PositiveMaterializationTimings"] = None


@dataclass
class NegativeBlockerSpec:
    target_class: URIRef
    blocker_classes: List[URIRef]
    blocker_nominal_members: List[Identifier]
    functional_data_requirements: List[Tuple[URIRef, Literal]]
    exact_property_requirements: List[Tuple[URIRef, Identifier]]
    skipped_negative_axioms: List[str]


@dataclass
class BlockedClassAssertion:
    node_term: Identifier
    target_class: URIRef
    blocker_class: URIRef


@dataclass
class NegativeBlockerResult:
    dataset: ReasoningDataset
    blocker_specs: Dict[URIRef, NegativeBlockerSpec]
    blocked_assertions: List[BlockedClassAssertion]
    conflicting_positive_assertions: List[BlockedClassAssertion]


@dataclass
class ClassAssignmentStatus:
    node_term: Identifier
    target_class: URIRef
    asserted: bool = False
    positively_derived: bool = False
    blocked: bool = False
    conflicted: bool = False
    blocker_classes: List[URIRef] = None

    def __post_init__(self) -> None:
        if self.blocker_classes is None:
            self.blocker_classes = []


class ConflictPolicy(Enum):
    REPORT_ONLY = "report_only"
    SUPPRESS_DERIVED_KEEP_ASSERTED = "suppress_derived_keep_asserted"
    STRICT_FAIL_ON_CONFLICT = "strict_fail_on_conflict"


@dataclass
class ConflictPolicyResult:
    policy: ConflictPolicy
    emitted_assignments: List[ClassAssignmentStatus]
    emitted_derived_assertions: List[Tuple[Identifier, URIRef]]
    suppressed_derived_assignments: List[ClassAssignmentStatus]
    asserted_conflicts: List[ClassAssignmentStatus]
    hard_conflicts: List[ClassAssignmentStatus]
    failed: bool = False
    failure_reason: Optional[str] = None


@dataclass
class StratifiedMaterializationResult:
    positive_result: ClassMaterializationResult
    negative_result: NegativeBlockerResult
    assignment_statuses: List[ClassAssignmentStatus]
    policy_result: ConflictPolicyResult
    timings: Optional["StratifiedMaterializationTimings"] = None


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
    HAS_SELF = "has_self"
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
    materialize_sameas: PreprocessingPassDecision
    materialize_haskey_equality: PreprocessingPassDecision
    materialize_reflexive_properties: PreprocessingPassDecision
    materialize_target_roles: PreprocessingPassDecision
    augment_property_domain_range: PreprocessingPassDecision


@dataclass
class PreprocessingTimings:
    ontology_merge_elapsed_ms: float = 0.0
    schema_cache_elapsed_ms: float = 0.0
    preprocessing_plan_elapsed_ms: float = 0.0
    hierarchy_elapsed_ms: float = 0.0
    atomic_domain_range_elapsed_ms: float = 0.0
    horn_safe_domain_range_elapsed_ms: float = 0.0
    sameas_elapsed_ms: float = 0.0
    reflexive_elapsed_ms: float = 0.0
    target_role_elapsed_ms: float = 0.0
    kgraph_build_elapsed_ms: float = 0.0
    mapping_vocab_collect_elapsed_ms: float = 0.0
    mapping_graph_scan_elapsed_ms: float = 0.0
    mapping_sort_elapsed_ms: float = 0.0
    mapping_index_elapsed_ms: float = 0.0
    kgraph_edge_bucket_elapsed_ms: float = 0.0
    kgraph_negative_helper_elapsed_ms: float = 0.0
    kgraph_literal_feature_elapsed_ms: float = 0.0
    kgraph_adjacency_elapsed_ms: float = 0.0
    sameas_passes_elapsed_ms: List[float] = field(default_factory=list)

    @property
    def preprocessing_elapsed_ms(self) -> float:
        return (
            self.ontology_merge_elapsed_ms
            + self.schema_cache_elapsed_ms
            + self.preprocessing_plan_elapsed_ms
            + self.hierarchy_elapsed_ms
            + self.atomic_domain_range_elapsed_ms
            + self.horn_safe_domain_range_elapsed_ms
            + self.sameas_elapsed_ms
            + self.reflexive_elapsed_ms
            + self.target_role_elapsed_ms
        )

    @property
    def dataset_build_elapsed_ms(self) -> float:
        return self.preprocessing_elapsed_ms + self.kgraph_build_elapsed_ms


@dataclass
class MaterializationIterationTiming:
    iteration: int
    dataset_refresh_count: int = 0
    dataset_build_elapsed_ms: float = 0.0
    ontology_merge_elapsed_ms: float = 0.0
    hierarchy_elapsed_ms: float = 0.0
    atomic_domain_range_elapsed_ms: float = 0.0
    horn_safe_domain_range_elapsed_ms: float = 0.0
    sameas_elapsed_ms: float = 0.0
    reflexive_elapsed_ms: float = 0.0
    target_role_elapsed_ms: float = 0.0
    kgraph_build_elapsed_ms: float = 0.0
    reasoner_setup_elapsed_ms: float = 0.0
    dag_compile_elapsed_ms: float = 0.0
    dag_eval_elapsed_ms: float = 0.0
    assertion_update_elapsed_ms: float = 0.0
    sameas_passes_elapsed_ms: List[float] = field(default_factory=list)


@dataclass
class ReasoningBuildCache:
    subproperty_supers: Dict[URIRef, set[URIRef]]
    property_axioms: Dict[URIRef, PropertyExpressionAxioms]
    atomic_domain_consequents: Dict[URIRef, Tuple[URIRef, ...]]
    atomic_range_consequents: Dict[URIRef, Tuple[URIRef, ...]]
    atomic_domain_range_predicates: Tuple[URIRef, ...]
    horn_domain_consequents: Dict[URIRef, Tuple[URIRef, ...]]
    horn_range_consequents: Dict[URIRef, Tuple[URIRef, ...]]
    horn_domain_range_predicates: Tuple[URIRef, ...]
    horn_safe_named_axiom_consequents: Dict[URIRef, Tuple[URIRef, ...]]
    preprocessing_class_supers: Dict[URIRef, set[URIRef]]
    singleton_nominals: Dict[URIRef, Identifier]
    has_key_axioms: Dict[URIRef, Tuple[URIRef, ...]]
    reflexive_props: Tuple[URIRef, ...]
    all_different_pairs: Tuple[Tuple[Identifier, Identifier], ...]
    schema_property_terms: frozenset[URIRef]
    schema_class_terms: frozenset[Identifier]
    schema_datatype_terms: frozenset[URIRef]


LiteralKeySignature = Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...]


@dataclass
class SameAsIncrementalState:
    relevant_classes: set[URIRef]
    members_by_class: Dict[URIRef, set[Identifier]]
    literal_only_key_classes: set[URIRef]
    object_key_classes: set[URIRef]
    property_values_by_subject_prop: Dict[Tuple[Identifier, URIRef], set[Identifier]]
    literal_signatures_by_class_subject: Dict[URIRef, Dict[Identifier, LiteralKeySignature]]
    literal_members_by_signature: Dict[URIRef, Dict[LiteralKeySignature, set[Identifier]]]


def _compute_literal_key_signature(
    subj: Identifier,
    key_props: Sequence[URIRef],
    property_values_by_subject: Dict[Tuple[Identifier, URIRef], set[Identifier]],
) -> Optional[LiteralKeySignature]:
    signature_parts: List[Tuple[str, Tuple[Tuple[str, str], ...]]] = []
    for prop in key_props:
        raw_values = property_values_by_subject.get((subj, prop), set())
        if not raw_values:
            return None
        literal_values = tuple(
            sorted(
                {
                    ("L", value.n3())
                    for value in raw_values
                    if isinstance(value, Literal)
                }
            )
        )
        signature_parts.append((str(prop), literal_values))
    return tuple(signature_parts)


def _build_sameas_incremental_state(
    materialized: Graph,
    *,
    singleton_nominals: Dict[URIRef, List[Identifier]],
    has_key_axioms: Dict[URIRef, Tuple[URIRef, ...]],
) -> SameAsIncrementalState:
    relevant_classes = set(singleton_nominals.keys()) | set(has_key_axioms.keys())
    members_by_class: Dict[URIRef, set[Identifier]] = defaultdict(set)
    for subj, _pred, obj in materialized.triples((None, RDF.type, None)):
        if isinstance(obj, URIRef) and obj in relevant_classes and not _is_datatype_term(obj):
            members_by_class[obj].add(subj)

    key_subjects: set[Identifier] = set()
    key_properties: set[URIRef] = set()
    for class_term, key_props in has_key_axioms.items():
        class_members = members_by_class.get(class_term, set())
        if len(class_members) < 2:
            continue
        key_subjects.update(class_members)
        key_properties.update(key_props)

    property_values_by_subject = _collect_property_values_by_subject(
        materialized,
        subjects=key_subjects if key_subjects else None,
        properties=key_properties if key_properties else None,
    )

    literal_only_key_classes: set[URIRef] = set()
    object_key_classes: set[URIRef] = set()
    literal_signatures_by_class_subject: Dict[URIRef, Dict[Identifier, LiteralKeySignature]] = defaultdict(dict)
    literal_members_by_signature: Dict[URIRef, Dict[LiteralKeySignature, set[Identifier]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for class_term, key_props in has_key_axioms.items():
        members = sorted(members_by_class.get(class_term, set()), key=_term_sort_key)
        if len(members) < 2:
            continue
        requires_object_iteration = False
        for subj in members:
            for prop in key_props:
                raw_values = property_values_by_subject.get((subj, prop), set())
                if any(not isinstance(value, Literal) for value in raw_values):
                    requires_object_iteration = True
                    break
            if requires_object_iteration:
                break
        if requires_object_iteration:
            object_key_classes.add(class_term)
            continue

        literal_only_key_classes.add(class_term)
        for subj in members:
            signature = _compute_literal_key_signature(subj, key_props, property_values_by_subject)
            if signature is None:
                continue
            literal_signatures_by_class_subject[class_term][subj] = signature
            literal_members_by_signature[class_term][signature].add(subj)

    return SameAsIncrementalState(
        relevant_classes=relevant_classes,
        members_by_class={class_term: set(members) for class_term, members in members_by_class.items()},
        literal_only_key_classes=literal_only_key_classes,
        object_key_classes=object_key_classes,
        property_values_by_subject_prop=property_values_by_subject,
        literal_signatures_by_class_subject={class_term: dict(subject_sigs) for class_term, subject_sigs in literal_signatures_by_class_subject.items()},
        literal_members_by_signature={
            class_term: {signature: set(subjects) for signature, subjects in sig_members.items()}
            for class_term, sig_members in literal_members_by_signature.items()
        },
    )


def _refresh_sameas_incremental_state_classes(
    materialized: Graph,
    *,
    state: SameAsIncrementalState,
    has_key_axioms: Dict[URIRef, Tuple[URIRef, ...]],
    classes: Sequence[URIRef],
) -> None:
    refresh_classes = {class_term for class_term in classes if class_term in state.relevant_classes}
    if not refresh_classes:
        return

    for class_term in refresh_classes:
        members: set[Identifier] = set()
        for subj, _pred, obj in materialized.triples((None, RDF.type, class_term)):
            members.add(subj)
        state.members_by_class[class_term] = members

    key_subjects: set[Identifier] = set()
    key_properties: set[URIRef] = set()
    for class_term in refresh_classes:
        if class_term not in has_key_axioms:
            continue
        members = state.members_by_class.get(class_term, set())
        if len(members) < 2:
            state.literal_signatures_by_class_subject.pop(class_term, None)
            state.literal_members_by_signature.pop(class_term, None)
            state.literal_only_key_classes.discard(class_term)
            state.object_key_classes.discard(class_term)
            continue
        key_subjects.update(members)
        key_properties.update(has_key_axioms[class_term])

    if key_subjects and key_properties:
        refreshed_values = _collect_property_values_by_subject(
            materialized,
            subjects=key_subjects,
            properties=key_properties,
        )
        for key, values in refreshed_values.items():
            state.property_values_by_subject_prop[key] = values

    for class_term in refresh_classes:
        key_props = has_key_axioms.get(class_term)
        if key_props is None:
            continue
        members = sorted(state.members_by_class.get(class_term, set()), key=_term_sort_key)
        if len(members) < 2:
            continue
        requires_object_iteration = False
        for subj in members:
            for prop in key_props:
                raw_values = state.property_values_by_subject_prop.get((subj, prop), set())
                if any(not isinstance(value, Literal) for value in raw_values):
                    requires_object_iteration = True
                    break
            if requires_object_iteration:
                break
        if requires_object_iteration:
            state.object_key_classes.add(class_term)
            state.literal_only_key_classes.discard(class_term)
            state.literal_signatures_by_class_subject.pop(class_term, None)
            state.literal_members_by_signature.pop(class_term, None)
            continue

        state.literal_only_key_classes.add(class_term)
        state.object_key_classes.discard(class_term)
        class_subject_signatures: Dict[Identifier, LiteralKeySignature] = {}
        class_signature_members: Dict[LiteralKeySignature, set[Identifier]] = defaultdict(set)
        for subj in members:
            signature = _compute_literal_key_signature(subj, key_props, state.property_values_by_subject_prop)
            if signature is None:
                continue
            class_subject_signatures[subj] = signature
            class_signature_members[signature].add(subj)
        state.literal_signatures_by_class_subject[class_term] = class_subject_signatures
        state.literal_members_by_signature[class_term] = {
            signature: set(subjects) for signature, subjects in class_signature_members.items()
        }


def _refresh_sameas_incremental_state_subjects(
    materialized: Graph,
    *,
    state: SameAsIncrementalState,
    has_key_axioms: Dict[URIRef, Tuple[URIRef, ...]],
    subjects: Sequence[Identifier],
    classes: Optional[Sequence[URIRef]] = None,
) -> None:
    active_class_filter = set(classes) if classes is not None else state.relevant_classes
    touched_subjects = [subj for subj in subjects if not isinstance(subj, Literal)]
    if not touched_subjects:
        return

    touched_classes: set[URIRef] = set()
    for subj in touched_subjects:
        for class_term in list(active_class_filter):
            if (subj, RDF.type, class_term) in materialized:
                state.members_by_class.setdefault(class_term, set()).add(subj)
                touched_classes.add(class_term)

    if not touched_classes:
        return

    for class_term in touched_classes:
        if class_term not in has_key_axioms:
            continue
        members = state.members_by_class.get(class_term, set())
        if len(members) < 2:
            state.literal_signatures_by_class_subject.pop(class_term, None)
            state.literal_members_by_signature.pop(class_term, None)
            state.literal_only_key_classes.discard(class_term)
            state.object_key_classes.discard(class_term)
            continue

        key_props = has_key_axioms[class_term]
        refreshed_values = _collect_property_values_by_subject(
            materialized,
            subjects=set(touched_subjects) & members,
            properties=set(key_props),
        )
        for key, values in refreshed_values.items():
            state.property_values_by_subject_prop[key] = values

        if class_term in state.object_key_classes:
            continue

        requires_object_iteration = False
        for subj in touched_subjects:
            if subj not in members:
                continue
            for prop in key_props:
                raw_values = state.property_values_by_subject_prop.get((subj, prop), set())
                if any(not isinstance(value, Literal) for value in raw_values):
                    requires_object_iteration = True
                    break
            if requires_object_iteration:
                break
        if requires_object_iteration:
            state.object_key_classes.add(class_term)
            state.literal_only_key_classes.discard(class_term)
            state.literal_signatures_by_class_subject.pop(class_term, None)
            state.literal_members_by_signature.pop(class_term, None)
            continue

        state.literal_only_key_classes.add(class_term)
        state.object_key_classes.discard(class_term)
        subject_signatures = state.literal_signatures_by_class_subject.setdefault(class_term, {})
        members_by_signature = state.literal_members_by_signature.setdefault(
            class_term,
            defaultdict(set),
        )
        for subj in touched_subjects:
            if subj not in members:
                continue
            old_signature = subject_signatures.get(subj)
            if old_signature is not None:
                existing = members_by_signature.get(old_signature)
                if existing is not None:
                    existing.discard(subj)
                    if not existing:
                        members_by_signature.pop(old_signature, None)
            new_signature = _compute_literal_key_signature(
                subj,
                key_props,
                state.property_values_by_subject_prop,
            )
            if new_signature is None:
                subject_signatures.pop(subj, None)
                continue
            subject_signatures[subj] = new_signature
            members_by_signature.setdefault(new_signature, set()).add(subj)


def _try_reuse_kgraph_with_updated_types(
    *,
    previous_dataset: ReasoningDataset,
    new_type_assertions: Optional[Sequence[Tuple[Identifier, URIRef]]] = None,
) -> Optional[KGraph]:
    """
    Reuse the previous KGraph structure when only rdf:type assertions changed.

    This is valid for the incremental stratified loop when:
    - equality was not rerun
    - non-type edges were not added or removed
    - include_type_edges is disabled, so rdf:type assertions affect only node_types
    """

    previous_mapping = previous_dataset.mapping
    previous_kg = previous_dataset.kg
    if new_type_assertions is not None:
        node_types = previous_kg.node_types.clone()
        for subj, class_term in new_type_assertions:
            subj_idx = previous_mapping.node_to_idx.get(subj)
            class_idx = previous_mapping.class_to_idx.get(class_term)
            if subj_idx is None or class_idx is None:
                return None
            node_types[subj_idx, class_idx] = 1.0
    else:
        num_nodes = len(previous_mapping.node_terms)
        num_classes = len(previous_mapping.class_terms)
        node_types = torch.zeros((num_nodes, num_classes), dtype=previous_kg.node_types.dtype)
        for subj, pred, obj in previous_dataset.data_graph:
            if pred != RDF.type:
                continue
            subj_idx = previous_mapping.node_to_idx.get(subj)
            class_idx = previous_mapping.class_to_idx.get(obj)
            if subj_idx is None or class_idx is None:
                return None
            node_types[subj_idx, class_idx] = 1.0

    return KGraph(
        num_nodes=previous_kg.num_nodes,
        offsets_p=previous_kg.offsets_p,
        neighbors_p=previous_kg.neighbors_p,
        node_types=node_types,
        literal_datatype_idx=previous_kg.literal_datatype_idx,
        literal_numeric_value=previous_kg.literal_numeric_value,
    )


def _try_reuse_kgraph_with_existing_mapping(
    *,
    effective_data_graph: Graph,
    previous_dataset: ReasoningDataset,
    include_literals: bool,
    include_type_edges: bool,
) -> Optional[KGraph]:
    """
    Rebuild adjacency against the existing mapping when node/property vocabulary
    stays stable, even if sameAs changed the edge set.
    """

    previous_mapping = previous_dataset.mapping
    previous_kg = previous_dataset.kg
    num_nodes = len(previous_mapping.node_terms)
    num_classes = len(previous_mapping.class_terms)
    node_types = torch.zeros((num_nodes, num_classes), dtype=previous_kg.node_types.dtype)
    indexed_edge_buckets: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for subj, pred, obj in effective_data_graph:
        subj_idx = previous_mapping.node_to_idx.get(subj)
        if subj_idx is None:
            return None

        if pred == RDF.type:
            class_idx = previous_mapping.class_to_idx.get(obj)
            if class_idx is None:
                return None
            node_types[subj_idx, class_idx] = 1.0
            if include_type_edges and not isinstance(obj, Literal):
                dst_idx = previous_mapping.node_to_idx.get(obj)
                prop_idx = previous_mapping.prop_to_idx.get(pred)
                if dst_idx is None or prop_idx is None:
                    return None
                indexed_edge_buckets[prop_idx].append((subj_idx, dst_idx))
            continue

        if isinstance(obj, Literal) and not include_literals:
            continue
        dst_idx = previous_mapping.node_to_idx.get(obj)
        prop_idx = previous_mapping.prop_to_idx.get(pred)
        if dst_idx is None or prop_idx is None:
            return None
        indexed_edge_buckets[prop_idx].append((subj_idx, dst_idx))

    for helper_prop, subj, obj in _collect_negative_property_assertion_edges(
        effective_data_graph,
        include_literals=include_literals,
    ):
        subj_idx = previous_mapping.node_to_idx.get(subj)
        dst_idx = previous_mapping.node_to_idx.get(obj)
        prop_idx = previous_mapping.prop_to_idx.get(helper_prop)
        if subj_idx is None or dst_idx is None or prop_idx is None:
            return None
        indexed_edge_buckets[prop_idx].append((subj_idx, dst_idx))

    offsets_p: List[torch.Tensor] = []
    neighbors_p: List[torch.Tensor] = []
    for prop_idx in range(len(previous_mapping.prop_terms)):
        edges = indexed_edge_buckets.get(prop_idx, [])
        if not edges:
            offsets_p.append(torch.zeros(num_nodes + 1, dtype=torch.int32))
            neighbors_p.append(torch.empty((0,), dtype=torch.int32))
            continue

        src_tensor = torch.tensor([src for src, _dst in edges], dtype=torch.int64)
        dst_tensor = torch.tensor([dst for _src, dst in edges], dtype=torch.int32)
        counts = torch.bincount(src_tensor, minlength=num_nodes)
        offsets = torch.zeros(num_nodes + 1, dtype=torch.int32)
        offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
        order = torch.argsort(src_tensor, stable=True)
        neighbors = dst_tensor[order]
        offsets_p.append(offsets)
        neighbors_p.append(neighbors)

    return KGraph(
        num_nodes=previous_kg.num_nodes,
        offsets_p=offsets_p,
        neighbors_p=neighbors_p,
        node_types=node_types,
        literal_datatype_idx=previous_kg.literal_datatype_idx,
        literal_numeric_value=previous_kg.literal_numeric_value,
    )


def _build_reasoning_dataset_from_preprocessed_graph(
    *,
    schema_graph: Graph,
    effective_data_graph: Graph,
    include_literals: bool,
    include_type_edges: bool,
    preprocessing_plan: PreprocessingPlan,
    build_cache: ReasoningBuildCache,
    rerun_sameas: bool,
    previous_dataset: Optional[ReasoningDataset] = None,
    new_type_assertions: Optional[Sequence[Tuple[Identifier, URIRef]]] = None,
    sameas_trigger_classes: Optional[Sequence[URIRef]] = None,
    build_ontology_graph: bool = True,
    sameas_state: Optional[SameAsIncrementalState] = None,
) -> ReasoningDataset:
    preprocessing_timings = PreprocessingTimings()
    effective_new_type_assertions: Optional[List[Tuple[Identifier, URIRef]]] = (
        list(new_type_assertions) if new_type_assertions is not None else None
    )

    if preprocessing_plan.materialize_hierarchy.enabled:
        t0 = perf_counter()
        _materialize_hierarchy_closure_in_place(
            effective_data_graph,
            subclass_supers=build_cache.preprocessing_class_supers,
            seed_type_assertions=new_type_assertions if not rerun_sameas else None,
            new_type_assertions=effective_new_type_assertions if not rerun_sameas else None,
        )
        preprocessing_timings.hierarchy_elapsed_ms += (perf_counter() - t0) * 1000.0

    sameas_changed = False
    if preprocessing_plan.materialize_sameas.enabled and rerun_sameas:
        trigger_set = set(sameas_trigger_classes or ())
        active_has_key_axioms = (
            build_cache.has_key_axioms
            if preprocessing_plan.materialize_haskey_equality.enabled
            else {}
        )
        selected_singleton_nominals = (
            {
                class_term: build_cache.singleton_nominals[class_term]
                for class_term in trigger_set
                if class_term in build_cache.singleton_nominals
            }
            if trigger_set
            else build_cache.singleton_nominals
        )
        selected_has_key_axioms = (
            {
                class_term: active_has_key_axioms[class_term]
                for class_term in trigger_set
                if class_term in active_has_key_axioms
            }
            if trigger_set
            else active_has_key_axioms
        )
        t0 = perf_counter()
        sameas_changed = _materialize_sameas_closure_in_place(
            effective_data_graph,
            ontology_graph=schema_graph,
            singleton_nominals=selected_singleton_nominals,
            has_key_axioms=selected_has_key_axioms,
            active_classes=trigger_set if trigger_set else None,
            incremental_state=sameas_state,
            new_type_assertions=new_type_assertions,
        )
        sameas_pass_ms = (perf_counter() - t0) * 1000.0
        preprocessing_timings.sameas_elapsed_ms += sameas_pass_ms
        preprocessing_timings.sameas_passes_elapsed_ms.append(sameas_pass_ms)

    if sameas_changed:
        sameas_new_domain_range_types: List[Tuple[Identifier, URIRef]] = []
        if preprocessing_plan.materialize_reflexive_properties.enabled:
            t0 = perf_counter()
            _materialize_reflexive_property_closure_in_place(
                effective_data_graph,
                reflexive_props=build_cache.reflexive_props,
            )
            preprocessing_timings.reflexive_elapsed_ms += (perf_counter() - t0) * 1000.0

        if preprocessing_plan.materialize_hierarchy.enabled:
            t0 = perf_counter()
            _materialize_hierarchy_closure_in_place(
                effective_data_graph,
                subclass_supers=build_cache.preprocessing_class_supers,
                subproperty_supers=build_cache.subproperty_supers,
            )
            preprocessing_timings.hierarchy_elapsed_ms += (perf_counter() - t0) * 1000.0

        if preprocessing_plan.materialize_horn_safe_domain_range.enabled:
            t0 = perf_counter()
            _materialize_domain_range_closure_in_place(
                effective_data_graph,
                domain_consequents=build_cache.horn_domain_consequents,
                range_consequents=build_cache.horn_range_consequents,
                active_predicates=build_cache.horn_domain_range_predicates,
                new_type_assertions=sameas_new_domain_range_types,
            )
            preprocessing_timings.horn_safe_domain_range_elapsed_ms += (perf_counter() - t0) * 1000.0
        elif preprocessing_plan.materialize_atomic_domain_range.enabled:
            t0 = perf_counter()
            _materialize_domain_range_closure_in_place(
                effective_data_graph,
                domain_consequents=build_cache.atomic_domain_consequents,
                range_consequents=build_cache.atomic_range_consequents,
                active_predicates=build_cache.atomic_domain_range_predicates,
                new_type_assertions=sameas_new_domain_range_types,
            )
            preprocessing_timings.atomic_domain_range_elapsed_ms += (perf_counter() - t0) * 1000.0

        if preprocessing_plan.materialize_hierarchy.enabled and sameas_new_domain_range_types:
            t0 = perf_counter()
            _materialize_hierarchy_closure_in_place(
                effective_data_graph,
                subclass_supers=build_cache.preprocessing_class_supers,
                seed_type_assertions=sameas_new_domain_range_types,
            )
            preprocessing_timings.hierarchy_elapsed_ms += (perf_counter() - t0) * 1000.0

    if build_ontology_graph:
        t0 = perf_counter()
        ontology_graph = aggregate_rdflib_graphs((schema_graph, effective_data_graph))
        preprocessing_timings.ontology_merge_elapsed_ms += (perf_counter() - t0) * 1000.0
    else:
        ontology_graph = previous_dataset.ontology_graph if previous_dataset is not None else schema_graph
    reused_kg: Optional[KGraph] = None
    mapping: Optional[RDFKGraphMapping] = None
    can_attempt_reuse = (
        previous_dataset is not None
        and previous_dataset.schema_graph is schema_graph
    )
    if can_attempt_reuse and not rerun_sameas and not include_type_edges:
        t0 = perf_counter()
        reused_kg = _try_reuse_kgraph_with_updated_types(
            previous_dataset=previous_dataset,
            new_type_assertions=effective_new_type_assertions,
        )
        preprocessing_timings.kgraph_build_elapsed_ms = (perf_counter() - t0) * 1000.0
        if reused_kg is not None:
            mapping = previous_dataset.mapping
    if reused_kg is None and can_attempt_reuse:
        t0 = perf_counter()
        reused_kg = _try_reuse_kgraph_with_existing_mapping(
            effective_data_graph=effective_data_graph,
            previous_dataset=previous_dataset,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
        )
        preprocessing_timings.kgraph_build_elapsed_ms = (perf_counter() - t0) * 1000.0
        if reused_kg is not None:
            mapping = previous_dataset.mapping

    if reused_kg is not None and mapping is not None:
        kg = reused_kg
    else:
        kg_source = effective_data_graph if len(effective_data_graph) > 0 else ontology_graph
        t0 = perf_counter()
        kg, mapping = rdflib_graph_to_kgraph(
            kg_source,
            vocab_graph=ontology_graph,
            vocab_source=ontology_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            preprocessing_timings=preprocessing_timings,
            schema_property_terms=build_cache.schema_property_terms,
            schema_class_terms=build_cache.schema_class_terms,
            schema_datatype_terms=build_cache.schema_datatype_terms,
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


@dataclass
class PositiveMaterializationTimings:
    iterations: int = 0
    initial_data_copy_elapsed_ms: float = 0.0
    initial_ontology_merge_elapsed_ms: float = 0.0
    rule_extraction_elapsed_ms: float = 0.0
    rule_index_elapsed_ms: float = 0.0
    schema_cache_elapsed_ms: float = 0.0
    preprocessing_plan_elapsed_ms: float = 0.0
    dependency_closure_elapsed_ms: float = 0.0
    sameas_state_init_elapsed_ms: float = 0.0
    ontology_merge_elapsed_ms: float = 0.0
    hierarchy_elapsed_ms: float = 0.0
    atomic_domain_range_elapsed_ms: float = 0.0
    horn_safe_domain_range_elapsed_ms: float = 0.0
    sameas_elapsed_ms: float = 0.0
    reflexive_elapsed_ms: float = 0.0
    target_role_elapsed_ms: float = 0.0
    kgraph_build_elapsed_ms: float = 0.0
    mapping_vocab_collect_elapsed_ms: float = 0.0
    mapping_graph_scan_elapsed_ms: float = 0.0
    mapping_sort_elapsed_ms: float = 0.0
    mapping_index_elapsed_ms: float = 0.0
    kgraph_edge_bucket_elapsed_ms: float = 0.0
    kgraph_negative_helper_elapsed_ms: float = 0.0
    kgraph_literal_feature_elapsed_ms: float = 0.0
    kgraph_adjacency_elapsed_ms: float = 0.0
    dataset_build_elapsed_ms: float = 0.0
    reasoner_setup_elapsed_ms: float = 0.0
    dag_compile_elapsed_ms: float = 0.0
    dag_eval_elapsed_ms: float = 0.0
    assertion_update_elapsed_ms: float = 0.0
    total_elapsed_ms: float = 0.0
    iteration_timings: List[MaterializationIterationTiming] = field(default_factory=list)
    final_dataset_timings: Optional[PreprocessingTimings] = None

    @property
    def avg_dataset_build_elapsed_ms(self) -> float:
        if not self.iteration_timings:
            return 0.0
        return sum(t.dataset_build_elapsed_ms for t in self.iteration_timings) / len(self.iteration_timings)

    @property
    def avg_dag_compile_elapsed_ms(self) -> float:
        if not self.iteration_timings:
            return 0.0
        return sum(t.dag_compile_elapsed_ms for t in self.iteration_timings) / len(self.iteration_timings)

    @property
    def avg_dag_eval_elapsed_ms(self) -> float:
        if not self.iteration_timings:
            return 0.0
        return sum(t.dag_eval_elapsed_ms for t in self.iteration_timings) / len(self.iteration_timings)

    @property
    def avg_reasoner_setup_elapsed_ms(self) -> float:
        if not self.iteration_timings:
            return 0.0
        return sum(t.reasoner_setup_elapsed_ms for t in self.iteration_timings) / len(self.iteration_timings)

    @property
    def avg_assertion_update_elapsed_ms(self) -> float:
        if not self.iteration_timings:
            return 0.0
        return sum(t.assertion_update_elapsed_ms for t in self.iteration_timings) / len(self.iteration_timings)

    @property
    def avg_total_elapsed_ms(self) -> float:
        if not self.iteration_timings:
            return 0.0
        return sum(
            t.dataset_build_elapsed_ms
            + t.reasoner_setup_elapsed_ms
            + t.dag_compile_elapsed_ms
            + t.dag_eval_elapsed_ms
            + t.assertion_update_elapsed_ms
            for t in self.iteration_timings
        ) / len(self.iteration_timings)


@dataclass
class StratifiedMaterializationTimings:
    positive_timings: PositiveMaterializationTimings
    negative_blocker_elapsed_ms: float = 0.0
    assignment_status_elapsed_ms: float = 0.0
    conflict_policy_elapsed_ms: float = 0.0
    total_elapsed_ms: float = 0.0


def _accumulate_preprocessing_timings(
    totals: PositiveMaterializationTimings,
    preprocessing: PreprocessingTimings,
    *,
    iteration_timing: Optional[MaterializationIterationTiming] = None,
) -> None:
    totals.ontology_merge_elapsed_ms += preprocessing.ontology_merge_elapsed_ms
    totals.hierarchy_elapsed_ms += preprocessing.hierarchy_elapsed_ms
    totals.atomic_domain_range_elapsed_ms += preprocessing.atomic_domain_range_elapsed_ms
    totals.horn_safe_domain_range_elapsed_ms += preprocessing.horn_safe_domain_range_elapsed_ms
    totals.sameas_elapsed_ms += preprocessing.sameas_elapsed_ms
    totals.reflexive_elapsed_ms += preprocessing.reflexive_elapsed_ms
    totals.target_role_elapsed_ms += preprocessing.target_role_elapsed_ms
    totals.kgraph_build_elapsed_ms += preprocessing.kgraph_build_elapsed_ms
    totals.mapping_vocab_collect_elapsed_ms += preprocessing.mapping_vocab_collect_elapsed_ms
    totals.mapping_graph_scan_elapsed_ms += preprocessing.mapping_graph_scan_elapsed_ms
    totals.mapping_sort_elapsed_ms += preprocessing.mapping_sort_elapsed_ms
    totals.mapping_index_elapsed_ms += preprocessing.mapping_index_elapsed_ms
    totals.kgraph_edge_bucket_elapsed_ms += preprocessing.kgraph_edge_bucket_elapsed_ms
    totals.kgraph_negative_helper_elapsed_ms += preprocessing.kgraph_negative_helper_elapsed_ms
    totals.kgraph_literal_feature_elapsed_ms += preprocessing.kgraph_literal_feature_elapsed_ms
    totals.kgraph_adjacency_elapsed_ms += preprocessing.kgraph_adjacency_elapsed_ms
    totals.dataset_build_elapsed_ms += preprocessing.dataset_build_elapsed_ms
    if iteration_timing is not None:
        iteration_timing.dataset_refresh_count += 1
        iteration_timing.dataset_build_elapsed_ms += preprocessing.dataset_build_elapsed_ms
        iteration_timing.ontology_merge_elapsed_ms += preprocessing.ontology_merge_elapsed_ms
        iteration_timing.hierarchy_elapsed_ms += preprocessing.hierarchy_elapsed_ms
        iteration_timing.atomic_domain_range_elapsed_ms += preprocessing.atomic_domain_range_elapsed_ms
        iteration_timing.horn_safe_domain_range_elapsed_ms += preprocessing.horn_safe_domain_range_elapsed_ms
        iteration_timing.sameas_elapsed_ms += preprocessing.sameas_elapsed_ms
        iteration_timing.reflexive_elapsed_ms += preprocessing.reflexive_elapsed_ms
        iteration_timing.target_role_elapsed_ms += preprocessing.target_role_elapsed_ms
        iteration_timing.kgraph_build_elapsed_ms += preprocessing.kgraph_build_elapsed_ms
        iteration_timing.sameas_passes_elapsed_ms.extend(preprocessing.sameas_passes_elapsed_ms)


def _ensure_sequence(paths: str | Path | Sequence[str | Path]) -> List[str]:
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(p) for p in paths]


def _finalize_rdflib_mapping(
    *,
    node_terms_set: set[Identifier],
    prop_terms_set: set[URIRef],
    class_terms_set: set[Identifier],
    datatype_terms_set: set[URIRef],
    preprocessing_timings: Optional[PreprocessingTimings] = None,
) -> RDFKGraphMapping:
    t0 = perf_counter()
    node_terms = sorted(node_terms_set, key=_term_sort_key)
    prop_terms = sorted(prop_terms_set, key=_term_sort_key)
    class_terms = sorted(class_terms_set, key=_term_sort_key)
    datatype_terms = sorted(datatype_terms_set, key=str)
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_sort_elapsed_ms += (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    mapping = RDFKGraphMapping(
        node_terms=node_terms,
        prop_terms=prop_terms,
        class_terms=class_terms,
        datatype_terms=datatype_terms,
        node_to_idx={term: idx for idx, term in enumerate(node_terms)},
        prop_to_idx={term: idx for idx, term in enumerate(prop_terms)},
        class_to_idx={term: idx for idx, term in enumerate(class_terms)},
        datatype_to_idx={term: idx for idx, term in enumerate(datatype_terms)},
    )
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_index_elapsed_ms += (perf_counter() - t0) * 1000.0
    return mapping


def _scan_graph_build_cache(
    graph: Graph,
    *,
    include_literals: bool,
    include_type_edges: bool,
    preprocessing_timings: Optional[PreprocessingTimings] = None,
    schema_property_terms: Optional[Sequence[URIRef]] = None,
    schema_class_terms: Optional[Sequence[Identifier]] = None,
    schema_datatype_terms: Optional[Sequence[URIRef]] = None,
    vocab_source: Optional[Graph] = None,
) -> GraphBuildScanCache:
    t0 = perf_counter()
    prop_terms_set: set[URIRef] = (
        set(schema_property_terms)
        if schema_property_terms is not None
        else _collect_property_terms(vocab_source if vocab_source is not None else graph)
    )
    class_terms_set: set[Identifier] = (
        set(schema_class_terms)
        if schema_class_terms is not None
        else _collect_named_class_terms(vocab_source if vocab_source is not None else graph)
    )
    datatype_terms_set: set[URIRef] = (
        set(schema_datatype_terms)
        if schema_datatype_terms is not None
        else _collect_datatype_terms(vocab_source if vocab_source is not None else graph)
    )
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_vocab_collect_elapsed_ms += (perf_counter() - t0) * 1000.0

    node_terms_set: set[Identifier] = set()
    type_pairs: List[Tuple[Identifier, Identifier]] = []
    type_edge_triples: List[Tuple[Identifier, URIRef, Identifier]] = []
    edge_triples: List[Tuple[Identifier, URIRef, Identifier]] = []
    edge_triples_by_pred: Dict[URIRef, List[Tuple[Identifier, Identifier]]] = defaultdict(list)

    t0 = perf_counter()
    for subj, pred, obj in graph:
        node_terms_set.add(subj)

        if pred == RDF.type:
            class_terms_set.add(obj)
            type_pairs.append((subj, obj))
            if include_type_edges and not isinstance(obj, Literal):
                node_terms_set.add(obj)
                prop_terms_set.add(pred)
                type_edge_triples.append((subj, pred, obj))
            continue

        prop_terms_set.add(pred)
        if isinstance(obj, Literal):
            if include_literals:
                node_terms_set.add(obj)
                if isinstance(obj.datatype, URIRef):
                    datatype_terms_set.add(obj.datatype)
                edge_triples.append((subj, pred, obj))
                edge_triples_by_pred[pred].append((subj, obj))
            continue

        node_terms_set.add(obj)
        edge_triples.append((subj, pred, obj))
        edge_triples_by_pred[pred].append((subj, obj))

    negative_helper_edges = _collect_negative_property_assertion_edges(
        graph,
        include_literals=include_literals,
    )
    for helper_prop, subj, obj in negative_helper_edges:
        prop_terms_set.add(helper_prop)
        node_terms_set.add(subj)
        node_terms_set.add(obj)
        if isinstance(obj, Literal) and isinstance(obj.datatype, URIRef):
            datatype_terms_set.add(obj.datatype)
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_graph_scan_elapsed_ms += (perf_counter() - t0) * 1000.0

    return GraphBuildScanCache(
        node_terms_set=node_terms_set,
        prop_terms_set=prop_terms_set,
        class_terms_set=class_terms_set,
        datatype_terms_set=datatype_terms_set,
        type_pairs=type_pairs,
        type_edge_triples=type_edge_triples,
        edge_triples=edge_triples,
        edge_triples_by_pred=edge_triples_by_pred,
        negative_helper_edges=negative_helper_edges,
    )


def _extend_graph_build_scan_cache(
    scan_cache: GraphBuildScanCache,
    added_triples: Sequence[Tuple[Identifier, Identifier, Identifier]],
    *,
    include_literals: bool,
    include_type_edges: bool,
) -> None:
    for subj, pred, obj in added_triples:
        scan_cache.node_terms_set.add(subj)

        if pred == RDF.type:
            scan_cache.class_terms_set.add(obj)
            scan_cache.type_pairs.append((subj, obj))
            if include_type_edges and not isinstance(obj, Literal):
                scan_cache.node_terms_set.add(obj)
                scan_cache.prop_terms_set.add(RDF.type)
                scan_cache.type_edge_triples.append((subj, RDF.type, obj))
            continue

        if not isinstance(pred, URIRef):
            continue
        scan_cache.prop_terms_set.add(pred)
        if isinstance(obj, Literal):
            if include_literals:
                scan_cache.node_terms_set.add(obj)
                if isinstance(obj.datatype, URIRef):
                    scan_cache.datatype_terms_set.add(obj.datatype)
                scan_cache.edge_triples.append((subj, pred, obj))
                scan_cache.edge_triples_by_pred[pred].append((subj, obj))
            continue

        scan_cache.node_terms_set.add(obj)
        scan_cache.edge_triples.append((subj, pred, obj))
        scan_cache.edge_triples_by_pred[pred].append((subj, obj))


def build_reasoning_build_cache(schema_graph: Graph) -> ReasoningBuildCache:
    schema_property_terms = frozenset(_collect_property_terms(schema_graph))
    schema_class_terms = frozenset(_collect_named_class_terms(schema_graph))
    schema_datatype_terms = frozenset(_collect_datatype_terms(schema_graph))
    subproperty_supers = _compute_transitive_super_map(schema_graph, RDFS.subPropertyOf)
    property_axioms = collect_property_expression_axioms(schema_graph)
    atomic_domain_consequents, atomic_range_consequents = _collect_atomic_property_consequents(
        schema_graph,
        property_axioms=property_axioms,
    )
    atomic_domain_range_predicates = tuple(
        sorted(
            set(atomic_domain_consequents.keys()) | set(atomic_range_consequents.keys()),
            key=str,
        )
    )
    horn_domain_consequents, horn_range_consequents = _collect_horn_safe_property_consequents(
        schema_graph,
        property_axioms=property_axioms,
    )
    horn_domain_range_predicates = tuple(
        sorted(
            set(horn_domain_consequents.keys()) | set(horn_range_consequents.keys()),
            key=str,
        )
    )
    horn_safe_named_axiom_consequents = _collect_horn_safe_named_class_axiom_consequents(schema_graph)
    preprocessing_class_supers = _compute_transitive_super_map_from_direct(
        {
            antecedent: set(consequents)
            for antecedent, consequents in horn_safe_named_axiom_consequents.items()
        }
    )
    singleton_nominals = _collect_singleton_nominal_axiom_consequents(schema_graph)
    has_key_axioms = _collect_has_key_axioms(schema_graph)
    reflexive_props = tuple(
        sorted(
            {
                prop
                for prop, _pred, _obj in schema_graph.triples((None, RDF.type, OWL.ReflexiveProperty))
                if isinstance(prop, URIRef)
            },
            key=str,
        )
    )
    all_different_pairs = tuple(_collect_all_different_pairs(schema_graph))
    return ReasoningBuildCache(
        subproperty_supers=subproperty_supers,
        property_axioms=property_axioms,
        atomic_domain_consequents=atomic_domain_consequents,
        atomic_range_consequents=atomic_range_consequents,
        atomic_domain_range_predicates=atomic_domain_range_predicates,
        horn_domain_consequents=horn_domain_consequents,
        horn_range_consequents=horn_range_consequents,
        horn_domain_range_predicates=horn_domain_range_predicates,
        horn_safe_named_axiom_consequents=horn_safe_named_axiom_consequents,
        preprocessing_class_supers=preprocessing_class_supers,
        singleton_nominals=singleton_nominals,
        has_key_axioms=has_key_axioms,
        reflexive_props=reflexive_props,
        all_different_pairs=all_different_pairs,
        schema_property_terms=schema_property_terms,
        schema_class_terms=schema_class_terms,
        schema_datatype_terms=schema_datatype_terms,
    )


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
        or _has_any_triple(graph, RDF.type, OWL.ReflexiveProperty)
    )


def _rdf_list_length(graph: Graph, head: Identifier) -> int:
    if not isinstance(head, BNode):
        return 0
    return sum(1 for _ in Collection(graph, head))


def _has_singleton_nominal_axioms(graph: Graph) -> bool:
    return any(
        _rdf_list_length(graph, head) == 1
        for head in graph.objects(None, OWL.oneOf)
    )


def _has_sameas_constructs(graph: Graph) -> bool:
    return (
        _has_any_triple(graph, OWL.sameAs)
        or _has_any_triple(graph, OWL.differentFrom)
        or _has_any_triple(graph, RDF.type, OWL.AllDifferent)
        or _has_singleton_nominal_axioms(graph)
    )


def _has_haskey_constructs(graph: Graph) -> bool:
    return _has_any_triple(graph, OWL.hasKey)


def _has_reflexive_property_axioms(graph: Graph) -> bool:
    return _has_any_triple(graph, RDF.type, OWL.ReflexiveProperty)


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
    materialize_sameas: Optional[bool] = None,
    materialize_sameas_policy: str = "auto",
    materialize_haskey_equality: Optional[bool] = None,
    materialize_haskey_equality_policy: str = "off",
    materialize_reflexive_properties: Optional[bool] = None,
    materialize_reflexive_properties_policy: str = "auto",
    materialize_target_roles: Optional[bool] = None,
    materialize_target_roles_policy: str = "auto",
    augment_property_domain_range: Optional[bool] = None,
    augment_property_domain_range_policy: str = "auto",
) -> PreprocessingPlan:
    has_hierarchy = _has_hierarchy_axioms(ontology_graph)
    has_domain_range = _has_domain_or_range_axioms(ontology_graph)
    has_negative = _has_negative_constructs(ontology_graph)
    has_sameas_constructs = _has_sameas_constructs(ontology_graph)
    has_haskey = _has_haskey_constructs(ontology_graph)
    has_reflexive_properties = _has_reflexive_property_axioms(ontology_graph)
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
    sameas_policy = _normalize_policy(
        materialize_sameas,
        materialize_sameas_policy,
        default="auto",
    )
    haskey_policy = _normalize_policy(
        materialize_haskey_equality,
        materialize_haskey_equality_policy,
        default="off",
    )
    reflexive_policy = _normalize_policy(
        materialize_reflexive_properties,
        materialize_reflexive_properties_policy,
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

    if sameas_policy == "on":
        sameas_decision = PreprocessingPassDecision(
            name="materialize_sameas",
            policy=sameas_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif sameas_policy == "off":
        sameas_decision = PreprocessingPassDecision(
            name="materialize_sameas",
            policy=sameas_policy,
            enabled=False,
            reason="explicitly disabled",
        )
    else:
        sameas_decision = PreprocessingPassDecision(
            name="materialize_sameas",
            policy=sameas_policy,
            enabled=has_sameas_constructs,
            reason=(
                "auto: sameAs-style constructs detected (sameAs, nominals, or differentFrom)"
                if has_sameas_constructs
                else "auto: no explicit sameAs-style constructs detected"
            ),
        )

    if not sameas_decision.enabled:
        haskey_decision = PreprocessingPassDecision(
            name="materialize_haskey_equality",
            policy=haskey_policy,
            enabled=False,
            reason="suppressed because sameAs materialization is off",
        )
    elif haskey_policy == "on":
        haskey_decision = PreprocessingPassDecision(
            name="materialize_haskey_equality",
            policy=haskey_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif haskey_policy == "off":
        haskey_decision = PreprocessingPassDecision(
            name="materialize_haskey_equality",
            policy=haskey_policy,
            enabled=False,
            reason="explicitly disabled or EL-friendly default",
        )
    else:
        haskey_decision = PreprocessingPassDecision(
            name="materialize_haskey_equality",
            policy=haskey_policy,
            enabled=has_haskey,
            reason=(
                "auto: hasKey axioms detected"
                if has_haskey
                else "auto: no hasKey axioms detected"
            ),
        )

    if reflexive_policy == "on":
        reflexive_decision = PreprocessingPassDecision(
            name="materialize_reflexive_properties",
            policy=reflexive_policy,
            enabled=True,
            reason="explicitly enabled",
        )
    elif reflexive_policy == "off":
        reflexive_decision = PreprocessingPassDecision(
            name="materialize_reflexive_properties",
            policy=reflexive_policy,
            enabled=False,
            reason="explicitly disabled",
        )
    else:
        reflexive_decision = PreprocessingPassDecision(
            name="materialize_reflexive_properties",
            policy=reflexive_policy,
            enabled=has_reflexive_properties,
            reason=(
                "auto: reflexive object property axioms detected"
                if has_reflexive_properties
                else "auto: no reflexive object property axioms detected"
            ),
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
        materialize_sameas=sameas_decision,
        materialize_haskey_equality=haskey_decision,
        materialize_reflexive_properties=reflexive_decision,
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
        plan.materialize_sameas,
        plan.materialize_haskey_equality,
        plan.materialize_reflexive_properties,
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


def _literal_to_bool(value: Identifier | None) -> Optional[bool]:
    if not isinstance(value, Literal):
        return None
    try:
        python_value = value.toPython()
    except Exception:
        return None
    if isinstance(python_value, bool):
        return python_value
    rendered = str(python_value).strip().lower()
    if rendered in {"true", "1"}:
        return True
    if rendered in {"false", "0"}:
        return False
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


def aggregate_rdflib_graphs(graphs: Sequence[Graph]) -> Graph:
    live_graphs = [graph for graph in graphs if graph is not None]
    if not live_graphs:
        return Graph()
    if len(live_graphs) == 1:
        return live_graphs[0]
    return ReadOnlyGraphAggregate(live_graphs)


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


def _compute_transitive_super_map_from_direct(
    direct_supers: Dict[URIRef, set[URIRef]],
) -> Dict[URIRef, set[URIRef]]:
    nodes: set[URIRef] = set(direct_supers.keys())
    for supers in direct_supers.values():
        nodes.update(supers)

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
    *,
    subclass_supers: Optional[Dict[URIRef, set[URIRef]]] = None,
    subproperty_supers: Optional[Dict[URIRef, set[URIRef]]] = None,
    all_different_pairs: Optional[Sequence[Tuple[Identifier, Identifier]]] = None,
) -> Graph:
    """
    Materialize simple hierarchy entailments into the instance graph:
    - rdf:type propagation over rdfs:subClassOf
    - property propagation over rdfs:subPropertyOf
    """

    if subclass_supers is None:
        subclass_supers = _compute_transitive_super_map(ontology_graph, RDFS.subClassOf)
    if subproperty_supers is None:
        subproperty_supers = _compute_transitive_super_map(ontology_graph, RDFS.subPropertyOf)

    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    if all_different_pairs is None:
        all_different_pairs = _collect_all_different_pairs(ontology_graph)
    for left, right in all_different_pairs:
        materialized.add((left, OWL.differentFrom, right))
        materialized.add((right, OWL.differentFrom, left))

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
    *,
    subclass_supers: Optional[Dict[URIRef, set[URIRef]]] = None,
) -> Graph:
    """
    Materialize only rdf:type propagation over rdfs:subClassOf.
    """

    if subclass_supers is None:
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


def _materialize_hierarchy_closure_in_place(
    materialized: Graph,
    *,
    subclass_supers: Dict[URIRef, set[URIRef]],
    subproperty_supers: Optional[Dict[URIRef, set[URIRef]]] = None,
    all_different_pairs: Optional[Sequence[Tuple[Identifier, Identifier]]] = None,
    seed_type_assertions: Optional[Sequence[Tuple[Identifier, URIRef]]] = None,
    type_assertions: Optional[Sequence[Tuple[Identifier, Identifier]]] = None,
    property_triples: Optional[Sequence[Tuple[Identifier, URIRef, Identifier]]] = None,
    new_type_assertions: Optional[List[Tuple[Identifier, URIRef]]] = None,
    new_triples: Optional[List[Tuple[Identifier, Identifier, Identifier]]] = None,
) -> bool:
    changed = False
    if all_different_pairs is not None:
        for left, right in all_different_pairs:
            triple = (left, OWL.differentFrom, right)
            if triple not in materialized:
                materialized.add(triple)
                changed = True
                if new_triples is not None:
                    new_triples.append(triple)
            triple = (right, OWL.differentFrom, left)
            if triple not in materialized:
                materialized.add(triple)
                changed = True
                if new_triples is not None:
                    new_triples.append(triple)

    if seed_type_assertions is not None:
        for subj, class_term in seed_type_assertions:
            for super_class in subclass_supers.get(class_term, ()):
                triple = (subj, RDF.type, super_class)
                if triple not in materialized:
                    materialized.add(triple)
                    changed = True
                    if new_type_assertions is not None:
                        new_type_assertions.append((subj, super_class))
                    if new_triples is not None:
                        new_triples.append(triple)
    else:
        if type_assertions is None or (subproperty_supers is not None and property_triples is None):
            for subj, pred, obj in list(materialized):
                if pred == RDF.type and isinstance(obj, URIRef):
                    for super_class in subclass_supers.get(obj, ()):
                        triple = (subj, RDF.type, super_class)
                        if triple not in materialized:
                            materialized.add(triple)
                            changed = True
                            if new_type_assertions is not None:
                                new_type_assertions.append((subj, super_class))
                            if new_triples is not None:
                                new_triples.append(triple)
                elif subproperty_supers is not None and isinstance(pred, URIRef):
                    for super_prop in subproperty_supers.get(pred, ()):
                        triple = (subj, super_prop, obj)
                        if triple not in materialized:
                            materialized.add(triple)
                            changed = True
                            if new_triples is not None:
                                new_triples.append(triple)
        else:
            for subj, class_term in type_assertions:
                if not isinstance(class_term, URIRef):
                    continue
                for super_class in subclass_supers.get(class_term, ()):
                    triple = (subj, RDF.type, super_class)
                    if triple not in materialized:
                        materialized.add(triple)
                        changed = True
                        if new_type_assertions is not None:
                            new_type_assertions.append((subj, super_class))
                        if new_triples is not None:
                            new_triples.append(triple)
            if subproperty_supers is not None:
                for subj, pred, obj in property_triples:
                    for super_prop in subproperty_supers.get(pred, ()):
                        triple = (subj, super_prop, obj)
                        if triple not in materialized:
                            materialized.add(triple)
                            changed = True
                            if new_triples is not None:
                                new_triples.append(triple)
    return changed


def materialize_atomic_domain_range_closure(
    ontology_graph: Graph,
    data_graph: Graph,
    *,
    property_axioms: Optional[Dict[URIRef, PropertyExpressionAxioms]] = None,
    atomic_domain_consequents: Optional[Dict[URIRef, Tuple[URIRef, ...]]] = None,
    atomic_range_consequents: Optional[Dict[URIRef, Tuple[URIRef, ...]]] = None,
) -> Graph:
    """
    Materialize only Horn-safe atomic domain/range consequences:
    - (s P o) and domain(P, C) -> (s rdf:type C)
    - (s P o) and range(P, C) -> (o rdf:type C)

    Complex expressions such as unions are intentionally left to the query-time
    augmentation path.
    """

    if property_axioms is None:
        property_axioms = collect_property_expression_axioms(ontology_graph)
    if atomic_domain_consequents is None or atomic_range_consequents is None:
        atomic_domain_consequents, atomic_range_consequents = _collect_atomic_property_consequents(
            ontology_graph,
            property_axioms=property_axioms,
        )
    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    for subj, pred, obj in data_graph:
        if pred == RDF.type or not isinstance(pred, URIRef):
            continue

        for class_term in atomic_domain_consequents.get(pred, ()):
            materialized.add((subj, RDF.type, class_term))

        if isinstance(obj, Literal):
            continue

        for class_term in atomic_range_consequents.get(pred, ()):
            materialized.add((obj, RDF.type, class_term))

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


def _collect_atomic_property_consequents(
    ontology_graph: Graph,
    *,
    property_axioms: Optional[Dict[URIRef, PropertyExpressionAxioms]] = None,
) -> Tuple[Dict[URIRef, Tuple[URIRef, ...]], Dict[URIRef, Tuple[URIRef, ...]]]:
    if property_axioms is None:
        property_axioms = collect_property_expression_axioms(ontology_graph)

    domain_consequents: Dict[URIRef, Tuple[URIRef, ...]] = {}
    range_consequents: Dict[URIRef, Tuple[URIRef, ...]] = {}
    for prop, axioms in property_axioms.items():
        domain_terms = tuple(
            sorted(
                {
                    expr
                    for expr in axioms.domain_expressions
                    if isinstance(expr, URIRef) and not _is_datatype_term(expr)
                },
                key=str,
            )
        )
        range_terms = tuple(
            sorted(
                {
                    expr
                    for expr in axioms.range_expressions
                    if isinstance(expr, URIRef) and not _is_datatype_term(expr)
                },
                key=str,
            )
        )
        if domain_terms:
            domain_consequents[prop] = domain_terms
        if range_terms:
            range_consequents[prop] = range_terms
    return domain_consequents, range_consequents


def _collect_horn_safe_property_consequents(
    ontology_graph: Graph,
    *,
    property_axioms: Optional[Dict[URIRef, PropertyExpressionAxioms]] = None,
) -> Tuple[Dict[URIRef, Tuple[URIRef, ...]], Dict[URIRef, Tuple[URIRef, ...]]]:
    if property_axioms is None:
        property_axioms = collect_property_expression_axioms(ontology_graph)

    domain_consequents: Dict[URIRef, Tuple[URIRef, ...]] = {}
    range_consequents: Dict[URIRef, Tuple[URIRef, ...]] = {}
    for prop, axioms in property_axioms.items():
        domain_terms: set[URIRef] = set()
        range_terms: set[URIRef] = set()
        for expr in axioms.domain_expressions:
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, expr)
            if consequents is not None:
                domain_terms.update(consequents)
        for expr in axioms.range_expressions:
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, expr)
            if consequents is not None:
                range_terms.update(consequents)
        if domain_terms:
            domain_consequents[prop] = tuple(sorted(domain_terms, key=str))
        if range_terms:
            range_consequents[prop] = tuple(sorted(range_terms, key=str))
    return domain_consequents, range_consequents


def materialize_horn_safe_domain_range_closure(
    ontology_graph: Graph,
    data_graph: Graph,
    *,
    property_axioms: Optional[Dict[URIRef, PropertyExpressionAxioms]] = None,
    horn_domain_consequents: Optional[Dict[URIRef, Tuple[URIRef, ...]]] = None,
    horn_range_consequents: Optional[Dict[URIRef, Tuple[URIRef, ...]]] = None,
    horn_domain_range_predicates: Optional[Sequence[URIRef]] = None,
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

    if property_axioms is None:
        property_axioms = collect_property_expression_axioms(ontology_graph)
    if horn_domain_consequents is None or horn_range_consequents is None:
        horn_domain_consequents, horn_range_consequents = _collect_horn_safe_property_consequents(
            ontology_graph,
            property_axioms=property_axioms,
        )
    if horn_domain_range_predicates is None:
        horn_domain_range_predicates = tuple(
            sorted(
                set(horn_domain_consequents.keys()) | set(horn_range_consequents.keys()),
                key=str,
            )
        )
    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    for pred in horn_domain_range_predicates:
        for class_term in horn_domain_consequents.get(pred, ()):
            for subj, obj in data_graph.subject_objects(pred):
                materialized.add((subj, RDF.type, class_term))

        range_classes = horn_range_consequents.get(pred, ())
        if not range_classes:
            continue
        for subj, obj in data_graph.subject_objects(pred):
            if isinstance(obj, Literal):
                continue
            for class_term in range_classes:
                materialized.add((obj, RDF.type, class_term))

    return materialized


def _materialize_domain_range_closure_in_place(
    materialized: Graph,
    *,
    domain_consequents: Dict[URIRef, Tuple[URIRef, ...]],
    range_consequents: Dict[URIRef, Tuple[URIRef, ...]],
    active_predicates: Optional[Sequence[URIRef]] = None,
    property_triples: Optional[Sequence[Tuple[Identifier, URIRef, Identifier]]] = None,
    property_pairs_by_pred: Optional[Dict[URIRef, Sequence[Tuple[Identifier, Identifier]]]] = None,
    new_type_assertions: Optional[List[Tuple[Identifier, URIRef]]] = None,
    new_triples: Optional[List[Tuple[Identifier, Identifier, Identifier]]] = None,
) -> bool:
    changed = False
    if active_predicates is None:
        active_predicates = tuple(
            sorted(
                set(domain_consequents.keys()) | set(range_consequents.keys()),
                key=str,
            )
        )
    if property_pairs_by_pred is not None:
        for pred in active_predicates:
            pred_pairs = property_pairs_by_pred.get(pred, ())
            if not pred_pairs:
                continue
            domain_classes = domain_consequents.get(pred, ())
            range_classes = range_consequents.get(pred, ())
            for subj, obj in pred_pairs:
                for class_term in domain_classes:
                    triple = (subj, RDF.type, class_term)
                    if triple not in materialized:
                        materialized.add(triple)
                        changed = True
                        if new_type_assertions is not None:
                            new_type_assertions.append((subj, class_term))
                        if new_triples is not None:
                            new_triples.append(triple)
                if isinstance(obj, Literal):
                    continue
                for class_term in range_classes:
                    triple = (obj, RDF.type, class_term)
                    if triple not in materialized:
                        materialized.add(triple)
                        changed = True
                        if new_type_assertions is not None:
                            new_type_assertions.append((obj, class_term))
                        if new_triples is not None:
                            new_triples.append(triple)
    elif property_triples is None:
        for pred in active_predicates:
            domain_classes = domain_consequents.get(pred, ())
            range_classes = range_consequents.get(pred, ())
            if not domain_classes and not range_classes:
                continue
            for subj, obj in materialized.subject_objects(pred):
                for class_term in domain_classes:
                    triple = (subj, RDF.type, class_term)
                    if triple not in materialized:
                        materialized.add(triple)
                        changed = True
                        if new_type_assertions is not None:
                            new_type_assertions.append((subj, class_term))
                        if new_triples is not None:
                            new_triples.append(triple)
                if isinstance(obj, Literal):
                    continue
                for class_term in range_classes:
                    triple = (obj, RDF.type, class_term)
                    if triple not in materialized:
                        materialized.add(triple)
                        changed = True
                        if new_type_assertions is not None:
                            new_type_assertions.append((obj, class_term))
                        if new_triples is not None:
                            new_triples.append(triple)
    else:
        active_set = set(active_predicates)
        for subj, pred, obj in property_triples:
            if pred not in active_set:
                continue
            domain_classes = domain_consequents.get(pred, ())
            range_classes = range_consequents.get(pred, ())
            for class_term in domain_classes:
                triple = (subj, RDF.type, class_term)
                if triple not in materialized:
                    materialized.add(triple)
                    changed = True
                    if new_type_assertions is not None:
                        new_type_assertions.append((subj, class_term))
                    if new_triples is not None:
                        new_triples.append(triple)
            if isinstance(obj, Literal):
                continue
            for class_term in range_classes:
                triple = (obj, RDF.type, class_term)
                if triple not in materialized:
                    materialized.add(triple)
                    changed = True
                    if new_type_assertions is not None:
                        new_type_assertions.append((obj, class_term))
                    if new_triples is not None:
                        new_triples.append(triple)
    return changed


def _collect_horn_safe_named_class_axiom_consequents(
    ontology_graph: Graph,
) -> Dict[URIRef, List[URIRef]]:
    consequents_by_antecedent: Dict[URIRef, set[URIRef]] = defaultdict(set)

    for antecedent_class in sorted(_collect_named_class_terms(ontology_graph), key=str):
        if not isinstance(antecedent_class, URIRef) or _is_datatype_term(antecedent_class):
            continue

        for consequent_expr in ontology_graph.objects(antecedent_class, RDFS.subClassOf):
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, consequent_expr)
            if consequents is None:
                continue
            for consequent_class in consequents:
                if consequent_class != antecedent_class:
                    consequents_by_antecedent[antecedent_class].add(consequent_class)

        equivalent_exprs = set(ontology_graph.objects(antecedent_class, OWL.equivalentClass))
        equivalent_exprs.update(ontology_graph.subjects(OWL.equivalentClass, antecedent_class))
        for consequent_expr in equivalent_exprs:
            if consequent_expr == antecedent_class:
                continue
            consequents = _extract_horn_safe_named_class_consequents(ontology_graph, consequent_expr)
            if consequents is None:
                continue
            for consequent_class in consequents:
                if consequent_class != antecedent_class:
                    consequents_by_antecedent[antecedent_class].add(consequent_class)

    return {
        antecedent: sorted(consequents, key=str)
        for antecedent, consequents in consequents_by_antecedent.items()
    }


def materialize_horn_safe_named_class_axiom_closure(
    ontology_graph: Graph,
    data_graph: Graph,
    *,
    max_iterations: int = 64,
    consequents_by_antecedent: Optional[Dict[URIRef, List[URIRef]]] = None,
) -> Graph:
    """
    Materialize Horn-safe positive named-class consequences from class axioms.

    Supported forms:
    - A ⊑ B
    - A ⊑ (B ⊓ C ...)
    - A ≡ B
    - A ≡ (B ⊓ C ...)

    This is intentionally restricted to named-class consequents so it stays a
    lightweight positive closure pass suitable for preprocessing.
    """

    if consequents_by_antecedent is None:
        consequents_by_antecedent = _collect_horn_safe_named_class_axiom_consequents(ontology_graph)
    if not consequents_by_antecedent:
        return _copy_graph(data_graph)

    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        asserted_types: Dict[Identifier, set[URIRef]] = defaultdict(set)
        for subj, _pred, obj in materialized.triples((None, RDF.type, None)):
            if isinstance(obj, URIRef) and not _is_datatype_term(obj):
                asserted_types[subj].add(obj)

        additions: List[Tuple[Identifier, URIRef]] = []
        for subj, types_for_subj in asserted_types.items():
            for antecedent_class in list(types_for_subj):
                for consequent_class in consequents_by_antecedent.get(antecedent_class, ()):
                    if consequent_class not in types_for_subj:
                        additions.append((subj, consequent_class))

        if not additions:
            break

        changed = False
        for subj, consequent_class in additions:
            triple = (subj, RDF.type, consequent_class)
            if triple not in materialized:
                materialized.add(triple)
                changed = True
        if not changed:
            break

    return materialized


def _extract_singleton_nominal_member(
    ontology_graph: Graph,
    expr: Identifier,
) -> Optional[Identifier]:
    if not isinstance(expr, BNode):
        return None
    one_of_head = ontology_graph.value(expr, OWL.oneOf)
    if not isinstance(one_of_head, BNode):
        return None
    members = list(Collection(ontology_graph, one_of_head))
    if len(members) != 1:
        return None
    member = members[0]
    if isinstance(member, Literal):
        return None
    return member


def _extract_singleton_oneof_member(
    ontology_graph: Graph,
    expr: Identifier,
) -> Optional[Identifier]:
    if not isinstance(expr, BNode):
        return None
    one_of_head = ontology_graph.value(expr, OWL.oneOf)
    if not isinstance(one_of_head, BNode):
        return None
    members = list(Collection(ontology_graph, one_of_head))
    if len(members) != 1:
        return None
    return members[0]


def _collect_singleton_nominal_axiom_consequents(
    ontology_graph: Graph,
) -> Dict[URIRef, List[Identifier]]:
    consequents_by_antecedent: Dict[URIRef, set[Identifier]] = defaultdict(set)

    for antecedent_class in sorted(_collect_named_class_terms(ontology_graph), key=str):
        if not isinstance(antecedent_class, URIRef) or _is_datatype_term(antecedent_class):
            continue

        for consequent_expr in ontology_graph.objects(antecedent_class, RDFS.subClassOf):
            singleton_member = _extract_singleton_nominal_member(ontology_graph, consequent_expr)
            if singleton_member is not None:
                consequents_by_antecedent[antecedent_class].add(singleton_member)

        equivalent_exprs = set(ontology_graph.objects(antecedent_class, OWL.equivalentClass))
        equivalent_exprs.update(ontology_graph.subjects(OWL.equivalentClass, antecedent_class))
        for consequent_expr in equivalent_exprs:
            if consequent_expr == antecedent_class:
                continue
            singleton_member = _extract_singleton_nominal_member(ontology_graph, consequent_expr)
            if singleton_member is not None:
                consequents_by_antecedent[antecedent_class].add(singleton_member)

    return {
        antecedent: sorted(consequents, key=_term_sort_key)
        for antecedent, consequents in consequents_by_antecedent.items()
    }


def _collect_has_key_axioms(
    ontology_graph: Graph,
) -> Dict[URIRef, Tuple[URIRef, ...]]:
    axioms: Dict[URIRef, Tuple[URIRef, ...]] = {}
    for class_term, _pred, key_head in ontology_graph.triples((None, OWL.hasKey, None)):
        if not isinstance(class_term, URIRef) or not isinstance(key_head, BNode):
            continue
        props = tuple(
            prop
            for prop in Collection(ontology_graph, key_head)
            if isinstance(prop, URIRef)
        )
        if not props:
            continue
        axioms[class_term] = tuple(sorted(dict.fromkeys(props), key=str))
    return axioms


def _collect_property_values_by_subject(
    data_graph: Graph,
    *,
    subjects: Optional[set[Identifier]] = None,
    properties: Optional[set[URIRef]] = None,
) -> Dict[Tuple[Identifier, URIRef], set[Identifier]]:
    values_by_subject_prop: Dict[Tuple[Identifier, URIRef], set[Identifier]] = defaultdict(set)
    for subj, pred, obj in data_graph:
        if pred == RDF.type or not isinstance(pred, URIRef):
            continue
        if subjects is not None and subj not in subjects:
            continue
        if properties is not None and pred not in properties:
            continue
        values_by_subject_prop[(subj, pred)].add(obj)
    return values_by_subject_prop


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Identifier, Identifier] = {}

    def add(self, item: Identifier) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: Identifier) -> Identifier:
        parent = self.parent.get(item)
        if parent is None:
            self.parent[item] = item
            return item
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: Identifier, right: Identifier) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        rep_left, rep_right = sorted((root_left, root_right), key=_term_sort_key)
        self.parent[rep_right] = rep_left
        return True

    def groups(self) -> Dict[Identifier, List[Identifier]]:
        grouped: Dict[Identifier, List[Identifier]] = defaultdict(list)
        for item in list(self.parent.keys()):
            grouped[self.find(item)].append(item)
        return {
            root: sorted(members, key=_term_sort_key)
            for root, members in grouped.items()
        }


def materialize_sameas_closure(
    ontology_graph: Graph,
    data_graph: Graph,
    *,
    singleton_nominals: Optional[Dict[URIRef, List[Identifier]]] = None,
    has_key_axioms: Optional[Dict[URIRef, Tuple[URIRef, ...]]] = None,
    active_classes: Optional[Sequence[URIRef]] = None,
    incremental_state: Optional[SameAsIncrementalState] = None,
    new_type_assertions: Optional[Sequence[Tuple[Identifier, URIRef]]] = None,
) -> Graph:
    """
    Materialize a lightweight sameAs-style equality closure.

    Supported equality sources:
    - explicit `owl:sameAs`
    - singleton nominal consequences of the form `A ⊑ {a}` / `A ≡ {a}`

    This preserves original node identities and expands facts across each
    discovered equivalence class, rather than collapsing nodes to a single
    representative. That keeps the existing per-node query machinery intact.
    """

    materialized = Graph()
    for triple in data_graph:
        materialized.add(triple)
    _materialize_sameas_closure_in_place(
        materialized,
        ontology_graph=ontology_graph,
        singleton_nominals=singleton_nominals,
        has_key_axioms=has_key_axioms,
        active_classes=active_classes,
        incremental_state=incremental_state,
        new_type_assertions=new_type_assertions,
    )
    return materialized


def _materialize_sameas_closure_in_place(
    materialized: Graph,
    *,
    ontology_graph: Graph,
    singleton_nominals: Optional[Dict[URIRef, List[Identifier]]] = None,
    has_key_axioms: Optional[Dict[URIRef, Tuple[URIRef, ...]]] = None,
    active_classes: Optional[Sequence[URIRef]] = None,
    incremental_state: Optional[SameAsIncrementalState] = None,
    new_type_assertions: Optional[Sequence[Tuple[Identifier, URIRef]]] = None,
) -> bool:
    before_len = len(materialized)
    union_find = _UnionFind()
    if singleton_nominals is None:
        singleton_nominals = _collect_singleton_nominal_axiom_consequents(ontology_graph)
    if has_key_axioms is None:
        has_key_axioms = _collect_has_key_axioms(ontology_graph)

    for subj, _pred, obj in ontology_graph.triples((None, OWL.sameAs, None)):
        if isinstance(subj, Literal) or isinstance(obj, Literal):
            continue
        union_find.add(subj)
        union_find.add(obj)
        union_find.union(subj, obj)

    relevant_classes = set(active_classes) if active_classes is not None else (
        set(singleton_nominals.keys()) | set(has_key_axioms.keys())
    )
    if incremental_state is not None:
        if new_type_assertions is not None:
            for subj, class_term in new_type_assertions:
                if class_term in incremental_state.relevant_classes:
                    incremental_state.members_by_class.setdefault(class_term, set()).add(subj)
        members_by_class = {
            class_term: sorted(incremental_state.members_by_class.get(class_term, set()), key=_term_sort_key)
            for class_term in relevant_classes
        }
        for members in members_by_class.values():
            for subj in members:
                union_find.add(subj)
    else:
        members_by_class: Dict[URIRef, List[Identifier]] = defaultdict(list)
        for subj, _pred, obj in materialized.triples((None, RDF.type, None)):
            if not isinstance(obj, URIRef) or _is_datatype_term(obj):
                continue
            union_find.add(subj)
            if obj in relevant_classes:
                members_by_class[obj].append(subj)

    for class_term, members in members_by_class.items():
        for subj in members:
            for singleton_member in singleton_nominals.get(class_term, ()):
                if _has_differentfrom_pair(materialized, subj, singleton_member):
                    continue
                union_find.add(singleton_member)
                union_find.union(subj, singleton_member)

    if has_key_axioms:
        key_subjects: set[Identifier] = set()
        key_properties: set[URIRef] = set()
        members_by_key_class: Dict[URIRef, List[Identifier]] = {}
        literal_only_key_classes: set[URIRef] = set()
        object_key_classes: set[URIRef] = set()
        for class_term, key_props in has_key_axioms.items():
            members = sorted(members_by_class.get(class_term, []), key=_term_sort_key)
            if len(members) < 2:
                continue
            members_by_key_class[class_term] = members
            key_subjects.update(members)
            key_properties.update(key_props)

        if incremental_state is not None:
            property_values_by_subject = incremental_state.property_values_by_subject_prop
            literal_only_key_classes = {
                class_term for class_term in incremental_state.literal_only_key_classes if class_term in relevant_classes
            }
            object_key_classes = {
                class_term for class_term in incremental_state.object_key_classes if class_term in relevant_classes
            }

            incremental_key_classes = {
                class_term
                for class_term in literal_only_key_classes
                if class_term in members_by_key_class
            }
            for class_term in incremental_key_classes:
                key_props = has_key_axioms[class_term]
                existing_signature_map = incremental_state.literal_signatures_by_class_subject.setdefault(class_term, {})
                existing_members_by_signature = incremental_state.literal_members_by_signature.setdefault(
                    class_term,
                    defaultdict(set),
                )
                new_subjects = [
                    subj
                    for subj in members_by_key_class.get(class_term, [])
                    if subj not in existing_signature_map
                ]
                if not new_subjects and new_type_assertions is not None:
                    new_subjects = [
                        subj
                        for subj, asserted_class in new_type_assertions
                        if asserted_class == class_term and subj in members_by_key_class.get(class_term, [])
                    ]
                if new_subjects:
                    refreshed_values = _collect_property_values_by_subject(
                        materialized,
                        subjects=set(new_subjects),
                        properties=set(key_props),
                    )
                    for key, values in refreshed_values.items():
                        property_values_by_subject[key] = values
                for subj in new_subjects:
                    union_find.add(subj)
                    signature = _compute_literal_key_signature(subj, key_props, property_values_by_subject)
                    if signature is None:
                        continue
                    existing_subjects = sorted(existing_members_by_signature.get(signature, set()), key=_term_sort_key)
                    for other in existing_subjects:
                        if _has_differentfrom_pair(materialized, subj, other):
                            continue
                        union_find.union(subj, other)
                    existing_signature_map[subj] = signature
                    existing_members_by_signature.setdefault(signature, set()).add(subj)
        else:
            property_values_by_subject = _collect_property_values_by_subject(
                materialized,
                subjects=key_subjects if key_subjects else None,
                properties=key_properties if key_properties else None,
            )

            for class_term, key_props in has_key_axioms.items():
                members = members_by_key_class.get(class_term, [])
                if len(members) < 2:
                    continue
                requires_object_iteration = False
                for subj in members:
                    for prop in key_props:
                        raw_values = property_values_by_subject.get((subj, prop), set())
                        if any(not isinstance(value, Literal) for value in raw_values):
                            requires_object_iteration = True
                            break
                    if requires_object_iteration:
                        break
                if requires_object_iteration:
                    object_key_classes.add(class_term)
                else:
                    literal_only_key_classes.add(class_term)

            for class_term in literal_only_key_classes:
                key_props = has_key_axioms[class_term]
                members = members_by_key_class.get(class_term, [])
                signatures: Dict[
                    Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...],
                    List[Identifier],
                ] = defaultdict(list)
                for subj in members:
                    union_find.add(subj)
                    signature = _compute_literal_key_signature(subj, key_props, property_values_by_subject)
                    if signature is not None:
                        signatures[signature].append(subj)

                for grouped_subjects in signatures.values():
                    if len(grouped_subjects) < 2:
                        continue
                    representative = grouped_subjects[0]
                    for other in grouped_subjects[1:]:
                        if _has_differentfrom_pair(materialized, representative, other):
                            continue
                        union_find.union(representative, other)

        changed = True
        while changed and object_key_classes:
            changed = False
            for class_term in object_key_classes:
                key_props = has_key_axioms[class_term]
                members = members_by_key_class.get(class_term, [])
                if len(members) < 2:
                    continue
                signatures: Dict[
                    Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...],
                    List[Identifier],
                ] = defaultdict(list)
                for subj in members:
                    union_find.add(subj)
                    signature_parts: List[Tuple[str, Tuple[Tuple[str, str], ...]]] = []
                    complete = True
                    for prop in key_props:
                        raw_values = property_values_by_subject.get((subj, prop), set())
                        if not raw_values:
                            complete = False
                            break
                        normalized_values: List[Tuple[str, str]] = []
                        for value in raw_values:
                            if isinstance(value, Literal):
                                normalized_values.append(("L", value.n3()))
                            else:
                                union_find.add(value)
                                representative = union_find.find(value)
                                normalized_values.append(("I", str(representative)))
                        signature_parts.append((str(prop), tuple(sorted(set(normalized_values)))))
                    if complete:
                        signatures[tuple(signature_parts)].append(subj)

                for grouped_subjects in signatures.values():
                    if len(grouped_subjects) < 2:
                        continue
                    representative = grouped_subjects[0]
                    for other in grouped_subjects[1:]:
                        if _has_differentfrom_pair(materialized, representative, other):
                            continue
                        if union_find.union(representative, other):
                            changed = True

    groups = union_find.groups()
    nontrivial_groups = [members for members in groups.values() if len(members) > 1]
    if not nontrivial_groups:
        return False

    base_triples = list(materialized)
    for subj, pred, obj in base_triples:
        subj_variants = groups.get(union_find.find(subj), [subj]) if not isinstance(subj, Literal) else [subj]
        if pred == RDF.type or isinstance(obj, Literal):
            for subj_variant in subj_variants:
                triple = (subj_variant, pred, obj)
                if triple not in materialized:
                    materialized.add(triple)
            continue

        obj_variants = groups.get(union_find.find(obj), [obj]) if not isinstance(obj, Literal) else [obj]
        for subj_variant in subj_variants:
            for obj_variant in obj_variants:
                triple = (subj_variant, pred, obj_variant)
                if triple not in materialized:
                    materialized.add(triple)

    for members in nontrivial_groups:
        representative = members[0]
        for left, right in (
            [(representative, representative)]
            + [(representative, member) for member in members[1:]]
            + [(member, representative) for member in members[1:]]
            + [(member, member) for member in members[1:]]
        ):
            triple = (left, OWL.sameAs, right)
            if triple not in materialized:
                materialized.add(triple)
    if incremental_state is not None:
        affected_subjects: List[Identifier] = []
        for members in nontrivial_groups:
            affected_subjects.extend(member for member in members if not isinstance(member, Literal))
        if new_type_assertions is not None:
            affected_subjects.extend(subj for subj, _class_term in new_type_assertions if not isinstance(subj, Literal))
        _refresh_sameas_incremental_state_subjects(
            materialized,
            state=incremental_state,
            has_key_axioms=has_key_axioms,
            subjects=list(dict.fromkeys(affected_subjects)),
            classes=list(active_classes) if active_classes is not None else list(relevant_classes),
        )
    return len(materialized) != before_len


def collect_sameas_equivalence_map(
    graph: Graph,
) -> Dict[Identifier, List[Identifier]]:
    union_find = _UnionFind()
    for subj, _pred, obj in graph.triples((None, OWL.sameAs, None)):
        if isinstance(subj, Literal) or isinstance(obj, Literal):
            continue
        union_find.add(subj)
        union_find.add(obj)
        union_find.union(subj, obj)

    groups = union_find.groups()
    equivalence_map: Dict[Identifier, List[Identifier]] = {}
    for members in groups.values():
        for member in members:
            equivalence_map[member] = members
    return equivalence_map


def _has_differentfrom_pair(
    graph: Graph,
    left: Identifier,
    right: Identifier,
) -> bool:
    return (
        (left, OWL.differentFrom, right) in graph
        or (right, OWL.differentFrom, left) in graph
    )


def _collect_all_different_pairs(
    graph: Graph,
) -> List[Tuple[Identifier, Identifier]]:
    pairs: set[Tuple[Identifier, Identifier]] = set()
    for all_diff in graph.subjects(RDF.type, OWL.AllDifferent):
        members_head = graph.value(all_diff, OWL.distinctMembers)
        if not isinstance(members_head, BNode):
            continue
        members = [member for member in Collection(graph, members_head) if not isinstance(member, Literal)]
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                ordered = tuple(sorted((left, right), key=_term_sort_key))
                pairs.add(ordered)
    return sorted(pairs, key=lambda pair: (_term_sort_key(pair[0]), _term_sort_key(pair[1])))


def materialize_reflexive_property_closure(
    ontology_graph: Graph,
    data_graph: Graph,
    *,
    reflexive_props: Optional[Sequence[URIRef]] = None,
) -> Graph:
    """
    Materialize reflexive object property assertions `(x, R, x)` for every
    non-literal node term appearing in the current ABox, whenever `R` is marked
    as `owl:ReflexiveProperty`.
    """

    if reflexive_props is None:
        reflexive_props = sorted(
            {
                prop
                for prop, _pred, _obj in ontology_graph.triples((None, RDF.type, OWL.ReflexiveProperty))
                if isinstance(prop, URIRef)
            },
            key=str,
        )
    if not reflexive_props:
        return _copy_graph(data_graph)

    node_terms: set[Identifier] = set()
    for subj, pred, obj in data_graph:
        if not isinstance(subj, Literal):
            node_terms.add(subj)
        if pred == RDF.type:
            continue
        if not isinstance(obj, Literal):
            node_terms.add(obj)

    materialized = _copy_graph(data_graph)
    _materialize_reflexive_property_closure_in_place(
        materialized,
        reflexive_props=reflexive_props,
        node_terms=node_terms,
    )
    return materialized


def _materialize_reflexive_property_closure_in_place(
    materialized: Graph,
    *,
    reflexive_props: Sequence[URIRef],
    node_terms: Optional[set[Identifier]] = None,
    new_triples: Optional[List[Tuple[Identifier, Identifier, Identifier]]] = None,
) -> bool:
    if node_terms is None:
        node_terms = set()
        for subj, pred, obj in materialized:
            if not isinstance(subj, Literal):
                node_terms.add(subj)
            if pred != RDF.type and not isinstance(obj, Literal):
                node_terms.add(obj)
    changed = False
    for prop in reflexive_props:
        for node_term in sorted(node_terms, key=_term_sort_key):
            triple = (node_term, prop, node_term)
            if triple not in materialized:
                materialized.add(triple)
                changed = True
                if new_triples is not None:
                    new_triples.append(triple)
    return changed


def _extract_conjunctive_singleton_nominal_members(
    ontology_graph: Graph,
    expr: Identifier,
    *,
    visited: Optional[set[Identifier]] = None,
) -> List[Identifier]:
    if visited is None:
        visited = set()
    if expr in visited:
        return []
    visited.add(expr)

    singleton_member = _extract_singleton_nominal_member(ontology_graph, expr)
    if singleton_member is not None:
        return [singleton_member]

    if not isinstance(expr, BNode):
        return []

    members: List[Identifier] = []
    intersection_head = ontology_graph.value(expr, OWL.intersectionOf)
    if isinstance(intersection_head, BNode):
        for member in Collection(ontology_graph, intersection_head):
            members.extend(
                _extract_conjunctive_singleton_nominal_members(
                    ontology_graph,
                    member,
                    visited=visited,
                )
            )
    return list(dict.fromkeys(members))


def _collect_functional_property_terms(ontology_graph: Graph) -> set[URIRef]:
    return {
        prop
        for prop, _pred, _obj in ontology_graph.triples((None, RDF.type, OWL.FunctionalProperty))
        if isinstance(prop, URIRef)
    }


def _extract_singleton_literal_oneof_member(
    ontology_graph: Graph,
    expr: Identifier,
) -> Optional[Literal]:
    member = _extract_singleton_oneof_member(ontology_graph, expr)
    if not isinstance(member, Literal):
        return None
    return member


def _extract_conjunctive_functional_data_requirements(
    ontology_graph: Graph,
    expr: Identifier,
    *,
    functional_properties: Optional[set[URIRef]] = None,
    visited: Optional[set[Identifier]] = None,
) -> List[Tuple[URIRef, Literal]]:
    if functional_properties is None:
        functional_properties = _collect_functional_property_terms(ontology_graph)
    if visited is None:
        visited = set()
    if expr in visited:
        return []
    visited.add(expr)

    if not isinstance(expr, BNode):
        return []

    requirements: List[Tuple[URIRef, Literal]] = []
    intersection_head = ontology_graph.value(expr, OWL.intersectionOf)
    if isinstance(intersection_head, BNode):
        for member in Collection(ontology_graph, intersection_head):
            requirements.extend(
                _extract_conjunctive_functional_data_requirements(
                    ontology_graph,
                    member,
                    functional_properties=functional_properties,
                    visited=visited,
                )
            )
        return sorted(
            dict.fromkeys(requirements),
            key=lambda item: (str(item[0]), item[1].n3()),
        )

    restriction_types = set(ontology_graph.objects(expr, RDF.type))
    if OWL.Restriction not in restriction_types:
        return []

    prop_expr = ontology_graph.value(expr, OWL.onProperty)
    try:
        prop_term, prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
    except ValueError:
        return []
    if prop_direction != TraversalDirection.FORWARD or prop_term not in functional_properties:
        return []

    has_value = ontology_graph.value(expr, OWL.hasValue)
    if isinstance(has_value, Literal):
        requirements.append((prop_term, has_value))
        return requirements

    some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
    singleton_literal = _extract_singleton_literal_oneof_member(ontology_graph, some_filler)
    if singleton_literal is not None:
        requirements.append((prop_term, singleton_literal))

    return sorted(
        dict.fromkeys(requirements),
        key=lambda item: (str(item[0]), item[1].n3()),
    )


def _extract_conjunctive_exact_property_requirements(
    ontology_graph: Graph,
    expr: Identifier,
    *,
    visited: Optional[set[Identifier]] = None,
) -> List[Tuple[URIRef, Identifier]]:
    if visited is None:
        visited = set()
    if expr in visited:
        return []
    visited.add(expr)

    if not isinstance(expr, BNode):
        return []

    requirements: List[Tuple[URIRef, Identifier]] = []
    intersection_head = ontology_graph.value(expr, OWL.intersectionOf)
    if isinstance(intersection_head, BNode):
        for member in Collection(ontology_graph, intersection_head):
            requirements.extend(
                _extract_conjunctive_exact_property_requirements(
                    ontology_graph,
                    member,
                    visited=visited,
                )
            )
        return sorted(
            dict.fromkeys(requirements),
            key=lambda item: (str(item[0]), _term_sort_key(item[1])),
        )

    restriction_types = set(ontology_graph.objects(expr, RDF.type))
    if OWL.Restriction not in restriction_types:
        return []

    prop_expr = ontology_graph.value(expr, OWL.onProperty)
    try:
        prop_term, prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
    except ValueError:
        return []
    if prop_direction != TraversalDirection.FORWARD:
        return []

    has_value = ontology_graph.value(expr, OWL.hasValue)
    if has_value is not None:
        requirements.append((prop_term, has_value))
        return requirements

    some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
    singleton_member = _extract_singleton_oneof_member(ontology_graph, some_filler)
    if singleton_member is not None:
        requirements.append((prop_term, singleton_member))

    return sorted(
        dict.fromkeys(requirements),
        key=lambda item: (str(item[0]), _term_sort_key(item[1])),
    )


def _collect_negative_property_assertions(
    graph: Graph,
) -> set[Tuple[Identifier, URIRef, Identifier]]:
    assertions: set[Tuple[Identifier, URIRef, Identifier]] = set()
    for assertion in graph.subjects(RDF.type, OWL.NegativePropertyAssertion):
        source = graph.value(assertion, OWL.sourceIndividual)
        prop = graph.value(assertion, OWL.assertionProperty)
        target_individual = graph.value(assertion, OWL.targetIndividual)
        target_value = graph.value(assertion, OWL.targetValue)
        if not isinstance(prop, URIRef) or source is None:
            continue
        if target_individual is not None:
            assertions.add((source, prop, target_individual))
        elif target_value is not None:
            assertions.add((source, prop, target_value))
    return assertions


def _negative_property_helper_term(prop_term: URIRef) -> URIRef:
    return URIRef(f"urn:dag:negative-property:{str(prop_term)}")


def _collect_negative_property_assertion_edges(
    graph: Graph,
    *,
    include_literals: bool,
) -> List[Tuple[URIRef, Identifier, Identifier]]:
    edges: List[Tuple[URIRef, Identifier, Identifier]] = []
    for source, prop_term, target in sorted(
        _collect_negative_property_assertions(graph),
        key=lambda item: (_term_sort_key(item[0]), str(item[1]), _term_sort_key(item[2])),
    ):
        if isinstance(target, Literal) and not include_literals:
            continue
        edges.append((_negative_property_helper_term(prop_term), source, target))
    return edges


def _collect_functional_property_terms_from_graph(graph: Graph) -> set[URIRef]:
    return {
        prop
        for prop, _pred, _obj in graph.triples((None, RDF.type, OWL.FunctionalProperty))
        if isinstance(prop, URIRef)
    }


def _extract_exact_restriction_filler(
    ontology_graph: Graph,
    expr: Identifier,
) -> Optional[Identifier]:
    if not isinstance(expr, BNode):
        return None
    restriction_types = set(ontology_graph.objects(expr, RDF.type))
    if OWL.Restriction not in restriction_types:
        return None

    has_value = ontology_graph.value(expr, OWL.hasValue)
    if has_value is not None:
        return has_value

    some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
    if some_filler is None:
        return None
    return _extract_singleton_oneof_member(ontology_graph, some_filler)


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
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
    compile_context: Optional[OntologyCompileContext] = None,
) -> List[Identifier]:
    if compile_context is not None:
        cache_key = (
            dependency_analysis.canonical_map.get(target_term, target_term)
            if dependency_analysis is not None
            else target_term
        )
        cached = compile_context.target_root_expressions_cache.get(cache_key)
        if cached is not None:
            return list(cached)

    if dependency_analysis is None:
        equivalent_members = collect_named_class_equivalence_members(ontology_graph)
        canonical_map = collect_named_class_canonical_map(ontology_graph)
    else:
        equivalent_members = dependency_analysis.equivalence_members
        canonical_map = dependency_analysis.canonical_map
    target_canonical = canonical_map.get(target_term, target_term)
    target_component = equivalent_members.get(target_canonical, [target_term])

    root_exprs: List[Identifier] = []
    for member in target_component:
        root_exprs.extend(ontology_graph.objects(member, RDFS.subClassOf))
        root_exprs.extend(
            expr
            for expr in ontology_graph.objects(member, OWL.equivalentClass)
            if canonical_map.get(expr, expr) != target_canonical
        )
        root_exprs.extend(ontology_graph.objects(member, OWL.intersectionOf))
        root_exprs.extend(
            expr
            for expr in ontology_graph.objects(member, OWL.disjointWith)
            if canonical_map.get(expr, expr) != target_canonical
        )
        root_exprs.extend(
            expr
            for expr in ontology_graph.subjects(OWL.disjointWith, member)
            if canonical_map.get(expr, expr) != target_canonical
        )
    if compile_context is not None:
        compile_context.target_root_expressions_cache[target_canonical] = list(root_exprs)
    return root_exprs


def collect_referenced_named_classes_for_class(
    ontology_graph: Graph,
    target_class: str | URIRef,
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
    compile_context: Optional[OntologyCompileContext] = None,
) -> List[URIRef]:
    """
    Collect named classes reachable from the target class's query surface.

    This follows the same compiler-visible surfaces as the current DAG path and
    recursively expands through referenced named-class definitions. The result
    is useful for targeted helper materialization.
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    if dependency_analysis is None:
        canonical_map = collect_named_class_canonical_map(ontology_graph)
        equivalent_members = collect_named_class_equivalence_members(ontology_graph)
    else:
        canonical_map = dependency_analysis.canonical_map
        equivalent_members = dependency_analysis.equivalence_members
    target_canonical = canonical_map.get(target_term, target_term)

    visited: set[Identifier] = set()
    class_terms: set[URIRef] = {target_canonical}
    stack: List[Identifier] = list(
        _collect_target_root_expressions(
            ontology_graph,
            target_term,
            dependency_analysis=dependency_analysis,
            compile_context=compile_context,
        )
    ) or [target_term]

    while stack:
        expr = stack.pop()
        if expr in visited:
            continue
        visited.add(expr)

        if isinstance(expr, URIRef) and not _is_datatype_term(expr):
            canonical_expr = canonical_map.get(expr, expr)
            class_terms.add(canonical_expr)

        for head in ontology_graph.objects(expr, OWL.intersectionOf):
            if isinstance(head, BNode):
                stack.extend(Collection(ontology_graph, head))

        for head in ontology_graph.objects(expr, OWL.unionOf):
            if isinstance(head, BNode):
                stack.extend(Collection(ontology_graph, head))

        stack.extend(ontology_graph.objects(expr, OWL.complementOf))

        restriction_types = set(ontology_graph.objects(expr, RDF.type))
        if OWL.Restriction in restriction_types:
            some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
            all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
            on_class = ontology_graph.value(expr, OWL.onClass)
            if some_filler is not None:
                stack.append(some_filler)
            if all_filler is not None:
                stack.append(all_filler)
            if on_class is not None:
                stack.append(on_class)

        if isinstance(expr, URIRef):
            expr_canonical = canonical_map.get(expr, expr)
            for member in equivalent_members.get(expr_canonical, [expr]):
                stack.extend(ontology_graph.objects(member, RDFS.subClassOf))
                stack.extend(
                    eq_expr
                    for eq_expr in ontology_graph.objects(member, OWL.equivalentClass)
                    if canonical_map.get(eq_expr, eq_expr) != expr_canonical
                )
                for head in ontology_graph.objects(member, OWL.intersectionOf):
                    if isinstance(head, BNode):
                        stack.extend(Collection(ontology_graph, head))
                for head in ontology_graph.objects(member, OWL.unionOf):
                    if isinstance(head, BNode):
                        stack.extend(Collection(ontology_graph, head))
                stack.extend(ontology_graph.objects(member, OWL.disjointWith))
                stack.extend(
                    disjoint_expr
                    for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, member)
                    if disjoint_expr != member
                )
        elif isinstance(expr, BNode):
            stack.extend(ontology_graph.objects(expr, RDFS.subClassOf))
            stack.extend(eq_expr for eq_expr in ontology_graph.objects(expr, OWL.equivalentClass) if eq_expr != expr)
            stack.extend(ontology_graph.objects(expr, OWL.disjointWith))
            stack.extend(
                disjoint_expr
                for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, expr)
                if disjoint_expr != expr
            )

    return sorted(class_terms, key=str)


def collect_referenced_named_classes_for_classes(
    ontology_graph: Graph,
    target_classes: Sequence[str | URIRef],
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
    compile_context: Optional[OntologyCompileContext] = None,
) -> List[URIRef]:
    class_terms: set[URIRef] = set()
    for target_class in target_classes:
        class_terms.update(
            collect_referenced_named_classes_for_class(
                ontology_graph,
                target_class,
                dependency_analysis=dependency_analysis,
                compile_context=compile_context,
            )
        )
    return sorted(class_terms, key=str)


def collect_referenced_properties_for_class(
    ontology_graph: Graph,
    target_class: str | URIRef,
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
) -> List[URIRef]:
    """
    Collect the properties directly referenced by a target class query.

    This follows the same definitional surfaces the current compiler uses:
    subclass/equivalent/intersection/complement/disjoint expressions and
    property restrictions reachable from the target class.
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    if dependency_analysis is None:
        canonical_map = collect_named_class_canonical_map(ontology_graph)
        equivalent_members = collect_named_class_equivalence_members(ontology_graph)
    else:
        canonical_map = dependency_analysis.canonical_map
        equivalent_members = dependency_analysis.equivalence_members
    visited: set[Identifier] = set()
    props: set[URIRef] = set()
    stack: List[Identifier] = list(
        _collect_target_root_expressions(
            ontology_graph,
            target_term,
            dependency_analysis=dependency_analysis,
        )
    ) or [target_term]

    while stack:
        expr = stack.pop()
        if expr in visited:
            continue
        visited.add(expr)

        for head in ontology_graph.objects(expr, OWL.intersectionOf):
            if isinstance(head, BNode):
                stack.extend(Collection(ontology_graph, head))

        for head in ontology_graph.objects(expr, OWL.unionOf):
            if isinstance(head, BNode):
                stack.extend(Collection(ontology_graph, head))

        stack.extend(ontology_graph.objects(expr, OWL.complementOf))

        restriction_types = set(ontology_graph.objects(expr, RDF.type))
        if OWL.Restriction in restriction_types:
            prop_expr = ontology_graph.value(expr, OWL.onProperty)
            prop, prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
            if isinstance(prop, URIRef):
                props.add(prop)
            some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
            all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
            if some_filler is not None:
                stack.append(some_filler)
            if all_filler is not None:
                stack.append(all_filler)

        if isinstance(expr, URIRef):
            expr_canonical = canonical_map.get(expr, expr)
            for member in equivalent_members.get(expr_canonical, [expr]):
                stack.extend(ontology_graph.objects(member, RDFS.subClassOf))
                stack.extend(
                    eq_expr
                    for eq_expr in ontology_graph.objects(member, OWL.equivalentClass)
                    if canonical_map.get(eq_expr, eq_expr) != expr_canonical
                )
                for head in ontology_graph.objects(member, OWL.intersectionOf):
                    if isinstance(head, BNode):
                        stack.extend(Collection(ontology_graph, head))
                for head in ontology_graph.objects(member, OWL.unionOf):
                    if isinstance(head, BNode):
                        stack.extend(Collection(ontology_graph, head))
                stack.extend(ontology_graph.objects(member, OWL.disjointWith))
                stack.extend(
                    disjoint_expr
                    for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, member)
                    if disjoint_expr != member
                )
        elif isinstance(expr, BNode):
            stack.extend(ontology_graph.objects(expr, RDFS.subClassOf))
            stack.extend(eq_expr for eq_expr in ontology_graph.objects(expr, OWL.equivalentClass) if eq_expr != expr)
            stack.extend(ontology_graph.objects(expr, OWL.disjointWith))
            stack.extend(
                disjoint_expr
                for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, expr)
                if disjoint_expr != expr
            )

    return sorted(props, key=str)


def collect_referenced_properties_for_classes(
    ontology_graph: Graph,
    target_classes: Sequence[str | URIRef],
) -> List[URIRef]:
    props: set[URIRef] = set()
    for target_class in target_classes:
        props.update(collect_referenced_properties_for_class(ontology_graph, target_class))
    return sorted(props, key=str)


def collect_direct_named_class_dependencies(
    ontology_graph: Graph,
    target_class: str | URIRef,
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
) -> List[URIRef]:
    """
    Collect only the direct named-class dependencies visible from the target
    class definition surface, without recursively expanding those dependencies.
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    if dependency_analysis is None:
        canonical_map = collect_named_class_canonical_map(ontology_graph)
        equivalent_members = collect_named_class_equivalence_members(ontology_graph)
    else:
        canonical_map = dependency_analysis.canonical_map
        equivalent_members = dependency_analysis.equivalence_members
    target_canonical = canonical_map.get(target_term, target_term)
    dependencies: set[URIRef] = set()
    visited: set[Identifier] = set()
    stack: List[Identifier] = list(
        _collect_target_root_expressions(
            ontology_graph,
            target_term,
            dependency_analysis=dependency_analysis,
        )
    ) or [target_term]

    while stack:
        expr = stack.pop()
        if expr in visited:
            continue
        visited.add(expr)

        if isinstance(expr, URIRef) and not _is_datatype_term(expr):
            expr_canonical = canonical_map.get(expr, expr)
            if expr_canonical != target_canonical:
                dependencies.add(expr_canonical)
            continue

        for head in ontology_graph.objects(expr, OWL.intersectionOf):
            if isinstance(head, BNode):
                stack.extend(Collection(ontology_graph, head))

        for head in ontology_graph.objects(expr, OWL.unionOf):
            if isinstance(head, BNode):
                stack.extend(Collection(ontology_graph, head))

        stack.extend(ontology_graph.objects(expr, OWL.complementOf))

        restriction_types = set(ontology_graph.objects(expr, RDF.type))
        if OWL.Restriction in restriction_types:
            some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
            all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
            on_class = ontology_graph.value(expr, OWL.onClass)
            if some_filler is not None:
                stack.append(some_filler)
            if all_filler is not None:
                stack.append(all_filler)
            if on_class is not None:
                stack.append(on_class)

        if isinstance(expr, BNode):
            stack.extend(ontology_graph.objects(expr, RDFS.subClassOf))
            stack.extend(eq_expr for eq_expr in ontology_graph.objects(expr, OWL.equivalentClass) if eq_expr != expr)
            stack.extend(ontology_graph.objects(expr, OWL.disjointWith))
            stack.extend(
                disjoint_expr
                for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, expr)
                if disjoint_expr != expr
            )
        elif isinstance(expr, URIRef):
            expr_canonical = canonical_map.get(expr, expr)
            for member in equivalent_members.get(expr_canonical, [expr]):
                stack.extend(ontology_graph.objects(member, RDFS.subClassOf))

    return sorted(dependencies, key=str)


def analyze_named_class_dependencies(
    ontology_graph: Graph,
) -> NamedClassDependencyAnalysis:
    named_classes = collect_named_class_terms(ontology_graph)

    union_find = _UnionFind()
    for class_term in named_classes:
        union_find.add(class_term)

    for subj, _pred, obj in ontology_graph.triples((None, OWL.equivalentClass, None)):
        if not isinstance(subj, URIRef) or not isinstance(obj, URIRef):
            continue
        if _is_datatype_term(subj) or _is_datatype_term(obj):
            continue
        union_find.add(subj)
        union_find.add(obj)
        union_find.union(subj, obj)

    groups: Dict[URIRef, List[URIRef]] = {}
    for class_term in named_classes:
        representative = union_find.find(class_term)
        groups.setdefault(representative, []).append(class_term)

    equivalence_members: Dict[URIRef, List[URIRef]] = {}
    for members in groups.values():
        canonical = min(members, key=str)
        equivalence_members[canonical] = sorted(members, key=str)
    canonical_map: Dict[URIRef, URIRef] = {}
    for canonical, members in equivalence_members.items():
        for member in members:
            canonical_map[member] = canonical

    partial_analysis = NamedClassDependencyAnalysis(
        named_classes=sorted({canonical_map.get(term, term) for term in named_classes}, key=str),
        equivalence_members=equivalence_members,
        canonical_map=canonical_map,
        direct_dependencies={},
        cycle_components=[],
        cycle_component_by_class={},
        reaches_cycle_by_class={},
    )

    direct_dependencies: Dict[URIRef, List[URIRef]] = {}
    reverse_adjacency: Dict[URIRef, Set[URIRef]] = {
        class_term: set() for class_term in partial_analysis.named_classes
    }
    for class_term in partial_analysis.named_classes:
        deps = collect_direct_named_class_dependencies(
            ontology_graph,
            class_term,
            dependency_analysis=partial_analysis,
        )
        direct_dependencies[class_term] = deps
        for dep in deps:
            reverse_adjacency.setdefault(dep, set()).add(class_term)

    visited: Set[URIRef] = set()
    finish_order: List[URIRef] = []
    for start in partial_analysis.named_classes:
        if start in visited:
            continue
        stack: List[Tuple[URIRef, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for neighbor in sorted(direct_dependencies.get(node, ()), key=str, reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, False))

    assigned: Set[URIRef] = set()
    cycle_components: List[List[URIRef]] = []
    cycle_component_by_class: Dict[URIRef, List[URIRef]] = {}
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component: List[URIRef] = []
        stack = [start]
        assigned.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(reverse_adjacency.get(node, ()), key=str, reverse=True):
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append(neighbor)

        component = sorted(component, key=str)
        if len(component) > 1 or (
            component and component[0] in set(direct_dependencies.get(component[0], ()))
        ):
            cycle_components.append(component)
            for class_term in component:
                cycle_component_by_class[class_term] = component

    reaches_cycle_by_class: Dict[URIRef, bool] = {}

    def reaches_cycle(class_term: URIRef) -> bool:
        cached = reaches_cycle_by_class.get(class_term)
        if cached is not None:
            return cached
        if class_term in cycle_component_by_class:
            reaches_cycle_by_class[class_term] = True
            return True
        reaches_cycle_by_class[class_term] = False
        result = any(reaches_cycle(dep) for dep in direct_dependencies.get(class_term, ()))
        reaches_cycle_by_class[class_term] = result
        return result

    for class_term in partial_analysis.named_classes:
        reaches_cycle(class_term)

    return NamedClassDependencyAnalysis(
        named_classes=partial_analysis.named_classes,
        equivalence_members=equivalence_members,
        canonical_map=canonical_map,
        direct_dependencies=direct_dependencies,
        cycle_components=cycle_components,
        cycle_component_by_class=cycle_component_by_class,
        reaches_cycle_by_class=reaches_cycle_by_class,
    )


def build_ontology_compile_context(
    ontology_graph: Graph,
    *,
    schema_graph: Optional[Graph] = None,
    sameas_source_graph: Optional[Graph] = None,
) -> OntologyCompileContext:
    analysis_graph = schema_graph if schema_graph is not None and len(schema_graph) > 0 else ontology_graph
    sameas_graph = sameas_source_graph if sameas_source_graph is not None else ontology_graph
    dependency_analysis = analyze_named_class_dependencies(analysis_graph)
    return OntologyCompileContext(
        dependency_analysis=dependency_analysis,
        direct_subs=_collect_direct_subproperties(analysis_graph),
        inverse_props=_collect_inverse_properties(analysis_graph),
        chains_by_conclusion=_collect_property_chains(analysis_graph),
        transitive_props=_collect_transitive_properties(analysis_graph),
        functional_props=_collect_functional_property_terms_from_graph(analysis_graph),
        sameas_equivalence_map=collect_sameas_equivalence_map(sameas_graph),
        property_axioms=collect_property_expression_axioms(analysis_graph),
        subclass_supers=_compute_transitive_super_map(analysis_graph, RDFS.subClassOf),
    )


def collect_named_class_dependency_cycles(
    ontology_graph: Graph,
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
) -> List[List[URIRef]]:
    analysis = dependency_analysis or analyze_named_class_dependencies(ontology_graph)
    return analysis.cycle_components


def class_is_in_named_dependency_cycle(
    ontology_graph: Graph,
    target_class: str | URIRef,
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
) -> bool:
    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    analysis = dependency_analysis or analyze_named_class_dependencies(ontology_graph)
    canonical_map = analysis.canonical_map
    target_canonical = canonical_map.get(target_term, target_term)
    return target_canonical in analysis.cycle_component_by_class


def get_named_class_dependency_cycle_component(
    ontology_graph: Graph,
    target_class: str | URIRef,
    *,
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
) -> List[URIRef]:
    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    analysis = dependency_analysis or analyze_named_class_dependencies(ontology_graph)
    canonical_map = analysis.canonical_map
    target_canonical = canonical_map.get(target_term, target_term)
    return analysis.cycle_component_by_class.get(target_canonical, [])


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


def collect_named_class_terms(ontology_graph: Graph) -> List[URIRef]:
    """
    Return all named class IRIs referenced in the ontology graph, excluding
    datatype terms.
    """
    return sorted(
        [
            class_term
            for class_term in _collect_named_class_terms(ontology_graph)
            if isinstance(class_term, URIRef) and not _is_datatype_term(class_term)
        ],
        key=str,
    )


def collect_named_class_equivalence_members(
    ontology_graph: Graph,
) -> Dict[URIRef, List[URIRef]]:
    union_find = _UnionFind()
    named_classes = collect_named_class_terms(ontology_graph)
    for class_term in named_classes:
        union_find.add(class_term)

    for subj, _pred, obj in ontology_graph.triples((None, OWL.equivalentClass, None)):
        if not isinstance(subj, URIRef) or not isinstance(obj, URIRef):
            continue
        if _is_datatype_term(subj) or _is_datatype_term(obj):
            continue
        union_find.add(subj)
        union_find.add(obj)
        union_find.union(subj, obj)

    groups: Dict[URIRef, List[URIRef]] = {}
    for class_term in named_classes:
        representative = union_find.find(class_term)
        groups.setdefault(representative, []).append(class_term)

    normalized_groups: Dict[URIRef, List[URIRef]] = {}
    for members in groups.values():
        canonical = min(members, key=str)
        normalized_groups[canonical] = sorted(members, key=str)

    return normalized_groups


def collect_named_class_canonical_map(
    ontology_graph: Graph,
) -> Dict[URIRef, URIRef]:
    canonical_map: Dict[URIRef, URIRef] = {}
    for canonical, members in collect_named_class_equivalence_members(ontology_graph).items():
        for member in members:
            canonical_map[member] = canonical
    return canonical_map


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
    vocab_source: Optional[Graph] = None,
    include_literals: bool = True,
    include_type_edges: bool = False,
    preprocessing_timings: Optional[PreprocessingTimings] = None,
    schema_property_terms: Optional[Sequence[URIRef]] = None,
    schema_class_terms: Optional[Sequence[Identifier]] = None,
    schema_datatype_terms: Optional[Sequence[URIRef]] = None,
    scan_cache: Optional[GraphBuildScanCache] = None,
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
    if vocab_source is None:
        vocab_source = graph if vocab_graph is None else aggregate_rdflib_graphs((vocab_graph, graph))
    if scan_cache is None:
        scan_cache = _scan_graph_build_cache(
            graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            preprocessing_timings=preprocessing_timings,
            schema_property_terms=schema_property_terms,
            schema_class_terms=schema_class_terms,
            schema_datatype_terms=schema_datatype_terms,
            vocab_source=vocab_source,
        )
    mapping = _finalize_rdflib_mapping(
        node_terms_set=scan_cache.node_terms_set,
        prop_terms_set=scan_cache.prop_terms_set,
        class_terms_set=scan_cache.class_terms_set,
        datatype_terms_set=scan_cache.datatype_terms_set,
        preprocessing_timings=preprocessing_timings,
    )

    node_terms = mapping.node_terms
    prop_terms = mapping.prop_terms
    class_terms = mapping.class_terms
    datatype_terms = mapping.datatype_terms

    node_to_idx = mapping.node_to_idx
    class_to_idx = mapping.class_to_idx
    datatype_to_idx = mapping.datatype_to_idx

    num_nodes = len(node_terms)
    num_classes = len(class_terms)
    node_types = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    literal_datatype_idx = torch.full((num_nodes,), -1, dtype=torch.int64)
    literal_numeric_value = torch.full((num_nodes,), float("nan"), dtype=torch.float32)

    edge_bucket_t0 = perf_counter()
    if scan_cache.type_pairs:
        type_src = np.fromiter(
            (node_to_idx[subj] for subj, _class_term in scan_cache.type_pairs),
            dtype=np.int64,
            count=len(scan_cache.type_pairs),
        )
        type_cls = np.fromiter(
            (class_to_idx[class_term] for _subj, class_term in scan_cache.type_pairs),
            dtype=np.int64,
            count=len(scan_cache.type_pairs),
        )
        node_types[type_src.tolist(), type_cls.tolist()] = 1.0

    edge_src_terms: List[Identifier] = []
    edge_prop_terms: List[URIRef] = []
    edge_dst_terms: List[Identifier] = []
    if scan_cache.type_edge_triples:
        for subj, pred, obj in scan_cache.type_edge_triples:
            edge_src_terms.append(subj)
            edge_prop_terms.append(pred)
            edge_dst_terms.append(obj)
    if scan_cache.edge_triples:
        for subj, pred, obj in scan_cache.edge_triples:
            edge_src_terms.append(subj)
            edge_prop_terms.append(pred)
            edge_dst_terms.append(obj)

    negative_helper_t0 = perf_counter()
    if scan_cache.negative_helper_edges:
        for helper_prop, subj, obj in scan_cache.negative_helper_edges:
            edge_src_terms.append(subj)
            edge_prop_terms.append(helper_prop)
            edge_dst_terms.append(obj)
    if preprocessing_timings is not None:
        preprocessing_timings.kgraph_negative_helper_elapsed_ms += (perf_counter() - negative_helper_t0) * 1000.0

    if edge_src_terms:
        src_arr = np.fromiter((node_to_idx[term] for term in edge_src_terms), dtype=np.int32, count=len(edge_src_terms))
        prop_arr = np.fromiter((mapping.prop_to_idx[term] for term in edge_prop_terms), dtype=np.int32, count=len(edge_prop_terms))
        dst_arr = np.fromiter((node_to_idx[term] for term in edge_dst_terms), dtype=np.int32, count=len(edge_dst_terms))
    else:
        src_arr = np.empty((0,), dtype=np.int32)
        prop_arr = np.empty((0,), dtype=np.int32)
        dst_arr = np.empty((0,), dtype=np.int32)
    if preprocessing_timings is not None:
        preprocessing_timings.kgraph_edge_bucket_elapsed_ms += (perf_counter() - edge_bucket_t0) * 1000.0

    literal_feature_t0 = perf_counter()
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
    if preprocessing_timings is not None:
        preprocessing_timings.kgraph_literal_feature_elapsed_ms += (perf_counter() - literal_feature_t0) * 1000.0

    adjacency_t0 = perf_counter()
    offsets_p: List[torch.Tensor] = []
    neighbors_p: List[torch.Tensor] = []
    if prop_arr.size:
        order = np.lexsort((src_arr, prop_arr))
        sorted_props = prop_arr[order]
        sorted_srcs = src_arr[order]
        sorted_dsts = dst_arr[order]
        prop_offsets = np.zeros(len(prop_terms) + 1, dtype=np.int64)
        prop_counts = np.bincount(sorted_props, minlength=len(prop_terms))
        prop_offsets[1:] = np.cumsum(prop_counts, dtype=np.int64)
    else:
        sorted_props = prop_arr
        sorted_srcs = src_arr
        sorted_dsts = dst_arr
        prop_offsets = np.zeros(len(prop_terms) + 1, dtype=np.int64)

    for prop_idx, _prop_term in enumerate(prop_terms):
        start = int(prop_offsets[prop_idx])
        end = int(prop_offsets[prop_idx + 1])
        if start == end:
            offsets_p.append(torch.zeros(num_nodes + 1, dtype=torch.int32))
            neighbors_p.append(torch.empty((0,), dtype=torch.int32))
            continue

        prop_srcs = sorted_srcs[start:end]
        prop_dsts = sorted_dsts[start:end]
        count_tensor = torch.from_numpy(
            np.bincount(prop_srcs, minlength=num_nodes).astype(np.int64, copy=False)
        )
        offsets = torch.zeros(num_nodes + 1, dtype=torch.int32)
        offsets[1:] = torch.cumsum(count_tensor.to(torch.int32), dim=0)
        neighbors = torch.from_numpy(prop_dsts.astype(np.int32, copy=False))
        offsets_p.append(offsets)
        neighbors_p.append(neighbors)
    if preprocessing_timings is not None:
        preprocessing_timings.kgraph_adjacency_elapsed_ms += (perf_counter() - adjacency_t0) * 1000.0

    kg = KGraph(
            num_nodes=num_nodes,
            offsets_p=offsets_p,
            neighbors_p=neighbors_p,
            node_types=node_types,
            literal_datatype_idx=literal_datatype_idx,
            literal_numeric_value=literal_numeric_value,
    )
    return (kg, mapping)


def build_rdflib_mapping(
    graph: Graph,
    *,
    vocab_graph: Optional[Graph] = None,
    vocab_source: Optional[Graph] = None,
    include_literals: bool = True,
    include_type_edges: bool = False,
    preprocessing_timings: Optional[PreprocessingTimings] = None,
    schema_property_terms: Optional[Sequence[URIRef]] = None,
    schema_class_terms: Optional[Sequence[Identifier]] = None,
    schema_datatype_terms: Optional[Sequence[URIRef]] = None,
) -> RDFKGraphMapping:
    node_terms_set: set[Identifier] = set()
    if vocab_source is None:
        vocab_source = graph if vocab_graph is None else aggregate_rdflib_graphs((vocab_graph, graph))
    t0 = perf_counter()
    prop_terms_set: set[URIRef] = (
        set(schema_property_terms)
        if schema_property_terms is not None
        else _collect_property_terms(vocab_source)
    )
    class_terms_set: set[Identifier] = (
        set(schema_class_terms)
        if schema_class_terms is not None
        else _collect_named_class_terms(vocab_source)
    )
    datatype_terms_set: set[URIRef] = (
        set(schema_datatype_terms)
        if schema_datatype_terms is not None
        else _collect_datatype_terms(vocab_source)
    )
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_vocab_collect_elapsed_ms += (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    for subj, pred, obj in graph:
        node_terms_set.add(subj)

        if pred == RDF.type:
            class_terms_set.add(obj)
            if include_type_edges and not isinstance(obj, Literal):
                node_terms_set.add(obj)
                prop_terms_set.add(pred)
            continue

        prop_terms_set.add(pred)
        if isinstance(obj, Literal) and not include_literals:
            continue
        node_terms_set.add(obj)

    for helper_prop, subj, obj in _collect_negative_property_assertion_edges(
        graph,
        include_literals=include_literals,
    ):
        prop_terms_set.add(helper_prop)
        node_terms_set.add(subj)
        node_terms_set.add(obj)
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_graph_scan_elapsed_ms += (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    node_terms = sorted(node_terms_set, key=_term_sort_key)
    prop_terms = sorted(prop_terms_set, key=_term_sort_key)
    class_terms = sorted(class_terms_set, key=_term_sort_key)
    datatype_terms = sorted(datatype_terms_set, key=str)
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_sort_elapsed_ms += (perf_counter() - t0) * 1000.0

    t0 = perf_counter()
    mapping = RDFKGraphMapping(
        node_terms=node_terms,
        prop_terms=prop_terms,
        class_terms=class_terms,
        datatype_terms=datatype_terms,
        node_to_idx={term: idx for idx, term in enumerate(node_terms)},
        prop_to_idx={term: idx for idx, term in enumerate(prop_terms)},
        class_to_idx={term: idx for idx, term in enumerate(class_terms)},
        datatype_to_idx={term: idx for idx, term in enumerate(datatype_terms)},
    )
    if preprocessing_timings is not None:
        preprocessing_timings.mapping_index_elapsed_ms += (perf_counter() - t0) * 1000.0
    return mapping


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
    materialize_sameas: Optional[bool] = None,
    materialize_sameas_policy: str = "auto",
    materialize_haskey_equality: Optional[bool] = None,
    materialize_haskey_equality_policy: str = "off",
    materialize_reflexive_properties: Optional[bool] = None,
    materialize_reflexive_properties_policy: str = "auto",
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
        materialize_sameas=materialize_sameas,
        materialize_sameas_policy=materialize_sameas_policy,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_haskey_equality_policy=materialize_haskey_equality_policy,
        materialize_reflexive_properties=materialize_reflexive_properties,
        materialize_reflexive_properties_policy=materialize_reflexive_properties_policy,
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
    materialize_sameas: Optional[bool] = None,
    materialize_sameas_policy: str = "auto",
    materialize_haskey_equality: Optional[bool] = None,
    materialize_haskey_equality_policy: str = "off",
    materialize_reflexive_properties: Optional[bool] = None,
    materialize_reflexive_properties_policy: str = "auto",
    materialize_target_roles: Optional[bool] = None,
    materialize_target_roles_policy: str = "auto",
    target_classes: Optional[Sequence[str | URIRef]] = None,
    dependency_closure: Optional[TargetDependencyClosure] = None,
    build_cache: Optional[ReasoningBuildCache] = None,
) -> ReasoningDataset:
    preprocessing_timings = PreprocessingTimings()
    t0 = perf_counter()
    ontology_graph = aggregate_rdflib_graphs((schema_graph, data_graph))
    preprocessing_timings.ontology_merge_elapsed_ms += (perf_counter() - t0) * 1000.0
    if build_cache is None:
        t0 = perf_counter()
        cache = build_reasoning_build_cache(schema_graph)
        preprocessing_timings.schema_cache_elapsed_ms += (perf_counter() - t0) * 1000.0
    else:
        cache = build_cache
    subproperty_supers = cache.subproperty_supers
    property_axioms = cache.property_axioms
    atomic_domain_consequents = cache.atomic_domain_consequents
    atomic_range_consequents = cache.atomic_range_consequents
    atomic_domain_range_predicates = cache.atomic_domain_range_predicates
    horn_domain_consequents = cache.horn_domain_consequents
    horn_range_consequents = cache.horn_range_consequents
    horn_domain_range_predicates = cache.horn_domain_range_predicates
    horn_safe_named_axiom_consequents = cache.horn_safe_named_axiom_consequents
    preprocessing_class_supers = cache.preprocessing_class_supers
    singleton_nominals = cache.singleton_nominals
    has_key_axioms = cache.has_key_axioms
    reflexive_props = cache.reflexive_props
    all_different_pairs = cache.all_different_pairs
    t0 = perf_counter()
    preprocessing_plan = plan_reasoning_preprocessing(
        ontology_graph,
        target_classes=target_classes,
        materialize_hierarchy=materialize_hierarchy,
        materialize_hierarchy_policy=materialize_hierarchy_policy,
        materialize_atomic_domain_range=materialize_atomic_domain_range,
        materialize_atomic_domain_range_policy=materialize_atomic_domain_range_policy,
        materialize_horn_safe_domain_range=materialize_horn_safe_domain_range,
        materialize_horn_safe_domain_range_policy=materialize_horn_safe_domain_range_policy,
        materialize_sameas=materialize_sameas,
        materialize_sameas_policy=materialize_sameas_policy,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_haskey_equality_policy=materialize_haskey_equality_policy,
        materialize_reflexive_properties=materialize_reflexive_properties,
        materialize_reflexive_properties_policy=materialize_reflexive_properties_policy,
        materialize_target_roles=materialize_target_roles,
        materialize_target_roles_policy=materialize_target_roles_policy,
    )
    preprocessing_timings.preprocessing_plan_elapsed_ms += (perf_counter() - t0) * 1000.0
    active_has_key_axioms = (
        has_key_axioms
        if preprocessing_plan.materialize_haskey_equality.enabled
        else {}
    )
    can_incrementally_build_scan_cache = (
        not preprocessing_plan.materialize_sameas.enabled
        and not preprocessing_plan.materialize_target_roles.enabled
    )
    graph_build_scan_cache: Optional[GraphBuildScanCache] = None
    if can_incrementally_build_scan_cache:
        graph_build_scan_cache = _scan_graph_build_cache(
            data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            preprocessing_timings=preprocessing_timings,
            schema_property_terms=cache.schema_property_terms,
            schema_class_terms=cache.schema_class_terms,
            schema_datatype_terms=cache.schema_datatype_terms,
            vocab_source=ontology_graph,
        )

    effective_data_graph = _copy_graph(data_graph)
    def apply_positive_preprocessing_pass(
        graph: Graph,
        *,
        include_property_hierarchy: bool,
    ) -> Graph:
        new_domain_range_types: List[Tuple[Identifier, URIRef]] = []
        added_triples: Optional[List[Tuple[Identifier, Identifier, Identifier]]] = (
            [] if graph_build_scan_cache is not None else None
        )
        cached_type_assertions = (
            graph_build_scan_cache.type_pairs if graph_build_scan_cache is not None else None
        )
        cached_property_triples = (
            graph_build_scan_cache.edge_triples if graph_build_scan_cache is not None else None
        )
        cached_property_pairs_by_pred = (
            graph_build_scan_cache.edge_triples_by_pred if graph_build_scan_cache is not None else None
        )
        if preprocessing_plan.materialize_hierarchy.enabled:
            t0 = perf_counter()
            if include_property_hierarchy:
                _materialize_hierarchy_closure_in_place(
                    graph,
                    subclass_supers=preprocessing_class_supers,
                    subproperty_supers=subproperty_supers,
                    all_different_pairs=all_different_pairs,
                    type_assertions=cached_type_assertions,
                    property_triples=cached_property_triples,
                    new_triples=added_triples,
                )
            else:
                _materialize_hierarchy_closure_in_place(
                    graph,
                    subclass_supers=preprocessing_class_supers,
                    type_assertions=cached_type_assertions,
                    new_triples=added_triples,
                )
            preprocessing_timings.hierarchy_elapsed_ms += (perf_counter() - t0) * 1000.0

        property_triples_for_domain_range = cached_property_triples
        property_pairs_by_pred_for_domain_range = cached_property_pairs_by_pred
        if property_triples_for_domain_range is not None and added_triples:
            added_property_triples = [
                (subj, pred, obj)
                for subj, pred, obj in added_triples
                if isinstance(pred, URIRef) and pred != RDF.type
            ]
            if added_property_triples:
                property_triples_for_domain_range = list(property_triples_for_domain_range) + added_property_triples
        if property_pairs_by_pred_for_domain_range is not None and added_triples:
            added_pairs_by_pred: Dict[URIRef, List[Tuple[Identifier, Identifier]]] = defaultdict(list)
            for subj, pred, obj in added_triples:
                if isinstance(pred, URIRef) and pred != RDF.type:
                    added_pairs_by_pred[pred].append((subj, obj))
            if added_pairs_by_pred:
                property_pairs_by_pred_for_domain_range = {
                    pred: list(property_pairs_by_pred_for_domain_range.get(pred, ()))
                    for pred in property_pairs_by_pred_for_domain_range.keys() | added_pairs_by_pred.keys()
                }
                for pred, pairs in added_pairs_by_pred.items():
                    property_pairs_by_pred_for_domain_range[pred].extend(pairs)

        if preprocessing_plan.materialize_horn_safe_domain_range.enabled:
            t0 = perf_counter()
            _materialize_domain_range_closure_in_place(
                graph,
                domain_consequents=horn_domain_consequents,
                range_consequents=horn_range_consequents,
                active_predicates=horn_domain_range_predicates,
                property_triples=property_triples_for_domain_range,
                property_pairs_by_pred=property_pairs_by_pred_for_domain_range,
                new_type_assertions=new_domain_range_types,
                new_triples=added_triples,
            )
            preprocessing_timings.horn_safe_domain_range_elapsed_ms += (perf_counter() - t0) * 1000.0
        elif preprocessing_plan.materialize_atomic_domain_range.enabled:
            t0 = perf_counter()
            _materialize_domain_range_closure_in_place(
                graph,
                domain_consequents=atomic_domain_consequents,
                range_consequents=atomic_range_consequents,
                active_predicates=atomic_domain_range_predicates,
                property_triples=property_triples_for_domain_range,
                property_pairs_by_pred=property_pairs_by_pred_for_domain_range,
                new_type_assertions=new_domain_range_types,
                new_triples=added_triples,
            )
            preprocessing_timings.atomic_domain_range_elapsed_ms += (perf_counter() - t0) * 1000.0

        if preprocessing_plan.materialize_hierarchy.enabled and new_domain_range_types:
            t0 = perf_counter()
            _materialize_hierarchy_closure_in_place(
                graph,
                subclass_supers=preprocessing_class_supers,
                seed_type_assertions=new_domain_range_types,
                new_triples=added_triples,
            )
            preprocessing_timings.hierarchy_elapsed_ms += (perf_counter() - t0) * 1000.0
        if graph_build_scan_cache is not None and added_triples:
            _extend_graph_build_scan_cache(
                graph_build_scan_cache,
                added_triples,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
            )
        return graph

    if preprocessing_plan.materialize_reflexive_properties.enabled:
        reflexive_added_triples: Optional[List[Tuple[Identifier, Identifier, Identifier]]] = (
            [] if graph_build_scan_cache is not None else None
        )
        t0 = perf_counter()
        _materialize_reflexive_property_closure_in_place(
            effective_data_graph,
            reflexive_props=reflexive_props,
            new_triples=reflexive_added_triples,
        )
        preprocessing_timings.reflexive_elapsed_ms += (perf_counter() - t0) * 1000.0
        if graph_build_scan_cache is not None and reflexive_added_triples:
            _extend_graph_build_scan_cache(
                graph_build_scan_cache,
                reflexive_added_triples,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
            )

    effective_data_graph = apply_positive_preprocessing_pass(
        effective_data_graph,
        include_property_hierarchy=not (
            preprocessing_plan.materialize_target_roles.enabled and target_classes
        ),
    )

    if preprocessing_plan.materialize_sameas.enabled:
        max_sameas_iterations = 4
        for _ in range(max_sameas_iterations):
            before_len = len(effective_data_graph)
            t0 = perf_counter()
            _materialize_sameas_closure_in_place(
                effective_data_graph,
                ontology_graph=ontology_graph,
                singleton_nominals=singleton_nominals,
                has_key_axioms=active_has_key_axioms,
            )
            sameas_pass_ms = (perf_counter() - t0) * 1000.0
            preprocessing_timings.sameas_elapsed_ms += sameas_pass_ms
            preprocessing_timings.sameas_passes_elapsed_ms.append(sameas_pass_ms)
            if len(effective_data_graph) == before_len:
                break
            effective_data_graph = apply_positive_preprocessing_pass(
                effective_data_graph,
                include_property_hierarchy=not (
                    preprocessing_plan.materialize_target_roles.enabled and target_classes
                ),
            )
            if len(effective_data_graph) == before_len:
                break

    ontology_graph = aggregate_rdflib_graphs((schema_graph, effective_data_graph))

    if preprocessing_plan.materialize_target_roles.enabled and target_classes:
        if dependency_closure is None:
            dependency_closure = TargetDependencyClosure(
                target_classes=sorted(
                    {
                        URIRef(target_class) if isinstance(target_class, str) else target_class
                        for target_class in target_classes
                    },
                    key=str,
                ),
                referenced_named_classes=[],
                inferable_named_classes=[],
                referenced_properties=collect_referenced_properties_for_classes(
                    ontology_graph,
                    target_classes,
                ),
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
        ontology_graph = aggregate_rdflib_graphs((schema_graph, effective_data_graph))

    kg_source = effective_data_graph if len(effective_data_graph) > 0 else ontology_graph
    t0 = perf_counter()
    kg, mapping = rdflib_graph_to_kgraph(
        kg_source,
        vocab_graph=ontology_graph,
        vocab_source=ontology_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        preprocessing_timings=preprocessing_timings,
        schema_property_terms=cache.schema_property_terms,
        schema_class_terms=cache.schema_class_terms,
        schema_datatype_terms=cache.schema_datatype_terms,
        scan_cache=graph_build_scan_cache,
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

        has_value = ontology_graph.value(expr, OWL.hasValue)
        if has_value is not None:
            return f"hasValue({prop_rendered}, {describe_owl_expression(ontology_graph, has_value)})"

        has_self = _literal_to_bool(ontology_graph.value(expr, OWL.hasSelf))
        if has_self is True:
            return f"hasSelf({prop_rendered})"

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

    has_value = ontology_graph.value(expr, OWL.hasValue)
    if has_value is not None:
        child_condition = NormalizedSufficientCondition(
            kind=SufficientConditionKind.NOMINAL,
            node_term=has_value,
        )
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.EXISTS,
                prop_term=prop_term,
                prop_direction=prop_direction,
                children=(child_condition,),
            )
        ]

    has_self = _literal_to_bool(ontology_graph.value(expr, OWL.hasSelf))
    if has_self is True:
        return [
            NormalizedSufficientCondition(
                kind=SufficientConditionKind.HAS_SELF,
                prop_term=prop_term,
                prop_direction=prop_direction,
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
    if condition.kind == SufficientConditionKind.HAS_SELF and condition.prop_term is not None:
        prop_rendered = (
            condition.prop_term.n3()
            if condition.prop_direction == TraversalDirection.FORWARD
            else f"{condition.prop_term.n3()}^-1"
        )
        return f"hasSelf({prop_rendered})"
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

    for antecedent_expr, _pred, consequent_expr in ontology_graph.triples((None, RDFS.subClassOf, None)):
        consequents = _extract_horn_safe_named_class_consequents(ontology_graph, consequent_expr)
        if consequents is None:
            continue
        normalized = _normalize_positive_sufficient_conditions(ontology_graph, antecedent_expr)
        if normalized is None:
            continue
        for consequent_class in consequents:
            for antecedent in normalized:
                add_rule(
                    consequent_class=consequent_class,
                    antecedent=antecedent,
                    source_kind="subClassOfHornSafeConsequent",
                    source_term=consequent_expr,
                    tags=[antecedent.kind.value, "horn-safe-consequent"],
                )

    for left, _pred, right in ontology_graph.triples((None, OWL.equivalentClass, None)):
        normalized_left = _normalize_positive_sufficient_conditions(ontology_graph, left)
        right_consequents = _extract_horn_safe_named_class_consequents(ontology_graph, right)
        if normalized_left is not None and right_consequents is not None:
            for consequent_class in right_consequents:
                for antecedent in normalized_left:
                    add_rule(
                        consequent_class=consequent_class,
                        antecedent=antecedent,
                        source_kind="equivalentClassHornSafeConsequent",
                        source_term=right,
                        tags=[antecedent.kind.value, "horn-safe-consequent"],
                    )

        normalized_right = _normalize_positive_sufficient_conditions(ontology_graph, right)
        left_consequents = _extract_horn_safe_named_class_consequents(ontology_graph, left)
        if normalized_right is not None and left_consequents is not None:
            for consequent_class in left_consequents:
                for antecedent in normalized_right:
                    add_rule(
                        consequent_class=consequent_class,
                        antecedent=antecedent,
                        source_kind="equivalentClassHornSafeConsequent",
                        source_term=left,
                        tags=[antecedent.kind.value, "horn-safe-consequent"],
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


def index_normalized_sufficient_rules_by_consequent(
    rule_set: NormalizedSufficientRuleSet,
) -> Dict[URIRef, List[NormalizedSufficientCondition]]:
    indexed: Dict[URIRef, List[NormalizedSufficientCondition]] = defaultdict(list)
    for rule in rule_set.rules:
        indexed[rule.consequent_class].append(rule.antecedent)
    return indexed


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


def _extract_negative_named_blockers_from_expr(
    ontology_graph: Graph,
    expr: Identifier,
    *,
    visited: set[Identifier],
    blocker_classes: set[URIRef],
    skipped_axioms: set[str],
) -> None:
    if expr in visited:
        return
    visited.add(expr)

    complement_target = ontology_graph.value(expr, OWL.complementOf)
    if complement_target is not None:
        if isinstance(complement_target, URIRef) and not _is_datatype_term(complement_target):
            blocker_classes.add(complement_target)
        else:
            skipped_axioms.add(
                "complementOf(" + describe_owl_expression(ontology_graph, complement_target) + ")"
            )
        return

    if isinstance(expr, URIRef):
        for disjoint_expr in ontology_graph.objects(expr, OWL.disjointWith):
            if isinstance(disjoint_expr, URIRef) and not _is_datatype_term(disjoint_expr):
                blocker_classes.add(disjoint_expr)
            else:
                skipped_axioms.add(
                    "disjointWith(" + describe_owl_expression(ontology_graph, disjoint_expr) + ")"
                )
        for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, expr):
            if disjoint_expr == expr:
                continue
            if isinstance(disjoint_expr, URIRef) and not _is_datatype_term(disjoint_expr):
                blocker_classes.add(disjoint_expr)
            else:
                skipped_axioms.add(
                    "disjointWith(" + describe_owl_expression(ontology_graph, disjoint_expr) + ")"
                )

    for head in ontology_graph.objects(expr, OWL.intersectionOf):
        if isinstance(head, BNode):
            for member in Collection(ontology_graph, head):
                _extract_negative_named_blockers_from_expr(
                    ontology_graph,
                    member,
                    visited=visited,
                    blocker_classes=blocker_classes,
                    skipped_axioms=skipped_axioms,
                )

    for sub_expr in ontology_graph.objects(expr, RDFS.subClassOf):
        _extract_negative_named_blockers_from_expr(
            ontology_graph,
            sub_expr,
            visited=visited,
            blocker_classes=blocker_classes,
            skipped_axioms=skipped_axioms,
        )
    for eq_expr in ontology_graph.objects(expr, OWL.equivalentClass):
        if eq_expr != expr:
            _extract_negative_named_blockers_from_expr(
                ontology_graph,
                eq_expr,
                visited=visited,
                blocker_classes=blocker_classes,
                skipped_axioms=skipped_axioms,
            )
    for eq_expr in ontology_graph.subjects(OWL.equivalentClass, expr):
        if eq_expr != expr:
            _extract_negative_named_blockers_from_expr(
                ontology_graph,
                eq_expr,
                visited=visited,
                blocker_classes=blocker_classes,
                skipped_axioms=skipped_axioms,
            )


def collect_negative_blocker_specs(
    ontology_graph: Graph,
    target_classes: Optional[Sequence[str | URIRef]] = None,
) -> Dict[URIRef, NegativeBlockerSpec]:
    target_terms = (
        [URIRef(term) if isinstance(term, str) else term for term in target_classes]
        if target_classes is not None
        else sorted(
            [
                class_term
                for class_term in _collect_named_class_terms(ontology_graph)
                if isinstance(class_term, URIRef) and _has_nontrivial_definition(ontology_graph, class_term)
            ],
            key=str,
        )
    )

    specs: Dict[URIRef, NegativeBlockerSpec] = {}
    functional_properties = _collect_functional_property_terms(ontology_graph)
    for target_term in target_terms:
        blocker_classes: set[URIRef] = set()
        blocker_nominal_members: List[Identifier] = []
        functional_data_requirements: List[Tuple[URIRef, Literal]] = []
        exact_property_requirements: List[Tuple[URIRef, Identifier]] = []
        skipped_axioms: set[str] = set()
        visited: set[Identifier] = set()

        for disjoint_expr in ontology_graph.objects(target_term, OWL.disjointWith):
            if isinstance(disjoint_expr, URIRef) and not _is_datatype_term(disjoint_expr):
                blocker_classes.add(disjoint_expr)
            else:
                skipped_axioms.add(
                    "disjointWith(" + describe_owl_expression(ontology_graph, disjoint_expr) + ")"
                )
        for disjoint_expr in ontology_graph.subjects(OWL.disjointWith, target_term):
            if disjoint_expr == target_term:
                continue
            if isinstance(disjoint_expr, URIRef) and not _is_datatype_term(disjoint_expr):
                blocker_classes.add(disjoint_expr)
            else:
                skipped_axioms.add(
                    "disjointWith(" + describe_owl_expression(ontology_graph, disjoint_expr) + ")"
                )

        for root_expr in _collect_target_root_expressions(ontology_graph, target_term):
            _extract_negative_named_blockers_from_expr(
                ontology_graph,
                root_expr,
                visited=visited,
                blocker_classes=blocker_classes,
                skipped_axioms=skipped_axioms,
            )
            blocker_nominal_members.extend(
                _extract_conjunctive_singleton_nominal_members(ontology_graph, root_expr)
            )
            functional_data_requirements.extend(
                _extract_conjunctive_functional_data_requirements(
                    ontology_graph,
                    root_expr,
                    functional_properties=functional_properties,
                )
            )
            exact_property_requirements.extend(
                _extract_conjunctive_exact_property_requirements(
                    ontology_graph,
                    root_expr,
                )
            )
        blocker_classes.discard(target_term)

        specs[target_term] = NegativeBlockerSpec(
            target_class=target_term,
            blocker_classes=sorted(blocker_classes, key=str),
            blocker_nominal_members=sorted(
                dict.fromkeys(blocker_nominal_members),
                key=_term_sort_key,
            ),
            functional_data_requirements=sorted(
                dict.fromkeys(functional_data_requirements),
                key=lambda item: (str(item[0]), item[1].n3()),
            ),
            exact_property_requirements=sorted(
                dict.fromkeys(exact_property_requirements),
                key=lambda item: (str(item[0]), _term_sort_key(item[1])),
            ),
            skipped_negative_axioms=sorted(skipped_axioms),
        )

    return specs


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
        elif term.kind == SufficientConditionKind.HAS_SELF:
            if term.prop_term is None or term.prop_term not in mapping.prop_to_idx:
                raise KeyError(f"Property {term.prop_term} not present in property mapping.")
            idx = new_node(
                ctype=ConstraintType.HAS_SELF_RESTRICTION,
                prop_idx=mapping.prop_to_idx[term.prop_term],
                prop_direction=term.prop_direction,
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
    antecedents_by_consequent: Optional[Dict[URIRef, List[NormalizedSufficientCondition]]] = None,
    include_atomic_seed: bool = True,
    intersection_agg: IntersectionAgg = IntersectionAgg.MIN,
    cardinality_agg: CardinalityAgg = CardinalityAgg.STRICT,
) -> ConstraintDAG:
    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    rules = rule_set or collect_normalized_sufficient_condition_rules(ontology_graph)
    indexed_antecedents = (
        antecedents_by_consequent
        if antecedents_by_consequent is not None
        else index_normalized_sufficient_rules_by_consequent(rules)
    )
    antecedents = list(indexed_antecedents.get(target_term, ()))

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
    compile_context: Optional[OntologyCompileContext] = None,
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
        target_roots = _collect_target_root_expressions(
            ontology_graph,
            target_term,
            dependency_analysis=(
                compile_context.dependency_analysis if compile_context is not None else None
            ),
            compile_context=compile_context,
        )
        if target_roots and all(
            _expression_entails_expression(
                ontology_graph,
                expr,
                root,
                subclass_supers,
                memo,
                active_pairs,
                compile_context,
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
                        compile_context,
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
                        compile_context,
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
    compile_context: Optional[OntologyCompileContext] = None,
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
            compile_context,
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
                        compile_context,
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
                        compile_context,
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
                        compile_context,
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
                        compile_context,
                    )
                    for member in lhs_members
                )

    active_pairs.remove(key)
    memo[key] = result
    return result
    return None


def query_target_is_obviously_supported(
    ontology_graph: Graph,
    mapping: RDFKGraphMapping,
    target_class: str | URIRef,
    *,
    augment_property_domain_range: bool = False,
    compile_context: Optional[OntologyCompileContext] = None,
) -> bool:
    """
    Conservative fast-path check for query target support.

    `True` means the target appears to use only compiler-supported structures
    over terms present in the current mapping. `False` means "unsupported or
    uncertain", and callers should fall back to the full compiler.
    """

    target_term = URIRef(target_class) if isinstance(target_class, str) else target_class
    if compile_context is None:
        compile_context = build_ontology_compile_context(ontology_graph)

    canonical_map = compile_context.dependency_analysis.canonical_map
    class_equivalence_members = compile_context.dependency_analysis.equivalence_members
    sameas_equivalence_map = compile_context.sameas_equivalence_map
    property_axioms = compile_context.property_axioms
    subclass_supers = compile_context.subclass_supers
    target_canonical = canonical_map.get(target_term, target_term)
    support_cache_key = (target_canonical, augment_property_domain_range)
    cached_target_support = compile_context.target_support_cache.get(support_cache_key)
    if cached_target_support is not None:
        return cached_target_support

    memo = compile_context.expression_support_cache
    active_exprs: set[Identifier] = set()

    def term_has_nominal_witness(term: Identifier) -> bool:
        expanded_members = sameas_equivalence_map.get(term, [term])
        return any(expanded_member in mapping.node_to_idx for expanded_member in expanded_members)

    def expr_supported(expr: Identifier) -> bool:
        cached = memo.get(expr)
        if cached is not None:
            return cached
        if expr in active_exprs:
            if isinstance(expr, URIRef):
                canonical_expr = canonical_map.get(expr, expr)
                return canonical_expr in mapping.class_to_idx or expr in mapping.node_to_idx
            return False

        active_exprs.add(expr)
        result = False
        try:
            if _parse_datatype_restriction(ontology_graph, expr, mapping) is not None:
                result = True
            elif isinstance(expr, Literal):
                result = expr in mapping.node_to_idx
            elif isinstance(expr, URIRef):
                canonical_expr = canonical_map.get(expr, expr)
                result = canonical_expr in mapping.class_to_idx or expr in mapping.node_to_idx
            elif isinstance(expr, BNode):
                one_of_heads = list(ontology_graph.objects(expr, OWL.oneOf))
                if one_of_heads:
                    member_count = 0
                    result = True
                    for head in one_of_heads:
                        members = _rdf_list_members(ontology_graph, head)
                        if not members:
                            result = False
                            break
                        for member in members:
                            member_count += 1
                            if not term_has_nominal_witness(member):
                                result = False
                                break
                        if not result:
                            break
                    result = result and member_count > 0
                else:
                    union_heads = list(ontology_graph.objects(expr, OWL.unionOf))
                    if union_heads:
                        members = [member for head in union_heads for member in _rdf_list_members(ontology_graph, head)]
                        result = bool(members) and all(expr_supported(member) for member in members)
                    else:
                        intersection_heads = list(ontology_graph.objects(expr, OWL.intersectionOf))
                        if intersection_heads:
                            members = [
                                member
                                for head in intersection_heads
                                for member in _rdf_list_members(ontology_graph, head)
                            ]
                            result = bool(members) and all(expr_supported(member) for member in members)
                        else:
                            complement_targets = list(ontology_graph.objects(expr, OWL.complementOf))
                            if complement_targets:
                                result = len(complement_targets) == 1 and expr_supported(complement_targets[0])
                            else:
                                restriction_types = set(ontology_graph.objects(expr, RDF.type))
                                if OWL.Restriction in restriction_types:
                                    prop_expr = ontology_graph.value(expr, OWL.onProperty)
                                    try:
                                        prop, _prop_direction = _resolve_property_expression(ontology_graph, prop_expr)
                                    except ValueError:
                                        result = False
                                    else:
                                        if prop not in mapping.prop_to_idx:
                                            result = False
                                        else:
                                            some_filler = ontology_graph.value(expr, OWL.someValuesFrom)
                                            all_filler = ontology_graph.value(expr, OWL.allValuesFrom)
                                            has_value = ontology_graph.value(expr, OWL.hasValue)
                                            has_self = _literal_to_bool(ontology_graph.value(expr, OWL.hasSelf))
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

                                            if some_filler is not None:
                                                result = expr_supported(some_filler)
                                            elif all_filler is not None:
                                                result = expr_supported(all_filler)
                                            elif has_value is not None:
                                                result = expr_supported(has_value)
                                            elif has_self is True:
                                                result = True
                                            elif min_card is not None or max_card is not None or exact_card is not None:
                                                result = len(
                                                    [
                                                        term
                                                        for term in (min_card, max_card, exact_card)
                                                        if term is not None
                                                    ]
                                                ) == 1
                                            elif (
                                                min_qualified is not None
                                                or max_qualified is not None
                                                or exact_qualified is not None
                                            ):
                                                result = (
                                                    len(
                                                        [
                                                            term
                                                            for term in (
                                                                min_qualified,
                                                                max_qualified,
                                                                exact_qualified,
                                                            )
                                                            if term is not None
                                                        ]
                                                    )
                                                    == 1
                                                    and on_class is not None
                                                    and expr_supported(on_class)
                                                )

            memo[expr] = result
            return result
        finally:
            active_exprs.discard(expr)

    target_component = class_equivalence_members.get(target_canonical, [target_term])
    saw_root = False
    target_roots = _collect_target_root_expressions(
        ontology_graph,
        target_term,
        dependency_analysis=compile_context.dependency_analysis,
        compile_context=compile_context,
    )
    if target_roots:
        for expr in target_roots:
            saw_root = True
            if not expr_supported(expr):
                compile_context.target_support_cache[support_cache_key] = False
                return False
    if not saw_root and not expr_supported(target_term):
        compile_context.target_support_cache[support_cache_key] = False
        return False

    if not augment_property_domain_range:
        compile_context.target_support_cache[support_cache_key] = True
        return True

    entailment_memo = compile_context.entailment_cache
    for prop, axioms in property_axioms.items():
        if prop not in mapping.prop_to_idx:
            compile_context.target_support_cache[support_cache_key] = False
            return False
        if any(
            _expression_entails_target_class(
                ontology_graph,
                expr,
                target_term,
                subclass_supers,
                entailment_memo,
                compile_context=compile_context,
            )
            for expr in axioms.domain_expressions
        ):
            for range_expr in axioms.range_expressions:
                if not expr_supported(range_expr):
                    compile_context.target_support_cache[support_cache_key] = False
                    return False
        if any(
            _expression_entails_target_class(
                ontology_graph,
                expr,
                target_term,
                subclass_supers,
                entailment_memo,
                compile_context=compile_context,
            )
            for expr in axioms.range_expressions
        ):
            for domain_expr in axioms.domain_expressions:
                if not expr_supported(domain_expr):
                    compile_context.target_support_cache[support_cache_key] = False
                    return False

    compile_context.target_support_cache[support_cache_key] = True
    return True


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
    compile_context: Optional[OntologyCompileContext] = None,
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
        compile_context=compile_context,
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
    dependency_analysis: Optional[NamedClassDependencyAnalysis] = None,
    compile_context: Optional[OntologyCompileContext] = None,
) -> ConstraintDAG:
    """
    Compile a small OWL fragment into a ConstraintDAG.

    Supported fragments:
    - named atomic classes
    - `owl:oneOf` nominals
    - `owl:hasValue` / datatype hasValue
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
    if compile_context is not None:
        if dependency_analysis is None:
            dependency_analysis = compile_context.dependency_analysis
    if dependency_analysis is None:
        class_equivalence_members = collect_named_class_equivalence_members(ontology_graph)
        canonical_class_map = {
            member: min(members, key=str)
            for members in class_equivalence_members.values()
            for member in members
        }
    else:
        class_equivalence_members = dependency_analysis.equivalence_members
        canonical_class_map = dependency_analysis.canonical_map
    nodes: List[ConstraintNode] = []
    memo: Dict[Identifier, int] = {}
    if compile_context is None:
        direct_subs = _collect_direct_subproperties(ontology_graph)
        inverse_props = _collect_inverse_properties(ontology_graph)
        chains_by_conclusion = _collect_property_chains(ontology_graph)
        transitive_props = _collect_transitive_properties(ontology_graph)
        functional_props = _collect_functional_property_terms_from_graph(ontology_graph)
        sameas_equivalence_map = collect_sameas_equivalence_map(ontology_graph)
    else:
        direct_subs = compile_context.direct_subs
        inverse_props = compile_context.inverse_props
        chains_by_conclusion = compile_context.chains_by_conclusion
        transitive_props = compile_context.transitive_props
        functional_props = compile_context.functional_props
        sameas_equivalence_map = compile_context.sameas_equivalence_map
    subproperty_closure_cache: Dict[URIRef, Tuple[URIRef, ...]] = {}
    restriction_memo: Dict[Tuple[str, URIRef, str, int], int] = {}
    active_exprs: set[Identifier] = set()

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
        stack: Optional[set[Tuple[str, URIRef, str]]] = None,
    ) -> int:
        key = (
            "forall" if universal else "exists",
            prop,
            prop_direction.value,
            child_idx,
        )
        cycle_key = (
            "forall" if universal else "exists",
            prop,
            prop_direction.value,
        )
        if key in restriction_memo:
            return restriction_memo[key]

        active_stack = set() if stack is None else set(stack)
        if cycle_key in active_stack:
            idx = direct_restriction_node(
                prop=prop,
                prop_direction=prop_direction,
                child_idx=child_idx,
                universal=universal,
            )
            restriction_memo[key] = idx
            return idx

        active_stack.add(cycle_key)
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

    def compile_exact_requirement(
        *,
        prop: URIRef,
        prop_direction: TraversalDirection,
        child_idx: int,
    ) -> int:
        branch_indices: List[int] = [
            compile_property_restriction(
                prop=prop,
                prop_direction=prop_direction,
                child_idx=child_idx,
                universal=False,
            )
        ]

        if prop_direction == TraversalDirection.FORWARD and prop in functional_props:
            branch_indices.append(
                compile_property_restriction(
                    prop=prop,
                    prop_direction=prop_direction,
                    child_idx=child_idx,
                    universal=True,
                )
            )

        negative_helper_prop = _negative_property_helper_term(prop)
        if (
            prop_direction == TraversalDirection.FORWARD
            and negative_helper_prop in mapping.prop_to_idx
        ):
            neg_exists_idx = compile_property_restriction(
                prop=negative_helper_prop,
                prop_direction=TraversalDirection.FORWARD,
                child_idx=child_idx,
                universal=False,
            )
            branch_indices.append(
                new_node(
                    ctype=ConstraintType.NEGATION,
                    child_indices=[neg_exists_idx],
                )
            )

        if len(branch_indices) == 1:
            return branch_indices[0]
        return new_node(
            ctype=ConstraintType.INTERSECTION,
            child_indices=branch_indices,
            intersection_agg=intersection_agg,
        )

    def compile_expr(expr: Identifier) -> int:
        if expr in active_exprs:
            if isinstance(expr, URIRef):
                canonical_expr = canonical_class_map.get(expr, expr)
                if canonical_expr in mapping.class_to_idx:
                    return new_node(
                        ctype=ConstraintType.ATOMIC_CLASS,
                        class_idx=mapping.class_to_idx[canonical_expr],
                    )
            raise ValueError(
                f"Cyclic anonymous class expression encountered while compiling {expr!r}."
            )
        if expr in memo:
            return memo[expr]
        active_exprs.add(expr)
        try:
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
                seen_nominal_terms: set[Identifier] = set()
                for head in one_of_heads:
                    for member in _rdf_list_members(ontology_graph, head):
                        expanded_members = sameas_equivalence_map.get(member, [member])
                        for expanded_member in expanded_members:
                            if expanded_member in seen_nominal_terms:
                                continue
                            if expanded_member not in mapping.node_to_idx:
                                if expanded_member == member:
                                    raise KeyError(
                                        f"Nominal member {expanded_member!r} not present in KGraph node mapping."
                                    )
                                continue
                            seen_nominal_terms.add(expanded_member)
                            members.append(
                                new_node(
                                    ctype=ConstraintType.NOMINAL,
                                    node_idx=mapping.node_to_idx[expanded_member],
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
                has_value = ontology_graph.value(expr, OWL.hasValue)
                has_self = _literal_to_bool(ontology_graph.value(expr, OWL.hasSelf))
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
                    and has_value is None
                    and has_self is not True
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
                    exact_filler = _extract_singleton_oneof_member(ontology_graph, some_filler)
                    if exact_filler is not None:
                        idx = compile_exact_requirement(
                            prop=prop,
                            prop_direction=prop_direction,
                            child_idx=child_idx,
                        )
                    else:
                        idx = compile_property_restriction(
                            prop=prop,
                            prop_direction=prop_direction,
                            child_idx=child_idx,
                            universal=False,
                        )
                elif has_value is not None:
                    child_idx = compile_expr(has_value)
                    idx = compile_exact_requirement(
                        prop=prop,
                        prop_direction=prop_direction,
                        child_idx=child_idx,
                    )
                elif has_self is True:
                    idx = new_node(
                        ctype=ConstraintType.HAS_SELF_RESTRICTION,
                        prop_idx=mapping.prop_to_idx[prop],
                        prop_direction=prop_direction,
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

            if isinstance(expr, Literal):
                if expr not in mapping.node_to_idx:
                    raise KeyError(f"Literal expression {expr!r} not present in KGraph node mapping.")
                idx = new_node(
                    ctype=ConstraintType.NOMINAL,
                    node_idx=mapping.node_to_idx[expr],
                )
                memo[expr] = idx
                return idx

            if isinstance(expr, (URIRef, BNode)):
                canonical_expr = canonical_class_map.get(expr, expr) if isinstance(expr, URIRef) else expr
                if canonical_expr in mapping.class_to_idx:
                    idx = new_node(
                        ctype=ConstraintType.ATOMIC_CLASS,
                        class_idx=mapping.class_to_idx[canonical_expr],
                    )
                elif expr in mapping.node_to_idx:
                    idx = new_node(
                        ctype=ConstraintType.NOMINAL,
                        node_idx=mapping.node_to_idx[expr],
                    )
                else:
                    raise KeyError(f"Expression {expr} not present in KGraph class or node mapping.")
                memo[expr] = idx
                return idx

            raise ValueError(f"Unsupported class expression: {expr!r}")
        finally:
            active_exprs.discard(expr)

    target_canonical = canonical_class_map.get(target_term, target_term)
    target_component = class_equivalence_members.get(target_canonical, [target_term])
    root_children: List[int] = []

    for member in target_component:
        for expr in ontology_graph.objects(member, RDFS.subClassOf):
            root_children.append(compile_expr(expr))

        for expr in ontology_graph.objects(member, OWL.equivalentClass):
            if canonical_class_map.get(expr, expr) != target_canonical:
                root_children.append(compile_expr(expr))

        for expr in ontology_graph.objects(member, OWL.disjointWith):
            if canonical_class_map.get(expr, expr) == target_canonical:
                continue
            root_children.append(
                new_node(
                    ctype=ConstraintType.NEGATION,
                    child_indices=[compile_expr(expr)],
                )
            )

        for expr in ontology_graph.subjects(OWL.disjointWith, member):
            if canonical_class_map.get(expr, expr) == target_canonical:
                continue
            root_children.append(
                new_node(
                    ctype=ConstraintType.NEGATION,
                    child_indices=[compile_expr(expr)],
                )
            )

        for expr in ontology_graph.objects(member, OWL.intersectionOf):
            for member_expr in _rdf_list_members(ontology_graph, expr):
                root_children.append(compile_expr(member_expr))

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
    if compile_context is None:
        property_axioms = collect_property_expression_axioms(ontology_graph)
        subclass_supers = _compute_transitive_super_map(ontology_graph, RDFS.subClassOf)
        entailment_memo: Dict[Tuple[str, str], bool] = {}
    else:
        property_axioms = compile_context.property_axioms
        subclass_supers = compile_context.subclass_supers
        entailment_memo = compile_context.entailment_cache

    for prop, axioms in property_axioms.items():
        if any(
            _expression_entails_target_class(
                ontology_graph,
                expr,
                target_term,
                subclass_supers,
                entailment_memo,
                compile_context=compile_context,
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
                compile_context=compile_context,
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
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
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

    t0 = perf_counter()
    current_data_graph = Graph()
    for triple in data_graph:
        current_data_graph.add(triple)
    timings.initial_data_copy_elapsed_ms = (perf_counter() - t0) * 1000.0

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
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
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
            materialize_sameas=materialize_sameas,
            materialize_haskey_equality=materialize_haskey_equality,
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
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
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

    total_t0 = perf_counter()
    timings = PositiveMaterializationTimings()

    current_data_graph = Graph()
    for triple in data_graph:
        current_data_graph.add(triple)

    inferred_assertions: List[Tuple[Identifier, URIRef]] = []
    iterations = 0
    closure: Optional[TargetDependencyClosure] = None
    target_terms = [URIRef(term) if isinstance(term, str) else term for term in target_classes] if target_classes else None
    t0 = perf_counter()
    base_rule_set = collect_normalized_sufficient_condition_rules(schema_graph)
    timings.rule_extraction_elapsed_ms = (perf_counter() - t0) * 1000.0
    t0 = perf_counter()
    antecedents_by_consequent = index_normalized_sufficient_rules_by_consequent(base_rule_set)
    timings.rule_index_elapsed_ms = (perf_counter() - t0) * 1000.0
    t0 = perf_counter()
    build_cache = build_reasoning_build_cache(schema_graph)
    timings.schema_cache_elapsed_ms = (perf_counter() - t0) * 1000.0
    t0 = perf_counter()
    initial_ontology_graph = aggregate_rdflib_graphs((schema_graph, data_graph))
    timings.initial_ontology_merge_elapsed_ms = (perf_counter() - t0) * 1000.0
    t0 = perf_counter()
    preprocessing_plan = plan_reasoning_preprocessing(
        initial_ontology_graph,
        target_classes=target_terms,
        materialize_hierarchy=materialize_hierarchy,
        materialize_horn_safe_domain_range=True if materialize_hierarchy else None,
        materialize_sameas=materialize_sameas,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_target_roles=materialize_target_roles,
    )
    timings.preprocessing_plan_elapsed_ms = (perf_counter() - t0) * 1000.0
    active_has_key_axioms = (
        build_cache.has_key_axioms
        if preprocessing_plan.materialize_haskey_equality.enabled
        else {}
    )
    loop_materialize_hierarchy = preprocessing_plan.materialize_hierarchy.enabled
    loop_materialize_sameas = preprocessing_plan.materialize_sameas.enabled
    loop_materialize_haskey_equality = preprocessing_plan.materialize_haskey_equality.enabled
    loop_materialize_horn_safe_domain_range = preprocessing_plan.materialize_horn_safe_domain_range.enabled
    loop_materialize_atomic_domain_range = preprocessing_plan.materialize_atomic_domain_range.enabled
    loop_preprocessing_plan = preprocessing_plan
    sameas_trigger_classes = set(build_cache.singleton_nominals.keys()) | set(active_has_key_axioms.keys())
    sameas_state: Optional[SameAsIncrementalState] = None
    if target_terms:
        t0 = perf_counter()
        closure = compute_sufficient_rule_dependency_closure(base_rule_set, target_terms)
        timings.dependency_closure_elapsed_ms = (perf_counter() - t0) * 1000.0
    target_classes_to_materialize = (
        [
            class_term
            for class_term in closure.inferable_named_classes
            if antecedents_by_consequent.get(class_term)
        ]
        if closure is not None
        else sorted(antecedents_by_consequent.keys(), key=str)
    )
    compiled_dags: Dict[URIRef, ConstraintDAG] = {}
    use_incremental_rebuild = False
    previous_incremental_dataset: Optional[ReasoningDataset] = None
    seeded_dataset: Optional[ReasoningDataset] = None
    pending_type_assertions: Optional[List[Tuple[Identifier, URIRef]]] = None

    while iterations < max_iterations:
        iterations += 1
        timings.iterations = iterations
        iteration_timing = MaterializationIterationTiming(iteration=iterations)
        if seeded_dataset is not None:
            dataset = seeded_dataset
            seeded_dataset = None
        elif use_incremental_rebuild:
            dataset = _build_reasoning_dataset_from_preprocessed_graph(
                schema_graph=schema_graph,
                effective_data_graph=current_data_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                preprocessing_plan=loop_preprocessing_plan,
                build_cache=build_cache,
                rerun_sameas=False,
                previous_dataset=previous_incremental_dataset,
                new_type_assertions=pending_type_assertions,
                build_ontology_graph=False,
                sameas_state=sameas_state,
            )
        else:
            dataset = build_reasoning_dataset_from_graphs(
                schema_graph=schema_graph,
                data_graph=current_data_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                materialize_hierarchy=loop_materialize_hierarchy,
                materialize_horn_safe_domain_range=loop_materialize_horn_safe_domain_range,
                materialize_atomic_domain_range=loop_materialize_atomic_domain_range,
                materialize_sameas=loop_materialize_sameas,
                materialize_haskey_equality=loop_materialize_haskey_equality,
                materialize_target_roles=materialize_target_roles,
                target_classes=target_terms,
                dependency_closure=closure,
                build_cache=build_cache,
            )
            if (
                sameas_state is None
                and preprocessing_plan.materialize_sameas.enabled
                and sameas_trigger_classes
            ):
                t0 = perf_counter()
                sameas_state = _build_sameas_incremental_state(
                    dataset.data_graph,
                    singleton_nominals=build_cache.singleton_nominals,
                    has_key_axioms=active_has_key_axioms,
                )
                timings.sameas_state_init_elapsed_ms += (perf_counter() - t0) * 1000.0
        if dataset.preprocessing_timings is not None:
            _accumulate_preprocessing_timings(
                timings,
                dataset.preprocessing_timings,
                iteration_timing=iteration_timing,
            )

        if not target_classes_to_materialize:
            timings.iteration_timings.append(iteration_timing)
            timings.total_elapsed_ms = (perf_counter() - total_t0) * 1000.0
            return ClassMaterializationResult(
                dataset=dataset,
                inferred_assertions=inferred_assertions,
                iterations=iterations,
                timings=timings,
            )

        from .dag_reasoner import DAGReasoner

        reasoner_t0 = perf_counter()
        reasoner = DAGReasoner(dataset.kg, device=device)
        reasoner_elapsed_ms = (perf_counter() - reasoner_t0) * 1000.0
        timings.reasoner_setup_elapsed_ms += reasoner_elapsed_ms
        iteration_timing.reasoner_setup_elapsed_ms += reasoner_elapsed_ms
        compile_t0 = perf_counter()
        for class_term in target_classes_to_materialize:
            dag = compiled_dags.get(class_term)
            if dag is None:
                dag = compile_sufficient_condition_dag(
                    schema_graph,
                    dataset.mapping,
                    class_term,
                    rule_set=base_rule_set,
                    antecedents_by_consequent=antecedents_by_consequent,
                )
                compiled_dags[class_term] = dag
            reasoner.add_concept(str(class_term), dag)
        compile_elapsed_ms = (perf_counter() - compile_t0) * 1000.0
        timings.dag_compile_elapsed_ms += compile_elapsed_ms
        iteration_timing.dag_compile_elapsed_ms += compile_elapsed_ms

        eval_t0 = perf_counter()
        scores = reasoner.evaluate_all().detach().cpu()
        eval_elapsed_ms = (perf_counter() - eval_t0) * 1000.0
        timings.dag_eval_elapsed_ms += eval_elapsed_ms
        iteration_timing.dag_eval_elapsed_ms += eval_elapsed_ms

        update_t0 = perf_counter()
        additions_this_round: List[Tuple[Identifier, URIRef]] = []
        node_terms = dataset.mapping.node_terms
        for class_col, class_term in enumerate(target_classes_to_materialize):
            class_idx = dataset.mapping.class_to_idx.get(class_term)
            candidate_mask = scores[:, class_col] >= threshold
            if class_idx is not None:
                candidate_mask = candidate_mask & (dataset.kg.node_types[:, class_idx].detach().cpu() < threshold)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()
            for node_idx in candidate_indices:
                additions_this_round.append((node_terms[node_idx], class_term))

        if not additions_this_round:
            update_elapsed_ms = (perf_counter() - update_t0) * 1000.0
            timings.assertion_update_elapsed_ms += update_elapsed_ms
            iteration_timing.assertion_update_elapsed_ms += update_elapsed_ms
            timings.iteration_timings.append(iteration_timing)
            timings.total_elapsed_ms = (perf_counter() - total_t0) * 1000.0
            return ClassMaterializationResult(
                dataset=dataset,
                inferred_assertions=inferred_assertions,
                iterations=iterations,
                timings=timings,
            )

        inferred_assertions.extend(additions_this_round)
        current_data_graph = dataset.data_graph
        triggering_sameas_classes = sorted(
            {
                class_term
                for _node_term, class_term in additions_this_round
                if class_term in sameas_trigger_classes
            },
            key=str,
        )
        rerun_sameas = bool(triggering_sameas_classes)
        for node_term, class_term in additions_this_round:
            current_data_graph.add((node_term, RDF.type, class_term))
        use_incremental_rebuild = True
        if rerun_sameas:
            dataset = _build_reasoning_dataset_from_preprocessed_graph(
                schema_graph=schema_graph,
                effective_data_graph=current_data_graph,
                include_literals=include_literals,
                include_type_edges=include_type_edges,
                preprocessing_plan=loop_preprocessing_plan,
                build_cache=build_cache,
                rerun_sameas=True,
                new_type_assertions=additions_this_round,
                sameas_trigger_classes=triggering_sameas_classes,
                sameas_state=sameas_state,
            )
            if dataset.preprocessing_timings is not None:
                _accumulate_preprocessing_timings(
                    timings,
                    dataset.preprocessing_timings,
                    iteration_timing=iteration_timing,
                )
            current_data_graph = dataset.data_graph
            previous_incremental_dataset = dataset
            seeded_dataset = dataset
            pending_type_assertions = None
        else:
            previous_incremental_dataset = dataset
            pending_type_assertions = additions_this_round
        update_elapsed_ms = (perf_counter() - update_t0) * 1000.0
        timings.assertion_update_elapsed_ms += update_elapsed_ms
        iteration_timing.assertion_update_elapsed_ms += update_elapsed_ms
        timings.iteration_timings.append(iteration_timing)

    dataset = (
        _build_reasoning_dataset_from_preprocessed_graph(
            schema_graph=schema_graph,
            effective_data_graph=current_data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            preprocessing_plan=loop_preprocessing_plan,
              build_cache=build_cache,
              rerun_sameas=False,
              previous_dataset=previous_incremental_dataset,
              new_type_assertions=pending_type_assertions,
              build_ontology_graph=True,
              sameas_state=sameas_state,
          )
        if use_incremental_rebuild
        else build_reasoning_dataset_from_graphs(
            schema_graph=schema_graph,
            data_graph=current_data_graph,
            include_literals=include_literals,
            include_type_edges=include_type_edges,
            materialize_hierarchy=loop_materialize_hierarchy,
            materialize_horn_safe_domain_range=loop_materialize_horn_safe_domain_range,
            materialize_atomic_domain_range=loop_materialize_atomic_domain_range,
            materialize_sameas=loop_materialize_sameas,
            materialize_haskey_equality=loop_materialize_haskey_equality,
            materialize_target_roles=materialize_target_roles,
            target_classes=target_terms,
            dependency_closure=closure,
            build_cache=build_cache,
        )
    )
    if dataset.preprocessing_timings is not None:
        _accumulate_preprocessing_timings(timings, dataset.preprocessing_timings)
        timings.final_dataset_timings = dataset.preprocessing_timings
    timings.total_elapsed_ms = (perf_counter() - total_t0) * 1000.0
    return ClassMaterializationResult(
        dataset=dataset,
        inferred_assertions=inferred_assertions,
        iterations=iterations,
        timings=timings,
    )


def materialize_negative_class_blockers(
    *,
    dataset: ReasoningDataset,
    target_classes: Optional[Sequence[str | URIRef]] = None,
) -> NegativeBlockerResult:
    """
    Compute blocked (node, class) assignments from the already-materialized
    positive closure for the currently supported negative fragment.

    Current blocker sources:
    - direct or inherited `owl:disjointWith`
    - direct or inherited named `owl:complementOf`

    This pass does not retract or materialize negative facts. It reports
    forbidden assignments and conflicts against the positive closure.
    """

    blocker_specs = collect_negative_blocker_specs(dataset.ontology_graph, target_classes)
    node_types = dataset.kg.node_types.detach().cpu()
    literal_values_by_subject_prop: Dict[Tuple[Identifier, URIRef], set[Literal]] = defaultdict(set)
    negative_property_assertions = _collect_negative_property_assertions(dataset.ontology_graph)

    for subj, pred, obj in dataset.ontology_graph:
        if isinstance(pred, URIRef) and isinstance(obj, Literal):
            literal_values_by_subject_prop[(subj, pred)].add(obj)

    blocked_assertions: List[BlockedClassAssertion] = []
    conflicting_positive_assertions: List[BlockedClassAssertion] = []

    for target_class, spec in blocker_specs.items():
        if target_class not in dataset.mapping.class_to_idx:
            continue
        target_class_idx = dataset.mapping.class_to_idx[target_class]
        for blocker_class in spec.blocker_classes:
            if blocker_class not in dataset.mapping.class_to_idx:
                continue
            blocker_idx = dataset.mapping.class_to_idx[blocker_class]
            for node_idx, node_term in enumerate(dataset.mapping.node_terms):
                if float(node_types[node_idx, blocker_idx].item()) < 0.999:
                    continue
                blocked = BlockedClassAssertion(
                    node_term=node_term,
                    target_class=target_class,
                    blocker_class=blocker_class,
                )
                blocked_assertions.append(blocked)
                if float(node_types[node_idx, target_class_idx].item()) >= 0.999:
                    conflicting_positive_assertions.append(blocked)

        for nominal_member in spec.blocker_nominal_members:
            for node_idx, node_term in enumerate(dataset.mapping.node_terms):
                if not (
                    (node_term, OWL.differentFrom, nominal_member) in dataset.ontology_graph
                    or (nominal_member, OWL.differentFrom, node_term) in dataset.ontology_graph
                ):
                    continue
                blocked = BlockedClassAssertion(
                    node_term=node_term,
                    target_class=target_class,
                    blocker_class=URIRef("urn:dag:blocker:differentFrom"),
                )
                blocked_assertions.append(blocked)
                if float(node_types[node_idx, target_class_idx].item()) >= 0.999:
                    conflicting_positive_assertions.append(blocked)

        for prop_term, required_literal in spec.functional_data_requirements:
            for node_idx, node_term in enumerate(dataset.mapping.node_terms):
                literal_values = literal_values_by_subject_prop.get((node_term, prop_term), set())
                if not literal_values:
                    continue
                if any(literal_value != required_literal for literal_value in literal_values):
                    blocked = BlockedClassAssertion(
                        node_term=node_term,
                        target_class=target_class,
                        blocker_class=URIRef("urn:dag:blocker:functionalDataProperty"),
                    )
                    blocked_assertions.append(blocked)
                    if float(node_types[node_idx, target_class_idx].item()) >= 0.999:
                        conflicting_positive_assertions.append(blocked)

        for prop_term, required_value in spec.exact_property_requirements:
            for node_idx, node_term in enumerate(dataset.mapping.node_terms):
                if (node_term, prop_term, required_value) not in negative_property_assertions:
                    continue
                blocker_suffix = (
                    "negativeDataPropertyAssertion"
                    if isinstance(required_value, Literal)
                    else "negativeObjectPropertyAssertion"
                )
                blocked = BlockedClassAssertion(
                    node_term=node_term,
                    target_class=target_class,
                    blocker_class=URIRef(f"urn:dag:blocker:{blocker_suffix}"),
                )
                blocked_assertions.append(blocked)
                if float(node_types[node_idx, target_class_idx].item()) >= 0.999:
                    conflicting_positive_assertions.append(blocked)

    blocked_assertions = list(
        {
            (item.node_term, item.target_class, item.blocker_class): item
            for item in blocked_assertions
        }.values()
    )
    conflicting_positive_assertions = list(
        {
            (item.node_term, item.target_class, item.blocker_class): item
            for item in conflicting_positive_assertions
        }.values()
    )

    blocked_assertions.sort(
        key=lambda item: (str(item.target_class), str(item.node_term), str(item.blocker_class))
    )
    conflicting_positive_assertions.sort(
        key=lambda item: (str(item.target_class), str(item.node_term), str(item.blocker_class))
    )
    return NegativeBlockerResult(
        dataset=dataset,
        blocker_specs=blocker_specs,
        blocked_assertions=blocked_assertions,
        conflicting_positive_assertions=conflicting_positive_assertions,
    )


def collect_assignment_statuses(
    *,
    original_data_graph: Graph,
    positive_result: ClassMaterializationResult,
    negative_result: NegativeBlockerResult,
    target_classes: Optional[Sequence[str | URIRef]] = None,
) -> List[ClassAssignmentStatus]:
    """
    Build explicit assignment-status bookkeeping for the stratified pipeline.

    Status bits:
    - asserted: present in the input ABox
    - positively_derived: added by the positive OWA fixpoint pass
    - blocked: forbidden by the negative blocker pass
    - conflicted: blocked and also present in the positive closure
    """

    target_filter = (
        {URIRef(term) if isinstance(term, str) else term for term in target_classes}
        if target_classes is not None
        else None
    )
    status_by_key: Dict[Tuple[Identifier, URIRef], ClassAssignmentStatus] = {}

    def ensure_status(node_term: Identifier, target_class: URIRef) -> ClassAssignmentStatus:
        if target_filter is not None and target_class not in target_filter:
            raise KeyError("status outside requested target filter")
        key = (node_term, target_class)
        status = status_by_key.get(key)
        if status is None:
            status = ClassAssignmentStatus(node_term=node_term, target_class=target_class)
            status_by_key[key] = status
        return status

    for subj, pred, obj in original_data_graph.triples((None, RDF.type, None)):
        if pred != RDF.type or not isinstance(obj, URIRef):
            continue
        if target_filter is not None and obj not in target_filter:
            continue
        ensure_status(subj, obj).asserted = True

    node_types = positive_result.dataset.kg.node_types.detach().cpu()
    if target_filter is None:
        target_terms = [
            class_term
            for class_term in positive_result.dataset.mapping.class_terms
            if isinstance(class_term, URIRef)
        ]
    else:
        target_terms = sorted(target_filter, key=str)

    for class_term in target_terms:
        class_idx = positive_result.dataset.mapping.class_to_idx.get(class_term)
        if class_idx is None:
            continue
        for node_idx, node_term in enumerate(positive_result.dataset.mapping.node_terms):
            if float(node_types[node_idx, class_idx].item()) < 0.999:
                continue
            status = ensure_status(node_term, class_term)
            if not status.asserted:
                status.positively_derived = True

    for blocked in negative_result.blocked_assertions:
        if target_filter is not None and blocked.target_class not in target_filter:
            continue
        status = ensure_status(blocked.node_term, blocked.target_class)
        status.blocked = True
        if blocked.blocker_class not in status.blocker_classes:
            status.blocker_classes.append(blocked.blocker_class)

    for blocked in negative_result.conflicting_positive_assertions:
        if target_filter is not None and blocked.target_class not in target_filter:
            continue
        ensure_status(blocked.node_term, blocked.target_class).conflicted = True

    for status in status_by_key.values():
        status.blocker_classes.sort(key=str)

    return sorted(
        status_by_key.values(),
        key=lambda status: (str(status.target_class), str(status.node_term)),
    )


def apply_conflict_policy(
    assignment_statuses: Sequence[ClassAssignmentStatus],
    *,
    policy: ConflictPolicy = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED,
) -> ConflictPolicyResult:
    """
    Apply a conflict-handling policy over stratified assignment statuses.

    Policies:
    - report_only: keep all asserted and positively-derived assignments; report blockers/conflicts separately
    - suppress_derived_keep_asserted: suppress blocked derived assignments, but never retract asserted ones
    - strict_fail_on_conflict: mark the run as failed if any blocked assignment is also present in the positive closure
    """

    asserted_conflicts = [
        status
        for status in assignment_statuses
        if status.asserted and status.blocked
    ]
    suppressed_derived_assignments = [
        status
        for status in assignment_statuses
        if status.positively_derived and status.blocked
    ]
    hard_conflicts = [
        status
        for status in assignment_statuses
        if status.conflicted
    ]

    if policy == ConflictPolicy.REPORT_ONLY:
        emitted_assignments = [
            status
            for status in assignment_statuses
            if status.asserted or status.positively_derived
        ]
        failed = False
        failure_reason = None
    elif policy == ConflictPolicy.STRICT_FAIL_ON_CONFLICT:
        emitted_assignments = [
            status
            for status in assignment_statuses
            if status.asserted or status.positively_derived
        ]
        failed = bool(hard_conflicts)
        failure_reason = (
            f"{len(hard_conflicts)} conflicting assignment(s) present in the positive closure."
            if failed
            else None
        )
    else:
        emitted_assignments = [
            status
            for status in assignment_statuses
            if status.asserted or (status.positively_derived and not status.blocked)
        ]
        failed = False
        failure_reason = None

    emitted_assignments = sorted(
        emitted_assignments,
        key=lambda status: (str(status.target_class), str(status.node_term)),
    )
    emitted_derived_assertions = sorted(
        [
            (status.node_term, status.target_class)
            for status in emitted_assignments
            if status.positively_derived
        ],
        key=lambda pair: (str(pair[1]), str(pair[0])),
    )

    return ConflictPolicyResult(
        policy=policy,
        emitted_assignments=emitted_assignments,
        emitted_derived_assertions=emitted_derived_assertions,
        suppressed_derived_assignments=sorted(
            suppressed_derived_assignments,
            key=lambda status: (str(status.target_class), str(status.node_term)),
        ),
        asserted_conflicts=sorted(
            asserted_conflicts,
            key=lambda status: (str(status.target_class), str(status.node_term)),
        ),
        hard_conflicts=sorted(
            hard_conflicts,
            key=lambda status: (str(status.target_class), str(status.node_term)),
        ),
        failed=failed,
        failure_reason=failure_reason,
    )


def materialize_stratified_class_inferences(
    *,
    schema_graph: Graph,
    data_graph: Graph,
    include_literals: bool = True,
    include_type_edges: bool = False,
    materialize_hierarchy: bool = True,
    materialize_sameas: Optional[bool] = None,
    materialize_haskey_equality: Optional[bool] = None,
    materialize_target_roles: bool = False,
    target_classes: Optional[Sequence[str | URIRef]] = None,
    threshold: float = 0.999,
    max_iterations: int = 10,
    device: str = "cpu",
    conflict_policy: ConflictPolicy = ConflictPolicy.SUPPRESS_DERIVED_KEEP_ASSERTED,
) -> StratifiedMaterializationResult:
    total_t0 = perf_counter()
    positive_result = materialize_positive_sufficient_class_inferences(
        schema_graph=schema_graph,
        data_graph=data_graph,
        include_literals=include_literals,
        include_type_edges=include_type_edges,
        materialize_hierarchy=materialize_hierarchy,
        materialize_sameas=materialize_sameas,
        materialize_haskey_equality=materialize_haskey_equality,
        materialize_target_roles=materialize_target_roles,
        target_classes=target_classes,
        threshold=threshold,
        max_iterations=max_iterations,
        device=device,
    )
    negative_t0 = perf_counter()
    negative_result = materialize_negative_class_blockers(
        dataset=positive_result.dataset,
        target_classes=target_classes,
    )
    negative_elapsed_ms = (perf_counter() - negative_t0) * 1000.0
    assignment_t0 = perf_counter()
    assignment_statuses = collect_assignment_statuses(
        original_data_graph=data_graph,
        positive_result=positive_result,
        negative_result=negative_result,
        target_classes=target_classes,
    )
    assignment_elapsed_ms = (perf_counter() - assignment_t0) * 1000.0
    policy_t0 = perf_counter()
    policy_result = apply_conflict_policy(
        assignment_statuses,
        policy=conflict_policy,
    )
    policy_elapsed_ms = (perf_counter() - policy_t0) * 1000.0
    timings = StratifiedMaterializationTimings(
        positive_timings=positive_result.timings or PositiveMaterializationTimings(iterations=positive_result.iterations),
        negative_blocker_elapsed_ms=negative_elapsed_ms,
        assignment_status_elapsed_ms=assignment_elapsed_ms,
        conflict_policy_elapsed_ms=policy_elapsed_ms,
        total_elapsed_ms=(perf_counter() - total_t0) * 1000.0,
    )
    return StratifiedMaterializationResult(
        positive_result=positive_result,
        negative_result=negative_result,
        assignment_statuses=assignment_statuses,
        policy_result=policy_result,
        timings=timings,
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
