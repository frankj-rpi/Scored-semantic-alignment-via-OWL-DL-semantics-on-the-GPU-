from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from rdflib import Graph, URIRef
from rdflib.term import Identifier

from .ontology_parse import load_rdflib_graph
from .oracle_compare import (
    EngineQueryResult,
    apply_engine_profile,
    normalize_engine_mode_name,
    normalize_engine_profile_name,
    resolve_super_dag_mode,
    resolve_target_classes,
    run_engine_queries,
)


@dataclass
class TensorKGResult:
    engine_result: EngineQueryResult
    target_classes: List[URIRef]

    @property
    def members_by_target(self) -> Dict[URIRef, Set[Identifier]]:
        return self.engine_result.members_by_target

    @property
    def scores_by_target(self) -> Dict[URIRef, Dict[Identifier, float]]:
        return self.engine_result.scores_by_target or {}

    @property
    def dataset(self):
        return self.engine_result.dataset

    def members_for(self, target_class: str | URIRef) -> Set[Identifier]:
        target_term = URIRef(str(target_class))
        return self.members_by_target.get(target_term, set())

    def scores_for(self, target_class: str | URIRef) -> Dict[Identifier, float]:
        target_term = URIRef(str(target_class))
        return self.scores_by_target.get(target_term, {})


class TensorKG:
    """
    Small prototype API for embedding TensorKG inside another Python project.

    This wrapper intentionally stays close to the existing engine behavior:
    users provide RDFLib graphs or file paths, choose a profile and mode, and
    then call `run()` or one of the mode-specific convenience methods.
    """

    def __init__(
        self,
        *,
        profile: str = "gpu-el",
        mode: str = "stratified",
        device: str = "cpu",
        threshold: float = 0.999,
        include_literals: Optional[bool] = None,
        include_type_edges: bool = False,
    ) -> None:
        self.profile = normalize_engine_profile_name(profile)
        self.mode = normalize_engine_mode_name(mode)
        self.device = device
        self.threshold = threshold
        self.include_literals = include_literals
        self.include_type_edges = include_type_edges
        self.schema_graph: Optional[Graph] = None
        self.data_graph: Optional[Graph] = None
        self.target_specs: List[str] = []

    def load_graphs(self, *, schema_graph: Graph, data_graph: Graph) -> "TensorKG":
        self.schema_graph = schema_graph
        self.data_graph = data_graph
        return self

    def load_files(
        self,
        *,
        schema_paths: str | Sequence[str],
        data_paths: str | Sequence[str],
    ) -> "TensorKG":
        schema_list = [schema_paths] if isinstance(schema_paths, str) else list(schema_paths)
        data_list = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        self.schema_graph = load_rdflib_graph(schema_list)
        self.data_graph = load_rdflib_graph(data_list)
        return self

    def set_targets(self, targets: Iterable[str | URIRef]) -> "TensorKG":
        self.target_specs = [str(target) for target in targets]
        return self

    def clear_targets(self) -> "TensorKG":
        self.target_specs = []
        return self

    def resolve_targets(
        self,
        targets: Optional[Iterable[str | URIRef]] = None,
        *,
        mode: Optional[str] = None,
    ) -> List[URIRef]:
        schema_graph, data_graph = self._require_graphs()
        effective_mode = self.mode if mode is None else normalize_engine_mode_name(mode)
        target_specs = self._normalize_target_specs(targets)
        profile_options = apply_engine_profile(
            profile=self.profile,
            materialize_hierarchy=None,
            materialize_horn_safe_domain_range=None,
            materialize_reflexive_properties=None,
            materialize_sameas=None,
            materialize_haskey_equality=None,
            materialize_target_roles=None,
            augment_property_domain_range=None,
            enable_negative_verification=None,
        )
        resolution = resolve_target_classes(
            schema_graph=schema_graph,
            data_graph=data_graph,
            target_class_specs=target_specs,
            engine_mode=effective_mode,
            include_literals=self._effective_include_literals(),
            include_type_edges=self.include_type_edges,
            materialize_hierarchy=profile_options.materialize_hierarchy,
            augment_property_domain_range=profile_options.augment_property_domain_range,
        )
        if resolution.skipped_targets:
            skipped_text = "; ".join(
                f"{target}: {reason}" for target, reason in resolution.skipped_targets
            )
            raise ValueError(f"Some targets could not be compiled in mode {effective_mode}: {skipped_text}")
        return resolution.resolved_targets

    def run(
        self,
        *,
        targets: Optional[Iterable[str | URIRef]] = None,
        mode: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> TensorKGResult:
        schema_graph, data_graph = self._require_graphs()
        effective_profile = self.profile if profile is None else normalize_engine_profile_name(profile)
        effective_mode = self.mode if mode is None else normalize_engine_mode_name(mode)
        target_terms = self.resolve_targets(targets, mode=effective_mode)
        profile_options = apply_engine_profile(
            profile=effective_profile,
            materialize_hierarchy=None,
            materialize_horn_safe_domain_range=None,
            materialize_reflexive_properties=None,
            materialize_sameas=None,
            materialize_haskey_equality=None,
            materialize_target_roles=None,
            augment_property_domain_range=None,
            enable_negative_verification=None,
        )
        super_dag = resolve_super_dag_mode("auto", effective_profile)
        engine_result = run_engine_queries(
            schema_graph=schema_graph,
            data_graph=data_graph,
            target_classes=target_terms,
            device=self.device,
            threshold=self.threshold,
            include_literals=self._effective_include_literals(effective_profile),
            include_type_edges=self.include_type_edges,
            materialize_hierarchy=profile_options.materialize_hierarchy,
            materialize_horn_safe_domain_range=profile_options.materialize_horn_safe_domain_range,
            materialize_sameas=profile_options.materialize_sameas,
            materialize_haskey_equality=profile_options.materialize_haskey_equality,
            materialize_reflexive_properties=profile_options.materialize_reflexive_properties,
            materialize_target_roles=profile_options.materialize_target_roles,
            materialize_supported_types=False,
            augment_property_domain_range=profile_options.augment_property_domain_range,
            native_sameas_canonicalization=profile_options.native_sameas_canonicalization,
            engine_mode=effective_mode,
            conflict_policy="suppress_derived_keep_asserted",
            enable_negative_verification=profile_options.enable_negative_verification,
            enable_negative_materialization=profile_options.enable_negative_materialization,
            enable_super_dag=(super_dag == "on"),
        )
        return TensorKGResult(engine_result=engine_result, target_classes=target_terms)

    def materialize(self, *, targets: Optional[Iterable[str | URIRef]] = None) -> TensorKGResult:
        return self.run(targets=targets, mode="stratified")

    def admissibility(self, *, targets: Optional[Iterable[str | URIRef]] = None) -> TensorKGResult:
        return self.run(targets=targets, mode="admissibility")

    def filtered_admissibility(self, *, targets: Optional[Iterable[str | URIRef]] = None) -> TensorKGResult:
        return self.run(targets=targets, mode="filtered_admissibility")

    def score(self, *, targets: Optional[Iterable[str | URIRef]] = None) -> TensorKGResult:
        return self.run(targets=targets, mode="scored_semantic_alignment")

    def _effective_include_literals(self, profile: Optional[str] = None) -> bool:
        if self.include_literals is not None:
            return self.include_literals
        return (profile or self.profile) == "gpu-dl"

    def _normalize_target_specs(
        self,
        targets: Optional[Iterable[str | URIRef]],
    ) -> List[str]:
        if targets is not None:
            normalized = [str(target) for target in targets]
            if normalized:
                return normalized
        if self.target_specs:
            return list(self.target_specs)
        return ["all-named-classes"]

    def _require_graphs(self) -> tuple[Graph, Graph]:
        if self.schema_graph is None or self.data_graph is None:
            raise RuntimeError("No graphs loaded. Call load_graphs(...) or load_files(...) first.")
        return self.schema_graph, self.data_graph
