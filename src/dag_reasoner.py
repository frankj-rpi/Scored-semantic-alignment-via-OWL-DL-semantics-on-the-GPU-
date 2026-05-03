from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import torch

from .graph import KGraph, prepare_kgraph_for_device
from .constraints import ConstraintDAG
from .dag_eval import eval_dag_score_matrix, eval_dag_scores


@dataclass
class ConceptSpec:
    """Container for a single concept (name + DAG)."""
    name: str
    dag: ConstraintDAG


class DAGReasoner:
    """
    Simple wrapper around the DAG evaluation engine.

    Responsibilities:
      - Owns a KGraph and optional similarity config.
      - Registers multiple concepts (each with a ConstraintDAG).
      - Evaluates all concepts over all nodes.
      - Provides helpers to inspect top-k scores.
    """

    def __init__(
        self,
        graph: KGraph,
        device: str = "cpu",
        sim_class: Optional[torch.Tensor] = None,
    ) -> None:
        self.graph = graph
        self.device = device
        self.sim_class = sim_class
        self.eval_graph = prepare_kgraph_for_device(graph, device)

        self._concepts: List[ConceptSpec] = []
        self._last_scores: Optional[torch.Tensor] = None  # [num_nodes, num_concepts]

    @property
    def concept_names(self) -> List[str]:
        return [c.name for c in self._concepts]

    def add_concept(self, name: str, dag: ConstraintDAG) -> None:
        """Register a new concept with its constraint DAG."""
        self._concepts.append(ConceptSpec(name=name, dag=dag))
        # Invalidate cached scores
        self._last_scores = None

    def clear_concepts(self) -> None:
        self._concepts.clear()
        self._last_scores = None

    def update_node_types(self, node_types: torch.Tensor) -> None:
        self.graph.node_types = node_types
        self.eval_graph.node_types = node_types.to(self.device)
        self._last_scores = None

    def apply_type_updates(
        self,
        added_node_indices: torch.Tensor,
        added_class_indices: torch.Tensor,
    ) -> None:
        if added_node_indices.numel() == 0 or added_class_indices.numel() == 0:
            return
        cpu_node_indices = added_node_indices.detach().to("cpu")
        cpu_class_indices = added_class_indices.detach().to("cpu")
        self.graph.node_types[cpu_node_indices, cpu_class_indices] = 1.0
        self.eval_graph.node_types[added_node_indices, added_class_indices] = 1.0
        self._last_scores = None

    def evaluate_all(self) -> torch.Tensor:
        """
        Evaluate all registered concepts over all nodes.

        Returns:
            scores: tensor of shape [num_nodes, num_concepts],
                scores[v, i] = s(v, concept_i).
        """
        if not self._concepts:
            raise ValueError("No concepts registered in DAGReasoner.")

        num_nodes = self.graph.num_nodes
        num_concepts = len(self._concepts)

        scores = torch.zeros((num_nodes, num_concepts), device=self.device)

        for i, spec in enumerate(self._concepts):
            s = eval_dag_scores(
                self.eval_graph,
                spec.dag,
                device=self.device,
                sim_class=self.sim_class,
            )
            scores[:, i] = s

        self._last_scores = scores
        return scores

    def evaluate_merged_roots(
        self,
        dag: ConstraintDAG,
        root_indices: List[int] | Tuple[int, ...],
    ) -> torch.Tensor:
        score_matrix = eval_dag_score_matrix(
            self.eval_graph,
            dag,
            device=self.device,
            sim_class=self.sim_class,
        )
        roots = torch.as_tensor(list(root_indices), device=score_matrix.device, dtype=torch.long)
        return score_matrix[:, roots]

    def _ensure_scores(self) -> torch.Tensor:
        if self._last_scores is None:
            return self.evaluate_all()
        return self._last_scores

    def top_k_for_concept(self, name: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Return top-k (node_id, score) for a given concept name.
        """
        if name not in self.concept_names:
            raise ValueError(f"Unknown concept name: {name}")

        scores = self._ensure_scores()  # [num_nodes, num_concepts]
        idx = self.concept_names.index(name)

        concept_scores = scores[:, idx]  # [num_nodes]
        values, indices = torch.topk(concept_scores, k)
        return [(int(i), float(v)) for i, v in zip(indices, values)]

    def top_k_for_node(self, node_id: int, k: int = 5) -> List[Tuple[str, float]]:
        """
        For a given node, return top-k (concept_name, score).
        """
        scores = self._ensure_scores()
        num_nodes, num_concepts = scores.shape

        if node_id < 0 or node_id >= num_nodes:
            raise ValueError(f"node_id {node_id} out of range [0, {num_nodes}).")

        node_scores = scores[node_id, :]  # [num_concepts]
        values, indices = torch.topk(node_scores, min(k, num_concepts))
        return [(self.concept_names[int(i)], float(v)) for i, v in zip(indices, values)]

    def satisfying_nodes(self, concept_name: str, threshold: float = 0.999):
        scores = self._ensure_scores()
        idx = self.concept_names.index(concept_name)
        mask = scores[:, idx] >= threshold
        return mask.nonzero(as_tuple=False).view(-1).tolist()
