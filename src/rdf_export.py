from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from .graph import KGraph



@dataclass
class KGraphMapping:
    """
    Keeps a stable mapping between KGraph indices and RDF IRIs.

    - node_iris[i]  is the IRI for node i
    - prop_iris[p]  is the IRI for property p
    - class_iris[c] is the IRI for class c
    """
    node_iris: List[URIRef]
    prop_iris: List[URIRef]
    class_iris: List[URIRef]
    base_ns: Namespace


def default_mapping_for_kgraph(
    kg: KGraph,
    base_uri: str = "http://example.org/",
    node_prefix: str = "n",
    prop_prefix: str = "p",
    class_prefix: str = "C",
) -> KGraphMapping:
    """
    Create a default mapping:

      node i  →  ex:n{i}
      prop p  →  ex:p{p}
      class c →  ex:C{c}

    where ex = Namespace(base_uri + "#").
    """
    ex = Namespace(base_uri.rstrip("/") + "#")

    num_nodes = kg.num_nodes
    # node_types: [num_nodes, num_classes]
    num_classes = kg.node_types.shape[1] if kg.node_types.dim() == 2 else 0
    num_props = len(kg.offsets_p)

    node_iris = [ex[f"{node_prefix}{i}"] for i in range(num_nodes)]
    prop_iris = [ex[f"{prop_prefix}{p}"] for p in range(num_props)]
    class_iris = [ex[f"{class_prefix}{c}"] for c in range(num_classes)]

    return KGraphMapping(
        node_iris=node_iris,
        prop_iris=prop_iris,
        class_iris=class_iris,
        base_ns=ex,
    )


def kgraph_to_rdflib_graph(
    kg: KGraph,
    mapping: Optional[KGraphMapping] = None,
    base_uri: str = "http://example.org/",
    type_threshold: float = 0.5,
) -> Tuple[Graph, KGraphMapping]:
    """
    Convert a KGraph into an rdflib.Graph using the given mapping.

    If 'mapping' is None, a default mapping is created:

      - node i  -> ex:n{i}
      - prop p  -> ex:p{p}
      - class c -> ex:C{c}

    Triples generated:

      - For each property p and edge (i -> j) in kg.offsets_p[p], kg.neighbors_p[p]:
          ex:ni ex:pp ex:nj .

      - For each node i and each class c where kg.node_types[i, c] > type_threshold:
          ex:ni rdf:type ex:Cc .
    """
    if mapping is None:
        mapping = default_mapping_for_kgraph(kg, base_uri=base_uri)

    g = Graph()
    ex = mapping.base_ns

    # Bind prefix for nicer Turtle output (e.g., ex:n0 instead of full URI)
    g.bind("ex", ex)
    g.bind("rdf", RDF)

    num_nodes = kg.num_nodes
    node_types = kg.node_types

    # 1. Add property edges
    for p, (offsets, neighbors) in enumerate(zip(kg.offsets_p, kg.neighbors_p)):
        prop_iri = mapping.prop_iris[p]

        # Ensure tensors are on CPU for iteration
        offsets_cpu = offsets.cpu()
        neighbors_cpu = neighbors.cpu()

        for i in range(num_nodes):
            start = int(offsets_cpu[i].item())
            end = int(offsets_cpu[i + 1].item())
            if start == end:
                continue

            subj = mapping.node_iris[i]
            for e in range(start, end):
                j = int(neighbors_cpu[e].item())
                obj = mapping.node_iris[j]
                g.add((subj, prop_iri, obj))

    # 2. Add rdf:type triples based on node_types
    #    We consider node_types[i, c] > type_threshold as "has type Cc".
    node_types_cpu = node_types.cpu()

    num_classes = node_types_cpu.shape[1] if node_types_cpu.dim() == 2 else 0
    for i in range(num_nodes):
        subj = mapping.node_iris[i]
        if num_classes == 0:
            continue

        # Find all classes with sufficient membership
        class_memberships = node_types_cpu[i]  # [num_classes]
        class_indices = (class_memberships > type_threshold).nonzero(as_tuple=False).view(-1)

        for c in class_indices.tolist():
            class_iri = mapping.class_iris[c]
            g.add((subj, RDF.type, class_iri))

    return g, mapping


def write_kgraph_as_turtle(
    kg: KGraph,
    path: str,
    mapping: Optional[KGraphMapping] = None,
    base_uri: str = "http://example.org/",
    type_threshold: float = 0.5,
) -> KGraphMapping:
    """
    Convenience helper:

      - Convert KGraph -> rdflib.Graph
      - Serialize as Turtle to 'path'
      - Return the mapping used
    """
    g, mapping = kgraph_to_rdflib_graph(
        kg,
        mapping=mapping,
        base_uri=base_uri,
        type_threshold=type_threshold,
    )
    g.serialize(destination=path, format="turtle")
    return mapping
