from __future__ import annotations

from typing import Set, List

import torch
from rdflib import URIRef

from .graph import KGraph
from .dag_reasoner import DAGReasoner
from .random_graphs import generate_random_kgraph, generate_random_path_pattern, PathPattern
from .patterns import compile_path_pattern_to_dag
from .rdf_export import write_kgraph_as_turtle, kgraph_to_rdflib_graph, default_mapping_for_kgraph, KGraphMapping
from .sparql_patterns import path_pattern_to_sparql

from .fuseki_client import fuseki_clear_dataset, fuseki_upload_turtle, fuseki_sparql_query

import requests
import time
from datetime import datetime
import csv
import os
import platform
from typing import Set, List, Dict, Any

CSV_FIELDS = [
    "timestamp",
    "cpu_name",
    "gpu_name",
    "device",

    "num_nodes",
    "num_props",
    "num_classes",
    "avg_degree_per_prop",
    "num_steps",
    "num_edges",

    "backend",
    "engine_time_ms",
    "rdf_build_time_ms",
    "sparql_time_ms",
    "ttl_export_time_ms",
    "fuseki_clear_time_ms",
    "fuseki_upload_time_ms",
    "agreement",
]

def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def now_timestamp() -> str:
    """
    Returns a filename-safe UTC timestamp.
    Format: YYYY-MM-DD_HH-MM-SS   (e.g., 2025-02-09_05-14-33)
    """
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def get_cpu_name() -> str:
    """
    Try to get a human-readable CPU name on Linux, macOS, and Windows.
    """
    # 1) Linux (most reliable)
    if os.path.isfile("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    # 2) Generic fallback
    cpu = platform.processor()
    if cpu:
        return cpu

    # 3) Fallback to uname
    return platform.uname().processor or "Unknown CPU"


def get_gpu_name(device: str = "cuda") -> str:
    """
    Returns GPU name if CUDA is available, otherwise 'None'.
    """
    if device == "cuda" and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "Unknown CUDA GPU"
    return "None"

def check_fuseki_alive(dataset_url: str) -> bool:
    """
    Returns True if the Fuseki dataset is reachable and can answer a trivial SPARQL query.

    Tries POSTing a small ASK {} query to:
      1) {dataset_url}/query
      2) {dataset_url}
    in that order.

    Example dataset_url: "http://localhost:3030/ds"
    """

    ask_query = "ASK {}"
    endpoints = [
        dataset_url.rstrip("/") + "/query",
        dataset_url.rstrip("/"),
    ]

    for ep in endpoints:
        try:
            r = requests.post(
                ep,
                data={"query": ask_query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=3.0,
            )
            if r.status_code == 200:
                return True
        except Exception:
            # Try the next endpoint
            continue

    return False

def run_sparql_remote(endpoint_url: str, query: str):
    r = requests.post(
        endpoint_url,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"},
    )
    r.raise_for_status()
    data = r.json()
    return [binding["v"]["value"] for binding in data["results"]["bindings"]]


def _iri_to_node_index(iri: URIRef, mapping: KGraphMapping) -> int:
    """
    Given an IRI (e.g. ex:n5) and a KGraphMapping, return the corresponding node index.

    We assume the mapping.node_iris list is exactly the IRIs we used during export.
    For efficiency, you might later cache a dict, but for small graphs this is fine.
    """
    try:
        return mapping.node_iris.index(iri)
    except ValueError:
        raise KeyError(f"IRI {iri} not found in node mapping.")

def compare_pattern_once(
    kg: KGraph,
    pattern: PathPattern,
    device: str = "cpu",
    threshold: float = 0.999,
) -> Dict[str, Any]:
    """
    Compare the result of your GPU-based reasoner vs rdflib SPARQL
    for a single KGraph + PathPattern pair, and return timing metrics.
    """

    # -------------------------
    # 1) DAG + reasoner (exact)
    # -------------------------
    dag = compile_path_pattern_to_dag(pattern)

    num_classes = kg.node_types.shape[1]
    sim_class = torch.eye(num_classes)

    reasoner = DAGReasoner(kg, device=device, sim_class=sim_class)
    reasoner.add_concept("Pattern", dag)

    t0 = time.perf_counter()
    scores = reasoner.evaluate_all()  # [num_nodes, 1]
    t1 = time.perf_counter()
    engine_time_ms = (t1 - t0) * 1000.0

    engine_nodes: List[int] = reasoner.satisfying_nodes("Pattern", threshold=threshold)
    engine_set: Set[int] = set(engine_nodes)

    #print("Engine scores for Pattern:")
    for v in range(kg.num_nodes):
        print(f"  v = {v}: s(v, Pattern) = {scores[v, 0].item():.4f}")
    #print("Engine satisfying nodes (score >= threshold):", sorted(engine_set))
    print(f"Engine evaluation time: {engine_time_ms:.3f} ms")

    # -------------------------
    # 2) RDF export + SPARQL (rdflib)
    # -------------------------
    mapping = default_mapping_for_kgraph(kg, base_uri="http://example.org/random")

    t2 = time.perf_counter()
    rdf_graph, mapping = kgraph_to_rdflib_graph(
        kg,
        mapping=mapping,
        base_uri="http://example.org/random",
        type_threshold=0.5,
    )
    t3 = time.perf_counter()
    rdf_build_time_ms = (t3 - t2) * 1000.0
    print(f"rdflib Graph build time: {rdf_build_time_ms:.3f} ms")

    sparql_query = path_pattern_to_sparql(pattern, mapping, start_var="v")
    print("\nSPARQL query used (rdflib):")
    print(sparql_query)

    t4 = time.perf_counter()
    results = rdf_graph.query(sparql_query)
    result_rows = list(results)  # force evaluation
    t5 = time.perf_counter()
    sparql_time_ms = (t5 - t4) * 1000.0
    print(f"rdflib SPARQL time: {sparql_time_ms:.3f} ms")

    sparql_node_indices: Set[int] = set()
    for row in result_rows:
        v_iri = row[0]
        if not isinstance(v_iri, URIRef):
            continue
        idx = mapping.node_iris.index(v_iri)
        sparql_node_indices.add(idx)

    print("rdflib satisfying nodes:", sorted(sparql_node_indices))

    # -------------------------
    # 3) Compare and report
    # -------------------------
    only_in_engine = engine_set - sparql_node_indices
    only_in_sparql = sparql_node_indices - engine_set
    intersection = engine_set & sparql_node_indices

    print("\nComparison (rdflib):")
    print("  Intersection:", sorted(intersection))
    print("  Only in engine:", sorted(only_in_engine))
    print("  Only in SPARQL:", sorted(only_in_sparql))

    agree = (not only_in_engine and not only_in_sparql)
    if agree:
        print("\n✅ Exact match between engine and rdflib for this pattern.")
    else:
        print("\n⚠️ Mismatch detected (engine vs rdflib).")

    return {
        "backend": "rdflib",
        "engine_time_ms": engine_time_ms,
        "rdf_build_time_ms": rdf_build_time_ms,
        "sparql_time_ms": sparql_time_ms,
        "ttl_export_time_ms": None,
        "fuseki_clear_time_ms": None,
        "fuseki_upload_time_ms": None,
        "agreement": agree,
    }

def compare_engine_only_once(
    kg: KGraph,
    pattern: PathPattern,
    device: str = "cpu",
    threshold: float = 0.999,
) -> Dict[str, Any]:
    """
    Measure only the DAGReasoner / engine runtime, without any RDF/SPARQL work.
    Returns a metrics dict shaped like the others, with non-engine fields set to None.
    """
    dag = compile_path_pattern_to_dag(pattern)

    num_classes = kg.node_types.shape[1]
    sim_class = torch.eye(num_classes)

    reasoner = DAGReasoner(kg, device=device, sim_class=sim_class)
    reasoner.add_concept("Pattern", dag)

    t0 = time.perf_counter()
    scores = reasoner.evaluate_all()  # [num_nodes, 1]
    t1 = time.perf_counter()
    engine_time_ms = (t1 - t0) * 1000.0

    engine_nodes: List[int] = reasoner.satisfying_nodes("Pattern", threshold=threshold)
    engine_set: Set[int] = set(engine_nodes)

    print("Engine-only scores for Pattern:")
    #for v in range(kg.num_nodes):
        # print(f"  v = {v}: s(v, Pattern) = {scores[v, 0].item():.4f}")
    print("Engine-only satisfying nodes (score >= threshold):", sorted(engine_set))
    print(f"Engine-only evaluation time: {engine_time_ms:.3f} ms")

    return {
        "backend": "engine_only",
        "engine_time_ms": engine_time_ms,
        "rdf_build_time_ms": None,
        "sparql_time_ms": None,
        "ttl_export_time_ms": None,
        "fuseki_clear_time_ms": None,
        "fuseki_upload_time_ms": None,
        # trivially "agree" because there's nothing to compare to
        "agreement": True,
    }



def compare_with_fuseki_once(
    kg: KGraph,
    pattern: PathPattern,
    dataset_url: str = "http://localhost:3030/ds",
    ttl_path: str = "random_graph.ttl",
    device: str = "cpu",
    threshold: float = 0.999,
) -> Dict[str, Any]:
    """
    Compare your GPU-based reasoner vs a remote Fuseki instance and return timing metrics.
    """

    # 1) Engine side
    dag = compile_path_pattern_to_dag(pattern)

    num_classes = kg.node_types.shape[1]
    sim_class = torch.eye(num_classes)

    reasoner = DAGReasoner(kg, device=device, sim_class=sim_class)
    reasoner.add_concept("Pattern", dag)

    t0 = time.perf_counter()
    scores = reasoner.evaluate_all()
    t1 = time.perf_counter()
    engine_time_ms = (t1 - t0) * 1000.0

    engine_nodes = reasoner.satisfying_nodes("Pattern", threshold=threshold)
    engine_set: Set[int] = set(engine_nodes)

    print("Engine scores for Pattern:")
    for v in range(kg.num_nodes):
        print(f"  v = {v}: s(v, Pattern) = {scores[v, 0].item():.4f}")
    print("Engine satisfying nodes (score >= threshold):", sorted(engine_set))
    print(f"Engine evaluation time: {engine_time_ms:.3f} ms")

    # 2) Export RDF + mapping
    print("\nExporting KGraph as Turtle...")
    t2 = time.perf_counter()
    mapping = write_kgraph_as_turtle(
        kg,
        path=ttl_path,
        base_uri="http://example.org/random",
        type_threshold=0.5,
    )
    t3 = time.perf_counter()
    ttl_export_time_ms = (t3 - t2) * 1000.0
    print(f"  Turtle export time: {ttl_export_time_ms:.3f} ms (file: {ttl_path})")

    # 3) Clear & upload to Fuseki
    print(f"\nClearing Fuseki dataset at {dataset_url} ...")
    t4 = time.perf_counter()
    fuseki_clear_dataset(dataset_url)
    t5 = time.perf_counter()
    clear_time_ms = (t5 - t4) * 1000.0
    print(f"  Clear time: {clear_time_ms:.3f} ms")

    print(f"\nUploading Turtle to Fuseki dataset at {dataset_url} ...")
    t6 = time.perf_counter()
    fuseki_upload_turtle(dataset_url, ttl_path)
    t7 = time.perf_counter()
    upload_time_ms = (t7 - t6) * 1000.0
    print(f"  Upload time: {upload_time_ms:.3f} ms")

    # 4) Compile SPARQL
    print("\nCompiling PathPattern to SPARQL...")
    sparql_query = path_pattern_to_sparql(pattern, mapping, start_var="v")
    print("\nSPARQL query (Fuseki):")
    print(sparql_query)

    # 5) Run SPARQL on Fuseki
    print("\nRunning SPARQL on Fuseki...")
    t8 = time.perf_counter()
    iri_values = fuseki_sparql_query(dataset_url, sparql_query)
    t9 = time.perf_counter()
    fuseki_sparql_time_ms = (t9 - t8) * 1000.0
    print(f"Fuseki SPARQL time: {fuseki_sparql_time_ms:.3f} ms")

    print("SPARQL returned IRIs:")
    for v in iri_values:
        print(" ", v)

    # 6) Map IRIs back to node indices
    iri_to_index: dict[str, int] = {
        str(iri): idx for idx, iri in enumerate(mapping.node_iris)
    }
    sparql_indices: Set[int] = set()
    for iri in iri_values:
        if iri in iri_to_index:
            sparql_indices.add(iri_to_index[iri])
        else:
            print(f"  Warning: IRI {iri} not found in mapping; ignoring.")

    print("Fuseki satisfying node indices:", sorted(sparql_indices))

    # 7) Compare
    only_in_engine = engine_set - sparql_indices
    only_in_sparql = sparql_indices - engine_set
    intersection = engine_set & sparql_indices

    print("\nComparison (Fuseki):")
    print("  Intersection:", sorted(intersection))
    print("  Only in engine:", sorted(only_in_engine))
    print("  Only in SPARQL:", sorted(only_in_sparql))

    agree = (not only_in_engine and not only_in_sparql)
    if agree:
        print("\n✅ Engine and Fuseki agree for this pattern.")
    else:
        print("\n⚠️ Mismatch between engine and Fuseki; inspect differences above.")

    return {
        "backend": "fuseki",
        "engine_time_ms": engine_time_ms,
        "rdf_build_time_ms": None,
        "sparql_time_ms": fuseki_sparql_time_ms,
        "ttl_export_time_ms": ttl_export_time_ms,
        "fuseki_clear_time_ms": clear_time_ms,
        "fuseki_upload_time_ms": upload_time_ms,
        "agreement": agree,
    }

import time
from typing import Dict, Any, Set, List
from rdflib import URIRef
import torch

from .rdf_export import kgraph_to_rdflib_graph, default_mapping_for_kgraph, KGraphMapping
from .sparql_patterns import path_pattern_to_sparql_partial_score
#from .dag_eval import DAGReasoner, compile_path_pattern_to_dag
from .graph import KGraph
from .random_graphs import PathPattern
# plus CSV_FIELDS, append_csv_row, etc., already present


def compare_partial_pattern_once(
    kg: KGraph,
    pattern: PathPattern,
    device: str = "cpu",
    threshold: float = 0.999,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compare GPU engine's partial path scores vs rdflib SPARQL scores.

    - Engine: uses DAGReasoner on the given PathPattern (with path normalization).
    - rdflib: uses path_pattern_to_sparql_partial_score and reads ?score.

    Checks:
      * Node sets with score >= threshold match
      * Max absolute score difference <= tol
    """

    # 1) Engine side
    dag = compile_path_pattern_to_dag(pattern)
    num_classes = kg.node_types.shape[1]
    sim_class = torch.eye(num_classes)

    reasoner = DAGReasoner(kg, device=device, sim_class=sim_class)
    reasoner.add_concept("Pattern", dag)

    t0 = time.perf_counter()
    scores = reasoner.evaluate_all()  # [num_nodes, 1]
    t1 = time.perf_counter()
    engine_time_ms = (t1 - t0) * 1000.0

    engine_scores = scores[:, 0].cpu().numpy()
    engine_set: Set[int] = {v for v, s in enumerate(engine_scores) if s >= threshold}

    print("Engine scores for Pattern (partial):")
    for v in range(kg.num_nodes):
        print(f"  v = {v}: s(v, Pattern) = {engine_scores[v]:.4f}")
    print("Engine satisfying nodes (score >= threshold):", sorted(engine_set))
    print(f"Engine evaluation time: {engine_time_ms:.3f} ms")

    # 2) rdflib side
    mapping = default_mapping_for_kgraph(kg, base_uri="http://example.org/random")

    t2 = time.perf_counter()
    rdf_graph, mapping = kgraph_to_rdflib_graph(
        kg,
        mapping=mapping,
        base_uri="http://example.org/random",
        type_threshold=0.5,
    )
    t3 = time.perf_counter()
    rdf_build_time_ms = (t3 - t2) * 1000.0
    print(f"rdflib Graph build time: {rdf_build_time_ms:.3f} ms")

    sparql_query = path_pattern_to_sparql_partial_score(pattern, mapping, start_var="v")
    print("\nSPARQL query used (rdflib, partial score):")
    print(sparql_query)

    t4 = time.perf_counter()
    results = rdf_graph.query(sparql_query)
    rows = list(results)  # materialize
    t5 = time.perf_counter()
    sparql_time_ms = (t5 - t4) * 1000.0
    print(f"rdflib SPARQL time (partial): {sparql_time_ms:.3f} ms")

    # Build SPARQL score array for all nodes, default 0.0
    sparql_scores = [0.0] * kg.num_nodes
    iri_to_index = {iri: idx for idx, iri in enumerate(mapping.node_iris)}

    for row in rows:
        v_iri = row["v"]
        score_lit = row["score"]
        if isinstance(v_iri, URIRef):
            idx = iri_to_index.get(v_iri)
        else:
            idx = iri_to_index.get(URIRef(str(v_iri)))
        if idx is None:
            continue
        sparql_scores[idx] = float(score_lit)

    print("rdflib scores for Pattern (partial):")
    for v in range(kg.num_nodes):
        print(f"  v = {v}: s_sparql(v, Pattern) = {sparql_scores[v]:.4f}")

    sparql_set: Set[int] = {v for v, s in enumerate(sparql_scores) if s >= threshold}

    # 3) Compare sets and numeric scores
    only_in_engine = engine_set - sparql_set
    only_in_sparql = sparql_set - engine_set
    intersection = engine_set & sparql_set

    print("\nSet comparison (rdflib, partial):")
    print("  Intersection (score >= threshold):", sorted(intersection))
    print("  Only in engine:", sorted(only_in_engine))
    print("  Only in SPARQL:", sorted(only_in_sparql))

    # numeric differences
    max_abs_diff = 0.0
    for v in range(kg.num_nodes):
        diff = abs(engine_scores[v] - sparql_scores[v])
        if diff > max_abs_diff:
            max_abs_diff = diff

    print(f"\nMax absolute score difference (engine vs rdflib): {max_abs_diff:.6g}")

    agree_sets = (not only_in_engine and not only_in_sparql)
    agree_scores = (max_abs_diff <= tol)
    agree = agree_sets and agree_scores

    if agree:
        print("\n✅ Engine and rdflib agree for partial scores (within tolerance).")
    else:
        print("\n⚠️ Mismatch detected (engine vs rdflib partial scores).")

    return {
        "backend": "rdflib_partial",
        "engine_time_ms": engine_time_ms,
        "rdf_build_time_ms": rdf_build_time_ms,
        "sparql_time_ms": sparql_time_ms,
        "ttl_export_time_ms": None,
        "fuseki_clear_time_ms": None,
        "fuseki_upload_time_ms": None,
        "agreement": agree,
    }

from .fuseki_client import (
    fuseki_clear_dataset,
    fuseki_upload_turtle,
    fuseki_sparql_query_with_scores,
)
from .sparql_patterns import path_pattern_to_sparql_partial_score


def compare_partial_with_fuseki_once(
    kg: KGraph,
    pattern: PathPattern,
    dataset_url: str = "http://localhost:3030/ds",
    ttl_path: str = "random_graph.ttl",
    device: str = "cpu",
    threshold: float = 0.999,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compare GPU engine's partial path scores vs a remote Fuseki instance.

    - Engine: DAGReasoner, path-normalized score.
    - Fuseki: partial-score SPARQL generator + MAX(path_score), COALESCE(...,0.0).
    """

    # 1) Engine side
    dag = compile_path_pattern_to_dag(pattern)
    num_classes = kg.node_types.shape[1]
    sim_class = torch.eye(num_classes)

    reasoner = DAGReasoner(kg, device=device, sim_class=sim_class)
    reasoner.add_concept("Pattern", dag)

    t0 = time.perf_counter()
    scores = reasoner.evaluate_all()
    t1 = time.perf_counter()
    engine_time_ms = (t1 - t0) * 1000.0

    engine_scores = scores[:, 0].cpu().numpy()
    engine_set: Set[int] = {v for v, s in enumerate(engine_scores) if s >= threshold}

    print("Engine scores for Pattern (partial):")
    for v in range(kg.num_nodes):
        print(f"  v = {v}: s(v, Pattern) = {engine_scores[v]:.4f}")
    print("Engine satisfying nodes (score >= threshold):", sorted(engine_set))
    print(f"Engine evaluation time: {engine_time_ms:.3f} ms")

    # 2) Export RDF
    print("\nExporting KGraph as Turtle...")
    t2 = time.perf_counter()
    mapping = write_kgraph_as_turtle(
        kg,
        path=ttl_path,
        base_uri="http://example.org/random",
        type_threshold=0.5,
    )
    t3 = time.perf_counter()
    ttl_export_time_ms = (t3 - t2) * 1000.0
    print(f"  Turtle export time: {ttl_export_time_ms:.3f} ms (file: {ttl_path})")

    # 3) Clear & upload
    print(f"\nClearing Fuseki dataset at {dataset_url} ...")
    t4 = time.perf_counter()
    fuseki_clear_dataset(dataset_url)
    t5 = time.perf_counter()
    clear_time_ms = (t5 - t4) * 1000.0
    print(f"  Clear time: {clear_time_ms:.3f} ms")

    print(f"\nUploading Turtle to Fuseki dataset at {dataset_url} ...")
    t6 = time.perf_counter()
    fuseki_upload_turtle(dataset_url, ttl_path)
    t7 = time.perf_counter()
    upload_time_ms = (t7 - t6) * 1000.0
    print(f"  Upload time: {upload_time_ms:.3f} ms")

    # 4) SPARQL (partial score)
    print("\nCompiling PathPattern to partial-score SPARQL...")
    sparql_query = path_pattern_to_sparql_partial_score(pattern, mapping, start_var="v")
    print("\nSPARQL query (Fuseki, partial score):")
    print(sparql_query)

    print("\nRunning SPARQL on Fuseki...")
    t8 = time.perf_counter()
    rows = fuseki_sparql_query_with_scores(dataset_url, sparql_query)
    t9 = time.perf_counter()
    fuseki_sparql_time_ms = (t9 - t8) * 1000.0
    print(f"Fuseki SPARQL time (partial): {fuseki_sparql_time_ms:.3f} ms")

    # 5) Build SPARQL score array
    sparql_scores = [0.0] * kg.num_nodes
    iri_to_index = {str(iri): idx for idx, iri in enumerate(mapping.node_iris)}

    print("Fuseki scores for Pattern (partial):")
    for v_iri, score in rows:
        idx = iri_to_index.get(v_iri)
        if idx is None:
            print(f"  Warning: IRI {v_iri} not found in mapping; ignoring.")
            continue
        sparql_scores[idx] = score

    for v in range(kg.num_nodes):
        print(f"  v = {v}: s_fuseki(v, Pattern) = {sparql_scores[v]:.4f}")

    sparql_set: Set[int] = {v for v, s in enumerate(sparql_scores) if s >= threshold}

    # 6) Compare sets + numeric scores
    only_in_engine = engine_set - sparql_set
    only_in_sparql = sparql_set - engine_set
    intersection = engine_set & sparql_set

    print("\nSet comparison (Fuseki, partial):")
    print("  Intersection (score >= threshold):", sorted(intersection))
    print("  Only in engine:", sorted(only_in_engine))
    print("  Only in SPARQL:", sorted(only_in_sparql))

    max_abs_diff = 0.0
    for v in range(kg.num_nodes):
        diff = abs(engine_scores[v] - sparql_scores[v])
        if diff > max_abs_diff:
            max_abs_diff = diff

    print(f"\nMax absolute score difference (engine vs Fuseki): {max_abs_diff:.6g}")

    agree_sets = (not only_in_engine and not only_in_sparql)
    agree_scores = (max_abs_diff <= tol)
    agree = agree_sets and agree_scores

    if agree:
        print("\n✅ Engine and Fuseki agree for partial scores (within tolerance).")
    else:
        print("\n⚠️ Mismatch detected (engine vs Fuseki partial scores).")

    return {
        "backend": "fuseki_partial",
        "engine_time_ms": engine_time_ms,
        "rdf_build_time_ms": None,
        "sparql_time_ms": fuseki_sparql_time_ms,
        "ttl_export_time_ms": ttl_export_time_ms,
        "fuseki_clear_time_ms": clear_time_ms,
        "fuseki_upload_time_ms": upload_time_ms,
        "agreement": agree,
    }


def demo_random_exact_comparison(
    num_nodes: int = 20,
    num_props: int = 3,
    num_classes: int = 4,
    avg_degree_per_prop: float = 2.0,
    num_steps: int = 3,
    seed_graph: int = 123,
    seed_pattern: int = 456,
    device: str = "cuda",
    dataset_url: str = "http://localhost:3030/ds",
    comparison: list = None,
    csv_path: str | None = None,
):
    """
    Generate a random graph + random path pattern and compare the engine results
    to rdflib and/or Fuseki for a single run. Optionally log timings to CSV.
    Comparison is a list including "rdflib" or "fuseki"
    """
    if comparison is None:
        comparison = ["rdflib"]

    if "fuseki" in comparison:
        print("\nChecking Fuseki availability...")

        if not check_fuseki_alive(dataset_url):
            print(f"❌ ERROR: Cannot contact Fuseki dataset at {dataset_url}")
            print("   Make sure Fuseki is running, and the dataset exists.")
            print("   Example: ./fuseki-server --update --mem /ds")
            return  # Exit early; do NOT attempt upload/query.

        print("Fuseki is online.\n")
    
    timestamp = now_timestamp()
    print(f"Run: {timestamp}")
    ttl_path = f"data/random-graphs/{timestamp}.ttl"

    print("Generating random KGraph...")
    kg = generate_random_kgraph(
        num_nodes=num_nodes,
        num_props=num_props,
        num_classes=num_classes,
        avg_degree_per_prop=avg_degree_per_prop,
        seed=seed_graph,
    )
    print("  num_nodes:", kg.num_nodes)
    print("  num_props:", len(kg.offsets_p))
    print("  num_classes:", kg.node_types.shape[1])

    # total edges across all properties
    num_edges = sum(neighbors.numel() for neighbors in kg.neighbors_p)
    print("  num_edges:", num_edges)

    print("\nGenerating random PathPattern...")
    pattern = generate_random_path_pattern(
        num_steps=num_steps,
        num_props=num_props,
        num_classes=num_classes,
        seed=seed_pattern,
        normalize=True,
    )
    print("  steps (prop_idx, class_idx):", pattern.steps)

    device_to_use = device
    if device_to_use == "cuda" and not torch.cuda.is_available():
        device_to_use = "cpu"
    print("\nUsing device:", device_to_use)

    cpu_name = get_cpu_name()
    gpu_name = get_gpu_name(device_to_use)

    common_meta = {
        "timestamp": timestamp,
        "cpu_name": cpu_name,
        "gpu_name": gpu_name,
        "device": device_to_use,

        "num_nodes": num_nodes,
        "num_props": num_props,
        "num_classes": num_classes,
        "avg_degree_per_prop": avg_degree_per_prop,
        "num_steps": num_steps,
        "num_edges": num_edges,
    }

    # --- NEW: engine-only benchmark (no RDF/SPARQL) ---
    if "engine_only" in comparison:
        print("\n=== Engine-only timing (no RDF/SPARQL comparison) ===")
        metrics = compare_engine_only_once(
            kg,
            pattern,
            device=device_to_use,
            threshold=0.999,
        )
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            append_csv_row(csv_path, row)

    if "fuseki" in comparison:
        print("\n=== Comparing engine vs Fuseki SPARQL on this random example ===")
        metrics = compare_with_fuseki_once(
            kg,
            pattern,
            dataset_url=dataset_url,
            ttl_path=ttl_path,
            device=device_to_use,
            threshold=0.999,
        )
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            append_csv_row(csv_path, row)

    if "rdflib" in comparison:
        print("\n=== Comparing engine vs RDFLib SPARQL on this random example ===")
        metrics = compare_pattern_once(kg, pattern, device=device_to_use, threshold=0.999)
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            append_csv_row(csv_path, row)

def demo_random_partial_comparison(
    num_nodes: int = 20,
    num_props: int = 3,
    num_classes: int = 4,
    avg_degree_per_prop: float = 2.0,
    num_steps: int = 3,
    seed_graph: int = 123,
    seed_pattern: int = 456,
    device: str = "cuda",
    dataset_url: str = "http://localhost:3030/ds",
    comparison: list = None,
    csv_path: str | None = None,
):
    print(comparison)
    if comparison is None:
        comparison = ["fuseki"]

    from .random_graphs import generate_random_kgraph, generate_random_path_pattern

    print("Generating random KGraph...")
    kg = generate_random_kgraph(
        num_nodes=num_nodes,
        num_props=num_props,
        num_classes=num_classes,
        avg_degree_per_prop=avg_degree_per_prop,
        seed=seed_graph,
    )
    print("  num_nodes:", kg.num_nodes)
    print("  num_props:", len(kg.offsets_p))
    print("  num_classes:", kg.node_types.shape[1])
    num_edges = sum(n.numel() for n in kg.neighbors_p)
    print("  num_edges:", num_edges)

    print("\nGenerating random PathPattern...")
    pattern = generate_random_path_pattern(
        num_steps=num_steps,
        num_props=num_props,
        num_classes=num_classes,
        seed=seed_pattern,
        normalize=True,
    )
    print("  steps (prop_idx, class_idx):", pattern.steps)

    device_to_use = device
    if device_to_use == "cuda" and not torch.cuda.is_available():
        device_to_use = "cpu"
    print("\nUsing device:", device_to_use)

    cpu_name = get_cpu_name()
    gpu_name = get_gpu_name(device_to_use)

    common_meta = {
        "cpu_name": cpu_name,
        "gpu_name": gpu_name,
        "device": device_to_use,
        "num_nodes": num_nodes,
        "num_props": num_props,
        "num_classes": num_classes,
        "avg_degree_per_prop": avg_degree_per_prop,
        "num_steps": num_steps,
        "num_edges": num_edges,
    }

    # rdflib exact
    if "rdflib" in comparison:
        print("\n=== Comparing engine vs RDFLib SPARQL (exact) ===")
        metrics = compare_pattern_once(kg, pattern, device=device_to_use, threshold=0.999)
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            row["timestamp"] = now_timestamp()
            append_csv_row(csv_path, row)

    # rdflib partial
    if "rdflib_partial" in comparison:
        print("\n=== Comparing engine vs RDFLib SPARQL (partial scores) ===")
        metrics = compare_partial_pattern_once(
            kg, pattern, device=device_to_use, threshold=0.999
        )
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            row["timestamp"] = now_timestamp()
            append_csv_row(csv_path, row)

    # Fuseki exact
    if "fuseki" in comparison:
        print("\n=== Checking Fuseki availability… ===")
        if not check_fuseki_alive(dataset_url):
            print(f"❌ ERROR: Cannot contact Fuseki dataset at {dataset_url}")
            print("   Make sure Fuseki is running, and the dataset exists.")
            print("   Example: ./fuseki-server --update --mem /ds")
            return

        print("\n=== Comparing engine vs Fuseki SPARQL (exact) ===")
        metrics = compare_with_fuseki_once(
            kg,
            pattern,
            dataset_url=dataset_url,
            ttl_path="random_graph.ttl",
            device=device_to_use,
            threshold=0.999,
        )
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            row["timestamp"] = now_timestamp()
            append_csv_row(csv_path, row)

    # Fuseki partial
    if "fuseki_partial" in comparison:
        print("\n=== Checking Fuseki availability… ===")
        if not check_fuseki_alive(dataset_url):
            print(f"❌ ERROR: Cannot contact Fuseki dataset at {dataset_url}")
            print("   Make sure Fuseki is running, and the dataset exists.")
            print("   Example: ./fuseki-server --update --mem /ds")
            return

        print("\n=== Comparing engine vs Fuseki SPARQL (partial scores) ===")
        metrics = compare_partial_with_fuseki_once(
            kg,
            pattern,
            dataset_url=dataset_url,
            ttl_path="random_graph.ttl",
            device=device_to_use,
            threshold=0.999,
        )
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            row["timestamp"] = now_timestamp()
            append_csv_row(csv_path, row)

    # engine-only for partial scoring
    if "engine_only" in comparison:
        print("\n=== Engine-only timing (partial scoring, no RDF/SPARQL) ===")
        metrics = compare_engine_only_once(
            kg,
            pattern,
            device=device_to_use,
            threshold=0.999,
        )
        if csv_path is not None:
            row = {field: None for field in CSV_FIELDS}
            row.update(common_meta)
            row.update(metrics)
            row["timestamp"] = now_timestamp()
            append_csv_row(csv_path, row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random KG + PathPattern comparison harness")

    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-props", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--avg-degree-per-prop", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=10)

    parser.add_argument("--seed-graph", type=int, default=123)
    parser.add_argument("--seed-pattern", type=int, default=456)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dataset-url", type=str, default="http://localhost:3030/ds")

    parser.add_argument(
        "--comparison",
        type=str,
        nargs="+",
        default=["fuseki"],
        choices=["rdflib", "fuseki", "rdflib_partial", "fuseki_partial", "engine_only"],
        help="Which backends to compare against.",
    )


    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional path to a CSV file to append results to.",
    )

    args = parser.parse_args()

    demo_random_partial_comparison(
        num_nodes=args.num_nodes,
        num_props=args.num_props,
        num_classes=args.num_classes,
        avg_degree_per_prop=args.avg_degree_per_prop,
        num_steps=args.num_steps,
        seed_graph=args.seed_graph,
        seed_pattern=args.seed_pattern,
        device=args.device,
        dataset_url=args.dataset_url,
        comparison=args.comparison,
        csv_path=args.csv_path,
    )
