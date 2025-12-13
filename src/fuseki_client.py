from __future__ import annotations

from typing import List
import requests


def fuseki_clear_dataset(dataset_url: str) -> None:
    """
    Clear all data from a Fuseki dataset using SPARQL Update: DROP ALL.

    dataset_url: base URL of the dataset, e.g. "http://localhost:3030/ds"
    """
    update_url = dataset_url.rstrip("/") + "/update"
    update = "DROP ALL"

    resp = requests.post(
        update_url,
        data={"update": update},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()


def fuseki_upload_turtle(dataset_url: str, ttl_path: str) -> None:
    """
    Upload a Turtle file as the default graph to a Fuseki dataset.

    dataset_url: base URL of the dataset, e.g. "http://localhost:3030/ds"
    ttl_path: path to a .ttl file on disk
    """
    data_url = dataset_url.rstrip("/") + "/data?default"

    with open(ttl_path, "rb") as f:
        data = f.read()

    resp = requests.post(
        data_url,
        data=data,
        headers={"Content-Type": "text/turtle"},
        timeout=60,
    )
    resp.raise_for_status()


def fuseki_sparql_query(dataset_url: str, query: str) -> List[str]:
    """
    Run a SPARQL SELECT query against a Fuseki dataset and return the
    raw string values for ?v bindings.

    Assumes the query has a variable ?v in the projection.

    dataset_url: base URL of the dataset, e.g. "http://localhost:3030/ds"
    """
    sparql_url = dataset_url.rstrip("/") + "/sparql"

    resp = requests.post(
        sparql_url,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    values: List[str] = []
    for binding in data["results"]["bindings"]:
        if "v" in binding:
            values.append(binding["v"]["value"])
    return values

def fuseki_sparql_query_with_scores(dataset_url: str, query: str):
    """
    Run a SPARQL SELECT query against Fuseki that returns ?v and ?score,
    and return a list of (v_iri, score_float).
    """
    sparql_url = dataset_url.rstrip("/") + "/query"
    resp = requests.post(
        sparql_url,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=36000, # 10 hours
    )
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for b in data["results"]["bindings"]:
        v_iri = b["v"]["value"]
        score = float(b["score"]["value"])
        rows.append((v_iri, score))
    return rows
