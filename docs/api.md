# Prototype Python API

The repository exposes a small Python API for embedding TensorKG inside another project. The command-line entry points remain the richest surface for benchmarking and oracle comparison, but many common in-process use cases can be handled directly from Python.

## What The Prototype API Is For

Use the prototype API when you want to:

- load RDFLib graphs directly from Python
- choose a TensorKG profile and mode programmatically
- resolve or set target classes
- run materialization, admissibility, filtered admissibility, or scored semantic alignment
- access the lowered dataset and mapping after execution

The API wraps the existing engine rather than introducing a separate execution path.

Use this API path for pipeline integration and in-process application logic. For benchmark reproduction, oracle comparison, and experiment-style timing runs, the CLI entry points remain the better fit.

## Basic Example

```python
from rdflib import Graph
from src import TensorKG

schema_graph = Graph().parse("ontology.owl", format="turtle")
data_graph = Graph().parse("abox.ttl", format="turtle")

engine = TensorKG(
    profile="gpu-el",
    mode="stratified",
    device="cpu",
)

result = (
    engine
    .load_graphs(schema_graph=schema_graph, data_graph=data_graph)
    .set_targets(["http://example.org/ontology#Employee"])
    .run()
)

members = result.members_for("http://example.org/ontology#Employee")
print("Materialized members:", sorted(map(str, members)))
```

## Loading From Files

```python
from src import TensorKG

engine = TensorKG(profile="gpu-el-full", mode="stratified", device="cuda")
result = engine.load_files(
    schema_paths=["data/owl2bench/UNIV-BENCH-OWL2EL.owl"],
    data_paths=["data/owl2bench/OWL2EL-1.owl"],
).run(targets=["http://benchmark/OWL2Bench#Employee"])
```

## Mode-Specific Convenience Methods

The wrapper includes a few convenience methods that simply call `run(...)` with a fixed mode:

- `materialize(...)`
- `admissibility(...)`
- `filtered_admissibility(...)`
- `score(...)`

Example:

```python
from src import TensorKG

engine = TensorKG(profile="gpu-dl", device="cpu")
engine.load_files(
    schema_paths=["data/negation_people.owl.ttl"],
    data_paths=["data/negation_people_data.ttl"],
)

score_result = engine.score(targets=["http://example.org/negation#TargetPerson"])
scores = score_result.scores_for("http://example.org/negation#TargetPerson")
```

Example shape of a scored semantic alignment result:

```text
ex:node123
  ex:Employee  score=1.00
  ex:Student   score=0.62
  ex:Course    score=0.00
```

That should be read as a structural compatibility view over the requested targets, not as a probability distribution.

## Target Resolution

If you do not set targets explicitly, the prototype API defaults to:

```python
["all-named-classes"]
```

You can also ask the engine to resolve targets first:

```python
targets = engine.resolve_targets(["all-named-classes"])
```

This uses the same target-resolution logic as the CLI and oracle-comparison tooling, including fragment-aware skipping for unsupported targets in a given mode.

## Accessing The Lowered Dataset

The returned `TensorKGResult` exposes the underlying engine result and dataset:

```python
dataset = result.dataset
kg = dataset.kg
mapping = dataset.mapping
```

This is useful when you want to:

- inspect the lowered tensor graph
- inspect the RDF-term mappings
- reuse the post-preprocessing dataset in surrounding code

## Current Scope

This API is intentionally small:

- it is meant to make the existing engine easy to import
- it is not yet a polished, versioned public SDK
- benchmark/oracle workflows still live most naturally in the CLI entry points

At the moment, the repo-local import path is:

```python
from src import TensorKG
```

If the project later grows a packaged distribution, that import surface can be formalized further.

For heavier experiment workflows, `src.oracle_compare` and `src.paper_benchmark` still provide the most complete surface.
