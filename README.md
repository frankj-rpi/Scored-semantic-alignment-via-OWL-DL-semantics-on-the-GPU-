# TensorKG

TensorKG is an experimental GPU-oriented reasoning and concept alignment scoring engine for OWL-backed KGs. It compiles OWL class restrictions into a directed acyclic graph of constraints, converts KGs into a GPU-friendly tensor representation, and then evaluates each layer of constraints against all KG nodes simultaneously on the CPU or GPU (via CUDA bindings).

The project supports [OWL 2 EL](https://www.w3.org/TR/owl2-profiles/#OWL_2_EL) semantics. It also includes limited experimental support for some OWL DL operations; **for more informarion on exactly what is supported and how, see [docs/semantics.md](docs/semantics.md)**. 

TensorKG has been evaluated against [ELK](https://github.com/liveontologies/elk-reasoner) and [openllet](https://github.com/Galigator/openllet) on the [OWL2bench](https://github.com/kracr/owl2bench) datasets, as well as randomly generated and [hand-built graphs](data/toys). **See [docs/evaluation.md](docs/evaluation.md) for more evaluation information, including how to recreate the evaluation results.**

**For usage information, see the [API](docs/api.md).** TensorKG can be used for the following reasoning tasks:

- Forward reasoning: materializing type assignments supported by sufficient conditions. In other words: can we infer *n* must be a *C* from known information?
- Admissibility testing: checking which KG nodes satisfy the necessary conditions of which classes, according to known information. In other words, it answers: "would asserting node *n* as class *C* be consistent with everything we know about *n* and *C*?" 
- Scored semantic alignment: scoring the degree to which each KG node satisfies the necessary conditions of each class. In other words, for some node *n* and class *C*, a score **s**(*n*,*C*)=0.0 means *n* is not known to satisfy any conditions of *C*, and a score **s**(*n*,*C*)=1.0 means *n* is known to satisfy all known requirements of *C*. 

## Documentation Guide

The two main documentation entry points are:

- [docs/semantics.md](docs/semantics.md): explains what TensorKG computes, how the engine modes differ, how to interpret scores, which feature families are supported in each profile, and why those support claims are justified. It also includes feature-by-feature notes describing how each construct family is implemented, which toy fixtures or harnesses validate it, and why any support is marked as limited or unsupported.
- [docs/evaluation.md](docs/evaluation.md): explains how to run the benchmark and validation tooling, where existing result artifacts live in the repository, how correctness is checked across the toy suite, OWL2Bench runs, and random-fragment harnesses, and how to reproduce the main experiment paths.
- [docs/api.md](docs/api.md): shows the prototype Python API for loading RDFLib graphs, selecting profiles and modes, resolving targets, and running TensorKG directly inside another Python project.

For implementation-level navigation, see [Repository Layout](#repository-layout) and [Implementation Pointers](#implementation-pointers).

## What TensorKG Is And Is Not

TensorKG is:

- a native graph/tensor execution engine for a restricted OWL-derived fragment
- a complete OWL 2 EL forward materializer
- an admissibility and scored-alignment engine for candidate node-class pairs
- a research codebase with explicit profiling, harnesses, and oracle-comparison tooling

TensorKG is not:

- an OWL 2 DL reasoner
- a drop-in replacement for tableau reasoners on arbitrary ontologies
- a probability model
- a calibrated entity-ranking model for knowledge graph completion

When the engine omits a conclusion outside the currently supported fragment, that should be interpreted as `unknown`, not as a proof that the conclusion is false.

## Engine Modes

| Mode | Best used for | Main output | Current intent |
| --- | --- | --- | --- |
| `stratified` | forward reasoning, especially complete OWL 2 EL materialization | materialized class assignments | complete on the supported OWL 2 EL path; strongest current mode beyond EL as well |
| `admissibility` | checking whether a candidate class assignment is locally compatible with known structure | admissible members or scores for requested targets | conservative necessary-condition evaluation |
| `scored_semantic_alignment` | dense node-to-class compatibility scoring | graded scores in `[0,1]` | same structural view as admissibility, but keeping fuzzy partial satisfaction |
| `filtered_admissibility` | candidate generation followed by stricter recheck/pruning | filtered admissible assignments | conservative query-style mode with synchronous recheck and closure-based filtering |

More precise semantics and caveats are documented in [docs/semantics.md](docs/semantics.md).

## Profiles At A Glance

| Profile | Summary |
| --- | --- |
| `gpu-el-lite` | lightest EL-oriented path; minimal preprocessing; no native ABox `sameAs` by default |
| `gpu-el` | EL-oriented path with native `sameAs` canonicalization and richer preprocessing |
| `gpu-el-full` | `gpu-el` plus native `HasKey`-driven equality generation |
| `gpu-dl` | broader, experimental OWL-DL-like fragment with literal-sensitive and negative/blocker-oriented machinery |

The full supported-fragment discussion is in [docs/semantics.md](docs/semantics.md#profiles-and-supported-constructs).

## Quick Start

### 1. Create an environment

On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the default install does not provide the correct CUDA build, install PyTorch explicitly from the official PyTorch index for your CUDA version.

### 2. Run the toy regression suite

```powershell
.\.venv\Scripts\python.exe -m src.test_gpu_dl_toys --device cpu --owlapi-home comparison/owlapi-5.5.1
```

### 3. Use the prototype Python API

```python
from rdflib import Graph
from src import TensorKG

schema_graph = Graph().parse("ontology.owl", format="turtle")
data_graph = Graph().parse("abox.ttl", format="turtle")

engine = TensorKG(profile="gpu-el", mode="stratified", device="cpu")
result = (
    engine
    .load_graphs(schema_graph=schema_graph, data_graph=data_graph)
    .set_targets(["http://example.org/ontology#Employee"])
    .run()
)

members = result.members_for("http://example.org/ontology#Employee")
print(sorted(map(str, members)))
```

Use this path when you want to integrate TensorKG into a Python pipeline or another application.

The full how-to is in [docs/api.md](docs/api.md).

### 4. Run a single-target ontology/data example

```powershell
.\.venv\Scripts\python.exe -m src.oracle_compare --show-timing-breakdown `
  --schema data/owl2bench/UNIV-BENCH-OWL2EL.owl --data data/owl2bench/OWL2EL-1.owl `
  --target-class http://benchmark/OWL2Bench#Employee `
  --engine-mode stratified --profile gpu-el --device cpu `
  --oracles elk --owlapi-home comparison/owlapi-5.5.1
```

Use the CLI entry points when you want benchmark-style runs, oracle comparison, timing breakdowns, or reproducible experiment workflows.

### 5. Inspect filtered admissibility on a small hand-built fixture

```powershell
.\.venv\Scripts\python.exe -m src.filtered_query_examples `
  --schema data\negation_people.owl.ttl `
  --data data\negation_people_data.ttl `
  --target-class http://example.org/negation#TargetPerson `
  --device cpu
```

### 6. Example scored semantic alignment output

An alignment-style result is typically read per node, for example:

```text
ex:node123
  ex:Employee  score=1.00
  ex:Student   score=0.62
  ex:Course    score=0.00
```

This means the node fully satisfies the compiled structural requirements for `ex:Employee`, partially matches `ex:Student`, and does not match the compiled requirements for `ex:Course`.

## Reproducing Benchmark Runs

### OWL2Bench benchmark sweep

```powershell
.\.venv\Scripts\python.exe -m src.paper_benchmark `
  --schema data/owl2bench/UNIV-BENCH-OWL2EL.owl `
  --datasets `
    data/owl2bench/OWL2EL-1.owl `
    data/owl2bench/OWL2EL-2.owl `
    data/owl2bench/OWL2EL-5.owl `
    data/owl2bench/OWL2EL-10.owl `
    data/owl2bench/OWL2EL-50.owl `
    data/owl2bench/OWL2EL-100.owl `
    data/owl2bench/OWL2EL-200.owl `
  -k 5 `
  --profiles gpu-el-lite gpu-el gpu-el-full `
  --devices cpu cuda `
  --engine-modes stratified admissibility filtered_admissibility `
  --reasoners elk openllet `
  --timeout-seconds 600 `
  --csv-path .results\paper-benchmark.csv `
  --log-path .results\paper-benchmark.jsonl `
  --owlapi-home comparison/owlapi-5.5.1
```

### Random-fragment coverage against Openllet

```powershell
.\.venv\Scripts\python.exe -m src.coverage_harness `
  --num-cases 20 --max-attempts 80 `
  --engine-mode stratified --engine-profile gpu-dl `
  --constructs OWL-EL `
  --device cpu `
  --timeout-seconds 20 `
  --owlapi-home comparison/owlapi-5.5.1
```

### Consistency-oriented admissibility checks on generated fragments

```powershell
.\.venv\Scripts\python.exe -m src.consistency_harness `
  --num-cases 5 --max-attempts 25 `
  --engine-mode query `
  --constructs subclass intersection union exists forall domain range disjoint `
  --device cpu
```

More detail on result artifacts, correctness checks, and result interpretation is in [docs/evaluation.md](docs/evaluation.md).

## Repository Layout

- `src/`: main implementation
- `docs/`: semantics and evaluation notes
- `specs/`: working semantic notes and design documents
- `data/`: toy ontologies, benchmark inputs, profiling output, and run artifacts
- `.results/`: benchmark outputs used during paper preparation
- `comparison/`: external reasoner and Fuseki assets used by the comparison tooling
- `methods.tex`: current methods-section draft for the paper

## Main Entry Points

- `src.oracle_compare`: main end-to-end runner for native engine and oracle comparisons
- `src.paper_benchmark`: batch benchmark runner across datasets, profiles, devices, and reasoners
- `src.test_gpu_dl_toys`: toy regression suite for correctness-oriented cases
- `src.coverage_harness`: random-fragment coverage checks against Openllet
- `src.consistency_harness`: random-fragment consistency checks for generated positive claims

## Known Limitations

- TensorKG is not complete for OWL 2 DL and should not be presented as such.
- TensorKG does make a strong positive claim for complete OWL 2 EL reasoning; the main incompleteness caveats begin when moving beyond that fragment.
- The strongest current non-EL semantic claim is in `stratified`; query-style modes are intentionally more conservative.
- `scored_semantic_alignment` scores are structural compatibility scores, not probabilities.
- GPU speedups are strongest when preprocessing is modest and the compiled execution is largely acyclic or batchable.
- CPU-side preprocessing, equality handling, fixpoint orchestration, and oracle interop can dominate end-to-end runtime on some workloads.
- Memory usage can become substantial because the evaluator stores graph structure plus per-node, per-DAG intermediate scores.

## Implementation Pointers

If you are reading the code, the most useful starting files are:

- `src/oracle_compare.py`: orchestration, mode handling, timing, oracle comparisons, reporting
- `src/ontology_parse.py`: preprocessing, lowering, helper generation, SCC analysis, compilation
- `src/dag_eval.py`: score-operator evaluation
- `src/dag_reasoner.py`: merged-root and DAG execution helpers
- `src/graph.py`: tensor graph representation

## Status

This is an active research codebase. Interfaces, supported fragments, and empirical results are still evolving, but the repository is organized to make the current guarantees, limitations, and evaluation pathways explicit.
