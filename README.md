# TensorKG

This repository contains an experimental GPU-oriented reasoning framework for ontology-guided knowledge graph analysis. The project focuses on compiling a restricted OWL-style fragment into native graph operators, preprocessing the graph into a tensor-friendly representation, and evaluating class constraints in parallel on CPU or GPU.

The codebase currently supports several related reasoning modes, including:

- `admissibility`: necessary-condition checking for candidate class assignments
- `scored_semantic_alignment`: graded necessary-condition scoring for fuzzy structural alignment
- `filtered_admissibility`: admissibility with synchronous recheck and blocker-based pruning
- `stratified`: positive sufficient-condition materialization followed by negative/blocker verification

It also exposes multiple execution profiles, including `gpu-el-lite`, `gpu-el`, `gpu-el-full`, and `gpu-dl`, which differ mainly in how much preprocessing, equality handling, and literal-sensitive support are enabled.

## Repository layout

- `src/`: main implementation
- `data/`: datasets, profiling output, and benchmark inputs
- `comparison/`: external reasoner integration assets
- `methods.tex`: current methods section draft for the paper
- `brainstorm/`, `specs/`: design notes, semantics notes, and working documents

## Setup

### Python environment

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

### PyTorch / CUDA

If the default install does not provide the right CUDA build, install PyTorch explicitly from the official index. For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Always confirm that the selected device is the one you intend to benchmark with.

## Quick start

### Synthetic exact-comparison demo
=======
####  Other tests

Batch correctness tests:

```
./.venv/Scripts/python.exe -m src.test_gpu_dl_toys --device cpu --owlapi-home comparison/owlapi-5.5.1
```

Oracle comparison for owl2bench, one class, stratified

```bash
python -m src.compare_exact \
  --num-nodes 10000 \
  --num-props 1000 \
  --num-classes 1000 \
  --avg-degree-per-prop 5.0 \
  --num-steps 10000 \
  --comparison engine_only \
  --csv-path ./data/runs/compare-exact/results.csv
```

### Single-target OWL2Bench run

```powershell
.\.venv\Scripts\python.exe -m src.oracle_compare --show-timing-breakdown `
  --schema data/owl2bench/UNIV-BENCH-OWL2EL.owl --data data/owl2bench/OWL2EL-1.owl `
  --target-class http://benchmark/OWL2Bench#Employee `
  --engine-mode stratified --profile gpu-el --device cpu `
  --oracles elk --owlapi-home comparison/owlapi-5.5.1
```

## Main entry points

### `src.oracle_compare`

Primary end-to-end runner for native engine/oracle comparisons.

Typical uses:

- run one ontology/data pair against a selected engine mode
- inspect timing breakdowns
- compare native results against ELK or Openllet

### `src.paper_benchmark`

Batch benchmark runner used for paper-style experiments across:

- multiple datasets
- multiple profiles
- multiple engine modes
- multiple devices
- multiple oracle reasoners

Example:

```bash
python -m src.paper_benchmark \
  --schema data/owl2bench/UNIV-BENCH-OWL2EL.owl \
  --datasets \
    data/owl2bench/OWL2EL-1.owl \
    data/owl2bench/OWL2EL-2.owl \
    data/owl2bench/OWL2EL-5.owl \
    data/owl2bench/OWL2EL-10.owl \
    data/owl2bench/OWL2EL-50.owl \
    data/owl2bench/OWL2EL-100.owl \
    data/owl2bench/OWL2EL-200.owl \
  -k 5 \
  --profiles gpu-el-lite gpu-el gpu-el-full \
  --devices cpu cuda \
  --engine-modes stratified admissibility filtered_admissibility \
  --reasoners elk openllet \
  --timeout-seconds 600 \
  --csv-path .results/laptop-benchmark.csv \
  --log-path .results/laptop-benchmark.jsonl \
  --owlapi-home comparison/owlapi-5.5.1
```

### `src.test_gpu_dl_toys`

Toy regression suite for small correctness-focused cases.

```powershell
.\.venv\Scripts\python.exe -m src.test_gpu_dl_toys --device cpu --owlapi-home comparison/owlapi-5.5.1
```

## Profiles at a glance

- `gpu-el-lite`: lightest preprocessing path; no ABox `sameAs` reasoning by default
- `gpu-el`: adds native `sameAs` canonicalization/expansion
- `gpu-el-full`: adds `HasKey`-driven equality generation on top of `gpu-el`
- `gpu-dl`: broader native OWL-DL-like fragment, including literal-sensitive support and negative/blocker-oriented machinery

## Notes

- This project is intentionally soundness-oriented and incomplete with respect to full OWL 2 DL.
- Some modes are stronger than others semantically; `stratified` is currently the strongest end-to-end reasoning path.
- Query-style modes are more conservative and may intentionally suppress conclusions outside the currently supported certified fragment.

## Related working files

If you are navigating the repo for implementation details, these files are the most useful starting points:

- `src/oracle_compare.py`: orchestration, mode handling, timing, reporting
- `src/ontology_parse.py`: preprocessing, compilation, rule extraction, and dataset construction
- `src/dag_eval.py`: native DAG score evaluation
- `src/graph.py`: tensor graph representation

## Status

This is an active research codebase. Interfaces, supported fragments, and performance characteristics are still evolving.
