# Evaluation, Reproducibility, And Result Artifacts

This note collects the main ways to inspect TensorKG's empirical behavior, correctness checks, and benchmark artifacts.

## What Is Already In The Repository

- `data/runs/`: benchmark, harness, and inspection outputs
- `.results/`: paper-preparation benchmark tables and spreadsheets
- `comparison/owlapi-5.5.1/`: Java-side oracle assets used by ELK/Openllet comparison scripts
- `comparison/apache-jena-fuseki-5.6.0/`: local Fuseki assets used for SPARQL-based comparison experiments

## Reproducing And Verifying Claims

For reproducing or verifying the claims made in the ISWC paper:

| Claim | Script / entry point | Output artifact |
| --- | --- | --- |
| OWL2Bench EL timing and profile comparison | `src.paper_benchmark` | `.results/paper-benchmark.csv` and `.results/paper-benchmark.jsonl` |
| One-off oracle comparison and timing breakdown | `src.oracle_compare` | console output plus optional timing JSON/CSV written via `--timing-json` and `--timing-csv` |
| Random-fragment coverage against Openllet | `src.coverage_harness` | `data/runs/coverage-harness/<timestamp>/run-summary.txt` and `run-summary.json` |
| Consistency / admissibility inspection on generated fragments | `src.consistency_harness` | `data/runs/consistency-harness/<timestamp>/run-summary.txt` and `run-summary.json` |
| Hand-built semantic feature regression checks | `src.test_gpu_dl_toys` | console regression summary over the toy fixtures in `data/toys/` |

## Main Evaluation Entry Points

### 1. `src.paper_benchmark`

Use this for batch benchmark experiments over:

- multiple datasets
- multiple devices
- multiple profiles
- multiple engine modes
- multiple oracle reasoners

It writes:

- append-only CSV summaries
- JSONL logs
- stage/profile timing trees

This is also the main path for the repository's EL validation story, since it is the benchmark surface used to compare the EL-oriented profiles against ELK on OWL2Bench.

### 2. `src.oracle_compare`

Use this for one-off end-to-end comparisons on a specific ontology/data pair. It is the most useful entry point when you want:

- timing breakdowns
- one target or a small set of targets
- direct comparison to ELK or Openllet

### 3. `src.coverage_harness`

Use this when the question is coverage rather than just speed.

What it does:

- generates small random fragment-bounded ontologies and ABoxes
- filters to base-consistent cases
- compares TensorKG's emitted novel `(node, target)` pairs to Openllet
- reports precision, recall, false positives, false negatives, and construct buckets

This is useful for making bounded statements such as:

- which construct combinations are currently strong
- where false positives are rare
- where recall is still incomplete

Example existing artifact:

- `data/runs/coverage-harness/2026-05-08_02-07-03/run-summary.txt`

That run reports, for `stratified` with `gpu-dl`, overall precision `0.9608` and recall `0.7424` on the completed generated cases, along with per-bucket examples.

That result should be read as a bounded statement about the broader experimental fragment, not as a replacement for the stronger EL validation story.

### 4. `src.consistency_harness`

Use this when the question is whether emitted high-confidence assignments survive local consistency checks.

What it does:

- generates random fragment-bounded cases
- records preprocessing plans
- asks the engine for perfect-score or emitted assignments
- checks those assignments with an external consistency test
- groups failures by construct bucket

This is especially useful for debugging policy-emitted assignments and inspecting construct buckets that still need stronger certification.

Example existing artifact:

- `data/runs/consistency-harness-inspect/2026-04-25_14-20-16/run-summary.txt`

That run is useful for inspecting construct buckets where local high-confidence outputs still needed stronger negative-side certification.

## Correctness Evaluation

TensorKG's correctness validation is split across several layers, because different parts of the system are validated in different ways.

### 1. Hand-built toy suite

The first line of validation is the toy regression suite:

```powershell
.\.venv\Scripts\python.exe -m src.test_gpu_dl_toys --device cpu --owlapi-home comparison/owlapi-5.5.1
```

What it covers:

- hand-built feature fixtures for equality, value restrictions, literal-sensitive cases, negative cases, and SCC/super-DAG behavior
- direct engine-versus-Openllet comparisons on the supported toy cases
- explicit skips and expected-failure markers for known unsupported frontiers

Why it matters:

- it keeps individual semantic features tied to named, inspectable fixtures
- it makes boundaries visible instead of burying them in aggregate benchmark numbers

Examples of toy fixtures already in the repo include:

- `data/toys/toy_sameas_native_*`
- `data/toys/data_some_oneof_people_*`
- `data/toys/functional_data_people_*`
- `data/toys/toy_negative_assertion_*`
- `data/toys/toy_negative_conclusion_*`
- `data/toys/toy_disjunction_chain_*`
- `data/toys/toy_disjunction_branching_*`
- `data/toys/toy_superdag_acyclic_*`
- `data/toys/toy_superdag_scc_*`

### 2. OWL2Bench and reasoner comparison

The second line of validation is larger-scale comparison on OWL2Bench through `src.paper_benchmark` and `src.oracle_compare`.

What it covers:

- end-to-end behavior of the EL-oriented profiles on realistic EL workloads
- direct comparison against ELK and, where appropriate, Openllet
- timing and stage-breakdown information alongside semantic comparison

Why it matters:

- this is the main evidence behind the repository's complete OWL 2 EL claim
- it complements the toy suite by checking the validated EL path on larger real datasets rather than only on hand-built fixtures

### 3. Random-fragment coverage checks

The third line of validation is the coverage harness.

What it covers:

- randomly generated fragment-bounded ontologies and ABoxes
- overlap between TensorKG outputs and Openllet outputs on novel `(node, target)` pairs
- precision/recall and per-construct-bucket diagnostics

Why it matters:

- it helps locate frontier construct combinations
- it gives broader empirical coverage than a fixed toy suite alone

### 4. Random-fragment consistency checks

The fourth line of validation is the consistency harness.

What it covers:

- whether emitted perfect-score or policy-emitted assignments survive external consistency checks
- preprocessing plans and construct buckets associated with any failures

Why it matters:

- it is especially useful for the admissibility-style and broader experimental paths, where omission and certification policy matter as much as raw overlap

## Recommended Reproduction Commands

### Toy regression suite

```powershell
.\.venv\Scripts\python.exe -m src.test_gpu_dl_toys --device cpu --owlapi-home comparison/owlapi-5.5.1
```

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

### Coverage harness

```powershell
.\.venv\Scripts\python.exe -m src.coverage_harness `
  --num-cases 20 --max-attempts 80 `
  --engine-mode stratified --engine-profile gpu-dl `
  --constructs OWL-EL `
  --device cpu `
  --timeout-seconds 20 `
  --owlapi-home comparison/owlapi-5.5.1
```

### Consistency harness

```powershell
.\.venv\Scripts\python.exe -m src.consistency_harness `
  --num-cases 5 --max-attempts 25 `
  --engine-mode query `
  --constructs subclass intersection union exists forall domain range disjoint `
  --device cpu
```

## Reading The Results

### End-to-end versus kernel-only timing

TensorKG is not just a GPU kernel. Real runs may include:

- RDF loading
- preprocessing-plan construction
- closure/materialization passes
- canonicalization and helper generation
- DAG compilation
- host-device transfer
- GPU evaluation
- mode-specific filtering or iteration

When reporting performance, it helps to say explicitly whether you mean:

- total engine runtime
- just compiled evaluation runtime
- oracle comparison runtime

For the paper results specifically, the reported reasoning times do not include RDF parsing time. They are reasoning/runtime measurements over already-loaded inputs, together with the selected preprocessing, lowering, and execution stages used by the run.

### Correctness, coverage, and performance answer different questions

The harnesses answer different questions:

- `src.test_gpu_dl_toys`: hand-built semantic feature regression checks
- `paper_benchmark`: speed and stage timing
- `coverage_harness`: overlap with an oracle on generated cases
- `consistency_harness`: whether emitted assignments appear locally safe on generated cases

No single harness should be treated as proving everything.

### Negative results matter

The repository intentionally keeps inspection artifacts for problematic construct buckets. That helps keep the current implementation boundaries visible.

## Current Validation Summary

At a high level, the repository currently supports the following validation story:

- the EL-oriented path is validated through repeated OWL2Bench comparison and smaller feature-specific fixtures
- the toy suite provides inspectable correctness checks for individual semantic features and known frontiers
- the random-fragment harnesses broaden that picture by measuring overlap and local safety on generated cases
- the broader `gpu-dl` path is empirical and bounded rather than presented as complete OWL 2 DL reasoning
