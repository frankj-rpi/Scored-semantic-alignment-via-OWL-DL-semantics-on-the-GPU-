# TensorKG Semantics

## Contents

- [Two Core Tasks](#two-core-tasks)
- [Engine Modes](#engine-modes)
- [Open-World Versus Current-Graph Semantics](#open-world-versus-current-graph-semantics)
- [Profiles And Supported Constructs](#profiles-and-supported-constructs)
- [Feature Notes And Validation Hooks](#feature-notes-and-validation-hooks)
- [What We Can Safely Claim Today](#what-we-can-safely-claim-today)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Where To Look In The Code](#where-to-look-in-the-code)

## Two Core Tasks

TensorKG is organized around two different tasks over node-class pairs `(n, C)`.

### 1. Type materialization and the forward score

This is the forward, sufficient-condition task.

Concretely:

- TensorKG computes a forward score for the pair `(n, C)` by propagating sufficient-condition support through the compiled DAG for `C`
- when that forward score reaches the engine's exact threshold, the engine treats `n` as supported strongly enough to materialize `C(n)`
- in other words, the forward score is a native execution view of the question "do the known sufficient conditions for `C` hold for `n`?"

Semantically, this is the task that is closest to ordinary OWL-style entailment. A materialized class assignment is meant to mean:

- inside the supported fragment, TensorKG has enough positive support to conclude `C(n)`
- for the validated EL path, this is the main place where the repo makes a completeness claim rather than only a conservative soundness-oriented claim

This is the semantic core of `stratified` mode.

How to read the forward score:

- near `1.0`: the compiled sufficient condition is fully satisfied
- intermediate score: useful internally for propagation or softer aggregations, but not by itself a claim of classical entailment
- `0.0`: no compiled sufficient support was found

### 2. Admissibility testing and the backward score

This is the backward, necessary-condition task.

Concretely:

- TensorKG computes a backward score for the pair `(n, C)` by propagating necessary-condition requirements for `C` through the compiled DAG
- this score answers the question "how well does the currently known graph satisfy what membership in `C` would require of `n`?"
- an exact admissibility-style result means that all compiled necessary conditions were satisfied strongly enough to pass the engine's threshold

Semantically, admissibility is weaker than materialization:

- an admissibility result does not, by itself, mean that `C(n)` is entailed
- it means that the known graph structure is compatible with `n` being an instance of `C`, as far as the current fragment and current graph state can certify
- when the engine suppresses a result here, that often means `unknown`, not "proved false"

This is the semantic core of `admissibility` and `scored_semantic_alignment`.

How to read the backward score:

- `1.0`: all compiled necessary conditions for the target were satisfied
- intermediate score: the node satisfies some but not all compiled structural requirements
- `0.0`: no compiled necessary-condition support was found

### Admissibility versus scored semantic alignment

`admissibility` and `scored_semantic_alignment` use the same backward, necessary-condition view, but they use it differently.

In `admissibility` mode:

- the backward score is used as a certification-style signal
- the main question is whether the node passes the engine's exact admissibility threshold for the requested class

In `scored_semantic_alignment` mode:

- the same backward score is kept as a graded output instead of being used only as an exact pass/fail signal
- the main question becomes which classes a node structurally resembles most strongly

So a semantic alignment score should be read as:

- not a probability
- not a claim of OWL entailment
- not a globally calibrated ranking value across arbitrary ontologies
- a structural compatibility score derived from the compiled necessary conditions of the target class

This is why a node can receive a meaningful semantic alignment score for a class even when TensorKG would not materialize that class for the node.

## Engine Modes

### `stratified`

This is the strongest current end-to-end reasoning mode.

Operationally it:

- builds the selected preprocessing closure
- compiles sufficient-condition DAGs
- materializes positive conclusions
- applies negative/blocker-oriented checks before emitting final assignments

This is the mode with the strongest soundness-oriented claim in the current codebase.

Within the EL-oriented profiles, this has been validated for OWL 2 EL materialization via oracle comparison on the OWL2Bench EL datasets, randomly generated graphs and ontologies, and the hand-built fixtures.

### `admissibility`

This mode evaluates target classes through the necessary-condition view.

Operationally it:

- prepares the graph under the selected profile
- compiles target DAGs in the backward/necessary-condition view
- returns admissible members or scores for the requested targets

It is intentionally conservative. Omission should be read as `unknown`, not `false`.

### `scored_semantic_alignment`

This mode uses the same structural view as `admissibility`, but keeps the graded output instead of using it only as a certified yes/no style result.

This is the mode to use when the question is:
- which ontology classes does this node most strongly resemble?
- how well does a node satisfy the structure expected by a class?

### `filtered_admissibility`

This mode starts from query-style necessary-condition candidates and then rechecks or prunes them through additional stabilization/filtering logic.

It is best understood as a more conservative admissibility pipeline rather than as a separate classical OWL reasoning task.

## Open-World Versus Current-Graph Semantics

Forward materialization is the part that is closest to standard OWL entailment behavior inside the supported fragment.

Admissibility-style modes are more local to the graph as currently known:

- adding new ABox facts later may change the score
- unsupported negative or branching effects may cause the engine to suppress a conclusion rather than guess
- the system prefers omission over an unsafe positive claim

That makes admissibility useful for candidate validation and semantic alignment, but it should not be confused with a complete open-world consistency test.

## Profiles And Supported Constructs

Support varies depending on the selected profile, in order to preserve tractability when more complex reasoning is not required.

- `yes` means there is direct native machinery for the construct family in the intended profile
- `limited` means support is partial, mode-dependent, preprocessing-dependent, or restricted to a safer subcase
- `no` means it should not be treated as supported

In particular:
- `universal restrictions` and `unions` are best understood as broader operator-family rows, not as rows that define the EL claim
- `owl:sameAs` and `HasKey` are profile extras layered on top of the core EL path, so their profile-sensitive support does not weaken the OWL 2 EL materialization claim
- several rows mix forward materialization support with query/admissibility/scored-alignment support, which is why the detailed notes below matter

| Construct family | `gpu-el-lite` | `gpu-el` | `gpu-el-full` | `gpu-dl` | Notes |
| --- | --- | --- | --- | --- | --- |
| named classes / subclassing] | yes | yes | yes | yes | core compiled fragment |
| intersections | yes | yes | yes | yes | native DAG aggregation |
| existential restrictions | yes | yes | yes | yes | core neighbor traversal operator |
| universal restrictions | limited | limited | limited | yes | used more defensibly in admissibility-style views |
| domain / range preprocessing | limited | yes | yes | yes | may be materialized or query-augmented depending on plan |
| hierarchy closure | yes | yes | yes | yes | preprocessing-stage support |
| reflexive roles | limited | yes | yes | yes | preprocessing plus native checks |
| target-role saturation | no | limited | limited | yes | profile- and plan-dependent |
| native `owl:sameAs` canonicalization | no | yes | yes | yes | schema-side and ABox-side equality handling differ by profile |
| `HasKey`-driven equality | no | no | yes | yes | native but more expensive |
| nominals / `oneOf` | limited | limited | limited | yes | supported in selected cases |
| `hasValue` | limited | limited | limited | yes | broader in `gpu-dl` |
| datatype and literal checks | no | no | limited | yes | literal-sensitive path is mainly `gpu-dl` |
| cardinality restrictions | no | no | limited | yes | stronger support on the `gpu-dl` side |
| unions / disjunction | no | limited | limited | limited | support is mode-sensitive and should be treated conservatively |
| complements / negative class effects | no | no | no | limited | mostly blocker- or negative-fragment-oriented, not unrestricted DL negation |
| general branching/tableau reasoning | no | no | no | no | unsupported frontier |

Note that the GPU-DL profile is specifically for **sound but incomplete** inferences. In other words: if the input KG/ontology are consistent, and some inference is inferred, then it is guaranteed to be a sound inference. However: if it is NOT inferred, that means it is unknown whether it is true or false. 

In other words: GPU-DL prioritizes tractability and soundness over completeness. If an inference would require general tableau reasoning in order to determine if it is true or false, we stop at that ``unkown" for that assertion and any other assertions that may directly or indirectly be influenced by it (conservatively estimated).

## Feature Notes And Validation Hooks

This section is the companion to the overview table. The goal is to make each feature family read like a concrete contract:

- if support is claimed, say how it is implemented
- if support is limited, say why the boundary exists
- if support is absent, say why it is intentionally absent
- where possible, point to hand-built toy fixtures that exercise the behavior

The toy fixtures below refer to files under `data/` and the regression runner `src/test_gpu_dl_toys.py`.

### Named classes, subclassing, and intersections

Status:

- supported

How it is supported:

- named classes compile to atomic lookup nodes
- subclass and equivalent-class structure are lowered into the compiled dependency graph
- intersections are evaluated natively as conjunction-style DAG aggregators

Validation hooks:

- `data/toys/toy_people.owl.ttl`
- `data/toys/toy_people_data.ttl`
- `data/toys/toy_superdag_acyclic_schema.ttl`
- `data/toys/toy_superdag_acyclic_data.ttl`

Why the claim is strong:

- this is part of the validated EL path

### Existential restrictions

Status:

- supported

How it is supported:

- existential restrictions compile to native neighbor-traversal operators over the tensor graph
- evaluation uses property-indexed CSR adjacency and layered DAG propagation

Validation hooks:

- `data/toys/toy_people.owl.ttl`
- `data/toys/toy_people_data.ttl`
- `data/toys/toy_hobby_chain_schema.ttl`
- `data/toys/toy_hobby_chain_data.ttl`

Why the claim is strong:

- existential restrictions are core to both the EL path and the scored path machinery

### Hierarchy closure

Status:

- supported

How it is supported:

- hierarchy effects are handled during preprocessing through schema-side closure and lowering
- compiled evaluation then runs over the lowered dataset rather than rediscovering the hierarchy online

Validation hooks:

- `data/hierarchy_people.owl.ttl`
- `data/hierarchy_people_data.ttl`
- OWL2Bench EL runs through `src/paper_benchmark`

Why the claim is strong:

- hierarchy handling is part of the normal validated EL workflow

### Domain and range effects

Status:

- supported, but through multiple mechanisms depending on fragment and mode

How it is supported:

- the preprocessing planner can choose domain/range materialization
- query-style execution can also use target-aware augmentation
- Horn-safe domain/range handling is surfaced as an explicit preprocessing stage

Validation hooks:

- `data/domain_range_people.owl.ttl`
- `data/domain_range_query_data.ttl`
- `data/domain_range_horn.owl.ttl`
- `data/domain_range_horn_data.ttl`

Why the table says `limited` in some columns:

- the limitation is not "we are unsure what happens"
- it means the effect may be realized by preprocessing or query augmentation rather than by a single uniform native operator in every profile/mode combination

### Reflexive roles

Status:

- supported in the profiles that enable reflexive-role preprocessing

How it is supported:

- reflexive property axioms are detected during preprocessing
- self-style checks then evaluate against explicit self-edges or reflexive-role metadata

Validation hooks:

- `data/reflexive_people.owl.ttl`
- `data/reflexive_people_data.ttl`
- generated reflexive construct buckets in `data/runs/coverage-harness/`

Why the table says `limited` in lighter profiles:

- the limitation is profile-policy related
- lighter profiles do not always turn on every closure pass by default

### `owl:sameAs`

Status:

- supported natively from `gpu-el` upward

How it is supported:

- schema-side equality facts are incorporated during schema cache extraction
- ABox equality is handled through native canonicalization and alias expansion
- reporting can expand results back across sameAs-equivalent aliases

Validation hooks:

- `data/toys/toy_sameas_native_schema.ttl`
- `data/toys/toy_sameas_native_data.ttl`
- `src/test_gpu_dl_toys.py` case `toy_sameas_native`

Why the table says `no` for `gpu-el-lite`:

- this is an intentional profile distinction
- `gpu-el-lite` is meant to be the leanest EL-oriented path and does not enable native ABox sameAs handling by default

### `HasKey`-driven equality

Status:

- supported from `gpu-el-full` upward

How it is supported:

- key-based equality generation is handled as a native profile feature layered on top of the EL/sameAs path
- it is explicitly opt-in because it adds additional host-side reasoning cost

Validation hooks:

- `data/has_key_people.owl.ttl`
- `data/has_key_people_data.ttl`
- generated construct buckets containing `has-key` in `data/runs/coverage-harness/`

Why the table says `no` in lighter profiles:

- this is a deliberate performance/feature split between `gpu-el`, `gpu-el-full`, and `gpu-dl`

### Nominals, `oneOf`, and exact named/value-style matches

Status:

- supported in selected cases

How it is supported:

- named nominals and data `oneOf` values lower to direct equality-style leaf checks
- exact value restrictions and related literal-sensitive leaves are available on the broader path

Validation hooks:

- `data/nominal_people.owl.ttl`
- `data/nominal_people_data.ttl`
- `data/toys/data_some_oneof_people.owl.ttl`
- `data/toys/data_some_oneof_people_data.ttl`
- `data/has_value_people.owl.ttl`
- `data/has_value_people_data.ttl`

Why the table says `limited` outside `gpu-dl`:

- the limitation is about breadth, not uncertainty
- selected nominal/value cases are native, but the broader anonymous and literal-sensitive space is not uniformly enabled across all profiles

### Datatype and literal predicates

Status:

- strongest in `gpu-dl`; absent or narrower in lighter profiles

How it is supported:

- datatype restrictions lower to literal-feature and datatype-check nodes
- the preprocessing pipeline explicitly includes literal-feature extraction where enabled

Validation hooks:

- `data/toys/functional_data_people.owl.ttl`
- `data/toys/functional_data_people_data.ttl`
- generated datatype buckets in `data/runs/coverage-harness/` and `data/runs/consistency-harness/`

Why the table says `no` or `limited` in EL profiles:

- the lighter profiles are intentionally optimized around the EL path rather than the full literal-sensitive path

### Cardinality restrictions

Status:

- supported more broadly in `gpu-dl`, narrower elsewhere

How it is supported:

- the evaluator includes native minimum, maximum, and exact-cardinality-style operators
- these rely on neighbor aggregation rather than tableau search

Validation hooks:

- `data/cardinality_people.owl.ttl`
- `data/cardinality_people_data.ttl`
- generated `geq-cardinality` buckets in `data/runs/consistency-harness/`

Why the table says `limited` outside `gpu-dl`:

- cardinality is not part of the core EL claim
- support exists, but it belongs to the broader experimental fragment rather than the cleanest validated EL story

### Universal restrictions

Status:

- partial / limited support

How it is supported:

- the evaluator includes a native universal-style operator
- it is most defensible in the necessary-condition view, where "all known neighbors satisfy the child" aligns naturally with admissibility-style checking

Validation hooks:

- generated `forall` buckets in `data/runs/consistency-harness/`
- domain/range plus `forall` generated cases under `data/runs/consistency-harness-inspect/`

Why support is limited:

- the limitation is semantic, not accidental
- universals interact more delicately with open-world reasoning and negative completeness than core EL constructs do
- because of that, the repo should describe them as supported in bounded ways rather than claiming uniform classical behavior across all modes

### Unions and disjunction

Status:

- partial / limited support

How it is supported:

- some disjunctive structure can be lowered through helper-backed or normalized forms
- forward materialization can handle selected positive disjunctive patterns more robustly than query-side certification can

Validation hooks:

- `data/union_people.owl.ttl`
- `data/union_people_data.ttl`
- `data/toys/toy_disjunction_chain_schema.ttl`
- `data/toys/toy_disjunction_chain_data.ttl`
- `data/toys/toy_disjunction_branching_schema.ttl`
- `data/toys/toy_disjunction_branching_data.ttl`

Why support is limited:

- query/admissibility certification over disjunction is not fully native today
- `src/test_gpu_dl_toys.py` explicitly marks some disjunction cases as expected failures with reasons such as:
  - "Disjunctive negative-side certification is not yet supported natively."
  - "Tableau-style branching over disjunction is not yet supported natively."

This is an intentional frontier, not an undocumented gap.

### Negative class effects, disjointness, and complements

Status:

- partial / limited support, strongest in stratified negative/blocker handling

How it is supported:

- explicit negative assertions, named disjointness, and exact named complement-style effects are handled through the negative/blocker-oriented path
- the system uses conservative certification and suppression rather than unrestricted negation propagation

Validation hooks:

- `data/negation_people.owl.ttl`
- `data/negation_people_data.ttl`
- `data/toys/toy_negative_assertion_schema.ttl`
- `data/toys/toy_negative_assertion_data.ttl`
- `data/toys/toy_negative_conclusion_schema.ttl`
- `data/toys/toy_negative_conclusion_data.ttl`

Why support is limited:

- unrestricted negative reasoning is not the current contract
- in the toy suite, negative class reasoning is explicitly skipped in some query-style modes because it is only implemented for stratified mode so far

### Target-role saturation, inverse roles, chains, and broader role reasoning

Status:

- partial / profile-dependent support

How it is supported:

- the preprocessing planner can enable target-role materialization where role axioms make it useful
- some broader role effects are surfaced through preprocessing rather than a single unified compiled operator family

Validation hooks:

- `data/role_saturation_people.owl.ttl`
- `data/role_saturation_people_data.ttl`
- `data/inverse_people.owl.ttl`
- `data/inverse_people_data.ttl`

Why support is limited:

- role machinery is more profile- and planning-sensitive than the core EL class fragment
- the limitation means "selectively enabled and bounded," not "unspecified"

### SCC handling and cyclic target dependencies

Status:

- supported

How it is supported:

- named-class dependency analysis computes SCCs and cycle reachability
- acyclic groups can be merged into super-DAG batches
- cyclic helper-relevant slices can be isolated and iterated to stabilization when needed

Validation hooks:

- `data/toys/toy_superdag_acyclic_schema.ttl`
- `data/toys/toy_superdag_acyclic_data.ttl`
- `data/toys/toy_superdag_scc_schema.ttl`
- `data/toys/toy_superdag_scc_data.ttl`
- `data/toys/toy_superdag_multiscc_schema.ttl`
- `data/toys/toy_superdag_multiscc_data.ttl`

Why one toy is skipped in the regression runner:

- `toy_superdag_multiscc` is skipped there because Openllet struggles on that hard SCC toy, not because TensorKG lacks SCC handling

### Unsupported frontier: general branching / tableau-style reasoning

Status:

- unsupported

Why it is not supported:

- general branching is exactly the style of reasoning the current GPU-oriented architecture is trying not to simulate heuristically
- forcing tableau-style case splitting into the native execution path would weaken the clarity of the current soundness-oriented contract

How the repo documents this:

- unsupported cases are called out explicitly in the toy suite rather than being silently treated as ordinary failures

## What We Can Safely Claim Today

### EL side

The EL-oriented profiles are the most mature part of the system. They are the right place to make the strongest coverage, correctness, performance, and native-execution claims.

In particular, the repo's intended positive claim is:

- complete OWL 2 EL reasoning in the validated EL execution path
- agreement with ELK on the tested OWL2Bench runs
- additional native support for some equality- and preprocessing-related effects that classical ELK-style comparisons do not always foreground

### `gpu-dl`

`gpu-dl` is best described as:

- native
- incomplete
- soundness-oriented
- strongest in `stratified`

It supports a broader OWL-DL-like fragment than the EL profiles, but the repo should not claim complete OWL 2 DL reasoning.

### Query-style modes

`admissibility` and `filtered_admissibility` are designed to be conservative:

- if they emit a result, that result is intended to have passed the currently supported certification path
- if they suppress a result, that often means `unknown`

## Preprocessing Pipeline

The codebase uses a staged preprocessing pipeline before execution. Depending on profile and ontology features, a run may include:

1. schema cache extraction, including static schema-side equality reasoning
2. preprocessing-plan selection
3. selected closure passes such as hierarchy, domain/range, reflexive-role, target-role, `sameAs`, or `HasKey`
4. graph lowering, canonicalization, helper generation, and literal-feature extraction
5. named-class dependency analysis and SCC detection
6. DAG compilation
7. merged-root or SCC-aware execution planning
8. layered CPU/GPU DAG evaluation
9. mode-specific stabilization, filtering, or blocker handling

Cycle handling is explicit:

- acyclic target groups can be merged into shared super-DAG execution batches
- cyclic target groups are identified through SCC analysis
- relevant monotone helper-cycle slices can be iterated to a fixpoint when needed

## Where To Look In The Code

- `src/oracle_compare.py`: mode orchestration and comparisons
- `src/ontology_parse.py`: preprocessing, helper generation, SCC analysis, DAG lowering
- `src/dag_eval.py`: native score operators
- `specs/gpu_dl_current_guarantees.md`: current conservative wording for `gpu-dl`

## Short Version

If you only need the bottom line:

- `stratified` is the strongest current reasoning mode
- EL profiles are where the repo makes its strongest positive claim: complete OWL 2 EL reasoning
- `gpu-dl` is experimental but native and increasingly robust
- scored semantic alignment is a structural compatibility score, not a probability and not a substitute for complete OWL entailment
