# `gpu-dl` Negative Reasoning Notes

This note records the current semantic plan for negative reasoning in the
experimental `gpu-dl` profile.

The goal is not full OWL DL negation reasoning. The goal is:

- preserve soundness
- stay compatible with the native `gpu-el`-style execution architecture
- avoid broad RDFLib graph crawling in the main reasoning loop
- accept incompleteness where necessary

In short:

- correctness over completeness
- native tensor/KGraph execution over legacy graph mutation


## State Model

For each `(node, class)` pair, we may eventually want four logical channels:

- `C_INFERRED`
- `NOT_C_INFERRED`
- `C_ADMISSIBLE`
- `NOT_C_ADMISSIBLE`

Interpretation:

- `C_INFERRED = 1`
  - the engine has a sound derivation that `C(node)` is entailed in the
    supported fragment
- `NOT_C_INFERRED = 1`
  - the engine has a sound derivation that `not C(node)` is entailed in the
    supported fragment
- `C_ADMISSIBLE = 1`
  - adding `C(node)` is certified to preserve consistency in the supported
    fragment
- `NOT_C_ADMISSIBLE = 1`
  - adding `not C(node)` is certified to preserve consistency in the supported
    fragment

Important:

- `INFERRED` and `ADMISSIBLE` are not the same thing
- admissibility is epistemic / proof-carrying
- a value of `0` means "unknown / not certified", not necessarily false


## Current Safe Fragment

The current negative machinery should remain limited to cases that are native
and clearly sound:

- explicit negative class assertions
  - encoded as class assertion to an anonymous `owl:complementOf` class
- named `owl:disjointWith`
- exact named `owl:complementOf`
- native negative blockers already supported by the stratified blocker pass
  - exact negative property assertions
  - supported functional-data blockers
  - supported nominal blockers

Not currently supported:

- arbitrary contraposition
- general negation-normal-form rewriting over arbitrary query DAGs
- branching disjunction / tableau case splits
- any reasoning that depends on open-world absence
- any claim that unsupported negative consequences are absent


## Safe Update Rules

These are the intended rules for a future unified four-channel implementation.

They are written here first as a semantic contract. Implementation can remain
 staged and partial.


### 1. Positive Inference

`C_INFERRED` is updated by the existing positive sufficient-condition forward
materialization machinery.

Safe examples:

- `A -> B`
  - if `A_INFERRED`, then `B_INFERRED`
- existential / intersection / role-based sufficient rules
  - same as current `gpu-el` / `gpu-dl` positive pass


### 2. Negative Inference

`NOT_C_INFERRED` is updated only by supported negative sources.

Safe rules:

- explicit negative assertion:
  - if input states `not C(n)`, then `NOT_C_INFERRED(n, C) = 1`
- disjointness:
  - if `A disjointWith B` and `A_INFERRED(n)`, then `NOT_B_INFERRED(n) = 1`
  - symmetrically, `B_INFERRED(n) -> NOT_A_INFERRED(n)`
- exact named complement:
  - if `A == not B` and `A_INFERRED(n)`, then `NOT_B_INFERRED(n) = 1`
  - if `B == not A` and `B_INFERRED(n)`, then `NOT_A_INFERRED(n) = 1`

Unsafe and forbidden:

- from `A -> B`, infer `not B -> not A`
- from `A -> B`, infer `not A -> not B`
- from failure to prove `C`, infer `not C`


### 3. Positive Inference From Negative Facts

This is allowed only for exact named complements.

Safe rule:

- if `A == not B` and `NOT_A_INFERRED(n)`, then `B_INFERRED(n) = 1`
- symmetrically, `NOT_B_INFERRED(n) -> A_INFERRED(n)`

This is intentionally narrow.

- disjointness alone does not justify the reverse direction
- arbitrary complement-like query shapes do not justify the reverse direction


### 4. Positive Admissibility

`C_ADMISSIBLE` is set only when admissibility is certified in the supported
negative fragment.

Safe sufficient conditions:

- `C_INFERRED -> C_ADMISSIBLE`
- or:
  - the necessary-side query for `C` succeeds
  - supported blocker / refutation checks for `C` do not fire
  - the negative side of `C` is complete in the supported fragment

If the target's negative side is not complete in the supported fragment:

- do not set `C_ADMISSIBLE`
- leave it unknown


### 5. Negative Admissibility

`NOT_C_ADMISSIBLE` is dual to positive admissibility, but only over the same
supported fragment.

Safe sufficient conditions:

- `NOT_C_INFERRED -> NOT_C_ADMISSIBLE`
- or:
  - the necessary-side query for `not C` succeeds in a supported compiled form
  - supported blocker / refutation checks for `not C` do not fire
  - the positive side needed to refute `not C` is complete in the supported
    fragment

Again, if certification is incomplete:

- do not set `NOT_C_ADMISSIBLE`
- leave it unknown


### 6. Inconsistency Detection

The following indicates a supported-fragment inconsistency:

- `C_INFERRED = 1` and `NOT_C_INFERRED = 1`

This should be treated as a strong signal:

- the engine has found an explicit contradiction in the fragment it knows how
  to reason about

The following is not itself an inconsistency:

- `C_ADMISSIBLE = 1` and `NOT_C_ADMISSIBLE = 1`

This only means:

- each assertion is individually consistency-preserving under the current
  certification rules
- not that both are jointly admissible at the same time


## Why Not Arbitrary Dualization?

It is tempting to say:

- necessary conditions for `C` are the complement of sufficient conditions for
  `not C`

This is only safe when the complement is itself expressible and complete inside
the supported fragment.

For example:

- exact named complement: okay
- direct named disjointness blocker: okay
- arbitrary DAG with disjunction, nested negation, or unsupported constructors:
  not safe to dualize blindly

So in practice:

- the engine must compile a restricted negative-side object
- if that restricted compilation is incomplete, the answer must remain unknown


## Recommended Implementation Plan

The implementation should stay staged.


### Stage 1: Native Negative Materialization In Stratified Mode

Current direction:

- positive forward pass runs as usual
- native negative closure runs afterward using:
  - explicit negative assertions
  - named disjointness
  - exact named complements
- blocker pass then reports conflicts / forbidden assignments

This stage is already the right foundation for `gpu-dl` stratified mode.


### Stage 2: Target-Level Negative Completeness Analysis

Before extending admissibility, add analysis for each target:

- can the relevant negative side of this target be fully represented in the
  supported fragment?

Possible result:

- `negative_fragment_complete_for_target = True/False`

Use this as a gate:

- only emit certified admissibility results when the answer is `True`
- otherwise downgrade to unknown unless direct supported refutation/proof exists


### Stage 3: Conservative Admissibility Filtering

Extend admissibility to use:

- current positive admissibility candidate generation
- plus native negative fact / blocker checks
- plus negative-fragment completeness gating

Conservative rule:

- only report `(node, C)` as admissible if:
  - positive candidate conditions hold
  - no supported refutation fires
  - negative completeness for `C` is certified

This will be incomplete, but sound.


### Stage 4: Unified Four-Channel Engine

Only after the earlier stages are stable should we consider a more unified
engine representation.

Possible directions:

- separate dense tensors for:
  - positive inferred
  - negative inferred
  - positive admissible
  - negative admissible
- or a logically equivalent packed representation

But this should come after the semantic rules are validated on toy cases.


## Design Guidance

- prefer native KGraph/tensor propagation over RDFLib mutation
- do not add generic tableau branching to `gpu-dl`
- do not let unsupported negative reasoning silently default to "safe"
- use `unknown` as the default when certification is incomplete
- keep `gpu-el` behavior untouched


## Immediate Next Steps

1. Keep the current stratified negative materialization narrow and tested.
2. Add target-level negative completeness analysis.
3. Use that analysis to gate `admissibility` results conservatively.
4. Expand support only when the semantic rule is clear and can be proven sound.


## Plan For Currently Skipped Toy Cases

These toys are intentionally hard. The policy for all of them is the same:

- it is acceptable to miss the intended inference or admissibility conclusion
- it is not acceptable to emit a false positive or claim admissibility without
  certification


### `toy_negative_assertion`

Current status:

- supported in `stratified`
- not yet supported in `admissibility` / `filtered_admissibility`

Safe plan:

- only extend beyond `stratified` once the target's negative side can be
  certified complete in the supported fragment
- otherwise leave the result unknown


### `toy_negative_conclusion`

Current status:

- supported in `stratified`
- not yet supported in `admissibility` / `filtered_admissibility`

Safe plan:

- reuse native negative inference as a refutation source
- only emit admissibility conclusions when the relevant negative-side fragment
  is certified complete


### `toy_disjunction_chain`

Why it is skipped:

- disjunctive conclusions appear in the reasoning path
- inference through those disjunctions is not yet handled natively

Safe plan:

- targets depending on these disjunctions should remain negative-incomplete
- admissibility should emit unknown, not a guessed answer
- stratified inference may miss the conclusion entirely until explicit
  disjunctive support exists


### `toy_disjunction_branching`

Why it is skipped:

- reaching the intended conclusion requires genuine branching / case analysis

Safe plan:

- do not attempt to simulate tableau branching with ad hoc heuristics
- mark the relevant targets unsupported / incomplete
- emit unknown rather than a speculative inference or admissibility judgment


### `toy_superdag_multiscc`

Why it is skipped:

- Openllet is too slow on this case for routine regression use

Safe plan:

- continue to use it as an occasional stress/correctness case
- do not require it in the fast regression suite
