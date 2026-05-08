# `gpu-dl` Current Claims, Guarantees, and Known Gaps

This note records what we can honestly claim about the current `gpu-dl`
implementation as of now.

It is intentionally conservative.

- if a behavior is only partially implemented, say so
- if a result is intended to be sound but incompletely certified, say so
- if a case is unsupported, prefer `unknown` / omission to a false positive


## Short Version

Current `gpu-dl` is best described as:

- a native, incomplete, soundness-oriented OWL-DL-like reasoning framework
- strongest today in `stratified` mode
- increasingly conservative and certification-driven in
  `admissibility` / `filtered_admissibility`

The core project claim that is already defensible is:

- when `gpu-dl` declines to conclude something, that is acceptable
- when `gpu-dl` does conclude something in the currently supported fragment, it
  is intended to be supported by explicit native machinery rather than heuristic
  fallback


## What We Can Claim Today


### 1. Native-First Architecture

The active implementation direction is now genuinely native-first:

- KGraph/tensor execution is the default reasoning substrate
- super-DAG execution is integrated into the native loop
- sameAs support is native and canonicalization-based
- HasKey support is native and opt-in
- negative stratified reasoning is implemented without falling back to a legacy
  RDFLib closure loop
- some anonymous disjunctive consequents are now handled through stable native
  helper classes

This means our main semantic experiments are no longer blocked on the old
default/RDFLib-heavy architecture.


### 2. `stratified` Is The Strongest Current Mode

The soundness story is strongest in `stratified`.

Currently supported in a native way:

- positive sufficient-condition forward materialization
- sameAs canonicalization
- HasKey equality generation when explicitly enabled
- safe negative materialization:
  - explicit negative class assertions
  - named `owl:disjointWith`
  - exact named `owl:complementOf`
- supported inconsistency detection when both positive and negative facts are
  derived in the supported fragment
- helper-backed forward propagation through at least some anonymous
  disjunctive consequents

Practical claim:

- `stratified` is currently the best-supported place to make a soundness claim
  for `gpu-dl`


### 3. Admissibility Modes Are Conservative By Design

`admissibility` and `filtered_admissibility` now use a more conservative policy
than earlier versions.

They currently:

- compute positive admissibility-style candidates natively
- compute target-level negative-fragment support/completeness information
- suppress output when the negative side is not certified complete
- prune candidates using currently supported native negative blockers and
  refutations

Current intended meaning:

- emitted results are meant to be the subset we can currently certify
- omitted results should be interpreted as `unknown`, not necessarily false

This is the right direction for paper-grade soundness, but it is not yet as
strongly mature as `stratified`.


## Current Guarantees


### Guaranteed Design Policy

The main guarantee we are actively enforcing is:

- when the engine cannot certify a conclusion in the supported fragment, it
  should prefer omission / unknown over a positive claim

This policy is already visible in:

- negative-fragment completeness gating in admissibility modes
- intentional `XFAIL` handling for hard disjunction / branching toys
- refusal to use arbitrary contraposition in negative reasoning


### Guaranteed Unsupported Areas Do Not Count As Proven Results

We do **not** currently claim support for:

- tableau-style branching
- arbitrary disjunctive reasoning in admissibility/query mode
- arbitrary negative dualization
- arbitrary open-world absence-based refutation

So:

- if the engine emits nothing there, that is consistent with the contract
- if the engine emits something there without certification, that would be a
  bug


### Current Toy-Suite Guarantee

The toy suite is now structured to support the soundness claim:

- supported cases must pass
- deliberately unsupported-but-interesting cases are tracked as `XFAIL`
- hard oracle-pathological cases can remain skipped when they are not useful as
  fast regressions

Current broad status:

- supported regression cases pass
- disjunctive / branching frontier cases are visible, not hidden


## What We Should Not Claim Yet


### 1. Full Soundness Claim For All Admissibility Outputs

We are moving toward this, but should not overstate it yet.

Why:

- target-level negative completeness is still coarse
- some query-mode filtering still depends on current fragment guards rather than
  richer proof objects
- branching/disjunctive admissibility cases are still fenced off rather than
  positively certified

Safer statement:

- admissibility-style modes are currently conservative and intended to be
  sound-by-omission, but the certification framework is still being expanded


### 2. General ALC/DL Completeness

We should not claim completeness for:

- disjunction
- negation
- branching
- arbitrary cyclic query interaction

At best, the correct claim is:

- partial support for a restricted, tractable subset of DL behavior


### 3. Arbitrary Anonymous-Class Materialization

We now support some anonymous helper-backed cases, especially around supported
unions, but we should not claim:

- arbitrary anonymous class materialization
- arbitrary anonymous-class dependencies in all modes


## Current Unsupported / Partially Supported Frontier


### Negative Reasoning

Supported:

- explicit negative class assertions
- named disjointness
- exact named complement

Not yet generally supported:

- broad negative admissibility
- arbitrary negative query compilation
- arbitrary propagation through unsupported negated structure


### Disjunction

Supported:

- union in normalized sufficient conditions
- native forward use of helper-backed anonymous disjunctive consequents in at
  least some `stratified` cases

Not yet generally supported:

- admissibility certification for disjunctive negative-side targets
- branching/case-split reasoning
- general disjunctive query support under cyclic helper conditions


### Branching

Still unsupported.

This should remain a bright red line:

- if a case fundamentally requires tableau-style branching, we should prefer
  `unknown` rather than attempt heuristic simulation


## Where To Push Next For More Robustness

These are the best next robustness targets if the goal is a stronger
incomplete-but-sound claim.


### 1. Richer Negative Completeness Metadata

Current completeness is a boolean plus a short reason.

We should move toward:

- per-target structured certification reasons
- per-target unsupported-constructor summaries
- explicit distinction between:
  - unsupported because of disjunction
  - unsupported because of branching risk
  - unsupported because blocker extraction is incomplete

This would make admissibility suppression easier to justify in the paper.


### 2. Proof-Like Admissibility Emission Criteria

Instead of just filtering candidates, we should gradually move toward:

- explicit certification for why each emitted admissibility result survived

For example:

- positive necessary condition satisfied
- no supported blocker fired
- target negative fragment certified complete

That would make the soundness story substantially stronger.


### 3. Helper-Aware Query/Admissibility Support

We have now improved helper-backed forward reasoning.

Next robustness step:

- decide which helper-backed anonymous constructs can safely participate in
  admissibility certification
- keep disallowing any case that would require branching


### 4. Clear Mode-Specific Claims

We should explicitly separate claims for:

- `stratified`
- `admissibility`
- `filtered_admissibility`

Right now, the modes do not all deserve equally strong claims.

The likely final shape is:

- strongest claim: `stratified`
- qualified conservative claim: `admissibility`
- even more qualified conservative claim: `filtered_admissibility`


## Recommended Wording For Current Paper/Repo Claims

A safe current formulation would be something like:

> `gpu-dl` currently provides a native, soundness-oriented, incomplete reasoning
> framework for a restricted OWL-DL-like fragment. The implementation is
> strongest in stratified materialization, and admissibility-style modes employ
> conservative certification and may suppress conclusions outside the currently
> supported negative and disjunctive fragment.

And, if we want to be even more explicit:

> The system is designed to prefer omission over false positive conclusions when
> reasoning leaves the currently certified fragment.


## Immediate Next Steps

1. Keep strengthening the explicit certification story for admissibility modes.
2. Preserve the current "unknown over unsafe" discipline.
3. Continue expanding support only where the semantic rule is clear enough to be
   defended as sound.
4. Treat branching/tableau-style cases as unsupported until there is a principled
   design for them.

