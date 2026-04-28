# Pipeline Summary

This repository does not implement just one execution path. It implements a shared loading/preprocessing foundation and then three closely related engine modes:

- `query`: evaluate necessary-condition DAGs for target classes over the current graph.
- `filtered_query`: run `query`, then repeatedly recheck provisional assignments until they are self-stable, then prune them with the negative/blocker pass.
- `stratified`: run a positive sufficient-condition fixpoint first, then apply a negative blocker stratum plus a conflict policy.

The code for almost all of this lives in [`src/ontology_parse.py`](./src/ontology_parse.py), with the mode orchestration in [`src/oracle_compare.py`](./src/oracle_compare.py) and the GPU evaluator in [`src/dag_eval.py`](./src/dag_eval.py).

## Category Legend

- `(1)` TBox-level preprocessing: reusable across different ABoxes.
- `(2)` ABox-level preprocessing: reusable across multiple iterations as long as the working ABox is unchanged.
- `(3)` Per-iteration preprocessing: CPU work that must run again when inferred facts are fed back into the graph.
- `(4)` DAG evaluation: graph/DAG scoring that can run on the GPU.

Several stages are hybrid. When that happens, the report calls out both the current implementation and the more ideal reassignment if parts of the semantic feature set were dropped or refactored.

## High-Level Pipeline Skeleton

1. Build reusable schema-side caches from the ontology.
2. Inspect the ontology and target set to choose which preprocessing passes to enable.
3. Apply enabled ABox preprocessing passes to the current working graph.
4. Convert the preprocessed RDF graph into the indexed `KGraph` representation.
5. Build target/dependency/compiler context.
6. Compile target semantics into DAGs.
7. Evaluate DAGs on the `KGraph`.
8. Depending on mode:
   - return direct query scores,
   - iterate a synchronous recheck loop (`filtered_query`),
   - or iterate a positive sufficient-condition materialization loop (`stratified`).
9. Optionally apply the negative blocker stratum and a conflict/emission policy.

## Stage 1: Schema Cache Extraction

Purpose: extract every reusable ontology-side summary that can be computed once from the schema and then reused across many data graphs.

Implementation: `build_reasoning_build_cache(...)` computes:

- transitive `rdfs:subPropertyOf` closure,
- parsed property axioms from `rdfs:domain` and `rdfs:range`,
- atomic and Horn-safe domain/range consequents,
- Horn-safe named-class consequents from subclass/equivalence axioms,
- a preprocessing-oriented class-super map,
- singleton nominal consequences,
- `owl:hasKey` axioms,
- reflexive properties,
- `owl:AllDifferent` / `differentFrom` pair summaries.

This is the first place where the implementation separates "what the ontology says" from "what the current graph contains". The cache is deliberately shaped to feed later passes without reparsing the full RDF graph every time.

Constructs or semantic capabilities dependent on this stage:

- subclass/subproperty preprocessing,
- domain/range preprocessing,
- Horn-safe helper materialization,
- singleton nominal equality effects,
- `hasKey`-based equality,
- reflexive property materialization,
- negative/equality bookkeeping involving `AllDifferent`.

Performance / stage interaction:

- Mostly CPU parsing/indexing over the schema.
- Strong positive effect on later stages because it prevents repeated graph scans.
- If the schema is fixed and many ABoxes are evaluated, this is one of the best places to cache aggressively.
- The more expressive the ontology, the more important this cache becomes.

Category assignment:

- Current implementation: `(1)`.
- Reassignment if semantics are dropped: if equality, domain/range, reflexivity, and Horn-safe helper inference were removed, this stage would shrink to a much smaller `(1)` hierarchy/compiler metadata pass.

## Stage 2: Preprocessing Plan Selection

Purpose: decide which preprocessing passes are worth enabling for the current workload instead of always running every pass.

Implementation: `plan_reasoning_preprocessing(...)` inspects the ontology for:

- hierarchy axioms,
- domain/range axioms,
- negative constructs,
- equality constructs,
- reflexive properties,
- role axioms,
- whether target classes were supplied.

It then emits a `PreprocessingPlan` with explicit decisions for:

- hierarchy materialization,
- atomic domain/range materialization,
- Horn-safe domain/range materialization,
- explicit sameAs-style materialization,
- `owl:hasKey`-driven equality generation,
- reflexive property materialization,
- target-role saturation,
- query-time domain/range augmentation.

This stage is important because later code branches on the plan rather than on raw ontology inspection.

Constructs or semantic capabilities dependent on this stage:

- all optional preprocessing features,
- auto/on/off policy behavior,
- target-aware role saturation,
- query-time domain/range augmentation.

Performance / stage interaction:

- Cheap CPU work compared to graph rewriting and DAG evaluation.
- Large indirect performance impact because it can suppress expensive passes.
- Especially important for preventing unnecessary role saturation, explicit sameAs expansion, and `hasKey`-driven equality generation.

Category assignment:

- Current implementation: mostly `(1)`, but target-dependent.
- If target-specific planning were separated from ontology-wide planning, the ontology-only portion would be pure `(1)` and the target-sensitive portion would still be schema-side control metadata rather than real ABox work.

## Stage 3: Initial ABox Preprocessing and Closure Construction

Purpose: rewrite the working data graph into a form that makes later evaluation simpler, more local, and more reusable.

Implementation: `build_reasoning_dataset_from_graphs(...)` applies enabled passes in roughly this order:

- ontology/schema+data merge bookkeeping for the current refresh,
- reflexive property materialization,
- hierarchy materialization,
- Horn-safe or atomic domain/range materialization,
- hierarchy replay after domain/range additions,
- explicit sameAs-style closure materialization, possibly iterated with replay of positive preprocessing,
- optional `hasKey`-driven equality generation inside that same closure pass,
- target-role closure materialization if role axioms matter for the requested targets.

The sameAs pass is lightweight rather than destructive: it expands facts across discovered equivalence classes instead of fully canonicalizing nodes. Role saturation is target-aware rather than global: only properties relevant to the queried targets are expanded.

Timing alignment:

- Each dataset refresh reports a single aggregate `preprocessing/dataset build` time.
- Inside that total, the engine now reports:
  - `merge`,
  - `hierarchy`,
  - `atomic domain/range`,
  - `horn-safe domain/range`,
  - `sameAs`,
  - `reflexive`,
  - `target roles`.
- Because sameAs may itself rerun multiple times inside one refresh, the engine also reports a per-refresh `sameAs passes:` list.

Constructs or semantic capabilities dependent on this stage:

- `rdfs:subClassOf`,
- `rdfs:subPropertyOf`,
- `rdfs:domain`,
- `rdfs:range`,
- `owl:sameAs`,
- singleton nominal equality effects,
- `owl:hasKey`,
- `owl:ReflexiveProperty`,
- `owl:inverseOf`,
- `owl:propertyChainAxiom`,
- `owl:TransitiveProperty`.

Performance / stage interaction:

- This is one of the most expensive CPU stages because it can grow the working graph.
- Positive effect: later DAG evaluation becomes simpler because many consequences are already explicit.
- Negative effect: graph blow-up increases memory use, KGraph size, and downstream evaluation cost.
- `hasKey`-driven equality and role saturation are the most likely sources of multiplicative growth.
- Target-aware role saturation is a major optimization because it avoids full-property closure.

Category assignment:

- Current implementation: `(2)` for a fixed working ABox.
- In iterative modes, any pass that must be replayed after newly inferred types effectively contributes to `(3)` as well.
- If support for equality and role axioms were dropped, most of the expensive replay logic would disappear and more of this stage would behave like a stable `(2)` preprocessing step.

## Stage 4: RDF Mapping and KGraph Construction

Purpose: convert RDF terms and triples into the tensor-friendly graph representation used by the evaluator.

Implementation:

- `build_rdflib_mapping(...)` assigns stable indices for nodes, properties, classes, and datatypes.
- `rdflib_graph_to_kgraph(...)` builds:
  - one CSR adjacency per property,
  - a `node_types` matrix,
  - lifted literal nodes when literals are included,
  - datatype and numeric metadata for literal nodes,
  - helper edges for negative property assertions where needed.

The `KGraph` format itself is defined in [`src/graph.py`](./src/graph.py): per-property CSR plus type and literal metadata tensors.

Timing alignment:

- The engine reports this stage as the `kgraph build` subcomponent of each dataset refresh.
- In iterative modes, this substage is shown once per refresh inside the per-iteration timing lines.

Constructs or semantic capabilities dependent on this stage:

- every DAG operator,
- datatype predicates,
- cardinality restrictions,
- nominals,
- exact value checks,
- negative property helper edges.

Performance / stage interaction:

- Usually the last major CPU cost before GPU evaluation.
- Bigger preprocessed graphs mean bigger CSR arrays and larger `node_types`.
- This stage controls memory layout, so it strongly affects DAG evaluation throughput.
- In iterative materialization, avoiding full rebuilds here is extremely valuable; the code already tries to reuse KGraphs and update types incrementally in `_build_reasoning_dataset_from_preprocessed_graph(...)`.

Category assignment:

- Current implementation: `(2)`.
- In iterative modes it partially becomes `(3)` because the graph must be rebuilt or refreshed after new type assertions.
- If the system were restricted to pure direct queries with no feedback loops, this would remain a clean `(2)` stage.

## Stage 5: Dependency Analysis and Target Relevance Closure

Purpose: compute which classes and properties actually matter for a target set, and detect class-dependency cycles that affect execution strategy.

Implementation:

- `analyze_named_class_dependencies(...)` computes class equivalence groups, canonical representatives, direct dependencies, strongly connected components, and whether classes reach cycles.
- `compute_target_dependency_closure(...)` and `compute_sufficient_rule_dependency_closure(...)` compute target-relevant helper classes and referenced properties.
- `build_ontology_compile_context(...)` packages direct subproperties, inverses, chains, transitive props, functional props, sameAs expansions, property axioms, and subclass supers for the compiler.

This is the stage that lets later passes stay targeted instead of global.

Constructs or semantic capabilities dependent on this stage:

- cycle-aware query support,
- target-restricted role saturation,
- target-restricted positive sufficient-condition materialization,
- equivalence-aware compilation,
- property-chain, inverse, and transitive compilation support.

Timing alignment:

- In stratified mode, the engine reports the one-off rule/dependency work as:
  - `stage 5 target/dependency prep`,
  - with subfields `rule extraction`, `rule index`, `dependency closure`, and `sameAs state init`.

Performance / stage interaction:

- Mostly CPU graph analysis over the schema.
- Positive effect on nearly every downstream stage because it shrinks the set of classes/properties that need helper work.
- Also determines when special cycle-handling loops are needed in `query` and `filtered_query` modes.

Category assignment:

- Current implementation: `(1)`.
- If only acyclic class definitions were supported, the cycle analysis portion would shrink or disappear, but relevance closure would still remain a valuable `(1)` stage.

## Stage 6A: Necessary-Condition DAG Compilation

Purpose: compile target class definitions into executable DAGs for the direct query path.

Implementation: `compile_class_to_dag(...)` turns the supported OWL-like fragment into `ConstraintDAG` nodes. Supported features include:

- atomic classes,
- nominals and `owl:oneOf`,
- datatype restrictions,
- `owl:unionOf`,
- `owl:intersectionOf`,
- `owl:complementOf`,
- direct `owl:disjointWith`,
- `someValuesFrom`,
- `allValuesFrom`,
- `hasValue`,
- `hasSelf`,
- min/max/exact cardinality,
- inverse-property traversal,
- subproperty expansion,
- property chains,
- transitive existentials,
- optional query-time domain/range augmentation.

The compiler also memoizes repeated subexpressions and uses the compile context so role semantics can be inlined into the DAG shape.

Constructs or semantic capabilities dependent on this stage:

- essentially the entire query-mode semantic fragment.

Performance / stage interaction:

- CPU-bound.
- Compilation cost scales with target count and expression complexity, not with graph size.
- Positive effect: once compiled, the evaluator sees a compact layered DAG rather than raw RDF syntax.
- Negative effect: the current compilation is concrete, not abstract; it depends on mapping indices, so DAGs cannot be fully reused across arbitrary ABoxes.

Category assignment:

- Current implementation: hybrid `(1)+(2)`, but practically closer to `(2)` because compilation needs the concrete mapping.
- Reassignment opportunity: if the system introduced an abstract symbol-based IR first and only performed a late index-binding pass, most of this stage could move cleanly to `(1)`.
- If nominals and literal value requirements were dropped, the ABox dependence would shrink further.

## Stage 6B: Positive Sufficient-Condition Rule Extraction and DAG Compilation

Purpose: compile the positive Horn-friendly stratum used by the `stratified` engine mode.

Implementation:

- `collect_normalized_sufficient_condition_rules(...)` extracts a normalized rule set from subclass/equivalence/domain/range axioms.
- Antecedents are represented as `NormalizedSufficientCondition` trees over a smaller positive fragment.
- `index_normalized_sufficient_rules_by_consequent(...)` groups antecedents by target class.
- `compile_sufficient_condition_dag(...)` compiles the disjunction of all antecedents for a target class into one executable DAG.

This stage is deliberately different from Stage 6A. The direct query compiler tries to represent necessary conditions for a class. The sufficient-condition compiler instead asks: "what positive evidence is enough to emit this class in the Horn-friendly fragment?"

Constructs or semantic capabilities dependent on this stage:

- Horn-style positive materialization,
- sufficient-condition inference from subclass/equivalence/domain/range,
- positive support for atomic classes, nominals, datatype checks, `hasSelf`, existentials, intersections, and min-cardinality antecedents.

Performance / stage interaction:

- CPU-bound.
- Strong positive effect in iterative materialization because the rule set can be built once and reused across iterations.
- The code already caches compiled sufficient-condition DAGs per class during materialization.

Category assignment:

- Current implementation: again hybrid `(1)+(2)`.
- Logically, the rule extraction portion is pure `(1)`.
- The final index-binding into concrete DAG node/property/class indices is `(2)`.
- If nominals/literal-specific antecedents were removed, more of the DAG compilation could be hoisted into `(1)`.

## Stage 7: GPU DAG Evaluation

Purpose: score every DAG node over every graph node efficiently.

Implementation: `eval_dag_score_matrix(...)` in [`src/dag_eval.py`](./src/dag_eval.py) evaluates the DAG layer by layer. Core operators include:

- atomic class lookup,
- nominal match,
- datatype predicate evaluation,
- negation,
- self restrictions,
- existential and transitive existential propagation,
- universal restriction via segmented minimum,
- min/max/exact cardinality via top-k neighborhood scores,
- intersection and union,
- typed path steps.

The implementation precomputes oriented edge lists and cached segment layouts so repeated reductions over the same property are cheaper.

Constructs or semantic capabilities dependent on this stage:

- all compiled DAG semantics,
- exact query evaluation,
- sufficient-condition evaluation,
- partial/fuzzy behavior where the DAG operators use non-binary aggregation.

Performance / stage interaction:

- This is the main `(4)` stage and the main place where GPU acceleration matters.
- Runtime scales primarily with:
  - number of nodes,
  - number of relevant edges,
  - DAG size,
  - the presence of operators like transitive existential or cardinality that require heavier neighborhood processing.
- Positive preprocessing can reduce DAG complexity but may enlarge the graph; performance is a tradeoff between richer explicit closure and bigger tensors.

Category assignment:

- Current implementation: `(4)`.
- If the system dropped expensive operators like transitive existential or cardinality, this stage would become simpler and more regular, but it would still be `(4)`.

## Stage 8: Query Snapshot Evaluation and Monotone Cycle Stabilization

Purpose: execute one full direct-query pass, including special handling for reachable monotone named-class cycles.

Implementation: `_evaluate_query_snapshot(...)` in [`src/oracle_compare.py`](./src/oracle_compare.py):

- builds a preprocessed dataset,
- compiles/evaluates helper cycle classes if needed,
- for monotone cycle components, seeds provisional memberships and iterates until the cycle stabilizes,
- compiles/evaluates the actual target DAGs,
- returns members and scores per target.

This stage exists because some target definitions are not just one-shot DAG evaluations; they may depend on helper classes in a positive cycle that still converges under a monotone interpretation.

Constructs or semantic capabilities dependent on this stage:

- cyclic positive named-class dependencies in query mode,
- monotone helper-class bootstrapping,
- general direct query semantics.

Performance / stage interaction:

- Hybrid CPU/GPU stage.
- CPU cost comes from dataset rebuilds and temporary ABox augmentation.
- GPU cost comes from reevaluating the helper and target DAGs.
- On acyclic target sets this stage is much cheaper.

Category assignment:

- Current implementation: `(3)` for the rebuild/reseed logic plus `(4)` for the evaluation.
- If cyclic helper classes were ruled out, this stage would collapse to a single query snapshot and lose most of its `(3)` character.

## Stage 9: Filtered Query Synchronous Recheck

Purpose: start from raw necessary-condition matches, then keep only candidates that remain valid after their own provisional type assignments are fed back into the graph.

Implementation: `run_engine_queries(..., engine_mode="filtered_query")` does:

1. one raw query snapshot,
2. repeatedly augment the original data graph with current provisional target assignments,
3. rerun query evaluation,
4. retract any candidate that no longer scores above threshold,
5. stop at a fixpoint,
6. then run the negative/blocker stratum to remove closure-blocked survivors.

This is a synchronous recheck design, not a pure forward materializer.

Constructs or semantic capabilities dependent on this stage:

- self-stability filtering for necessary-condition assignments,
- pruning of assignments that only looked valid before feedback,
- final combination with the blocker stratum.

Performance / stage interaction:

- Expensive because it repeatedly reruns Stage 8.
- Positive effect: improves semantic conservatism and removes unstable candidates.
- Negative effect: CPU rebuild and GPU reevaluation costs multiply by the number of necessary-fixpoint iterations.

Category assignment:

- Current implementation: `(3)+(4)`.
- If feedback-sensitive query filtering were dropped, this entire stage would disappear and raw query mode could stop after a single Stage 8 pass.

## Stage 10: Positive Sufficient-Condition Materialization Loop

Purpose: compute the positive OWA-style closure used by the `stratified` mode.

Implementation: `materialize_positive_sufficient_class_inferences(...)` does the following:

- build the normalized sufficient-rule set once,
- compute target dependency closure once,
- build preprocessing caches once,
- compile sufficient DAGs lazily and cache them by class,
- repeatedly:
  - rebuild or incrementally refresh the dataset,
  - evaluate all target/helper sufficient DAGs,
  - add any `(node, rdf:type, class)` pairs above threshold,
- rerun equality-sensitive rebuilding only when newly inferred types trigger explicit sameAs or `hasKey`-related classes,
  - stop when no new positive assertions appear.

This is the core forward-chaining stage of the stratified engine.

Constructs or semantic capabilities dependent on this stage:

- positive Horn-style inference,
- chained helper-class derivations,
- target-aware helper inference,
- explicit sameAs-triggered propagation during inference,
- optional `hasKey`-triggered equality generation during inference.

Performance / stage interaction:

- Usually the dominant cost in `stratified` mode.
- CPU costs:
  - dataset rebuild / incremental refresh,
- occasional explicit sameAs / `hasKey` replay,
  - bookkeeping for new type assertions.
- GPU costs:
  - evaluating the cached sufficient DAG bank each iteration.
- Positive optimization already present: DAG compilation is cached; KGraph/type reuse is attempted; equality replay is conditional instead of unconditional.

Timing alignment:

- The engine reports the whole loop as `stratified positive OWA loop`.
- It also reports:
  - total iteration count,
  - average per-iteration `dataset build`, `dag compile`, and `dag eval`,
  - one line per iteration with:
    - refresh count,
    - dataset build total,
    - merge,
    - hierarchy,
    - atomic domain/range,
    - horn-safe domain/range,
    - sameAs,
    - reflexive,
    - target roles,
    - KGraph build,
    - DAG compile,
    - DAG eval,
  - and, when present, the per-iteration `sameAs passes:` list.
- If the loop ends with an additional final dataset rebuild, that refresh is reported separately as `final dataset refresh`.

Category assignment:

- Current implementation: `(3)+(4)`, with some reused substructure from `(1)` and `(2)`.
- Reassignment if semantics are dropped:
- without sameAs/hasKey/nominal-triggered equality, more of the rebuild path becomes cheap,
- without iterative inference itself, the stage disappears and only single-pass query evaluation remains.

## Practical Optimization Outlook

The current implementation already moved a substantial amount of work into the more ideal categories:

- TBox-once:
  - schema cache extraction,
  - preprocessing-plan selection,
  - dependency analysis,
  - sufficient-rule extraction and indexing.
- ABox-once where possible:
  - direct query preprocessing,
  - KGraph construction for one-shot query snapshots.
- DAG evaluation:
  - direct role semantics,
  - necessary-condition checks,
  - sufficient-condition scoring.

The expensive work that most stubbornly remains in `(3)` is the work whose input genuinely changes when new positive type assertions are fed back into the graph:

- hierarchy replay over newly added types,
- Horn-safe domain/range replay over newly added types,
- explicit sameAs / `hasKey` replay when new types activate keyed classes or singleton nominal consequences,
- KGraph refresh after the working graph changes.

Without equality-style reasoning (`sameAs`, singleton-nominal identity effects, `hasKey`), much more of Stage 10 can be pushed toward a stable `(2)` shape:

- dataset refreshes become close to type-delta updates,
- equality-triggered rebuilds disappear,
- hierarchy/domain-range replay becomes easier to keep incremental,
- KGraph reuse becomes dramatically more effective.

By contrast, with equality fully enabled, the strongest remaining reason for `(3)` is that equality can change which graph facts are visible at all, not merely which scores are assigned to already-fixed nodes. That makes explicit sameAs/HasKey work much harder to migrate cleanly into `(4)`, since the DAG evaluator assumes a fixed tensor graph.

In practical terms, the most plausible future performance gains are therefore:

- further shrinking or specializing equality replay,
- improving type-delta refreshes for hierarchy/domain-range,
- reducing how often KGraph refreshes are needed,
- and hoisting more compiler logic into a schema-only abstract IR.

The least plausible large gain is moving full equality bookkeeping into DAG evaluation. That would require the GPU stage to become a graph-rewriting / identity-management engine rather than a scorer over a fixed graph, which is not the current design.

## Stage 11: Negative Blocker Specification Extraction

Purpose: compile the negative fragment into a simple blocker representation that can be checked after positive closure is known.

Implementation: `collect_negative_blocker_specs(...)` extracts for each target class:

- blocker classes from direct/inherited `owl:disjointWith`,
- blocker classes from named `owl:complementOf`,
- singleton nominal blockers,
- exact literal requirements on functional data properties,
- exact-value requirements that can be invalidated by negative property assertions,
- a list of skipped negative axioms outside the supported blocker fragment.

This stage intentionally does not implement a full negative reasoner. It compiles only the negative forms the project can safely check after the positive stratum.

Constructs or semantic capabilities dependent on this stage:

- disjointness blockers,
- simple named complements,
- some nominal exclusion behavior,
- limited `FunctionalDataProperty` blockers,
- limited negative object/data property blockers.

Performance / stage interaction:

- CPU-bound and relatively cheap compared with full graph rewriting.
- Strong positive effect because it converts awkward negative semantics into simple scans over the positive closure.

Category assignment:

- Current implementation: mostly `(1)`.
- The extracted specs are schema-side.
- If the negative fragment were dropped, this stage would disappear entirely.

## Stage 12: Negative Blocker Application

Purpose: take the final positive closure and mark assignments that should be considered blocked or conflicting.

Implementation: `materialize_negative_class_blockers(...)` scans:

- the positive `node_types` closure in the current dataset,
- explicit `differentFrom` facts,
- observed literal values,
- collected negative property assertions.

It produces:

- blocked assertions,
- conflicting positive assertions where a blocked assignment is already present in the positive closure.

Constructs or semantic capabilities dependent on this stage:

- all supported negative/blocker semantics,
- detection of positive/negative conflicts after closure.

Performance / stage interaction:

- CPU scan over the realized closure.
- Cheaper than replaying negative logic during every positive iteration.
- Its cost scales with node count, target count, and blocker richness.

Category assignment:

- Current implementation: `(2)` if you treat the positive closure as a stable dataset, or `(3)` when embedded in a single end-to-end run because it must be rerun after the final positive result is known.
- If all negative semantics were removed, the pipeline could stop after Stage 10 in `stratified` mode.

## Stage 13: Assignment Status Construction and Conflict Policy

Purpose: decide what gets emitted to the caller after both strata have run.

Implementation:

- `collect_assignment_statuses(...)` labels each target assignment as asserted, positively derived, blocked, and/or conflicted.
- `apply_conflict_policy(...)` then chooses one of:
  - `report_only`,
  - `suppress_derived_keep_asserted`,
  - `strict_fail_on_conflict`.

This is the final semantic policy layer. The positive and negative stages compute facts; this stage decides how those facts are used operationally.

Constructs or semantic capabilities dependent on this stage:

- stratified final output behavior,
- conflict reporting,
- conservative emission of derived types,
- strict failure mode for contradiction-sensitive workflows.

Performance / stage interaction:

- Cheap CPU bookkeeping.
- High semantic impact because it determines whether contradictions are merely reported, filtered, or treated as hard failures.

Category assignment:

- Current implementation: `(3)` in an end-to-end run, because it depends on the freshly computed closure and blocker result.
- If the workflow always used a single fixed policy and never exposed suppressed/conflicted assignments, this stage could be simplified heavily, but some output-layer bookkeeping would still remain.

## Practical Reverse-Engineering Notes

If someone wanted to reconstruct the design from scratch, the most important architectural ideas are:

- The real boundary is not "parser vs evaluator"; it is "schema-derived reusable summaries" vs "current working ABox" vs "feedback-driven iterative reruns".
- The project uses two different semantic compilation views:
  - necessary-condition DAGs for direct query scoring,
  - sufficient-condition DAGs for positive Horn-style materialization.
- Preprocessing is not a generic ETL step. It is semantically aware and deliberately selective.
- Role semantics are handled in two ways:
  - native DAG compilation for direct query behavior,
  - target-aware role saturation when reusable explicit closure is more useful.
- Equality is handled as fact expansion over equivalence classes rather than destructive node merging.
- The GPU stage is intentionally narrow: it evaluates fixed DAGs over a fixed tensor graph. Everything difficult about OWL support is pushed into earlier compilation, closure, and control-flow stages.

## Where the Category Boundaries Really Fall

The cleanest persistent partition is:

- `(1)`: schema cache extraction, dependency analysis, rule extraction, most abstract compilation logic.
- `(2)`: preprocessed working ABox, KGraph construction, blocker application against a fixed closure.
- `(3)`: loops that feed inferred types back into the graph and force rebuild/recheck.
- `(4)`: the actual DAG scoring kernel.

The least clean boundary in the current implementation is DAG compilation, because the current compiler binds directly to concrete mapping indices. If the system ever introduces an abstract symbolic DAG IR, a meaningful amount of today's `(2)` compile work could migrate to `(1)`.
