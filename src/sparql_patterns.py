from __future__ import annotations

from typing import Optional, List

from rdflib.namespace import RDF

from .random_graphs import PathPattern
from .rdf_export import KGraphMapping


def path_pattern_to_sparql(
    pattern: PathPattern,
    mapping: KGraphMapping,
    start_var: str = "v",
    use_prefixes: bool = True,
    prefix_ex: str = "ex",
) -> str:
    """
    Compile a PathPattern into a SPARQL SELECT query that returns
    all starting nodes ?{start_var} that *fully* satisfy the pattern.

    Pattern:
      steps = [(p0, Ck0), (p1, Ck1), ..., (p_{L-1}, Ck_{L-1})]

    Semantics of the query:

      SELECT DISTINCT ?v WHERE {
        ?v    <p0> ?x1 .
        ?x1   rdf:type <Ck0> .
        ?x1   <p1> ?x2 .
        ?x2   rdf:type <Ck1> .
        ...
        ?xL   rdf:type <Ck_{L-1}> .
      }

    i.e., ?v has a path of length L via those properties, landing
    on nodes with the specified classes at each step. This corresponds
    to the "fully matching path" case (score = L, or normalized score=1.0)
    in your engine.

    Parameters:
      - pattern: PathPattern with .steps = [(prop_idx, class_idx), ...]
      - mapping: KGraphMapping used when exporting the KGraph to RDF.
      - start_var: name of the SPARQL variable for the starting node (default "v").
      - use_prefixes: if True, emits PREFIX lines and uses CURIEs (ex:pn, ex:Cn).
                     if False, uses full IRIs in angle brackets (<...>).
      - prefix_ex: the prefix label for the base namespace (default "ex").

    Returns:
      A SPARQL query string.
    """
    steps = pattern.steps
    L = len(steps)
    if L <= 0:
        raise ValueError("PathPattern must have at least one step to compile to SPARQL.")

    # --- Prefixes ---
    lines: List[str] = []

    base_ns = mapping.base_ns  # e.g. http://example.org/random#
    rdf_ns = RDF

    if use_prefixes:
        lines.append(f"PREFIX {prefix_ex}: <{str(base_ns)}>")
        lines.append(f"PREFIX rdf: <{str(rdf_ns)}>")

        def prop_term(prop_idx: int) -> str:
            # mapping.prop_iris[prop_idx] should be something like base_ns["p0"]
            local = str(mapping.prop_iris[prop_idx]).split(str(base_ns))[-1]
            return f"{prefix_ex}:{local}"

        def class_term(class_idx: int) -> str:
            local = str(mapping.class_iris[class_idx]).split(str(base_ns))[-1]
            return f"{prefix_ex}:{local}"

    else:
        def prop_term(prop_idx: int) -> str:
            return f"<{str(mapping.prop_iris[prop_idx])}>"

        def class_term(class_idx: int) -> str:
            return f"<{str(mapping.class_iris[class_idx])}>"

        lines.append(f"PREFIX rdf: <{str(rdf_ns)}>")

    # --- SELECT header ---
    start_var_q = f"?{start_var}"
    lines.append("")
    lines.append(f"SELECT DISTINCT {start_var_q} WHERE {{")

    # --- Body: chain of triple patterns ---
    # Variables: ?v (start), then ?x1, ?x2, ..., ?xL
    prev_var = start_var_q
    for i, (prop_idx, class_idx) in enumerate(steps):
        step_var = f"?x{i+1}"  # ?x1, ?x2, ..., ?xL

        p_term = prop_term(prop_idx)
        c_term = class_term(class_idx)

        # Property edge
        lines.append(f"  {prev_var} {p_term} {step_var} .")
        # Type constraint
        lines.append(f"  {step_var} rdf:type {c_term} .")

        prev_var = step_var

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


from typing import List
from .rdf_export import KGraphMapping
from .random_graphs import PathPattern  # adjust import if PathPattern is elsewhere


def path_pattern_to_sparql_partial_score(
    pattern: PathPattern,
    mapping: KGraphMapping,
    start_var: str = "v",
) -> str:
    """
    Generate a SPARQL query that computes a partial-typed-path similarity score
    for each node ?{start_var}, using:

      - paths: ?v p0 ?x1 . ?x1 p1 ?x2 . ... ?x_{L-1} p_{L-1} ?xL
      - for each path, path_score = (# of matching class constraints) / L
      - for each ?v, score(v) = COALESCE( MAX(path_score over all paths), 0.0 )

    Nodes with no matching path get score 0.0.
    """

    steps: List[tuple[int, int]] = pattern.steps
    L = len(steps)
    if L == 0:
        raise ValueError("PathPattern must have at least one step")

    lines: List[str] = []
    lines.append("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
    lines.append("")

    v_var = f"?{start_var}"
    score_var = "?score"
    path_score_var = "?path_score"

    lines.append(f"SELECT {v_var} (COALESCE(MAX({path_score_var}), 0.0) AS {score_var})")
    lines.append("WHERE {")
    # "All nodes" – here: any subject of at least one triple
    lines.append(f"  {v_var} ?any ?any2 .")
    lines.append("")
    lines.append("  OPTIONAL {")

    for i, (prop_idx, class_idx) in enumerate(steps, start=1):
        prop_iri = mapping.prop_iris[prop_idx]
        class_iri = mapping.class_iris[class_idx]

        x_prev = v_var if i == 1 else f"?x{i-1}"
        x_curr = f"?x{i}"
        t_curr = f"?t{i}"
        sum_prev = "0.0" if i == 1 else f"?sum{i-1}"
        sum_curr = f"?sum{i}"

        lines.append(f"    {x_prev} <{prop_iri}> {x_curr} .")
        lines.append(f"    {x_curr} rdf:type {t_curr} .")
        lines.append(
            f"    BIND( {sum_prev} + IF({t_curr} = <{class_iri}>, 1.0, 0.0) AS {sum_curr} )"
        )
        lines.append("")

    lines.append(f"    BIND( ?sum{L} / {float(L):.1f} AS {path_score_var} )")
    lines.append("  }")
    lines.append("}")
    lines.append(f"GROUP BY {v_var}")

    return "\n".join(lines)
