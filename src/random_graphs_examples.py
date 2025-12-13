from __future__ import annotations

import torch

from .graph import KGraph
from .constraints import (
    ConstraintNode,
    ConstraintDAG,
    ConstraintType,
    IntersectionAgg,
)
from .dag_reasoner import DAGReasoner
from .random_graphs import generate_random_kgraph, generate_random_path_pattern
from .patterns import compile_path_pattern_to_dag
from .rdf_export import write_kgraph_as_turtle
from .sparql_patterns import path_pattern_to_sparql



# ✅ (done) GPU-vectorized EXISTS and PATH_STEP

# ✅ Add satisfying_nodes() helper and a clean “exact mode”

# ✅ Graph generator (random KGraph )

# ✅ KGraph → RDF

# ✅ PathPattern → ConstraintDAG compiler

# ✅ PathPattern → SPARQL compiler

# ⬜ RDF → KGraph loader (for real RDF, beyond synthetic)

# ⬜ Triplestore setup + simple Python SPARQL client

# ⬜ Benchmark harness that ties it all together

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    # 3) Random graph + random path pattern
    print("\n=== Random KGraph + random PathPattern (demo) ===")

    num_nodes = 20
    num_props = 3
    num_classes = 4

    rg = generate_random_kgraph(
        num_nodes=num_nodes,
        num_props=num_props,
        num_classes=num_classes,
        avg_degree_per_prop=2.0,
        seed=123,
    )

    pattern = generate_random_path_pattern(
        num_steps=3,
        num_props=num_props,
        num_classes=num_classes,
        seed=456,
        normalize=True,
    )

    print (f"Random graph generated with {num_nodes} nodes, {num_props} properties, {num_classes} classes.")

    print ("Random graph: offsets tensor")
    print( rg.offsets_p )

    print ("Random graph: neighbors tensor")
    print( rg.neighbors_p )

    print("Random graph: classes", torch.argmax(rg.node_types, dim=1))

    p = 1
    offsets = rg.offsets_p[p]
    neigh = rg.neighbors_p[p]

    v = 0
    start = offsets[v].item()
    end = offsets[v+1].item()
    print(f"Neighbors via p{p} from v={v}:", neigh[start:end].tolist())
    v = 3
    start = offsets[v].item()
    end = offsets[v+1].item()
    print(f"Neighbors via p{p} from v={v}:", neigh[start:end].tolist())
    v = 8
    start = offsets[v].item()
    end = offsets[v+1].item()
    print(f"Neighbors via p{p} from v={v}:", neigh[start:end].tolist())
    v = 12
    start = offsets[v].item()
    end = offsets[v+1].item()
    print(f"Neighbors via p{p} from v={v}:", neigh[start:end].tolist())    


    print("Random PathPattern steps (prop_idx, class_idx):", pattern.steps)

    dag = compile_path_pattern_to_dag(pattern)

    rand_reasoner = DAGReasoner(rg, device=device, sim_class=None)
    rand_reasoner.add_concept("RandomPath", dag)
    rand_scores = rand_reasoner.evaluate_all()  # [num_nodes, 1]

    print("\nScores s(v, RandomPath) for each node v:")
    for v in range(rg.num_nodes):
        print(f"  v = {v}: s(v, RandomPath) = {rand_scores[v, 0].item():.4f}")

    satisfying = rand_reasoner.satisfying_nodes("RandomPath", threshold=0.999)
    print("\nNodes with score ~1.0 (fully matching the path):", satisfying)

    # Export random graph to Turtle
    print("\nWriting random KGraph to 'random_graph.ttl' ...")
    mapping = write_kgraph_as_turtle(
        rg,
        path="random_graph.ttl",
        base_uri="http://example.org/random",
        type_threshold=0.5,
    )
    print("Done. Turtle written as 'random_graph.ttl'.")

    # Compile the same PathPattern to SPARQL using the same mapping
    sparql_query = path_pattern_to_sparql(pattern, mapping, start_var="v")
    print("\nEquivalent SPARQL query for fully matching nodes:")
    print(sparql_query)


if __name__ == "__main__":
    main()
