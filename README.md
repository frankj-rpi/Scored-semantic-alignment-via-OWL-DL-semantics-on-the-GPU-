# Scored-semantic-alignment-via-OWL-DL-semantics-on-the-GPU-
CSCI 6960 project.

example execution:
```
python -m src.compare_exact   --num-nodes 10000   --num-props 1000   --num-classes 1000   --avg-degree-per-prop 5.0   --num-steps 10000   --comparison engine_only --csv-path ./data/runs/compare-exact/results.csv
```

example setup, because I keep forgetting how python works:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA may encounter issues. try this:
```
pip install torch --index-url
https://download.pytorch.org/whl/cu121
```

watch that the device is correct. see jupyter notebook if it doesnt print that automatically i suppose

---

####  Other tests

Oracle comparison for owl2bench, one class, stratified

```
./.venv/Scripts/python.exe -m src.oracle_compare --schema data/owl2bench/UNIV-BENCH-OWL2EL.owl --data data/owl2bench/O
WL2EL-1.owl --target-class http://benchmark/OWL2Bench#Employee --engine-mode stratified --oracles elk --owlapi-home com
parison/owlapi-5.5.1 --device cpu --show-timing-breakdown
```
