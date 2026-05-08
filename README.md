# Scored-semantic-alignment-via-OWL-DL-semantics-on-the-GPU-

Example execution:
```
python -m src.compare_exact   --num-nodes 10000   --num-props 1000   --num-classes 1000   --avg-degree-per-prop 5.0   --num-steps 10000   --comparison engine_only --csv-path ./data/runs/compare-exact/results.csv
```

Example setup, because I keep forgetting how python works:
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
./.venv/Scripts/python.exe -m src.oracle_compare --show-timing-breakdown \
  --schema data/owl2bench/UNIV-BENCH-OWL2EL.owl --data data/owl2bench/OWL2EL-1.owl \
  --target-class http://benchmark/OWL2Bench#Employee \
  --engine-mode stratified --profile gpu-el --device cpu \
  --oracles elk --owlapi-home comparison/owlapi-5.5.1
```

Benchmark:
```
python -m src.paper_benchmark \
  --schema data/owl2bench/UNIV-BENCH-OWL2EL.owl \
  --datasets data/owl2bench/OWL2EL-1.owl data/owl2bench/OWL2EL-2.owl  data/owl2bench/OWL2EL-5.owl data/owl2bench/OWL2EL-10.owl data/owl2bench/OWL2EL-50.owl  data/owl2bench/OWL2EL-100.owl data/owl2bench/OWL2EL-200.owl\
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
