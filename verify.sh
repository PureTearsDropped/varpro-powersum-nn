#!/usr/bin/env bash
set -u; PY="${PY:-python3}"; cd "$(dirname "$0")"
echo "== [1/2] total_bench.py  (VarPro power-sum, CPU/numpy) =="; $PY -u total_bench.py || exit 1
echo; echo "== [2/2] physics_tot_train.py  (totalized-gradient demo; needs CUDA) =="; $PY -u physics_tot_train.py
