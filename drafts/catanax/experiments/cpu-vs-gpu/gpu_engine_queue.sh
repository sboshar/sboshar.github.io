#!/bin/bash
# GPU engine-only random play (catanatron rules), queued after the forced-moves experiment.
set -u
R=~/engine-bench; mkdir -p $R
until grep -q DONE ~/exp-forced-results/progress.log 2>/dev/null; do sleep 60; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
cd ~/catanax-main-adfe 2>/dev/null || cd ~/catanax
W=~/engine-bench-wt
[ -d $W ] || git -C ~/catanax worktree add -q --detach $W 08caf35
cd $W
export JAX_COMPILATION_CACHE_DIR=~/.cache/catanax-jax XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD
PY=~/venv-catanax-bench/bin/python
echo "$(date -u +%FT%TZ) start commit $(git rev-parse --short HEAD)" >> $R/progress.log
for eng in xla default; do
  for b in 4096 16384; do
    until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
    $PY ~/engine-bench/engine_random_play.py catanax $R/gpu_${eng}_b${b}.json --engine $eng --batch $b --steps 2000 > $R/gpu_${eng}_b${b}.log 2>&1
    echo "$(date -u +%FT%TZ) $eng b$b exit=$?" >> $R/progress.log
  done
done
echo DONE >> $R/progress.log
