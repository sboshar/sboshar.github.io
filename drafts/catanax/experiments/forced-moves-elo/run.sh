#!/bin/bash
# skip_forced_plies off vs on, one after the other, each alone on an idle GPU (PLAN.md).
set -u
R=~/exp-forced-results; W=~/exp-forced; COMMIT=d2bdfff
mkdir -p $R
log() { echo "$(date -u +%FT%TZ) $*" >> $R/progress.log; }
wait_idle() {  # two consecutive empty checks, a minute apart
  while :; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
      sleep 60; [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return
    fi
    sleep 60
  done
}
[ -d $W ] || git -C ~/catanax worktree add -q --detach $W $COMMIT
git -C $W checkout -q --detach $COMMIT
log "commit $(git -C $W rev-parse HEAD)"
cd $W
export JAX_COMPILATION_CACHE_DIR=~/.cache/catanax-jax XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD PYTHONUNBUFFERED=1
PY=~/venv-catanax-bench/bin/python
CFG=configs/colonist_paper_lite_normalized_anchor.yaml
COMMON="iterations=2000 checkpoint_every=250 eval_every=100000"
nvidia-smi pmon -s um -d 5 > $R/pmon.log 2>&1 &
PMON=$!
for arm in off on; do
  extra=""; [ $arm = on ] && extra="skip_forced_plies=true"
  log "waiting for an idle GPU before $arm"; wait_idle
  log "start $arm: $COMMON $extra"
  $PY -m catanax.run --config $CFG --out $R/$arm $COMMON $extra 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > $R/$arm.log
  log "end $arm exit=${PIPESTATUS[0]}"
done
kill $PMON
log "round robin"
$PY ~/exp-forced-results/round_robin.py > $R/round_robin.log 2>&1
log "round robin exit=$?"
log DONE
