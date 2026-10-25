# Experiment: skip_forced_plies off vs on, one run after the other

## Question

At equal training steps, and at equal real time, does colonist paper_lite learn as well when forced moves are played inside the engine step (`skip_forced_plies=true`, #154) as when they're searched like any other position?

## Design

- **Code:** the #154 branch head `d2bdfff` (`main` + #154), in its own detached worktree `~/exp-forced` so nothing else can change it mid-run. With the flag off this code behaves exactly like `main`; #154 adds code only on the flag-on path.
- **The only difference between the arms is `skip_forced_plies`.** Everything else is the recipe's defaults in both. That includes `max_forced_plies` (8, used only when the flag is on), `search.candidate_capacity` (0.4) and `search.dice_capacity` (0.02 for colonist).
  - So the on arm runs with chunk sizes tuned for the off arm. With forced moves skipped, almost every root is a decision, so each step overflows into a second candidate and dice chunk. That makes the on arm slower than it could be. This is deliberate: it's the no-other-difference comparison.
  - Learning isn't affected by chunk sizes (packing is exact), only speed.
- **Recipe:** `configs/colonist_paper_lite_normalized_anchor.yaml`, same seed (the config's) for both.
  - `iterations=2000 checkpoint_every=250`
  - `eval_every=100000`, so the only evaluation is the recipe's iteration-0 one, the same in both arms.
- **Order:** off first, then on. Each run alone on an idle GPU:
  - the script waits until no other process uses the GPU before each run
  - `nvidia-smi pmon` is logged throughout, so any contention is visible afterwards
- **Expected time:** about 1.35 h off, about 1.6–1.9 h on (slower without tuned capacities), plus compile. Then the round robin.

## What gets recorded (`~/exp-forced-results/` on the VM)

| File | Contents |
|---|---|
| `off.log`, `on.log` | the training log, every line prefixed with a Unix timestamp (unbuffered), giving real time per iteration |
| `off/`, `on/` | checkpoints at 249, 499, ..., 1999 (`ckpt_<i>.pkl`), resolved config, provenance |
| `timings.json` | per arm: start time, time of each iteration's log line, the 8 checkpoint times |
| `pmon.log` | GPU process monitor for the whole experiment |
| `progress.log` | start/end times and exit codes |
| `round_robin.json` | joint Bradley-Terry Elo (random = 0) over both arms' 8 checkpoints, 512 games per pair (17 agents, 136 pairs) |

## Deliverables

1. **Elo vs training iteration**, both arms as lines on one plot.
2. **Elo vs real training hours**, measured from each run's iteration-0 line, so compile time is excluded and both arms start at 0. Both arms on one plot.
3. A table: iteration, real hours and Elo per checkpoint, per arm.
