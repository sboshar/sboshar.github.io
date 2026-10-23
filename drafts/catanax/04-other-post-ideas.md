# Other post ideas (undecided)

Each could be its own post, a section of another post, or dropped.

## Where does the learning signal come from?
Bigger nets didn't help, so the bottleneck was the training target. Covers:
- the one-ply search
- the near-uniform softmax(q) baseline
- the prior-anchored target (+311 Elo)
- per-state Q normalization rescuing colonist (rung 445 → 885)
- OFAT ablations (depth flips sign between training and inference)
- dice noise being ~55% of value loss (rollout denoising +211)
- the Gumbel MCTS value_scale dose-response

## How I fooled myself
Every time a measurement or bug made me believe something false:
- the seat-leak observation bug (#88)
- misdiagnosed horizon starvation
- win rate vs random misranking models
- sampling artifacts and max-of-N reporting
- the 36× timing artifact
- recipe drift
- search peeking at hidden cards (#114)
- the open large-batch kernel divergence
- "bit-identical to main" failing by 25% for a correct change: main itself was batch-dependent (split-K, a recomputed tanh, norm tiling); see 05
- "it's just rounding" for a 5-ulp difference: check the magnitude before accepting the explanation

Ends with the habits that came out of it.

## Is the engine actually right?
How bit-exactness against catanatron was established:
- board ids generated from its map
- the lockstep bridge
- 70k+ state checks
- deliberately reproducing catanatron's quirks (cut award, seat-0 ties)

Could be a section of the kernels post.

## Porting AlphaBeta to JAX (numerics sidebar)
catanatron's heuristic weights span 3e14 down to 1. float32 collapses them to 33% decision agreement, and splitting VP out reaches 84%. Turning on x64 broke the engine through an int64 leak.

## Where it stands now
- the current leaderboard and the best result vs AlphaBeta
- what's open: fair-information search (#117), the unrated Gumbel MCTS runs, PPO / diverse resets, the truncation ratchet

## Play against it
The replay viewer and the Hugging Face Space bot. An embed for the top of the series.

## Learning from expert Colonist replays
Scraping replays, converting them to the 371-wide observation and the 294-action space, and a behavior-cloning dataset. A teaser, or a future post once something is trained on it.

## Research infrastructure for a solo RL project
- the per-ruleset Elo ladder with a gate
- checkpoints carrying config + git provenance
- obs-schema versioning
- config-driven runs
- human-readable run names

## Related work
catanatron, pgx, Gumbel MuZero / mctx, Stochastic MuZero, prior Catan RL attempts.

## Compute budget
Hardware, total GPU-hours, rough cost.

## Running research with AI coding agents (only if true and you want to)
The issue-per-branch workflow, the CLAUDE.md rules, what worked, and where agents fooled you.
