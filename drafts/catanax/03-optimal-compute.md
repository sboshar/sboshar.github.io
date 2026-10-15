# How to spend a compute budget at inference: K/R/D rollout scaling

<!-- Bullet outline. Source: catanax experiments/krd_rollout_sweep/ (#69) + PR60 battery.
Figures in ./figures. [CHECK] = verify before publishing. -->

## TL;DR bullets
- search at move time: +150 Elo at 32 ms, +260 at ~0.9 s per batched move (vs greedy)
- roughly +25 Elo per doubling of compute, no plateau yet
- optimal allocation grows on both axes: R* ∝ B^0.6, D* ∝ B^0.4
- spend it badly and you finish below no search at all

## Setup
- the net: `paper_lite`, played greedy = top-16 one-ply value search
  - 1.9 ms per batched move (batch 96, A100)
  - Elo anchor = 0
- the search: flat Monte-Carlo rollouts
  - top-K candidates by prior
  - R playouts per candidate, D plies each, net policy picks the moves
  - bootstrap with the value net at the cutoff; average; argmax
  - no tree, no UCB
- the three knobs as three questions
  - K: how many options really matter?
  - R: how much noise needs averaging?
  - D: how far ahead until the value net can be trusted?
- budget B = R·D engine steps per candidate
- [TODO] why rollouts rather than MCTS here (mctx s64 rated 578 vs rollout R64 670 at inference, PR60)

## Methodology: rating ~50 agents without a round-robin
- a full round-robin costs years (each pairing scales with K·R·D)
- every variant plays greedy, 192 seat-swapped games
- strong variants also play anchors R64/D20 and R128/D20
- one joint Bradley–Terry fit over the sparse graph, greedy = 0
- **noise floor measured by repeat**: R32/D10 rated 153 and 194 (41 Elo apart)
  - treat ±40 as the band for one pairing
- timing: steady-state (≥5 warmups, mean of 10–20 calls)

## Result 1: the R×D map
- [FIGURE] figures/krd_map.png
- top-right green, edges red: strength needs both breadth and depth
- R ≤ 4 is worse than no search (R1/D10 = −262)
  - a noisy argmax loses to a stable, biased one
- deep-and-thin collapses: R8/D128 −140, R4/D80 −169
- depth floor ~D10: R512/D5 (+140) loses to R64/D20 (+197) at ¼ the cost
- [TODO] intuition for D≈10: about one turn cycle? with 64% forced plies, it's only ~3–4 real decisions

## Result 2: iso-compute curves are inverted U's
- [FIGURE] figures/krd_iso_alloc.png
- fix R·D, slide from shallow-wide to deep-thin
- the peak moves deeper as the budget grows: D≈10 at 320 → D≈20 at 1–5k → D≈40 at 10k
- mechanism
  - playout variance grows with D; averaging R playouts pays it down as 1/√R
  - depth reduces bias (you bootstrap from a more resolved position) but adds variance
  - the optimal split = the bias–variance trade-off traced over budget

## Result 3: an allocation scaling law
- [FIGURE] figures/krd_optimal_alloc.png
- per-line argmax: R* 32 → 256, D* 10 → 40 over a 32× budget
- robustness check: a smooth quadratic surface in (log R, log D)
  - fit to 36 K=6 cells with R ≥ 8, D ≥ 4; residual SD 24 Elo
  - 2,000 bootstrap resamples (script: figures/fit_krd_surface.py)
  - D* exponent 0.36 [0.28, 0.44]; R* exponent 0.64
  - Elo per doubling 23 [17.5, 28] vs ~30 from argmaxes
  - argmax over noisy cells biases upward; quote "~+25 per doubling"
- [TODO] decide whether to draw the Chinchilla analogy ("grow samples and horizon together")
- [FIGURE TODO] a smoothed surface with the ridge path + iso-budget lines; could replace the map and optimal-alloc plots

## Result 4: K barely matters
- [FIGURE] figures/krd_axes.png
- K from 2 to 64 at R8/D10: all within noise; K = 4–6 is enough
- ties back to the setup post: the median real decision has 6 options, and the prior ranks them well
- caveat: the D-axis panel at R8 zig-zags within noise; don't over-read it

## Result 5: the Pareto frontier
- [FIGURE] figures/krd_pareto_final.png
- greedy 1.9 ms → R8/D2 → R8/D8 → R16/D10 → … → R256/D40 at 890 ms
- caveat: the frontier is drawn through noisy maxima, including R32/D10's lucky repeat (194 vs 153)
  - [TODO] redraw from the smoothed surface or the averaged repeat
- retracted claim: "greedy is Pareto-dominated"
  - a 3-call post-compile timing inflated greedy 36× (65 ms vs 1.9 ms)
  - lesson: never time JAX without a warmup

## Does it beat AlphaBeta?
- rollouts R64 and R256 each won 14/24 (58%) vs real catanatron AlphaBeta(n=2)
- greedy never beat it
- 95% CI roughly 37–77%; [TODO] run a bigger match before headlining
- [CHECK] numbers against PR60

## The catch: this gain can't be distilled
- expert iteration with R8/D10 and R32/D10 rollout targets → nets indistinguishable from the one-ply control (794 vs 742–786)
- so the +260 is pay-per-move
- [TODO] hypotheses
  - the rollout argmax is ~one-hot over ~4 real options and adds little beyond the anchored target
  - value noise at D ≥ 10 makes the target too noisy to fit
  - the gains come from averaging away dice, which a policy can't learn
- segue to the learning-signal post / MCTS training

## Related context
- EXP-A: mctx has ~22× the throughput of rollouts at large batch, but is slower below batch ~64
- eval-time depth-2 search +52, but as a training operator depth hurts (−78): depth flips sign between training and inference

## Takeaways to box
- measure the noise floor first (rate one config twice)
- too little search is worse than none
- scale breadth and depth together; depth only pays once you have the samples
- fit a surface; don't take argmaxes
- time in steady state
