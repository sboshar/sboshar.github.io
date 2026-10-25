# catanax blog series: drafts

Planning notes for a blog series about [catanax](https://github.com/sboshar/catanax), a JAX
Catan engine plus AlphaZero-style training. These files live outside `src/`, so Astro does
not publish them.

| File | Status |
|---|---|
| [01-problem-setup.md](01-problem-setup.md) | bullet outline: problem overview and formulation |
| [02-kernels.md](02-kernels.md) | bullet outline: vectorizing the engine, what's clever in the kernels, speedups |
| [03-optimal-compute.md](03-optimal-compute.md) | bullet outline: K/R/D rollout scaling, Pareto front, optimal allocation |
| [05-packing-and-batch-invariance.md](05-packing-and-batch-invariance.md) | bullet outline: one-ply search packing only the candidates that matter (#144), and the three XLA effects that made a row's network output depend on its batch neighbours |
| [04-other-post-ideas.md](04-other-post-ideas.md) | titles + one-line ideas for the remaining posts |
| [viz/catan-to-kernel.html](viz/catan-to-kernel.html) | interactive Distill-style draft: board → row, one move in the kernel, longest road, the A100, the warp race, packing candidates, batch-size dependence of the network |
| [kernel-improvement-ideas.md](kernel-improvement-ideas.md) | notes from reading the engine kernel: latency diagnosis + six improvement ideas |
| [experiments/forced-moves-elo/](experiments/forced-moves-elo/) | experiment: playing forced moves inside the engine step (catanax #154), off vs on, run one after the other, Elo by iteration and by real hours. Result: faster per decision, weaker agent. Plots, logs, ratings, plan |
| [experiments/cpu-vs-gpu/](experiments/cpu-vs-gpu/) | measurements: self-play with the same network on an M3 Pro vs an A100, and engine-only random play (catanatron, vmap, kernel) |
| [figures/](figures/) | plots from `catanax/experiments/krd_rollout_sweep` + a surface-fit script |
