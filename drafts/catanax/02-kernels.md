# Making Catan fast: vmap's lockstep tax and the kernels that remove it

<!-- Interactive draft of the figures: viz/catan-to-kernel.html (open locally or serve the folder). -->

<!-- Bullet outline. Fill in prose later. Numbers: A100-40GB unless noted. [CHECK] = verify before publishing. -->

## Main idea of the post
- **`vmap` is lockstep. A Pallas `grid=(B,)` gives every game its own control flow.**
  - under vmap, a `while_loop` runs until the slowest lane finishes
  - under vmap, a `lax.switch` evaluates every branch on every lane
- Catan is full of data-dependent loops whose length varies wildly per game
- possible framing: "I wrote the same fix three times before I recognized it as one fix"
  - longest-road DFS
  - MCTS descent / backup
  - action dispatch
  - (fourth, future: forced-ply collapse; see the end)

## Baseline and headline numbers
- catanatron: ~100 games/s on 16 CPU cores
- end state: ~7M engine steps/s (~5,700 games/s) on an A100 for the bare engine
- MCTS: stock mctx 1,462 ms/move → fused kernels
- time to 1M gradient steps
  - stock mctx: ~50 days
  - wide fused search: 70 h
  - + engine kernel: 47 h
  - + batch 4096: 22 h
  - + 16 sims: 4.9 h (not Elo-gated)
- [FIGURE] waterfall of that table
- [FIGURE] steps/s timeline across the optimizations

## Act 1: the engine under vmap
- longest road ran 3× per step (once per branch of the switch)
  - fix: classify once before dispatch (NONE / INITIAL / BUILD / SYNC / CUT)
  - result: 2.8× (CPU, batch 1024: 19.9k → 51.9k steps/s)
- incremental "longest trail through the new edge" + `shard_map` across cores
  - 316 games/s on Apple Silicon (3.1× catanatron)
  - phased two-arm expansion avoids C(a+b, a) duplicate interleavings
- profiling: longest road = ~85% of step time
  - straggler amplification: mean 73 vs max 796 node visits (~11×)
  - [FIGURE] histogram of DFS cost per board, with the batch max marked
- failed idea: a closed-form parity/odd-degree formula ("provably over-counts")
- red herring: profiling blamed `_component_walk` for "61%", but removing it changed nothing (XLA overlapped it). Lesson: re-measure.
- memory: a (54,3,16) tensor took ~340 MB at batch 131k → cumsum slot map (6.66M → 7.05M steps/s)

## Focused section: longest road (could be its own post)
- the rule: 2 VP for the longest edge-simple trail ≥ 5; can't pass through an opponent's building
- why it's hard
  - longest trail is a search over trails; its cost depends on network shape (a line is trivial, a tangle with branches explodes)
  - NP-hard on general graphs; tiny here (≤ 15 roads, degree ≤ 3), but the per-board variance is what kills vmap
- catanatron semantics reproduced on purpose: never step *into* an enemy node; the cut award skips the ≥5 check; ties favor seat 0
- the history, in order
  - ran 3× per step (once per switch branch) → classify once before dispatch (NONE / INITIAL / BUILD / SYNC / CUT): 2.8×
  - incremental: a new road only extends trails through itself → search from that edge; phased two-arm expansion
  - failed: a closed-form degree-parity formula ("provably over-counts")
  - per-game Pallas kernel (below): 9.5–12×
  - inside the v2 engine kernel: used-set indexed by edge id in 31-bit limbs (no prefix sum or popcount in Triton)
  - future: seed only from nodes with degree ≠ 2 (exact; fewer duplicates; lower stack occupancy); add a stack-overflow flag
- interactive: viz figures 3 (straggler race) and 4 (DFS stack)

## Act 2: the longest-road kernel: clever bits
- `grid=(B,)`: one program per game, so each DFS retires independently
- **a DFS frame in one int32**: `node | length<<6 | used_mask<<11`
  - 6 bits node, 5 bits length, 16 bits used edges (≤15 roads, compacted)
  - the whole stack is a `(32,)` int32 vector
- **push without scatter: each slot pulls its value**
  - `match = (push_pos[None,:] == slots[:,None]) & can[None,:]`
  - `stack = where(match.any(1), sum(where(match, entry, 0), 1), stack)`
  - turns a scatter into a comparison matrix plus a reduction
- Triton constraints (jaxlib 0.10.2): no scatter, no dynamic index, power-of-2 shapes
- setup stays in vmapped XLA; only the data-dependent loop is in the kernel
- result: bit-exact on 73,728 states; 9.5× at batch 16k (556k → 5.29M steps/s), 12× at 32k
- ceiling check: with no longest road at all, 6.6M steps/s

## Act 3: search is the bottleneck (#82)
- stock mctx Gumbel: 1,462 ms/move at n=64, batch 1024; 96% mctx internals, 4% net + engine
- cost scales ~n^2.4. Why?
  - forced moves make trees chains: mean depth 35 of 64 sims
  - ~25k kernel launches per move
- ruled out by measurement: batch scaling, CUDA graphs (1.01×), action width, embedding bytes, leaf count
- top-K value search: 28 → 561 games/s
- int8 obs + sparse targets: buffer 17 GB → 2.6 GB

## Act 4: fused descent + backup kernels
- discipline: a pure-JAX reference first, **proven integer-exact against stock mctx**, then kernelize
- descent kernel
  - walks root → leaf in one launch
  - completed-Q transform + both Gumbel rules (root sequential halving, interior deterministic) in-kernel
- backup kernel: in-place statistics update via `input_output_aliases`
- tree arrays stay in global memory (`memory_space=pl.ANY`); windowing needed 418 KB vs 167 KB of shared memory
- Mosaic GPU rejected the kernels → force Triton; actions padded 294 → 512 with -1e9
- small wins: descent returns `next_node`; flat-packed embeddings removed ~46 gathers/scatters per expand
- result: 54.5 / 247 / 1471 → 24.8 / 58.5 / 150.6 ms/move at n = 16 / 32 / 64 (2.2× / 4.2× / 9.8×)
- compile wall: a 12-way aliased commit kernel hung for > 45 min; splitting it → 8–13 s

## Act 5: the whole engine step in one kernel
- v1 failed: 1.13×, and its "XLA twin" control was 0.95×
  - the Triton body was ~8% of its own pipeline
  - every field access was a masked reduction over 512 lanes
  - all 13 families ran on every lane
  - lesson: profile the pipeline, not the kernel
- **probe the constraint list before designing**: five of six assumed limits were false
  - dynamic ref load/store, `fori_loop`, `while_loop` and table lookups all work
  - only register-vector gathers are impossible
  - [FIGURE] capability table (assumed vs reality)
- **cost-curve probe sized the design before building it**: 0.091 ms at 0 RMWs → 0.481 ms at 1600; a step needs ~500–900
- v2 design
  - the state row is addressed in place
  - `pl.when` per action family = a real branch (one game per program)
  - component walk + longest road + winner check, all in one launch
- **bitmasks as gathers**: `values[idx]` → `(mask >> idx) & 1`
  - build with `sum` (distinct bits, so sum == OR; there's no reduce_or lowering)
  - 31-bit int32 limbs (x64 off, bit 31 = sign)
  - index the used-edge set by edge id, so no prefix sum or popcount is needed
- **road-cut connectivity as bitset relaxation**: 15 fixed rounds of `reach |= any(nbr_reach & owned & ~enemy)`
- static rule flags compile code out: 5.19× catanatron vs 4.04× colonist
- results
  - engine step 1.40 → 0.347 ms (4.04×); 731k → 2.95M env-steps/s
  - MCTS move 87 → 49.6 ms (1.76×)
  - projected self-play 37.3 h → 21.2 h

## War stories (bugs)
- **the warp race** (best story; draw a two-warp timeline)
  - a program = `num_warps × 32` threads running scalar code
  - every thread does the same load → add → store
  - independent warps: a lagging warp writes back a stale value
  - ~11 of 4096 lanes wrong, different lanes each run
  - fix: `num_warps=1`, and it's a correctness requirement
- corrupted `(B,1)` side outputs → carry the values in the row's padding
- intermittent illegal memory access → clamp traced offsets; assert static ones at trace time
- dev-deck parity: `jnp` clamps out-of-range gathers, so match that to be bit-exact
- lowering papercuts: `where` with Python ints → i1; no `reduce_or`; `functools.cache` leaking tracers
- two bugs only appeared at batch 4096 (v1 had been tested at 64). Lesson: benchmark at the regime you ship.

## Testing approach
- exhaustive: every action id × 512 states × 2 rule sets × 3 phases, legal or not
- one batch shape + persistent compile cache: ~4 min instead of 33+
- "a test that cannot fail is not a test" (the road-cut test that never produced a cut)

## Amdahl moves: what's left
- per-move shares after v2: net + observe 37%, tree 29%, engine 20%, legal mask 14%
- unpack measured more than the whole engine step in isolation, but it's free in the loop (XLA fuses it; 13 of the 46 leaves are never read)
- tf32 / bf16 gave bit-identical output at the same speed: the trunk is launch-bound (~33 MFLOP matmuls)
- "the config switch alone was worth 11× more than the engine kernel"
- trivial recurrent function: the tree machinery is only ~9% of a move

## Dead ends (short list)
- merging 13 switch branches into 6: 1.01×
- search-only legal mask: 1.72× in isolation, 1.01× per move
- forced-ply unroll (#105): projected 2.2–2.5×, measured 1.18–1.25×
- full per-sim megakernel (#87): never built
- JAX AlphaBeta port (#38): a numerics story; maybe a sidebar

## What I'd do next (could close the post)
- [CHECK FIRST] the search backup and expand-write kernels still use the default 4 warps and do in-place RMW
  - possibly the same race; possibly linked to the #87/#89 large-batch divergence
- forced-ply collapse is still a lockstep `while jnp.any(forced)` over the batch, i.e. the same straggler pattern
  - move it into the engine kernel as a per-game loop
  - "the fourth time I wrote the same fix"
- emit the legal mask (and maybe the observation) from the engine kernel, as pgx does
- longest road: seed the DFS only from nodes whose degree ≠ 2 (exact; fewer duplicates; lower stack occupancy)

## Epilogue: the network was the next problem (see 05)
- packing one-ply candidates (only 21–35% of slots matter) gave 1.36–1.57× self-play
- it surfaced batch-size-dependent network numerics; that story is post 05

## Lessons to box
- profile the whole pipeline, not just the kernel
- verify the constraint list before designing around it
- size the design with probes before building
- benchmark at the regime you ship
- prove exactness against a reference at every stage
