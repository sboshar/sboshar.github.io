# Only score what can change the answer, and the 0.018 that wasn't a bug

<!-- Bullet outline. Source: catanax #144 (PR + ~/bench144review/REPORT.md on the A100 VM). Numbers: A100-40GB, jaxlib 0.10.2. [CHECK] = verify before publishing. -->
<!-- Interactive figures: viz/catan-to-kernel.html sections 7 and 8 (Figures 10-12). -->

## Main idea of the post
- **The cheapest work is work you prove can't matter.** One-ply search scored every candidate child, then threw most of the values away.
- Removing that work exposed a second, deeper fact: on a GPU, **a row's network output depends on how many other rows share the call**.
- Possible framing: "the speedup took a day; proving it was correct took the rest of the week, and found three compiler effects and a bug-shaped feature in `main`."
- Could be its own post, or the closing act of the kernels post (02). It is the first change where the network, not the engine, was the problem.

## Part 1: most of one-ply search's work is thrown away
- **The search:** for each game, the top K = 16 legal moves by prior. Each child is stepped and scored by the value network. Then a softmax over `logits + q_scale·q` gives the policy target, and a Gumbel top-k picks the move.
- **Child values that can't affect anything:**
  - **padding slots:** positions with fewer than 16 legal moves
  - **single-candidate games:** a softmax over one entry is exactly 1, so the target and the sampled move are fixed
  - **ROLL slots:** with dice enumeration, the value is overwritten by the expectation over 11 dice outcomes
- **What survives, measured on real self-play:**

  | | colonist (fresh net) | catanatron (trained net) |
  |---|---|---|
  | real slots | 25% | 38% |
  | needed slots | **21.5%** (max 25%) | **35%** (max 38%) |
  | games with more than one candidate | 45% | 70% |

- Colonist wastes more: its games more often have a single legal move.
- [FIGURE] toy step-through, 6 games × 4 slots, buffer of 8: no packing (24 network rows), packing that fits (8), packing that overflows (2 chunks, 16) (viz Figure 10).
- [FIGURE] the same at real scale: 32 games × 16 slots, colored padding / single / roll / needed, packed into a 40% buffer (viz Figure 11).
- **Intro needs to define packing for a general reader:** stream compaction under static shapes = prefix sum → gather into a fixed buffer → compute → scatter back; overflow runs another chunk.

## Part 2: packing with static shapes
- **JAX needs fixed shapes,** so "compute only the needed rows" becomes a fixed-size buffer:
  - `rank = cumsum(needed) - 1` gives each needed (game, move, key) triple a slot
  - per chunk j: `idx = nonzero(needed & (rank // C == j), size=C, fill_value=n)`
  - gather `parent[game[idx]]`, step it, score it
  - scatter back with `.at[idx].set(out, mode="drop")`, so padding rows go nowhere
  - `fori_loop(0, ceil(needed / C), …)` with a traced trip count, which lowers to a `while_loop`: overflow is exact, it just costs a second chunk
- **Keys:** `split(key, B·K)` stays in the original (game, slot) order, so the children are bit-identical to the unpacked path.
- **Sizing the buffer:** chunks are not free.

  | Buffer (fraction of B·K) | colonist ms/step | catanatron ms/step |
  |---|---|---|
  | off | 3.47 | 2.74 |
  | 10% (many small chunks) | 2.60 | 2.85 |
  | 30% | **2.03** | 2.45 (overflows into a 2nd chunk on most steps) |
  | **40% (shipped)** | 2.21 | **2.00** |

- Lesson: size the buffer for the usual case, from measured fractions, rather than for the worst case or for tightness.
- **The dice version didn't pay.** Packing the 11-outcome dice enumeration looked like the next big win, but only **1%** of colonist games (12% catanatron) need it: they have ROLL *and* another candidate. Most games where ROLL is a candidate have *only* ROLL, so they're already skipped. A buffer big enough to be safe cost more than it saved. Dropped.
- **Result:**

  | | before | after | speedup |
  |---|---|---|---|
  | colonist | 832 games/s | 1,307 games/s | **1.57×** |
  | catanatron | 3,057 games/s | 4,166 games/s | **1.36×** |
  | colonist paper_lite run | 4.0 h | 2.5 h | |
  | catanatron paper_lite run | 2.1 h | 1.55 h | |

## Part 3: "bit-identical to main" failed
- **The check:** tiny self-play (32 games × 24 steps, fixed seeds), main vs branch.
- **The result:** 25% of policy targets differed, and observations diverged.
- **The bisection, each step one experiment:**

  | Check | Result |
  |---|---|
  | one big chunk vs unpacked | identical |
  | three chunks vs unpacked | differ, π by 0.146 |
  | child states | bit-identical |
  | the **same 204 observations**, alone vs as rows 0–203 of a 512-row call | value differs by **0.0178** |

- **Why "it's rounding" wasn't good enough:** 0.0178 is about 5 bf16 ulps for a value near 0.5. One matmul's rounding should be under one ulp. I wrote an independent review brief and had a second agent on the VM check it from scratch. [Maybe a sidebar on using a second agent as a skeptical reviewer.]
- **The divergence wasn't only in setup:** at small chunk sizes, mid-game diverged at 76 of 77 checked steps. What mattered was rows per call, not the number of chunks or the phase.

## Part 4: three ways a row depends on its neighbours
In exact arithmetic, row i of `x @ W` depends only on row i of x. On a GPU, XLA picks the kernel per shape.

### 1. Split-K below ~450 rows
- With few rows there aren't enough output tiles to fill 108 SMs. XLA's `splitk-rewriter` pass cuts the K = 256 reduction into 4 slices, which separate thread blocks sum in fp32, then reduces them.
- Side effect: it skips the bf16 rounding of the layer output (XLA is allowed "excess precision"). At other sizes, a single Triton GEMM writes bf16.
- So **every element** differs by up to half a bf16 ulp (99.99% of elements), in **every** layer.
- **Layer by layer**, batch 204 vs 512:

  | layer | max abs diff | median activation | note |
  |---|---|---|---|
  | input dense | 1.95e-3 | 0.13 | exactly ½ ulp: rounded vs not |
  | block 0 | 1.27e-2 → 1.38e-2 | 0.45 | the RMS norm rescales small activations by ~6×, and the error with them |
  | block 1 residual | 2.70e-2 | 0.65 | each layer adds another ½-ulp mismatch to the residual stream |
  | block 2 residual | 3.07e-2 | 0.77 | |
  | block 3 residual | 4.03e-2 | 0.89 | |
  | value (tanh) | **1.78e-2** | 0.54 | |

  - [FIGURE] error growth through the layers, split-K on vs off (viz Figure 12).
- **Amplification:** paper_lite's anchor `logits + 50·q` turns 0.018 into ~0.9 logits, enough to flip sampled moves.
- **What didn't help:**
  - `--xla_gpu_deterministic_ops`: it makes one shape reproducible, not two shapes agree
  - an fp32 network at `highest`: still 1.3e-6, because the order still varies
  - cuBLAS instead of Triton: moves the boundary (256–1,024 rows vs 2,048+)
  - the split-K autotuner override: the split is an HLO pass that runs before autotuning
  - the old `split_k_autotuning` flag: gone
- **Fix:** `compiler_options={"xla_disable_hlo_passes": "splitk-rewriter"}` on the **self-play jit only**. Training's weight-gradient matmul reduces over the 8,192-row minibatch and needs split-K. It costs nothing at batch 2,048: those calls are 13k–33k rows, which XLA doesn't split anyway.

### 2. `tanh` computed twice (colonist only)
- With split-K off, catanatron matched bit for bit. Colonist still differed by ≤ 1.1e-5.
- The network's child values were identical, packed vs unpacked. But in the **unpacked path, i.e. main's code**, `q = where(valid, cv, -inf)` differed from `cv` itself by 1 ulp in 302 of 512 slots.
- Why: XLA recomputed the value head's tanh inside a Triton fusion, along with colonist's dice expectation and the `q` selects. Triton's tanh differs from XLA's in the last bit.
- Colonist's `q_rescale` divides by the spread of child values, which magnifies 1 ulp ~1000×.
- Fix: `jax.lax.optimization_barrier` on the value output. It is free. A barrier around the whole net also works, but costs 4.5% by blocking fusions.

### 3. The norm's reduction tiling below ~120 rows
- With 102-row chunks, every matmul matched but `final_norm` was 1–2 fp32 ulps off.
- XLA emits the RMS norm as a Triton fusion with a per-shape tile: `[2,256]` / 2 warps at 2,048 rows, `[1,256]` / 1 warp at 102 rows. That gives a different reduction tree for the 256-feature sum.
- Disabling that fusion was worse: the native reduction emitter changes order at almost every size.
- A fixed-order pairwise-tree norm is invariant everywhere, but it made the forward pass 1.5× slower and the training step 2.9× slower.
- Pragmatic fix: the stock norm is identical from 120 rows up (checked row by row to 135, then up to 50,000), so chunks get at least 128 rows.

## Part 5: main was already batch-dependent
- Searching 16 games on their own, vs as the first 16 of 64, changed **main's** policy targets by up to 0.10. With split-K off: identical.
- 32 vs 64 games was identical either way: both are above the split-K range (512 and 1,024 child rows).
- So "bit-identical to main" was never a reachable bar for *any* change in how rows are batched. Main itself depended on batch size and on fusion context.
- **Now:** with the batch-invariant compile, "packed == unpacked" holds bit for bit:
  - 300 steps, all six outputs (obs, policy ids and weights, value targets, valid mask, aux)
  - catanatron and colonist, fresh and trained nets
  - capacities 0.05–1.0, up to 21 chunks per step

## Part 6: how to test a batching change
- **Separate the logic from the numerics.** Swap the network for an **integer-hash value function**. Integer sums don't depend on order, so any packed-vs-unpacked difference is a real packing bug. It ran at capacities 0.05–1.0, up to 21 chunks, from the opening to step 400: identical.
- **Then test the numerics on their own:** the network is batch-invariant from 128 to 1,024 rows under the compile options.
- **Negative control:** the regression test fails on the unfixed branch, with all 512 candidate weights differing by up to 0.89. A test that can't fail isn't a test (see 02).
- **Watch for uninformative fields:** z was all zeros in every comparison, because no game ends in 24 or 300 steps. "z identical" meant nothing.

## Lessons to box
- The cheapest work is work you can prove doesn't matter. Measure what fraction that is before building.
- Size fixed buffers from measured distributions. Overflow should be exact but rare.
- "It's floating-point rounding" is a hypothesis, not an explanation. Check the magnitude in ulps.
- `deterministic_ops` gives run-to-run reproducibility, not shape invariance.
- Before demanding bit-identity with a baseline, check that the baseline is invariant to the thing you're changing.
- Turn off compiler passes per jit, not globally: training wanted the same pass that self-play had to avoid.

## Open / caveats
- This all depends on how jaxlib 0.10.2 compiles these shapes on an A100. An upgrade can rename the pass or change the tiling; the GPU tests are there to catch it.
- Possible follow-up: is batch invariance worth enforcing everywhere, e.g. for MCTS leaf evaluation or eval vs training batch sizes? [CHECK]
