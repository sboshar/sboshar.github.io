# Engine kernel: improvement ideas from a code read

Source: `catanax/_src/engine_step_pallas.py` (main, Oct 2026) plus `docs/engine_step_pallas.md`.
None of this has been run (no GPU here). Every item says what to probe first, in the
repo's own style.

## The diagnosis in one paragraph

The v2 kernel is written as **scalar code over memory**: there are 144 `row.get/add/set`
sites and 8 vector accesses. Each `row.add` is a dependent load → add → store to the row
in global memory. The repo's own cost-curve probe gives
(0.481 − 0.091) ms / 1600 RMWs ≈ **244 ns per dependent read-modify-write** at batch 1024.
At ~1.4 GHz that's roughly 340 cycles, which is about an L2 round trip, not an L1 hit
(estimate; unverified). At batch 1024 an A100 holds only ~9.5 games per SM, so nothing
hides that latency: **step time ≈ length of the longest dependent chain × L2 latency**.
Meanwhile, with `num_warps=1`, 31 of the warp's 32 lanes sit idle during every scalar op.
So the lever is to **shorten dependent chains and use the idle lanes**, not to cut
arithmetic.

## 1. Load once, compute in registers, store once (the "header" layout)

- The per-move hot scalars (hands, bank, VP, stocks, phase machine, award colors and
  lengths, win_vp, hold_limit, winner) already sit almost contiguously: everything outside
  the board arrays lives in offsets **336–464**, which is four 32-wide vectors.
- Proposal:
  - at kernel start, issue those 4 vector loads (independent, so they're in flight together)
  - do every scalar update in registers with `where(iota == k, new, header)` on ≤ 32-wide
    vectors, which is one instruction for a one-warp program
  - store the 4 vectors once at the end
- This turns chains like `_pay_cost` (8 serial RMWs), `_finish` (~20) and
  `_record_balanced_roll` (~35, incl. an 11-way loop and a 4-step shift of `recent_rolls`)
  into register math.
- It's a middle ground between v1 and v2. v1 did masked reductions over all 512 lanes for
  every access; this only touches 128 lanes and only for the header, while the board arrays
  stay addressed in memory.
- Make it cheap and explicit: reorder `GameState` fields so the hot header is one
  aligned 32/64/128 block (pack order follows field order; bump `LAYOUT_VERSION`).
- Probe first: a kernel doing N `where`-updates on a 128-int register header vs N `row.add`s.

## 2. Vectorize the two big loops with the bitmask trick you already have

- `_distribute_roll`: `fori_loop` over 19 tiles × 6 corners × (table load + 2 row loads)
  is ~400 dependent scalar loads, on **every non-7 ROLL**, and ROLL happens every turn.
  - Vector form, using the existing limb machinery:
    1. load `tile_numbers` / `tile_resources` as 32-wide vectors
    2. build 5 "hit tiles with resource r" masks (one limb each, 19 bits)
    3. for each of the 3 `NODE_TILES` slots, `(mask_r >> node_tiles[:, k]) & 1` over a
       64-wide node vector
    4. multiply by each player's building vector and reduce
  - That's ~40 vector ops instead of ~400 dependent loads.
- `_free_road_bookkeeping`'s `any_buildable` is a `fori_loop` over 72 edges with 3 dependent
  loads each. It's the same pattern `_component_walk` already vectorizes: a component limb
  mask, then `_mask_bit(mask, EDGE_NODES[:, side])` over a 128-wide edge vector.
- `_apply_move_robber` (6 corners × 2 loads, then 5 hand loads) and the 7-roll hand totals
  become one vector load each.

## 3. Probe: vector-indexed loads from a *ref* (memory gather)

- The design note's key distinction is that a register vector can't be gathered, but a ref
  can be addressed. The kernel only ever addresses refs at **scalar** offsets.
- Triton itself is built on pointer-tensor loads (`tl.load(base + offsets_vector)`), so a
  memory gather such as `node_owner[TILE_NODES[t]]` *should* be expressible as one vector
  load from the row ref.
- If Pallas-Triton lowers an array indexer on a ref (worth a 20-line probe, exactly like
  the original capability table), many bitmask workarounds become plain gathers. It also
  makes item 2 even simpler.

## 4. Store the board as bitboards (the chess-engine idea)

- Today `node_owner`, `node_building`, `edge_owner` and `node_in_component` are int32 per
  element (288 ints), and the kernel rebuilds bitmasks from them on every call
  (`_mask_from_flags` = masked full-width sums per limb in `_longest_road` and
  `_component_walk`).
- Bitboard layout per player:
  - settlements: 2 limbs
  - cities: 2 limbs
  - roads: 3 limbs
  - component: 2 limbs
  - that's ~18 ints for both players instead of 288
- Every predicate (owned, enemy, occupied, buildable) becomes one load + shifts/ANDs.
- Constant per-node neighbor masks (54 × 2 limbs) turn the distance rule into
  `occupied & NEIGHBOR_MASK[n]`. That's literally a chess "attack table". It's also good blog
  material.
- Cost: the XLA reference engine, `observe` and `legal_action_mask` read these arrays. Either
  convert at the boundary or port them too. This is the biggest refactor on the list.
- Minor: use `uint32` limbs (no sign bit) for cleaner code. It's the same limb count for 54
  and 72 bits, so this is about readability, not speed.

## 5. The presample is two-thirds of the step

- Measured: presample 0.232 ms vs fused kernel ~0.15 ms, inside a 0.347 ms step. It's the
  single largest piece of the engine step.
- Proposal: in XLA, presample only **raw random bits** for all lanes (one batched threefry
  call), then convert bits → dice / steal *inside* the kernel.
  - catanatron `randint` and the steal's uniform → cumulative-threshold are integer/bit
    exact, so bit-exactness survives.
  - colonist's `jax.random.categorical` uses Gumbel noise (`-log(-log(u))`). Triton's `log`
    may differ by an ulp from XLA's and flip a near-tie, so either keep balanced dice in
    XLA or change the contract (define the sampler once, shared by the reference and the
    kernel).
- Also drops two `lax.cond`s, a cumsum, and the vmapped `balanced_roll_distribution` from
  the XLA side.

## 6. Smaller items

- `out_ref[...] = state_ref[...]` copies the full 2 KB row in and out. Aliasing the input
  (`input_output_aliases={0: 0}`) avoids the second buffer. It's tiny compared to latency
  but free.
- Longest road: seed only from nodes with degree ≠ 2 (exact; see the earlier notes), and
  add an overflow flag for the 32-slot stack.
- Measure per-family cost × frequency (batches of all-ROLL, all-END, all-road …). ROLL and
  END run every turn, so they matter more than rare dev cards. Prioritize from that table.

## Honest sizing

- The engine step is ~20% of an MCTS move after v2 (net + observe 37%, tree 29%, legal mask
  14%).
- If items 1, 2 and 5 cut the step from 0.35 ms to, say, ~0.12 ms (a guess, not a
  measurement), the move gets ~13% faster.
- Bigger move-level wins remain:
  - emit the legal mask from this kernel (it already has the state in hand)
  - collapse forced plies per game inside the kernel (the lockstep `while jnp.any(forced)`
    loop is the longest-road straggler pattern again)
  - fewer, wider network layers
- Item 1 and the bitboard layout are prerequisites for doing the legal mask in-kernel
  cheaply, so they compound.

## Blog angle

"The kernel is latency-bound scalar code: each `row.add` costs about one L2 round trip, and
31 of 32 lanes idle." That one sentence explains why v1 (all-vector, masked reductions) and
v2 (all-scalar, addressed memory) both leave room, and points to the hybrid: a header in
registers, bitboards for the board.

## Measured: per-family kernel cost (Oct 23, A100, colonist, 2,048 lanes)

From `measure_kernel_families.py` on real self-play snapshots: every lane plays one action
family; when a lane has no legal action of that family, it plays its first legal action instead.

| family | early (step 40) | middle (250) | late (700) | training mix |
|---|---|---|---|---|
| roll | 0.134 ms | 0.161 | 0.133 | 0.119 |
| end turn | 0.133 | 0.176 | 0.135 | 0.131 |
| build road | 0.136 | 0.224 | **0.279** | **0.256** |
| build settlement | 0.135 | 0.176 | 0.189 | 0.165 |
| maritime trade | 0.133 | 0.175 | 0.135 | 0.141 |
| real one-ply candidate load (~8.5k rows) | 0.22 | 0.49 | 0.42 | 0.50 |

- **There is a flat floor of about 0.13 ms per 2,048 lanes** that every family pays. This is the
  "latency-bound scalar code" diagnosis above, now measured: rare dev-card families cost the
  same as ROLL.
- **Build road roughly doubles late in the game** (the longest-road search grows with the road
  network). Settlement adds ~30%.
- After #144, the engine's share of a colonist one-ply step is ~0.5 of ~2.2 ms (~23%).
- Next ("F"): attribute the floor (launch, dispatch, finish, row load/store), then vectorize
  longest road, the component walk and the roll payout across the warp's idle lanes.

## Follow-up (#146): what the floor actually was, and what moved

- Of the ~0.13 ms "floor": 0.052 ms XLA chance presample, ~0.022 ms the measuring harness, 0.056 ms the kernel
  (launch + row copy 0.017, slowest lane's applier 0.025, `_finish` 0.013).
- Items 2–4 of the old list (component walk, roll payout, free-road check) measured ≤ 0.025 ms or run on almost no lanes: skipped.
- Road search on the one-ply candidate load: 0.313 → 0.104 ms (colonist), 0.234 → 0.113 ms (catanatron), from
  3-limb masks (E1) and arms + pairs (E2). End to end −8.4 / −8.7 / −8.7% (colonist one-ply / catanatron one-ply / colonist MCTS).
- Remaining engine-side items, now small: the XLA presample (0.04–0.08 ms per call), `init_board` on every lane for resets
  (~2.4% of a one-ply step), full sorts over 294 actions (~4.6%).

