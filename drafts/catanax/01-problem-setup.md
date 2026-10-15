# Catan as a tensor: problem overview and setup

<!-- Bullet outline. Fill in prose later. [CHECK] = verify the number before publishing. -->

## Hook / motivation
- Goal: an AlphaZero-style agent for 1v1 Catan that beats catanatron's `AlphaBetaPlayer`
- Why it's hard: branchy rules, dice, hidden cards, multi-action turns, long games
- Why existing engines don't work for this: catanatron runs at ~100 games/s on CPU, far too slow for self-play
- The pgx recipe: write the game as a pure `step(state, action, key)` over fixed-shape arrays, then `vmap` + `jit` the whole self-play loop
- The catch: every state needs the same shape, and every rule has to avoid data-dependent Python control flow
- [TODO] personal motivation: why Catan, why now
- [TODO] embed the playable demo / replay viewer if it's still live

## Scoping decisions
- 1v1 only
  - zero-sum, so a single value head, negamax, and well-defined Elo
  - colonist.io has a ranked 1v1 mode (and replays)
  - no player-to-player trading in either rule set; maritime trades with the bank stay
- Two rule sets, each with its own ladder
  - catanatron: 10 VP, discard on > 7 cards, i.i.d. dice
  - colonist: 15 VP, discard on ≥ 10 cards, friendly robber (can't target a player below 3 VP), balanced dice, public opponent hand
- Rules are a static `Rules` NamedTuple
  - JIT specializes on them; unused code paths compile out
  - the catanatron observation stays byte-identical to the original

## The board is lookup tables, not geometry
- 19 tiles, 54 nodes, 72 edges, 9 ports
- static incidence tables: `TILE_NODES (19,6)`, `EDGE_NODES (72,2)`, `NODE_TILES / NODE_EDGES / NODE_NEIGHBORS (54,3)`, padded with -1
- generated from catanatron's map so ids match the oracle 1:1
- every rule becomes a gather
  - distance rule: `occupied[NODE_NEIGHBORS].any(-1)`
  - production: a gather over `TILE_NODES` + a masked sum
- [FIGURE] hex board with tile/node/edge ids overlaid next to the tables

## Game state: 46 fixed-shape arrays
- ~465 int32 per game; a batch is the same pytree with a leading B axis
- highlights
  - `node_owner`, `node_building (54)`, `edge_owner (72)`, with -1 for empty
  - `hands (2,5)`, `bank (5)`, `dev_hand / dev_played (2,5)`
  - `dev_deck (25)` + pointer: shuffled at reset, so drawing is a pointer increment (this matters for hidden info later)
  - phase machine: `prompt`, `current_seat` vs `current_turn_seat` (discards on a 7), `has_rolled`
  - longest-road bookkeeping, which turns out to be the most expensive part (see the kernels post)
- packed into one `(B, 465→512)` int32 row for the kernels

## Action space: 294 flat ids
- main idea: split compound choices into sequences of single choices, so a turn spans many plies
- table of blocks
  - ROLL 0 | END 1 | road 2–73 | settlement 74–127 | city 128–181
  - buy dev 182 | knight 183 | robber 184–202 (the steal is automatic in 1v1)
  - discard 203–207 (one card per action)
  - year of plenty 208–227 (15 pairs + 5 doubles) | monopoly 228–232 | road building 233
  - maritime trades 234–293, only at your best rate
- why: removes dominated actions, turns C(n,k) discards into k small choices
- dispatch: `searchsorted` → 13-way `lax.switch`
  - under vmap, every branch runs on every lane, which sets up the kernels post

## Legal actions: the game is mostly forced
- `legal_action_mask` = `lax.switch` over 5 prompts
- over 40,960 self-play states: median legal = 1, 64% forced, p90 18, max 54; median among real decisions 6; 33% of decisions have > 16 options
- [CHECK] #105 measured 38.9% / 28.4% forced under trained policies; reconcile the two
- consequence: MCTS trees become chains (mean depth 35 of 64 sims), which drives the kernel work
- [FIGURE] histogram of legal-action counts

## Chance is an action
- ROLL is its own action
  - post-roll states get observed and trained on
  - search can enumerate all 11 sums exactly (`roll_outcomes`); worth about +60 Elo vs sampling (OFAT ablation)
- other randomness (steals, dev draws): one sample from the step key
- the balanced-dice rabbit hole (colonist)
  - a 36-combo bag, reshuffled below 13 cards
  - a recent-roll penalty ×(1 − 0.34·count over the last 5 rolls)
  - 7-balancing between players with a streak term
  - history-dependent, so it breaks the Markov property unless observed: a 21-int dice-context block
  - one `balanced_roll_distribution` shared by sampling and enumeration, so they can't drift apart
- [TODO] does balanced dice matter in practice? any number

## Observation
- flat int vector from the mover's perspective
  - own/opp buildings and roads, tiles, numbers, robber, ports
  - own hand, opp hand count, own dev cards, opp dev count, played devs, VP, flags
- widths: catanatron 345 → colonist 366 (dice block) → 371 (+ opp exact hand, which is public in colonist 1v1)
- stored as int8; with sparse top-16 targets the replay buffer went from 17 GB to 2.6 GB
- perspective bug (#88): `visible_vp` was in absolute seat order
  - now schema-versioned, and the loaders refuse mismatches
  - teaser for the "fooled myself" post

## Hidden information
- the net sees only public info + its own hand
- but search stepped the true state, which includes the dev deck order and the opponent's cards (#114)
- fix: determinization
  - resample the opponent's hand and dev cards with a multivariate hypergeometric draw
  - rejection sampling to prevent "phantom wins"
- no learned belief model (yet)
- [TODO] how candid to be: fair-info results (#117) aren't in yet

## Reward, value target, horizon
- ±1 at game end; truncated games score sign(VP lead)
- self-play = a fixed-length `lax.scan` with auto-reset; samples from unfinished games are masked
- horizon is a first-class hyperparameter
  - at 200 plies only 0.4% of catanatron games finish
  - needs ~700 (catanatron) / ~1400 (colonist)

## Network and training loop (brief: details live in the learning-signal post)
- residual MLP, pre-norm RMSNorm + ReLU, bf16 matmuls, 256×4 (~0.69M params), 294 logits + tanh value
- scaling 2×128 → 8×512 didn't move peak Elo (~630–670), so the bottleneck is the training signal
- self-play → sparse top-16 policy target + MC value → CE + MSE
- one-ply top-K value search as the operator; anchored target softmax(logits + q_scale·q)

## Evaluation setup (brief)
- per-ruleset Elo ladder
  - greedy top-16 one-ply search, 512 seat-swapped games per match
  - Bradley–Terry with random = 0; +15 Elo gate
- external opponent: catanatron AlphaBeta(n=2) via a lockstep bridge

## Recap bullets to close
- fixed-shape everything; 294 flat actions; mostly forced; chance as an action
- hidden info leaks easily through search; long horizon, sparse reward
- next: making `step` fast
