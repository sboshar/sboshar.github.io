"""Engine-only throughput: random 1v1 games, no network, no search.

Fair engine comparison for the blog (catanatron rules, first to 10 points, 1,000-turn
cap in both engines). Reports finished games per second and moves (engine steps) per
second.

usage:
  engine_random_play.py catanatron OUT_JSON [--procs N] [--games-per-proc N]
  engine_random_play.py catanax OUT_JSON --engine xla|default [--batch B] [--steps S]
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import time

TURNS_LIMIT = 1000  # catanatron's own turn cap; catanax uses the same below


def _catanatron_worker(args):
    seed, games = args
    import random

    from catanatron import Color, Game, RandomPlayer

    random.seed(seed)
    moves = 0
    for i in range(games):
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed * 100_000 + i)
        game.play()
        moves += len(game.state.actions)
    return games, moves


def run_catanatron(a):
    procs = a.procs or os.cpu_count()
    work = [(p + 1, a.games_per_proc) for p in range(procs)]
    with mp.Pool(procs) as pool:
        pool.map(_catanatron_worker, [(1000 + p, 2) for p in range(procs)])  # warm-up imports
        t0 = time.perf_counter()
        results = pool.map(_catanatron_worker, work)
        dt = time.perf_counter() - t0
    games = sum(g for g, _ in results)
    moves = sum(m for _, m in results)
    return {"engine": "catanatron (Python)", "procs": procs, "games": games, "moves": moves, "seconds": dt,
            "games_per_s": games / dt, "moves_per_s": moves / dt, "moves_per_game": moves / games}


def run_catanax(a):
    import jax
    import jax.numpy as jnp

    from catanax.engine import rules
    from catanax.training import self_play

    g = rules.game_for("catanatron")
    step = {"xla": g.step_batched_xla, "default": g.step_batched}[a.engine]
    batch = a.batch
    win_vp = int(g.rules.win_vp)

    def body(carry, _):
        states, key, games, moves = carry
        key, ak, sk, rk = jax.random.split(key, 4)
        legal = jax.vmap(g.legal_action_mask)(states) if a.engine == "xla" else g.legal_action_mask_batched(states)
        actions = jax.random.categorical(ak, jnp.where(legal, 0.0, -jnp.inf)).astype(jnp.int32)
        states = step(states, actions, jax.random.split(sk, batch))
        over = self_play.game_over(states.actual_vp, states.num_turns, win_vp, TURNS_LIMIT)
        fresh = jax.vmap(g.init)(jax.random.split(rk, batch))
        states = jax.tree.map(lambda n, o: jnp.where(over.reshape((-1,) + (1,) * (o.ndim - 1)), n, o), fresh, states)
        return (states, key, games + over.sum(), moves + batch), None

    run = jax.jit(lambda s, k: jax.lax.scan(body, (s, k, jnp.int32(0), jnp.int32(0)), None, length=a.steps)[0])
    states = jax.vmap(g.init)(jax.random.split(jax.random.PRNGKey(0), batch))
    # Warm up long enough that the batch holds a steady mix of game stages.
    states, *_ = jax.block_until_ready(run(states, jax.random.PRNGKey(1)))
    states, *_ = jax.block_until_ready(run(states, jax.random.PRNGKey(2)))
    t0 = time.perf_counter()
    _, _, games, moves = jax.block_until_ready(run(states, jax.random.PRNGKey(3)))
    dt = time.perf_counter() - t0
    games, moves = int(games), int(moves)
    return {"engine": f"catanax {a.engine}", "device": jax.devices()[0].device_kind, "batch": batch, "steps": a.steps,
            "games": games, "moves": moves, "seconds": dt, "games_per_s": games / dt, "moves_per_s": moves / dt,
            "moves_per_game": moves / max(games, 1)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("catanatron", "catanax"))
    p.add_argument("out")
    p.add_argument("--procs", type=int, default=0)
    p.add_argument("--games-per-proc", type=int, default=20)
    p.add_argument("--engine", choices=("xla", "default"), default="xla")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--steps", type=int, default=2000)
    a = p.parse_args()
    res = run_catanatron(a) if a.which == "catanatron" else run_catanax(a)
    res["machine"] = platform.platform()
    res["cpu"] = platform.processor() or platform.machine()
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
