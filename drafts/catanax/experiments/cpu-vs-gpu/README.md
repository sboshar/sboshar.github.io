# CPU vs GPU, like for like

These are measurements for "Catan on a GPU". They answer: how much faster is the GPU, comparing the same work on both? Two comparisons, each measured with one setup on both sides.

## 1. Self-play with the network: the same code, search and network

Catanatron rules, trained 256×4 network (`paper_lite_obsv2`), one-ply top-16 search, 2,048 games, `benchmarks/training.py`. Both machines play bit-identical games, about 8.3 finished per step.

| Setup | ms per step | Games/s | catanatron paper_lite run (3,800 iterations) |
|---|---|---|---|
| M3 Pro laptop CPU, #137 code (`6cb7bdd`), JAX CPU | 639 | **13** | ~476 h |
| A100, #137 code, `vmap` XLA engine | — | 126 | 49 h (from PR #137) |
| M3 Pro laptop CPU, today's code (`08caf35`) | 237 | 35 | ~179 h |
| A100, today's code (`08caf35`) | 1.47 | **5,687** | 1.16 h (from PR #147) |

- **The first GPU version** was about 10× the laptop on the same code.
- **Today's GPU** is about 160× today's code on the laptop.

Files: `selfplay_cpu_catanatron_pr137.json` and `selfplay_cpu_catanatron.json`.

## 2. Engine only: random play, no network

Catanatron rules, first to 10 points, 1,000-turn cap. Random legal moves, finished games per second. `engine_random_play.py`.

| Engine | Hardware | Random games/s | Moves/s | Moves per game |
|---|---|---|---|---|
| catanatron 3.2.1 (Python) | M3 Pro, 12 processes | 519 | 468k | 903 |
| catanax `vmap` | M3 Pro CPU, 4,096 games | 83 | 75k | 914 |
| catanax `vmap` | A100, 4,096 games | 822 | 751k | 914 |
| catanax `vmap` | A100, 16,384 games | 499 | 455k | 911 |
| catanax kernel, one program per game | A100, 4,096 games | 13,333 | 12.2M | 914 |
| catanax kernel, one program per game | A100, 16,384 games | 19,184 | 17.5M | 911 |

- **`vmap` on the A100 is only 1.6× catanatron's Python on a laptop, and it gets slower with a bigger batch.**
  - Each step waits for the slowest game's longest-road search, a loop with a data-dependent count.
  - On a GPU, each iteration of that loop costs a host round trip.
  - A bigger batch has a slower slowest game.
- **The kernel** is 37× catanatron and 23× `vmap` on the same GPU, and it speeds up with batch size.
- **Same game:** every row averages 903–914 moves per game.

catanatron is the PyPI release 3.2.1. The project pins a local 3.3.0, which isn't on PyPI.

Files:
- `random_catanatron_m3pro.json`
- `random_catanax_xla_cpu_b4096.json`
- `gpu_xla_b4096_rerun.json`. The first XLA run at batch 4,096 overlapped another job by a few seconds; the clean rerun gave 822 against 809.
- `gpu_xla_b16384.json`
- `gpu_default_b4096.json`
- `gpu_default_b16384.json`
- `gpu_engine_queue.sh`, the GPU runner
