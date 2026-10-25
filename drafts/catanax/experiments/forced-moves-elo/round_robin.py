"""Joint round robin over both arms' checkpoints, Bradley-Terry Elo with random = 0 (PLAN.md)."""
import json
import pathlib

import jax
import numpy as np

from catanax import eval as E
from catanax.elo import fit_elo
from catanax.engine import rules as catan

base = pathlib.Path.home() / "exp-forced-results"
game = catan.game_for("colonist")
labels, agents, archs = ["random"], [None], [E.T.Arch()]
for arm in ("off", "on"):
    for it in range(250, 2001, 250):
        name = "ckpt_1999.pkl" if it == 2000 else f"ckpt_{it}.pkl"  # the last is saved at 1999
        params, arch, _ = E.load_model(str(base / arm / name))
        labels.append(f"{arm} @{it}")
        agents.append(params)
        archs.append(arch)
schemas = [catan.OBS_SCHEMA_VERSION] * len(agents)
pairs = len(agents) * (len(agents) - 1) // 2
print(f"{len(agents)} agents, {pairs} pairs, {pairs * 2 * E.RR_BATCH} games", flush=True)
w, n = E.round_robin(agents, archs, jax.random.PRNGKey(7), game, schemas=schemas)
elo = fit_elo(w, n, anchor=0)
for i in np.argsort(-elo):
    print(f"  {labels[i]:12s} {elo[i]:+6.0f}")
json.dump(
    {"labels": labels, "elo": [float(x) for x in elo], "wins": np.asarray(w).tolist(), "games": np.asarray(n).tolist()},
    open(base / "round_robin.json", "w"),
    indent=1,
)
