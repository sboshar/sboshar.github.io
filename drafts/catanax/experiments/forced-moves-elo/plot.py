"""Plots and table for PLAN.md: Elo vs training iteration and vs real hours, both arms.

Inputs (copied from ~/exp-forced-results on the VM): off.log / on.log (each line prefixed
with a Unix timestamp) and round_robin.json. Writes elo_vs_iteration.png,
elo_vs_hours.png, timings.json and results.md.
"""
import json
import re

import matplotlib.pyplot as plt

ARMS = {"off": ("skip_forced_plies off", "#2a78d6", "o"), "on": ("skip_forced_plies on", "#eb6834", "s")}
ITER_LINE = re.compile(r"^(\d+\.\d+)\s+(\d+)\s+[-\d.]+\s+[-\d.]+\s+[\d.]+%")


def iteration_times(path):
    """Seconds since iteration 0's log line, for each iteration."""
    times = {}
    for line in open(path):
        m = ITER_LINE.match(line)
        if m:
            times.setdefault(int(m.group(2)), float(m.group(1)))
    start = times[0]
    return {i: t - start for i, t in times.items()}


rr = json.load(open("round_robin.json"))
elo = dict(zip(rr["labels"], rr["elo"]))
timings, points = {}, {}
for arm in ARMS:
    t = iteration_times(f"{arm}.log")
    timings[arm] = {"seconds_per_iteration": t[max(t)] / max(t), "iteration_seconds": t}
    # ckpt_<it> is saved after iteration <it> is logged; the last one is ckpt_1999.
    points[arm] = [(it, t[min(it, 1999)] / 3600, elo[f"{arm} @{it}"]) for it in range(250, 2001, 250)]
json.dump(timings, open("timings.json", "w"), indent=1)

for axis, xlabel, out in (
    (0, "training iteration (same gradient steps per iteration in both arms)", "elo_vs_iteration.png"),
    (1, "real training hours since iteration 0 (each run alone on the A100)", "elo_vs_hours.png"),
):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for arm, (label, color, marker) in ARMS.items():
        x = [p[axis] for p in points[arm]]
        y = [p[2] for p in points[arm]]
        ax.plot(x, y, "-", color=color, marker=marker, lw=2, ms=6, label=label)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("joint Elo (random = 0)")
    ax.set_title("Colonist paper_lite: skip_forced_plies off vs on, only difference the flag", fontsize=10.5)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)

with open("results.md", "w") as f:
    f.write("| Iteration | Off: hours | Off: Elo | On: hours | On: Elo |\n|---|---|---|---|---|\n")
    for (it, h0, e0), (_, h1, e1) in zip(points["off"], points["on"]):
        f.write(f"| {it} | {h0:.2f} | {e0:.0f} | {h1:.2f} | {e1:.0f} |\n")
    for arm in ARMS:
        f.write(f"\n{arm}: {timings[arm]['seconds_per_iteration']:.2f} s per iteration\n")
print(open("results.md").read())
