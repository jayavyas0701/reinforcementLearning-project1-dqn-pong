#!/usr/bin/env python3
"""
CMPE260: summarise the TensorBoard logs in runs/ into a results table and two plots.
Reads the reward_100 curve the repo scripts already log, so nothing extra is needed in training.

    python compare.py --logdir runs --target 19
"""
import argparse
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def method_of(run_dir):
    name = os.path.basename(run_dir)
    if name.endswith("-maxboltz") or "-maxboltz-" in name:
        m = "Max-Boltzmann"
    elif "-boltzmann" in name:
        m = "Boltzmann"
    else:
        m = "epsilon-greedy"
    return m + (" + PER" if name.endswith("-per") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="runs")
    ap.add_argument("--target", type=float, default=19.0)
    args = ap.parse_args()

    runs = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(args.logdir, "*"))):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if "reward_100" not in ea.Tags()["scalars"]:
            continue
        ev = ea.Scalars("reward_100")
        t0 = ea.FirstEventTimestamp()
        minutes = np.array([(e.wall_time - t0) / 60 for e in ev])
        frames = np.array([e.step for e in ev])
        values = np.array([e.value for e in ev])
        hit = np.nonzero(values >= args.target)[0]
        runs[method_of(d)].append(dict(
            dir=d, minutes=minutes, frames=frames, values=values,
            min_to_target=minutes[hit[0]] if len(hit) else None,
            frames_to_target=frames[hit[0]] if len(hit) else None,
            final=values[-1], best=values.max()))
    if not runs:
        print("No runs with reward_100 found in", args.logdir)
        return

    order = ["epsilon-greedy", "Boltzmann", "Max-Boltzmann"]
    keys = sorted(runs, key=lambda k: ("PER" in k, order.index(k.split(" +")[0])))

    def avg(rs, key):
        xs = [r[key] for r in rs if r[key] is not None]
        return float(np.mean(xs)) if xs else None

    base = runs.get("epsilon-greedy")
    b_min, b_frames = (avg(base, "min_to_target"), avg(base, "frames_to_target")) if base else (None, None)
    rows = []
    for k in keys:
        rs = runs[k]
        m, f = avg(rs, "min_to_target"), avg(rs, "frames_to_target")
        rows.append({
            "Method": k,
            "Runs reaching target": f"{sum(r['min_to_target'] is not None for r in rs)}/{len(rs)}",
            f"Minutes to {args.target:g}": f"{m:.1f}" if m else "not reached",
            f"Frames to {args.target:g} (k)": f"{f / 1000:.0f}" if f else "not reached",
            "Best mean-100 reward": f"{avg(rs, 'best'):.2f}",
            "Speed-up vs baseline": f"{(b_min - m) / b_min * 100:+.1f}%" if b_min and m else "–",
            "Fewer frames vs baseline": f"{(b_frames - f) / b_frames * 100:+.1f}%" if b_frames and f else "–",
        })
    table = pd.DataFrame(rows).to_markdown(index=False)
    print(table)
    with open(os.path.join(args.logdir, "results_table.md"), "w") as fh:
        fh.write(table + "\n")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for x_key, x_label, fname in [("minutes", "Wall-clock time (minutes)", "compare_time.png"),
                                  ("frames", "Frames (k)", "compare_frames.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, k in enumerate(keys):
            for j, r in enumerate(runs[k]):
                x = r[x_key] / 1000 if x_key == "frames" else r[x_key]
                ax.plot(x, r["values"], color=colors[i % len(colors)], label=k if j == 0 else None)
        ax.axhline(args.target, ls="--", color="gray", lw=1, label=f"target ({args.target:g})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean reward (last 100 games)")
        ax.set_title("DQN on Pong")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.logdir, fname), dpi=150)
        plt.close(fig)
    print("Wrote results_table.md, compare_time.png, compare_frames.png to", args.logdir)


if __name__ == "__main__":
    main()
