"""Preview a SaR map to figures/map_example.png. Run: python show_map.py [size] [seed]."""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: save to file instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

from src.simulator import Simulator
from src.constants import VictimPresence, VulnerabilityLevel, FireLevel
from src.rl.configs import sar_config


def render(width, height, config, seed, start=(0, 0), path="figures/map_example.png"):
    sim = Simulator(width, height)
    sim.generate_ground_truth(config, seed=seed)
    gt = sim.ground_truth
    trav = gt.traversability.matrix.astype(int)    # 0 floor, 1 wall
    vict = gt.victims.matrix.astype(int)
    vuln = gt.vulnerability.matrix.astype(float)    # 0.0 / 0.6 / 1.0
    fire = gt.fire.matrix.astype(int)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(trav, cmap="binary", interpolation="nearest", vmin=0, vmax=1)

    vmask = np.where(vuln > VulnerabilityLevel.SAFE.value, vuln, np.nan)
    ax.imshow(vmask, cmap="autumn_r", interpolation="nearest", alpha=0.35, vmin=0.5, vmax=1.0)

    for level, colour, alpha in [
        (FireLevel.FLAMMABLE, "gold", 0.5),
        (FireLevel.BURNING, "darkorange", 0.85),
        (FireLevel.BURNT, "dimgray", 0.9),
    ]:
        mask = np.where(fire == int(level), 1, np.nan)
        ax.imshow(mask, cmap=mcolors.ListedColormap([colour]), interpolation="nearest", alpha=alpha)

    vy, vx = np.nonzero(vict == VictimPresence.PRESENT)
    ax.scatter(vx, vy, marker="o", s=90, c="red", edgecolors="black", linewidths=0.6, zorder=5)
    ax.scatter([start[0]], [start[1]], marker="*", s=280, c="dodgerblue",
               edgecolors="black", linewidths=0.7, zorder=6)

    n_vict = int((vict == VictimPresence.PRESENT).sum())
    n_floor = int((trav == 0).sum())
    ax.set_title(f"{width}x{height} SaR map (seed {seed})  -  {n_vict} victims, {n_floor} floor cells")
    legend = [
        mpatches.Patch(facecolor="white", edgecolor="gray", label="floor"),
        mpatches.Patch(facecolor="black", label="wall"),
        mpatches.Patch(facecolor="red", edgecolor="black", label="victim"),
        mpatches.Patch(facecolor="dodgerblue", edgecolor="black", label="agent start"),
        mpatches.Patch(facecolor="orange", alpha=0.4, label="vulnerable terrain"),
        mpatches.Patch(facecolor="gold", alpha=0.6, label="flammable"),
        mpatches.Patch(facecolor="darkorange", alpha=0.85, label="burning"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}  ({n_vict} victims, {n_floor} floor cells)")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    render(size, size, sar_config(size, num_victims=10), seed=seed)
