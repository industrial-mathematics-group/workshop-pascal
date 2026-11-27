"""Plot the measured coffee temperatures with the Newton's law of cooling fit.

Uses the fit from fit_cooling.py, so the parameters match results/summary.md.
The lower panel shows the residuals against the probe's +/-0.5 C accuracy band.
Needs numpy and matplotlib:

    python analysis/plot_cooling.py data/coffee-cooling.csv results/coffee-cooling-fit.png
"""

import math
import sys

sys.dont_write_bytecode = True  # importing fit_cooling shouldn't leave a __pycache__ behind

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_cooling as fc

PROBE_ACCURACY = 0.5  # degrees C (notes/lab-notes.md)
DRINKABLE = 60.0  # degrees C

BLUE, ORANGE = "#257de0", "#f47230"
INK, INK2, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def main(data_path, out_path):
    t, T = fc.read_measurements(data_path)
    a, b, k_err = fc.fit_line(t, [math.log(x - fc.T_ROOM) for x in T])
    k, T0 = -b, fc.T_ROOM + math.exp(a)

    def model(x):
        return fc.T_ROOM + (T0 - fc.T_ROOM) * np.exp(-k * np.asarray(x))

    t60 = math.log((T0 - fc.T_ROOM) / (DRINKABLE - fc.T_ROOM)) / k
    residuals = np.array(T) - model(t)

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax, ar) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax.axhline(fc.T_ROOM, color=INK2, lw=1, ls="--")
    ax.text(0.5, fc.T_ROOM - 1, f"Room {fc.T_ROOM} °C", ha="left", va="top", color=INK2, fontsize=10)
    tt = np.linspace(min(t), max(t), 300)
    ax.plot(tt, model(tt), color=ORANGE, lw=2, zorder=2,
            label=f"Newton fit: k = {k:.4f} ± {k_err:.4f} /min, T₀ = {T0:.1f} °C")
    ax.scatter(t, T, s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
               label=f"Measured (probe ±{PROBE_ACCURACY} °C)")
    ax.plot([t60, t60], [fc.T_ROOM, DRINKABLE], color=INK2, lw=1, ls=":")
    ax.plot([0, t60], [DRINKABLE, DRINKABLE], color=INK2, lw=1, ls=":")
    ax.annotate(f"{DRINKABLE:.0f} °C at {t60:.1f} min", (t60, DRINKABLE),
                xytext=(t60 + 3, DRINKABLE + 3), color=INK, fontsize=10)
    ax.set_ylabel("Temperature (°C)")
    ax.set_ylim(15, 92)
    ax.set_title("Coffee cooling: data and Newton's law fit", loc="left", color=INK, fontsize=13)
    ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK)

    ar.axhspan(-PROBE_ACCURACY, PROBE_ACCURACY, color=GRID, lw=0)
    ar.axhline(0, color=INK2, lw=1)
    ar.scatter(t, residuals, s=30, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    ar.set_ylabel("Residual (°C)")
    ar.set_xlabel("Time (min)")
    ar.set_xlim(-1.5, 61.5)
    ar.set_xticks(range(0, 61, 6))

    for axis in (ax, ar):
        axis.grid(axis="y", color=GRID, lw=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path} (residual RMS {np.sqrt((residuals ** 2).mean()):.2f} C)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/coffee-cooling.csv",
         sys.argv[2] if len(sys.argv) > 2 else "results/coffee-cooling-fit.png")
