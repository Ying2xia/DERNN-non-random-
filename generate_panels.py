#!/usr/bin/env python3
"""Six publication-quality sub-panels for the DPS-ERNN sensitivity figures.
Pilot:   Ex1/Ex2/Ex3  (Centralized + DPS-ERNN over r), MAE+RMSE boxes + DPS time line.
Machine: S1/S2/S3      (Centralized + DPS-ERNN over M), MAE+RMSE boxes + DPS time line
                        with the centralized time marked for the speedup reference.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 11.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": 0.9, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE_E, BLUE_F = "#2B5D8A", "#BAD0E6"
ORNG_E, ORNG_F = "#BE6A2B", "#F0D3B2"
RED   = "#B23A3A"
SHADE = "#F4F4F2"
GRID  = "#CCCCCC"

os.makedirs("out/images", exist_ok=True)
rng = np.random.default_rng(0)

def style_box(bp, edge, face, lw=1.05):
    for b in bp["boxes"]:  b.set(facecolor=face, edgecolor=edge, linewidth=lw, alpha=0.95)
    for w in bp["whiskers"]: w.set(color=edge, linewidth=lw)
    for c in bp["caps"]:   c.set(color=edge, linewidth=lw)
    for md in bp["medians"]: md.set(color=edge, linewidth=1.5)
    for fl in bp["fliers"]: fl.set(marker="o", markersize=2.4, markerfacecolor=edge,
                                   markeredgecolor="none", alpha=0.45)

def jit(ax, x, v, color, w=0.07):
    ax.scatter(x + rng.uniform(-w, w, size=len(v)), v, s=8, color=color,
               edgecolor="white", linewidth=0.3, alpha=0.7, zorder=5)

def panel(fname, *, cen_mae, cen_rmse, dps_mae, dps_rmse, right_vals, swept_labels,
          right_label, right_max, right_legend, cen_time=None, annotate=None,
          left_max, xlabel=None, show_legend=False, legend_ctime=False):
    k = len(swept_labels)
    xpos = np.arange(1, k + 1, dtype=float)
    off, w = 0.205, 0.34
    fig, ax = plt.subplots(figsize=(7.0, 2.52))
    fig.subplots_adjust(left=0.10, right=0.90, top=0.875, bottom=0.185)
    ax.axvspan(-0.62, 0.62, color=SHADE, zorder=0)

    b = ax.boxplot([cen_mae],  positions=[-off], widths=w, patch_artist=True, manage_ticks=False)
    style_box(b, BLUE_E, BLUE_F); jit(ax, -off, cen_mae, BLUE_E)
    b = ax.boxplot([cen_rmse], positions=[ off], widths=w, patch_artist=True, manage_ticks=False)
    style_box(b, ORNG_E, ORNG_F); jit(ax, off, cen_rmse, ORNG_E)
    bm = ax.boxplot(dps_mae,  positions=xpos - off, widths=w, patch_artist=True, manage_ticks=False)
    style_box(bm, BLUE_E, BLUE_F)
    for x, v in zip(xpos - off, dps_mae): jit(ax, x, v, BLUE_E)
    br = ax.boxplot(dps_rmse, positions=xpos + off, widths=w, patch_artist=True, manage_ticks=False)
    style_box(br, ORNG_E, ORNG_F)
    for x, v in zip(xpos + off, dps_rmse): jit(ax, x, v, ORNG_E)

    ax.set_xticks([0] + list(xpos))
    ax.set_xticklabels(["Centralized"] + swept_labels)
    ax.set_xlim(-0.72, k + 0.7)
    ax.set_ylim(0, left_max)
    ax.set_ylabel("Test error")
    ax.yaxis.grid(True, color=GRID, linestyle=(0, (4, 3)), linewidth=0.55, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    if xlabel: ax.set_xlabel(xlabel)

    ax2 = ax.twinx()
    if cen_time is not None:
        ax2.scatter([0], [cen_time], color=RED, marker="D", s=34, zorder=7)
    ax2.plot(xpos, right_vals, color=RED, marker="o", markersize=4.8, markerfacecolor="white",
             markeredgecolor=RED, markeredgewidth=1.3, linewidth=1.7, zorder=6)
    if annotate:
        ax2.text(xpos.mean(), float(np.mean(right_vals)) + 0.55, annotate, color=RED,
                 ha="center", va="bottom", fontsize=9.5)
    ax2.set_ylim(0, right_max)
    ax2.set_ylabel(right_label, color=RED)
    ax2.tick_params(axis="y", colors=RED, labelsize=9.5)
    ax2.spines["right"].set_color(RED); ax2.spines["top"].set_visible(False)

    if show_legend:
        h = [Patch(facecolor=BLUE_F, edgecolor=BLUE_E, label="MAE"),
             Patch(facecolor=ORNG_F, edgecolor=ORNG_E, label="RMSE"),
             Line2D([0], [0], color=RED, marker="o", markerfacecolor="white",
                    markeredgecolor=RED, lw=1.7, label=right_legend)]
        if legend_ctime:
            h.append(Line2D([0], [0], color=RED, marker="D", linestyle="none", label="Centralized time"))
        ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=len(h),
                  frameon=True, framealpha=0.95, edgecolor="#BBBBBB", fontsize=9,
                  columnspacing=1.2, handletextpad=0.45, borderpad=0.4)

    fig.savefig(f"out/images/{fname}.pdf")
    fig.savefig(f"out/{fname}.png", dpi=150)
    plt.close(fig)

# ---------------- load data ----------------
p = pd.read_csv("results/pilot_sensitivity.csv")
m = pd.read_csv("results/machine_sensitivity.csv")
r = pd.read_csv("results/raw_results.csv")
ratios = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
rlab = ["0.5", "1", "2", "5", "10", "20"]
Ms = [5, 10, 20, 50, 100]

def cen_example(ex, col):
    c = r[(r.method == "Centralized") & (r.error == "normal") & (np.isclose(r.tau, 0.5))
          & (r.example == ex) & (r.storage_strategy == 2)]
    return c[col].values

# pilot panels: common time axis 0-13; per-example error axis
PILOT_RMAX = 13.0
pilot_lmax = {1: 0.082, 2: 0.097, 3: 0.92}
for i, ex in enumerate([1, 2, 3]):
    pe = p[p.example == ex]
    dmae  = [pe[pe.pilot_ratio == rr].MAE.values  for rr in ratios]
    drmse = [pe[pe.pilot_ratio == rr].RMSE.values for rr in ratios]
    dtime = [pe[pe.pilot_ratio == rr].time_seconds.mean() for rr in ratios]
    panel(f"fig_pilot_ex{ex}",
          cen_mae=cen_example(ex, "MAE"), cen_rmse=cen_example(ex, "RMSE"),
          dps_mae=dmae, dps_rmse=drmse, right_vals=dtime, swept_labels=rlab,
          right_label="Mean computation time (s)", right_max=PILOT_RMAX,
          right_legend="DPS-ERNN time",
          left_max=pilot_lmax[ex],
          xlabel=(r"Pilot ratio $r$ (%)" if ex == 3 else None),
          show_legend=(ex == 1))

# machine panels: common error axis with headroom for the RACT line; RACT axis 0-13.5
cen2 = m[m.method == "Centralized"]
cm = cen2.groupby("replication").MAE.first().values
cr = cen2.groupby("replication").RMSE.first().values
MACH_LMAX = 0.0705   # headroom so the RACT curve sits above the boxes
for s in [1, 2, 3]:
    ms = m[(m.method == "DPS-ERNN") & (m.storage_strategy == s)]
    dmae  = [ms[ms.M == M].MAE.values  for M in Ms]
    drmse = [ms[ms.M == M].RMSE.values for M in Ms]
    ract  = [ms[ms.M == M].RACT.mean() for M in Ms]
    panel(f"fig_machine_s{s}",
          cen_mae=cm, cen_rmse=cr, dps_mae=dmae, dps_rmse=drmse, right_vals=ract,
          swept_labels=[str(M) for M in Ms],
          right_label=r"Speedup over centralized (RACT, $\times$)", right_max=13.5,
          right_legend="DPS-ERNN speedup (RACT)", annotate=r"$\approx 11\times$ speedup",
          left_max=MACH_LMAX,
          xlabel=(r"Number of worker machines $M$" if s == 3 else None),
          show_legend=(s == 1))

print("6 panels written to out/images/.")
