# -*- coding: utf-8 -*-
"""
Regenerates the four manuscript figures (leakage forest plot, calibration
reliability diagrams, 5-dimension robustness bars, masking ablation bars)
from the experiment deliverables' source CSVs.

Expects the experiment_{a,e,f}_deliverables/ folders as siblings of this
repository's root (i.e. the same layout used throughout the study). Override
with the PNEUMONIA_DELIVERABLES_ROOT environment variable if your checkout
places them elsewhere. Output PNGs are written to OUT (defaults to a
"figures_out" folder next to this script; point it at your paper's
figures/ folder if regenerating in place).
"""
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("PNEUMONIA_DELIVERABLES_ROOT", str(REPO_ROOT.parent))
OUT = os.environ.get("PNEUMONIA_FIGURES_OUT", str(Path(__file__).resolve().parent / "figures_out"))
Path(OUT).mkdir(parents=True, exist_ok=True)

COLORS = {
    "densenet121": "#2a78d6",
    "efficientnet_b0": "#eb6834",
    "swin_tiny": "#1baf7a",
}
LABELS = {
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
    "swin_tiny": "Swin-Tiny",
}
ARCH_ORDER = ["densenet121", "efficientnet_b0", "swin_tiny"]

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_COLOR = "#0b0b0b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "text.color": TEXT_COLOR,
    "axes.edgecolor": AXIS_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

LEGEND_KW = dict(loc="lower left", frameon=False, fontsize=6.5)


# ---------------------------------------------------------------
# Figure 1: Leakage forest plot
# ---------------------------------------------------------------
def fig_leakage_forest():
    per_run = pd.read_csv(f"{BASE}/experiment_a_deliverables/source_csv/statistics/clean_vs_leaky_per_run.csv")
    agg = pd.read_csv(f"{BASE}/experiment_a_deliverables/source_csv/statistics/clean_vs_leaky_per_architecture.csv")
    agg = agg.set_index("architecture")

    # display order top-to-bottom: Swin-Tiny, EfficientNet-B0, DenseNet121
    display_order = ["swin_tiny", "efficientnet_b0", "densenet121"]
    y_pos = {a: i for i, a in enumerate(reversed(display_order))}

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=300)

    # per-seed dots
    for arch in display_order:
        sub = per_run[per_run["architecture"] == arch]
        y = y_pos[arch]
        ax.scatter(sub["dAUROC"], [y] * len(sub), s=18, color=COLORS[arch],
                   alpha=0.45, zorder=2, edgecolors="none")
    seed_dot_handle = ax.scatter([], [], s=18, color="#8a8d91", alpha=0.7, edgecolors="none")

    # mean marker + 95% CI error bar
    for arch in display_order:
        y = y_pos[arch]
        mean = agg.loc[arch, "dAUROC_mean"]
        sd = agg.loc[arch, "dAUROC_sd"]
        n = agg.loc[arch, "n_seeds"]
        se = sd / np.sqrt(n)
        ci = 1.96 * se
        ax.errorbar([mean], [y], xerr=[[ci], [ci]], fmt="D", color=COLORS[arch],
                    markersize=6, capsize=3, elinewidth=1.4, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)

    ax.axvline(0, color=AXIS_COLOR, linestyle="--", linewidth=1, zorder=1)
    ax.set_yticks([y_pos[a] for a in display_order])
    ax.set_yticklabels([LABELS[a] for a in display_order])
    ax.set_xlabel(r"$\Delta$AUROC (Leaky $-$ Clean)")
    ax.set_ylim(-0.6, 2.6)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="y", visible=False)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend([seed_dot_handle], ["Per-seed run"], loc="lower left",
               bbox_to_anchor=(0.02, 0.0), frameon=False, fontsize=6.5)
    fig.savefig(f"{OUT}/fig_leakage_forest.png", dpi=300)
    plt.close(fig)
    print("saved fig_leakage_forest.png")


# ---------------------------------------------------------------
# Figure 2: Calibration reliability diagrams
# ---------------------------------------------------------------
def fig_calibration_reliability():
    rel = pd.read_csv(f"{BASE}/experiment_a_deliverables/source_csv/statistics/reliability_diagram_data.csv")
    rel = rel[rel["count"] >= 3]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), dpi=300, sharey=True)

    clean_color = "#8a8d91"
    leaky_color = "#e34948"

    handles = None
    for ax, arch in zip(axes, ARCH_ORDER):
        sub = rel[rel["architecture"] == arch].copy()
        sub["bin_mid"] = (sub["bin_low"] + sub["bin_high"]) / 2
        ax.plot([0, 1], [0, 1], linestyle="--", color=AXIS_COLOR, linewidth=1, zorder=1)

        clean = sub[sub["condition"] == "clean"].sort_values("bin_mid")
        leaky = sub[sub["condition"] == "leaky"].sort_values("bin_mid")
        h1, = ax.plot(clean["mean_confidence"], clean["empirical_accuracy"], "-o",
                      color=clean_color, markersize=4, linewidth=1.4, zorder=2)
        h2, = ax.plot(leaky["mean_confidence"], leaky["empirical_accuracy"], "-s",
                      color=leaky_color, markersize=4, linewidth=1.4, zorder=2)
        if handles is None:
            handles = (h1, h2)

        ax.set_title(LABELS[arch], fontsize=8, color=TEXT_COLOR, pad=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Empirical accuracy")

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(handles, ["Clean", "Leaky"], loc="lower left",
               bbox_to_anchor=(0.01, 0.0), ncol=2, frameon=False, fontsize=6.5)
    fig.savefig(f"{OUT}/fig_calibration_reliability.png", dpi=300)
    plt.close(fig)
    print("saved fig_calibration_reliability.png")


# ---------------------------------------------------------------
# Figure 3: Robustness across 5 corruption dimensions
# ---------------------------------------------------------------
def fig_robustness_5dim():
    df = pd.read_csv(f"{BASE}/experiment_e_deliverables/source_csv/exp_e_robustness_score_5dim.csv")
    dims = ["noise", "blur", "jpeg", "res", "contrast"]
    dim_labels = ["Noise", "Blur", "JPEG", "Resolution", "Bright./\nContr."]

    fig, ax = plt.subplots(figsize=(4.2, 3.15), dpi=300)
    x = np.arange(len(dims))
    width = 0.26

    for i, arch in enumerate(ARCH_ORDER):
        sub = df[df["arch"] == arch].set_index("transform")
        vals = [sub.loc[d, "robustness_meanAUROC"] for d in dims]
        ax.bar(x + (i - 1) * width, vals, width, color=COLORS[arch], label=LABELS[arch])

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=6.5)
    ax.set_ylabel("Robustness score\n(mean AUROC across severities)")
    ax.set_ylim(0.5, 1.02)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.0),
               ncol=3, frameon=False, fontsize=6.5, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(f"{OUT}/fig_robustness_5dim.png", dpi=300)
    plt.close(fig)
    print("saved fig_robustness_5dim.png")


# ---------------------------------------------------------------
# Figure 4: Masking ablation
# ---------------------------------------------------------------
def fig_masking_ablation():
    df = pd.read_csv(f"{BASE}/experiment_f_deliverables/source_csv/ablation_summary.csv")
    conds = ["clean", "lung_blackout", "nonlung_blackout"]
    cond_labels = ["Clean\n(full image)", "Lung blackout\n(lungs removed)", "Nonlung blackout\n(lungs only)"]

    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=300)
    x = np.arange(len(conds))
    width = 0.26

    for i, arch in enumerate(ARCH_ORDER):
        sub = df[df["arch"] == arch].set_index("mask_condition")
        means = [sub.loc[c, "auroc_mean"] for c in conds]
        stds = [sub.loc[c, "auroc_std"] for c in conds]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=2,
               color=COLORS[arch], label=LABELS[arch],
               error_kw=dict(elinewidth=1, ecolor="#333333"))

    chance_line = ax.axhline(0.5, color=AXIS_COLOR, linestyle="--", linewidth=1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=6.5)
    ax.set_ylabel("AUROC (mean \u00b1 SD, n=5 seeds)")
    ax.set_ylim(0.4, 1.02)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    handles = [chance_line] + handles
    labels = ["Chance (AUROC=0.5)"] + labels
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.0),
               ncol=2, frameon=False, fontsize=6.5, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(f"{OUT}/fig_masking_ablation.png", dpi=300)
    plt.close(fig)
    print("saved fig_masking_ablation.png")


if __name__ == "__main__":
    fig_leakage_forest()
    fig_calibration_reliability()
    fig_robustness_5dim()
    fig_masking_ablation()
    print("all done")
