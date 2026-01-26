#!/usr/bin/env python3
"""
Generate MI plots from a mi_metrics.csv file.

Usage:
  python plot_mi_metrics.py /path/to/mi_metrics.csv

Optional:
  --out1 PATH   Output PDF for the "conditioned on z" plot
  --out2 PATH   Output PDF for the "conditioned on carries" plot
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for CLI / headless runs
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------- Defaults / styling ----------
FIGSIZE = (8, 4)
LINEWIDTH = 2.5
ALPHA_BASE = 0.6
LEGEND_FONTSIZE = 12

HGRID_KW = dict(axis="y", linestyle=":", linewidth=0.5, alpha=0.18, color="gray")

LABEL_FONTSIZE = 13
TICK_FONTSIZE = 12
MAX_X_DISPLAY = 200_000

# small amount of padding when saving the figure
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.02)
# --------------------------------------


def get_series(df: pd.DataFrame, col: str):
    if col not in df.columns:
        raise KeyError(
            f"Column '{col}' not found in CSV. Available columns: {list(df.columns)}"
        )
    return df[col].to_numpy()


def k_formatter(x_val, pos=None):
    if x_val == 0:
        return "0"
    k = x_val / 1000.0
    if abs(k - round(k)) < 1e-6:
        return f"{int(round(k))}K"
    return f"{k:.1f}K"


def parse_args():
    p = argparse.ArgumentParser(description="Plot MI metrics from a CSV.")
    p.add_argument(
        "csv_path",
        type=Path,
        help="Path to mi_metrics.csv",
    )
    p.add_argument(
        "--out1",
        type=Path,
        default=None,
        help="Output PDF path for MI conditioned on z (default: alongside CSV).",
    )
    p.add_argument(
        "--out2",
        type=Path,
        default=None,
        help="Output PDF path for MI conditioned on carries (default: alongside CSV).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out1 = (args.out1.expanduser().resolve() if args.out1 else (csv_path.parent / "mi_conditioned_on_z_plot_1M.pdf"))
    out2 = (args.out2.expanduser().resolve() if args.out2 else (csv_path.parent / "mi_conditioned_on_carries_plot_1M.pdf"))

    df = pd.read_csv(csv_path)
    x = get_series(df, "iter")

    kfmt = FuncFormatter(k_formatter)

    colors = plt.get_cmap("tab10").colors
    place_to_color = {
        "Thousands": colors[0],
        "Hundreds": colors[1],
        "Tens": colors[2],
        "Units": colors[3],
    }

    legend_kwargs = dict(
        fontsize=LEGEND_FONTSIZE,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.9),
        frameon=True,
        framealpha=0.92,
        fancybox=True,
        borderaxespad=0.4,
    )

    # ---------------- FIGURE 1 ----------------
    pairs_first = [
        {"place": "Thousands", "base_col": "mi/thousands-base",   "model_col": "mi/thousands"},
        {"place": "Hundreds",  "base_col": "mi/hundreds-z-base",  "model_col": "mi/hundreds-z"},
        {"place": "Tens",      "base_col": "mi/tens-z-base",      "model_col": "mi/tens-z"},
        {"place": "Units",     "base_col": "mi/units-z-base",     "model_col": "mi/units-z"},
    ]

    fig, axs = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [2.25, 1]}
    )
    ax_top, ax_bottom = axs

    base_handles, model_handles, base_labels, model_labels = [], [], [], []

    for p in pairs_first:
        place = p["place"]
        color = place_to_color[place]
        base_y = get_series(df, p["base_col"])
        h_base, = ax_top.plot(
            x, base_y, linestyle="--", linewidth=LINEWIDTH, alpha=ALPHA_BASE, color=color
        )
        base_handles.append(h_base)
        base_labels.append(f"{place} place (data)")

    for p in pairs_first:
        place = p["place"]
        color = place_to_color[place]
        y_model = get_series(df, p["model_col"])
        h_model, = ax_top.plot(x, y_model, linewidth=LINEWIDTH, color=color)
        model_handles.append(h_model)
        model_labels.append(f"{place} place (model)")

    ax_top.legend(base_handles + model_handles, base_labels + model_labels, **legend_kwargs)
    ax_top.set_ylabel("Mutual information", fontsize=LABEL_FONTSIZE)
    ax_top.grid(**HGRID_KW)
    ax_top.relim()
    ax_top.autoscale_view()
    ymin, ymax = ax_top.get_ylim()
    ax_top.set_ylim(max(0.0, ymin), ymax)

    ax_bottom.plot(x, get_series(df, "train_loss"), linewidth=LINEWIDTH)
    ax_bottom.set_ylabel("Train loss", fontsize=LABEL_FONTSIZE)
    ax_bottom.set_xlabel("Training steps", fontsize=LABEL_FONTSIZE)
    ax_bottom.grid(**HGRID_KW)
    ax_bottom.set_xlim(0, MAX_X_DISPLAY)
    ax_bottom.xaxis.set_major_formatter(kfmt)

    ax_bottom.relim()
    ax_bottom.autoscale_view()
    bymin, bymax = ax_bottom.get_ylim()
    ax_bottom.set_ylim(max(0.0, bymin), bymax)

    ax_top.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax_bottom.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(bottom=0.07, top=0.95, hspace=0.12)
    fig.savefig(str(out1), **SAVE_KW)
    plt.close(fig)
    print(f"Saved first figure to {out1}")

    # ---------------- FIGURE 2 ----------------
    pairs_second = [
        {"place": "Thousands", "base_col": "mi/thousands-base",          "model_col": "mi/thousands"},
        {"place": "Hundreds",  "base_col": "mi/hundreds-carries-base",  "model_col": "mi/hundreds-carries"},
        {"place": "Tens",      "base_col": "mi/tens-carries-base",      "model_col": "mi/tens-carries"},
        {"place": "Units",     "base_col": "mi/units-carries-base",     "model_col": "mi/units-carries"},
    ]

    fig2, axs2 = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [2.25, 1]}
    )
    ax2_top, ax2_bottom = axs2

    base_handles, model_handles, base_labels, model_labels = [], [], [], []

    for p in pairs_second:
        place = p["place"]
        color = place_to_color[place]
        base_y = get_series(df, p["base_col"])
        h_base, = ax2_top.plot(
            x, base_y, linestyle="--", linewidth=LINEWIDTH, alpha=ALPHA_BASE, color=color
        )
        base_handles.append(h_base)
        base_labels.append(f"{place} place (data)")

    for p in pairs_second:
        place = p["place"]
        color = place_to_color[place]
        y_model = get_series(df, p["model_col"])
        h_model, = ax2_top.plot(x, y_model, linewidth=LINEWIDTH, color=color)
        model_handles.append(h_model)
        model_labels.append(f"{place} place (model)")

    ax2_top.legend(base_handles + model_handles, base_labels + model_labels, **legend_kwargs)
    ax2_top.set_ylabel("Mutual information", fontsize=LABEL_FONTSIZE)
    ax2_top.grid(**HGRID_KW)
    ax2_top.relim()
    ax2_top.autoscale_view()
    ymin2, ymax2 = ax2_top.get_ylim()
    ax2_top.set_ylim(max(0.0, ymin2), ymax2)

    ax2_bottom.plot(x, get_series(df, "train_loss"), linewidth=LINEWIDTH)
    ax2_bottom.set_ylabel("Train loss", fontsize=LABEL_FONTSIZE)
    ax2_bottom.set_xlabel("Training steps", fontsize=LABEL_FONTSIZE)
    ax2_bottom.grid(**HGRID_KW)
    ax2_bottom.set_xlim(0, MAX_X_DISPLAY)
    ax2_bottom.xaxis.set_major_formatter(kfmt)

    ax2_bottom.relim()
    ax2_bottom.autoscale_view()
    bymin2, bymax2 = ax2_bottom.get_ylim()
    ax2_bottom.set_ylim(max(0.0, bymin2), bymax2)

    ax2_top.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax2_bottom.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    fig2.subplots_adjust(bottom=0.07, top=0.95, hspace=0.12)
    fig2.savefig(str(out2), **SAVE_KW)
    plt.close(fig2)
    print(f"Saved second figure to {out2}")


if __name__ == "__main__":
    main()
