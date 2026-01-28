#!/usr/bin/env python3
"""
Make a bar plot of model accuracy vs. number of added clauses (k1..k6).

- Reads a CSV with columns: model,k1,k2,k3,k4,k5,k6 (raw correct counts out of 50)
- Bars are colored by k (k=1..6), with a legend at the top
- No title

Usage:
  python plot_accuracy_bars.py path/to/results.csv
Optional:
  python plot_accuracy_bars.py path/to/results.csv --out plot.png
  python plot_accuracy_bars.py path/to/results.csv --show

If --out is omitted, the plot is saved as a PDF next to the input CSV
(e.g. results.csv -> results.pdf).

Configurable at top:
  AXIS_LABEL_FONT_SIZE - font size used for x/y axis labels
  TICKS_FONT_SIZE      - font size used for tick labels
  LEGEND_FONT_SIZE     - font size used for legend
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# User-configurable appearance
# ----------------------------
AXIS_LABEL_FONT_SIZE = 16
TICKS_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
X_TICK_LABEL_FONT_SIZE = 13   # <- new: smaller font for model names

# ----------------------------


def plot_from_csv(csv_path: str, out_path: Optional[str] = None, show: bool = False) -> None:
    df = pd.read_csv(csv_path)

    required = ["model", "k1", "k2", "k3", "k4", "k5", "k6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    k_cols = [f"k{i}" for i in range(1, 7)]

    long_df = df.melt(id_vars="model", value_vars=k_cols, var_name="k", value_name="correct")
    long_df["k_num"] = long_df["k"].str.replace("k", "", regex=False).astype(int)
    long_df["accuracy"] = long_df["correct"] / 50.0

    model_order = df["model"].tolist()
    long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)
    long_df = long_df.sort_values(["model", "k_num"]).reset_index(drop=True)

    n_models = len(model_order)
    bars_per_model = 6

    # --- helper: wrap labels into (at most) 2 lines ---
    def wrap_label_two_lines(label: str) -> str:
        if "\n" in label:
            return label
        if "-" in label:
            left, right = label.split("-", 1)
            return left + "\n" + right
        words = label.split()
        if len(words) <= 2:
            return label
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

    wrapped_labels = [wrap_label_two_lines(l) for l in model_order]

    # --- define colors (this fixes the NameError) ---
    cmap = plt.get_cmap("tab10")
    k_to_color: Dict[int, tuple] = {i: cmap((i - 1) % 10) for i in range(1, bars_per_model + 1)}

    # Use a slightly shorter figure height to save vertical space
    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.9), 4.2))

    group_positions = np.arange(n_models)
    group_width = 0.85
    bar_width = group_width / bars_per_model

    acc_by_k: List[np.ndarray] = []
    for k in range(1, bars_per_model + 1):
        acc = long_df[long_df["k_num"] == k].sort_values("model")["accuracy"].to_numpy()
        if acc.shape[0] != n_models:
            raise RuntimeError(f"expected {n_models} accuracy values for k={k}, got {acc.shape[0]}")
        acc_by_k.append(acc)

    offsets = (np.arange(bars_per_model) - (bars_per_model - 1) / 2.0) * bar_width

    for idx_k, (offset, acc) in enumerate(zip(offsets, acc_by_k), start=1):
        ax.bar(group_positions + offset, acc, width=bar_width * 0.95, color=k_to_color[idx_k], label=f"k={idx_k}")

    # Apply wrapped labels, horizontally, centered
    ax.set_xticks(group_positions)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=X_TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICKS_FONT_SIZE)

    for i in range(1, n_models):
        ax.axvline(i - 0.5, linewidth=1, color="lightgray")

    ax.set_ylabel("Accuracy", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, n_models - 0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=k_to_color[i]) for i in range(1, 7)]
    labels = [f"k={i}" for i in range(1, 7)]
    ax.legend(
        handles,
        labels,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        columnspacing=1.0,
        title='k = Number of newly added clauses',
        title_fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.1,
        handletextpad=0.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")

    if show:
        plt.show()

    plt.close(fig)




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV with columns: model,k1,k2,k3,k4,k5,k6 (raw correct counts out of 50)")
    ap.add_argument(
        "--out",
        help="Output image path (e.g., plot.png). If omitted, saves a PDF next to the CSV.",
        default=None,
    )
    ap.add_argument("--show", action="store_true", help="Show plot window even if --out is provided.")
    args = ap.parse_args()

    in_path = Path(args.csv)
    if not in_path.exists():
        print(f"ERROR: CSV not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Decide output path:
    if args.out is None:
        out_path = in_path.with_suffix(".pdf")
    else:
        out_path = Path(args.out)
        # If user passed a directory, place file inside it using CSV stem & .pdf
        if out_path.exists() and out_path.is_dir():
            out_path = out_path / in_path.with_suffix(".pdf").name

    plot_from_csv(str(in_path), out_path=str(out_path), show=args.show)


if __name__ == "__main__":
    main()
