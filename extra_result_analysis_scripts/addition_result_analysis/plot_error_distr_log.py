#!/usr/bin/env python3
"""
plot_error_distr_log.py

Usage:
    python plot_error_distr_log.py [input.csv] [output_dir] [--log] [--use-columns COL1 COL2 ...]

Reads a CSV with a 'difference' column and iteration columns (e.g. iter_8000...).
- Plots each iteration column as its own line (with markers).
- Also saves a summed_counts.csv and plots the summed totals as a separate image.
--log : plot y-axis on a logarithmic scale using y+1 to handle zeros.
--use-columns : optionally provide a list of iteration columns to plot (defaults to all except 'difference').
"""

import sys
import os
from typing import List, Optional
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Defaults (you can override from CLI)
Input_file_path = '/Users/perfectpeter/Library/CloudStorage/GoogleDrive-llmfunexperiment@gmail.com/My Drive/addition/results/4_operands_0_to_999_uniform_wo_padding/reverse_out_with_complete_stats_measure/4_operands_0_to_999_uniform_wo_padding_with_complete_stats_measure_reverse_2/difference_histograms_8000_160000.csv'
Output_file_path = '/Users/perfectpeter/Library/CloudStorage/GoogleDrive-llmfunexperiment@gmail.com/My Drive/addition/results/4_operands_0_to_999_uniform_wo_padding/reverse_out_with_complete_stats_measure/4_operands_0_to_999_uniform_wo_padding_with_complete_stats_measure_reverse_2/plots'


def discover_iter_columns(df: pd.DataFrame, diff_col_name: str = "difference") -> List[str]:
    """Return columns to plot (all columns except the difference column)."""
    return [c for c in df.columns if c != diff_col_name]


def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_per_column_lines(df: pd.DataFrame, diffs: np.ndarray, iter_cols: List[str],
                          outpath: str, title: str, use_log: bool = False):
    """
    Plot each column in iter_cols as its own line (with markers).
    If use_log is True, plot y + 1 and set the y-axis to log scale.
    """
    plt.figure(figsize=(11, 6))

    if use_log:
        plt.yscale('log')
        ylabel = "count (plotted on log scale; shown as y + 1)"
    else:
        ylabel = "count"

    # Choose a colormap that can handle many lines
    cmap = plt.get_cmap('tab20')
    ncols = len(iter_cols)

    for i, col in enumerate(iter_cols):
        y = df[col].to_numpy(dtype=float)
        y_plot = y + 1 if use_log else y
        # cycle colors from colormap
        color = cmap(i % cmap.N)
        plt.plot(diffs, y_plot, marker='o', linestyle='-', label=col, markersize=4, linewidth=1, color=color, alpha=0.9)

    plt.xlabel("difference (actual - predicted)")
    plt.ylabel(ylabel)
    plt.title(title)

    # sample xticks so they are readable
    N = max(1, len(diffs) // 12)
    plt.xticks(diffs[::N], rotation=45)

    plt.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    # Place legend outside the plot when there are several columns
    if ncols <= 8:
        plt.legend(loc='best', fontsize='small')
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summed_counts(diffs: np.ndarray, totals: np.ndarray, outpath: str,
                       title: str = "Summed totals (points)", use_log: bool = False):
    plt.figure(figsize=(10, 5))
    if use_log:
        totals_plot = totals + 1
        plt.yscale('log')
        plt.ylabel("total count (plotted on log scale; shown as y + 1)")
    else:
        totals_plot = totals
        plt.ylabel("total count")
    plt.plot(diffs, totals_plot, marker='o', linestyle='-', markersize=5)
    plt.xlabel("difference (actual - predicted)")
    plt.title(title)
    N = max(1, len(diffs) // 12)
    plt.xticks(diffs[::N], rotation=45)
    plt.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main(infile: str, outdir: str = "plots", use_columns: Optional[List[str]] = None, use_log: bool = False):
    df = pd.read_csv(infile)
    df.columns = [c.strip() for c in df.columns]

    if "difference" not in df.columns:
        raise ValueError("CSV must contain a 'difference' column.")

    df["difference"] = df["difference"].astype(int)
    df = df.sort_values("difference").reset_index(drop=True)

    iter_cols = use_columns if use_columns is not None else discover_iter_columns(df, diff_col_name="difference")
    if len(iter_cols) == 0:
        raise ValueError("No iteration columns found to plot.")

    # Convert to numeric and fill NaN with 0
    for c in iter_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Sum across the selected columns (row-wise) and save CSV including the individual columns
    df["total_count"] = df[iter_cols].sum(axis=1)

    make_output_dir(outdir)
    summed_csv = os.path.join(outdir, "summed_counts.csv")
    df[["difference", "total_count"] + iter_cols].to_csv(summed_csv, index=False)
    print(f"Saved summed counts CSV to: {summed_csv}")

    diffs = df["difference"].to_numpy()
    totals = df["total_count"].to_numpy()

    # Plot per-column lines
    percol_png = os.path.join(outdir, f"per_column_lines{'_log' if use_log else ''}.png")
    plot_per_column_lines(df, diffs, iter_cols, percol_png,
                          title="Per-column error distribution", use_log=use_log)
    print(f"Saved per-column plot to: {percol_png}")

    # Plot summed totals (marker/line)
    summed_png = os.path.join(outdir, f"summed_totals{'_log' if use_log else ''}.png")
    plot_summed_counts(diffs, totals, summed_png, use_log=use_log)
    print(f"Saved summed plot to: {summed_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot difference -> counts per iteration column (and summed totals).")
    parser.add_argument('infile', nargs='?', default=Input_file_path, help='Input CSV file path')
    parser.add_argument('outdir', nargs='?', default=Output_file_path, help='Output directory')
    parser.add_argument('--log', action='store_true', help='Plot y-axis on log scale (uses y+1 to handle zeros).')
    parser.add_argument('--use-columns', nargs='+', help='Specific iteration columns to use (optional).')
    args = parser.parse_args()

    main(args.infile, args.outdir, use_columns=args.use_columns, use_log=args.log)
