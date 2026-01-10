#!/usr/bin/env python3
"""
plot_error_distr.py

Usage:
    python plot_error_distr.py input.csv [output_dir]

Reads a CSV with a 'difference' column and one or more count columns (e.g. 'first_phase'
or 'iter_8000', ...). Draws:
  - a bar chart (PNG) for the column(s), saved to output_dir/
  - a combined line plot overlaying all columns (if there's only one column it still
    writes a combined plot with that single line)

Notes:
  - By default this script excludes the row where difference == 0 from plots.
  - Keeps numeric counts as floats (no forced integer conversion).
"""

import sys
import os
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default paths (you can override by passing args)
Input_file_path = '/Users/perfectpeter/Library/CloudStorage/GoogleDrive-llmfunexperiment@gmail.com/My Drive/addition/extra_result_analysis_scripts/addition_result_analysis/difference_histograms_zero_phase_single_iter.csv'
Output_dir = '/Users/perfectpeter/Library/CloudStorage/GoogleDrive-llmfunexperiment@gmail.com/My Drive/addition/extra_result_analysis_scripts/addition_result_analysis/plots_zero_phase_single_iter'

def discover_iter_columns(df: pd.DataFrame, diff_col_name: str = "difference") -> List[str]:
    """Return columns to plot (all columns except the difference column)."""
    return [c for c in df.columns if c != diff_col_name]

def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_single_iteration(diffs: np.ndarray, counts: np.ndarray, iter_name: str, outpath: str):
    """
    Create a single bar chart for one iteration and save to outpath (PNG).
    Counts may be floats.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(diffs, counts, width=1.0, align="center")
    # plt.title(f"Error Difference distribution — {iter_name}")
    plt.xlabel("difference (actual - predicted)")
    plt.ylabel("Empirical error counts")
    N = max(1, len(diffs) // 12)
    plt.xticks(diffs[::N], rotation=45)
    plt.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_combined(diffs: np.ndarray, columns: List[str], matrix: np.ndarray, outpath: str):
    """
    Create a combined line plot with one line per column (overlay).
    matrix shape: (len(diffs), len(columns))
    """
    plt.figure(figsize=(11, 6))
    for i, col in enumerate(columns):
        plt.plot(diffs, matrix[:, i], marker="o", linewidth=1, label=col)
    plt.title("Difference distributions (combined)")
    plt.xlabel("difference (actual - predicted)")
    plt.ylabel("count")
    N = max(1, len(diffs) // 12)
    plt.xticks(diffs[::N], rotation=45)
    plt.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    if len(columns) > 1:
        plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main(infile: str, outdir: str = "plots"):
    # read CSV
    df = pd.read_csv(infile)
    df.columns = [c.strip() for c in df.columns]

    if "difference" not in df.columns:
        raise ValueError("CSV must contain a 'difference' column.")

    # convert difference to int and sort
    df["difference"] = df["difference"].astype(int)
    df = df.sort_values("difference", ascending=True).reset_index(drop=True)

    # detect data columns
    iter_cols = discover_iter_columns(df, diff_col_name="difference")
    if len(iter_cols) == 0:
        raise ValueError("No data columns found (expected something like 'first_phase' or 'iter_8000').")

    # convert counts to numeric floats (keep decimal averages)
    for c in iter_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    diffs = df["difference"].to_numpy()
    matrix = df[iter_cols].to_numpy()  # shape (n_diff, n_cols)

    # Exclude difference == 0 if present (per your requirement)
    mask = diffs != 0
    diffs_plot = diffs[mask]
    matrix_plot = matrix[mask, :]

    make_output_dir(outdir)

    # If there's only one data column, produce one bar chart named after it.
    if len(iter_cols) == 1:
        col = iter_cols[0]
        outpath = os.path.join(outdir, f"{col}.png")
        plot_single_iteration(diffs_plot, matrix_plot[:, 0], col, outpath)
        print(f"Saved {outpath}")

        # Also save a combined line plot (single line), for consistency with multi-column workflow
        combined_path = os.path.join(outdir, f"combined_{col}.png")
        plot_combined(diffs_plot, [col], matrix_plot, combined_path)
        print(f"Saved combined plot to {combined_path}")
    else:
        # multiple columns: create one bar chart per column and a combined overlay plot
        for i, col in enumerate(iter_cols):
            outpath = os.path.join(outdir, f"{col}.png")
            plot_single_iteration(diffs_plot, matrix_plot[:, i], col, outpath)
            print(f"Saved {outpath}")

        combined_path = os.path.join(outdir, "combined_iterations.png")
        plot_combined(diffs_plot, iter_cols, matrix_plot, combined_path)
        print(f"Saved combined plot to {combined_path}")

if __name__ == "__main__":
    # allow overriding via CLI: python plot_diff_histograms.py infile.csv outdir
    infile = Input_file_path
    outdir = Output_dir
    if len(sys.argv) >= 2:
        infile = sys.argv[1]
    if len(sys.argv) >= 3:
        outdir = sys.argv[2]
    main(infile, outdir)

