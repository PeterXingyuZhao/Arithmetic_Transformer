#!/usr/bin/env python3
"""
multiple_csv_error_vs_iter.py

Usage:
    python multiple_csv_error_vs_iter.py file1.csv file2.csv ...
    Optional:
      --output /path/to/output.pdf
      --max-step N        Only include numeric steps <= N
      --interval I        Only include numeric steps that are multiples of I (default 1)

Behavior:
  - Each input CSV becomes one curve.
  - Columns named like pred_iter_0, pred_iter_25, ... are parsed; numeric suffix becomes x-axis steps.
  - For each row, the 'actual' column must contain one of '<', '>', '=' (possibly embedded in other text).
    If actual cannot be parsed, that row is skipped.
  - For each pred column, we extract the first occurrence of '<', '>', '='. If none found, it's treated as incorrect.
  - Output plot is saved as a PDF. By default the PDF is named "error_rate_vs_step.pdf" in the directory
    containing the first CSV file, unless --output is provided.
"""
import argparse
import os
import sys
import re
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

LABEL_CHARS = ['<', '>', '=']

# -------------------------
# User-configurable values
# Edit these at the top of the program (no CLI changes required).
# -------------------------
# Optional: set legend names to a list of strings matching the CSV order.
# Example:
#   LEGEND_NAMES = ['baseline', 'modelA', 'modelB']
# or leave None to use the CSV basenames as legend labels.
LEGEND_NAMES = ['Thousands different only', 'Hundreds different only', 'Tens different only', 'Units different only']
# LEGEND_NAMES = ['No condition (1000-9999 uniform)', 'Thousands equal', 'Thousands, hundreds equal', 'Thousands, hundreds, tens equal', 'All digits equal']

# Whether to draw markers on each data point. Use None for no marker (default), or e.g. 'o', 's', '^'
MARKER = None

# Thickness of plotted lines (float). e.g. 1.0 thin, 2.0 medium, 3.0 thick
LINE_WIDTH = 4.0

# X axis numeric range and preferred tick spacing.
# If you want the x axis to run 0..2000 and ticks every 200, set:
X_AXIS_MIN = 0
X_AXIS_MAX = 2000
X_TICK_STEP = 200   # set to an integer step (e.g. 200) or None to auto-select ticks based on data

# Maximum number of x ticks to display. If the computed tick set exceeds this,
# ticks will be sub-sampled to not exceed this count.
MAX_X_TICKS = 10

# Font sizes
FONT_SIZE_XY_LABEL = 26
FONT_SIZE_TICKS = 24
FONT_SIZE_LEGEND = 24
# -------------------------

def extract_label(cell):
    """
    Robustly extract the first comparison label (<, >, =) from a cell.
    Handles leading single-quote (') escaping like "'=" by stripping leading quotes.
    Returns the label character or None if not found.
    """
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    # remove a single leading quote or double quote commonly used as escape
    if s and s[0] in ("'", '"'):
        s = s[1:].lstrip()
    m = re.search(r'[<>=]', s)
    if m:
        return m.group(0)
    return None

def find_pred_columns(columns):
    """Return ordered list of prediction columns (keep original order).
    Accept any column whose name contains 'pred' (case-insensitive) AND 'iter' or 'iter_'.
    """
    preds = []
    for c in columns:
        if re.search(r'pred', c, re.I) and re.search(r'iter', c, re.I):
            preds.append(c)
    return preds

def step_from_col(colname):
    """Extract numeric step from a column name like 'pred_iter_25'.
    If no numeric suffix found, return None.
    """
    m = re.search(r'(\d+)', colname)
    if m:
        return int(m.group(1))
    return None

def try_read_csv_variants(path):
    """
    Try multiple ways to read a CSV that may use single-quote quoting or double-quote quoting
    or need sniffing. Returns a DataFrame.
    """
    # 1) try pandas auto-detect (engine='python', sep=None)
    tried = []
    try:
        tried.append(("auto", {"engine": "python", "sep": None}))
        df = pd.read_csv(path, engine="python", sep=None, skipinitialspace=True)
        return df
    except Exception:
        pass

    # 2) try explicit comma separator + single-quote quotechar (your sample appears to use single quotes)
    try:
        tried.append(("single-quote", {"sep": ",", "quotechar": "'", "engine": "python"}))
        df = pd.read_csv(path, sep=",", quotechar="'", engine="python", skipinitialspace=True)
        return df
    except Exception:
        pass

    # 3) try explicit comma separator + double-quote (standard)
    try:
        tried.append(("double-quote", {"sep": ",", "quotechar": '"'}))
        df = pd.read_csv(path, sep=",", quotechar='"', engine="python", skipinitialspace=True)
        return df
    except Exception:
        pass

    # 4) final fallback: read with csv module treating no quoting
    try:
        with open(path, newline='') as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        if rows:
            header = rows[0]
            data = rows[1:]
            # pad short rows to header length with empty strings
            normed = [r + [''] * (len(header) - len(r)) if len(r) < len(header) else r[:len(header)] for r in data]
            df = pd.DataFrame(normed, columns=header)
            return df
    except Exception:
        pass

    raise ValueError(f"Unable to parse CSV '{path}' (tried methods: {tried}). Please check quoting/delimiters.")

def compute_error_rates_for_file(path, max_step=None, interval=1):
    """
    Read CSV robustly, extract 'actual' and 'pred' columns, compute error rates.
    Returns ordered results, basename, dirname as before.
    """
    df = try_read_csv_variants(path)
    columns = list(df.columns)

    # find actual column case-insensitively
    real_actual = next((c for c in columns if c.lower() == 'actual'), None)
    if real_actual is None:
        raise ValueError(f"Couldn't find an 'actual' column in {path}. Columns: {columns}")

    # find prediction columns
    pred_cols = find_pred_columns(columns)
    if not pred_cols:
        pred_cols = [c for c in columns if re.search(r'pred', c, re.I) or re.search(r'iter', c, re.I)]
    if not pred_cols:
        raise ValueError(f"No prediction columns found in {path}. Columns: {columns}")

    # extract actual labels robustly
    actual_labels = df[real_actual].map(extract_label)
    valid_rows_mask = actual_labels.notna()
    n_valid_rows = int(valid_rows_mask.sum())
    if n_valid_rows == 0:
        raise ValueError(f"No valid actual labels parsed in {path}; cannot compute error rates.")

    results = {}
    for col in pred_cols:
        # Determine numeric step (if any)
        step = step_from_col(col)

        # filter by max_step and interval for numeric columns
        if step is not None:
            if (max_step is not None) and (step > max_step):
                continue
            if interval is not None and interval > 1:
                if (step % interval) != 0:
                    continue
        # Non-numeric columns are kept always

        preds = df[col].map(extract_label) if col in df.columns else pd.Series([None] * len(df))
        preds = preds.reindex(df.index)
        actuals = actual_labels[valid_rows_mask]
        preds = preds[valid_rows_mask]

        # count missing pred as error
        errors = (preds != actuals) | (preds.isna())
        n_errors = int(errors.sum())
        error_rate = n_errors / len(actuals)

        sort_key = (0, step) if step is not None else (1, col)
        display_key = step if step is not None else col
        results[col] = {
            'step': step,
            'sort_key': sort_key,
            'display_key': display_key,
            'error_rate': error_rate,
            'n_examples': int(len(actuals)),
            'n_errors': n_errors
        }

    if not results:
        raise ValueError(f"After applying max_step/interval filters, no prediction columns remain for {path}.")

    ordered = OrderedDict(sorted(results.items(), key=lambda kv: kv[1]['sort_key']))
    return ordered, os.path.basename(path), os.path.dirname(path)

def plot_error_rates(file_results, output_pdf_path):
    """file_results: list of tuples (ordered_results_dict, legend_label, dir)
       ordered_results_dict: OrderedDict as returned above
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # collect global x positions for consistent ticks
    all_x_positions = set()
    per_file_points = []

    for ordered_dict, legend_label, _dir in file_results:
        xs = []
        ys = []
        xtick_labels = []
        for colname, info in ordered_dict.items():
            step = info['step']
            display = info['display_key']
            xs.append(step)  # numeric or None
            ys.append(info['error_rate'])
            xtick_labels.append(str(display))
        per_file_points.append((legend_label, xs, ys, xtick_labels))

    # Convert None steps to positions after numeric max
    # Find max numeric step across everything
    numeric_steps = [s for tpl in per_file_points for s in tpl[1] if isinstance(s, int)]
    max_numeric = max(numeric_steps) if numeric_steps else -1
    # Map None to increasing integers starting at max_numeric + 1 (stable order by column)
    for legend_label, xs, ys, labels in per_file_points:
        transformed_xs = []
        next_pos = max_numeric + 1
        for x in xs:
            if isinstance(x, int):
                transformed_xs.append(x)
                all_x_positions.add(x)
            else:
                transformed_xs.append(next_pos)
                all_x_positions.add(next_pos)
                next_pos += 1
        # sort by x for plotting
        pairs = sorted(zip(transformed_xs, ys, labels))
        if pairs:
            tx, ty, tlabels = zip(*pairs)
        else:
            tx, ty, tlabels = ([], [], [])
        # Use configured MARKER (None => no marker) and LINE_WIDTH
        ax.plot(tx, ty, marker=MARKER, linewidth=LINE_WIDTH, label=legend_label)
        # we no longer set per-point tick labels here; we'll set ticks globally below

    # set up pos->label mapping, preferring first file's labels
    sorted_positions = sorted(all_x_positions)
    pos2label = {}
    for legend_label, xs, ys, labels in per_file_points:
        next_pos = max_numeric + 1
        for x, lab in zip(xs, labels):
            pos = x if isinstance(x, int) else next_pos
            pos2label.setdefault(pos, str(lab))
            if not isinstance(x, int):
                next_pos += 1

    # Decide tick positions:
    if X_TICK_STEP is not None:
        # prefer user-specified tick spacing over data-driven ticks
        # generate ticks in the configured axis range
        tick_positions = list(np.arange(X_AXIS_MIN, X_AXIS_MAX + 1, X_TICK_STEP, dtype=int))
        # If there are more ticks than MAX_X_TICKS, subsample evenly
        if len(tick_positions) > MAX_X_TICKS:
            step = math.ceil(len(tick_positions) / MAX_X_TICKS)
            tick_positions = tick_positions[::step]
    else:
        # auto: base ticks off available positions but limit to MAX_X_TICKS
        tick_positions = sorted_positions.copy()
        if len(tick_positions) > MAX_X_TICKS:
            step = math.ceil(len(tick_positions) / MAX_X_TICKS)
            tick_positions = tick_positions[::step]

    # Build labels for the ticks: use pos2label if available, else fall back to numeric string
    tick_labels = [pos2label.get(p, str(p) if isinstance(p, int) else '') for p in tick_positions]

    # Set horizontal tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')

    # Apply x axis range if desired (this may clip any non-numeric columns mapped beyond X_AXIS_MAX)
    try:
        ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    except Exception:
        pass

    # Ensure x-axis label is horizontal
    ax.set_xlabel('Training step', fontsize=FONT_SIZE_XY_LABEL)
    ax.xaxis.label.set_rotation(0)   # explicitly horizontal
    ax.xaxis.label.set_ha('center')

    ax.set_ylabel('Error rate', fontsize=FONT_SIZE_XY_LABEL)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)

    # tick font size
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS)

    # legend font size
    ax.legend(fontsize=FONT_SIZE_LEGEND)

    plt.tight_layout()
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()
    print(f"Saved plot to {output_pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot error rate vs training steps from one or more CSVs.")
    parser.add_argument('csv_files', nargs='+', help='CSV files to process (each becomes one curve)')
    parser.add_argument('--output', '-o', help='Output PDF path. Defaults to error_rate_vs_step.pdf in the directory of the first CSV.')
    parser.add_argument('--max-step', type=int, default=None, help='Maximum numeric training step to include (inclusive).')
    parser.add_argument('--interval', type=int, default=1, help='Interval for numeric steps (keep steps where step %% interval == 0). Default 1 means keep all.')
    args = parser.parse_args()

    if args.interval is not None and args.interval <= 0:
        parser.error("--interval must be a positive integer.")

    csv_files = args.csv_files
    if not csv_files:
        parser.error("Please supply at least one CSV file.")

    for p in csv_files:
        if not os.path.exists(p):
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(2)

    # If LEGEND_NAMES is set, validate length matches number of CSVs
    if LEGEND_NAMES is not None:
        if not isinstance(LEGEND_NAMES, (list, tuple)):
            print("Error: LEGEND_NAMES must be a list or tuple of strings (or None).", file=sys.stderr)
            sys.exit(4)
        if len(LEGEND_NAMES) != len(csv_files):
            print(f"Error: LEGEND_NAMES has length {len(LEGEND_NAMES)} but {len(csv_files)} CSV files were provided.", file=sys.stderr)
            sys.exit(5)

    file_results = []
    first_dir = None
    for idx, p in enumerate(csv_files):
        try:
            ordered, basename, dirname = compute_error_rates_for_file(p, max_step=args.max_step, interval=args.interval)
            # choose legend label: from LEGEND_NAMES if provided, otherwise basename
            legend_label = LEGEND_NAMES[idx] if LEGEND_NAMES is not None else basename
            file_results.append((ordered, legend_label, dirname))
            if first_dir is None:
                first_dir = dirname
        except Exception as e:
            print(f"Error processing '{p}': {e}", file=sys.stderr)
            sys.exit(3)

    if args.output:
        out_pdf = args.output
    else:
        out_pdf = os.path.join(first_dir if first_dir else '.', 'error_rate_vs_step.pdf')

    plot_error_rates(file_results, out_pdf)

if __name__ == '__main__':
    main()
