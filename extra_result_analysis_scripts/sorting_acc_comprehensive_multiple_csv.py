#!/usr/bin/env python3
"""
sorting_acc_comprehensive.py

Now supports multiple CSVs and multiple modes. Plots ONLY joint-accuracy curves
(one curve per CSV/mode pair) on the same figure.

Usage examples:
    python sorting_acc_comprehensive.py \
        --csv file1.csv file2.csv file3.csv \
        --positions 1,2,3,4 \
        --mode length first second \
        --out joint_acc.png \
        --max-iter 3000 \
        --title "My Joint Accuracy Plot" \
        --labels "Run A" "Run B" "Run C"

    # Provide one --mode it will be applied to all CSVs:
    python sorting_acc_comprehensive.py \
        --csv a.csv b.csv c.csv \
        --positions 1 2 3 4 \
        --mode strict \
        --out joint_acc.png \
        --labels "experiment"   # same label applied to all CSVs

Notes:
 - If --labels is provided with more than one label, the number of labels
   must be either 1 (broadcast to all CSVs) or equal to the number of CSVs.
 - If --title is provided it will replace the default generated title.
"""
from __future__ import annotations
import re
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

# regexes
INT_TOKEN_RE = re.compile(r'-?\d+')   # used for numeric parsing (keeps sign if any)
DIGIT_TOKEN_RE = re.compile(r'\d+')   # used for digit-string extraction (for length / digit comparison)

def extract_token_ints(s: object) -> List[int]:
    """Extract integer tokens (as ints) from messy cell string."""
    if pd.isna(s) or (isinstance(s, str) and s.strip() == ""):
        return []
    s = str(s)
    found = INT_TOKEN_RE.findall(s)
    out: List[int] = []
    for t in found:
        try:
            out.append(int(t))
        except Exception:
            continue
    return out

def extract_token_strs(s: object) -> List[str]:
    """Extract digit-only token strings (no sign) from messy cell string.
    Preserves leading zeros if present in the original cell.
    """
    if pd.isna(s) or (isinstance(s, str) and s.strip() == ""):
        return []
    s = str(s)
    found = DIGIT_TOKEN_RE.findall(s)
    return found  # list of strings, e.g. ["062","361","428"]

def token_int_at_pos(cell: object, pos: int) -> Optional[int]:
    toks = extract_token_ints(cell)
    if len(toks) >= pos:
        return toks[pos - 1]
    return None

def token_str_at_pos(cell: object, pos: int) -> Optional[str]:
    toks = extract_token_strs(cell)
    if len(toks) >= pos:
        return toks[pos - 1]
    return None

def find_pred_columns(df: pd.DataFrame) -> List[str]:
    pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]
    def iter_of(col: str) -> int:
        m = re.search(r'pred_iter_(\d+)', col)
        return int(m.group(1)) if m else 10**9
    return sorted(pred_cols, key=iter_of)

def parse_iter_number(colname: str) -> int:
    m = re.search(r'pred_iter_(\d+)', colname)
    return int(m.group(1)) if m else -1

def compute_accuracies_for_positions(df: pd.DataFrame, positions: List[int], mode: str = "strict") -> Dict[int, pd.DataFrame]:
    """
    Compute per-position and joint accuracies. Returns a dict mapping each pos -> DataFrame,
    with pos==0 reserved for the joint metric.
    (Function logic largely unchanged; kept here for reuse.)
    """
    if 'actual' not in df.columns:
        raise ValueError("CSV must contain an 'actual' column with the true tokens.")

    pred_cols = find_pred_columns(df)
    if not pred_cols:
        raise ValueError("No columns named pred_iter_* found in CSV.")

    # Pre-extract actual tokens (ints and strings) so we don't re-parse repeatedly
    actual_ints_series = df['actual'].apply(extract_token_ints)
    actual_strs_series = df['actual'].apply(extract_token_strs)

    # canonical whole-result target: sorted numeric tokens (ascending)
    canonical_sorted_series = actual_ints_series.apply(lambda toks: sorted(toks) if toks else [])

    results_by_pos: Dict[int, pd.DataFrame] = {}

    digit_mode_map = {"first": 1, "second": 2, "third": 3, "fourth": 4}

    # compute per-position series (unused for plotting in the new flow, but we keep it)
    for pos in positions:
        if mode == "strict":
            actual_at_pos = actual_ints_series.apply(lambda toks: toks[pos - 1] if len(toks) >= pos else None)
        elif mode == "length":
            actual_at_pos = actual_strs_series.apply(lambda toks: toks[pos - 1] if len(toks) >= pos else None)
        elif mode in digit_mode_map:
            k = digit_mode_map[mode]
            def actual_kth(toks: List[str]) -> Optional[str]:
                if len(toks) >= pos and toks[pos - 1] != "" and len(toks[pos - 1]) >= k:
                    return toks[pos - 1][k - 1]
                return None
            actual_at_pos = actual_strs_series.apply(actual_kth)
        else:
            raise ValueError("Unknown mode: choose 'strict', 'length', 'first', 'second', 'third', or 'fourth'")

        results = []
        for col in pred_cols:
            iter_num = parse_iter_number(col)
            if mode == "strict":
                pred_at_pos = df[col].apply(lambda s: token_int_at_pos(s, pos))
                mask_valid_actual = actual_at_pos.notna()
                total = int(mask_valid_actual.sum())
                if total == 0:
                    accuracy = np.nan
                    matches = 0
                else:
                    matches = int((pred_at_pos[mask_valid_actual] == actual_at_pos[mask_valid_actual]).sum())
                    accuracy = matches / total

            elif mode == "length":
                pred_str_at_pos = df[col].apply(lambda s: token_str_at_pos(s, pos))
                mask_valid_actual = actual_at_pos.notna()
                total = int(mask_valid_actual.sum())
                if total == 0:
                    accuracy = np.nan
                    matches = 0
                else:
                    def match_len(a_str: Optional[str], p_str: Optional[str]) -> bool:
                        if a_str is None or p_str is None:
                            return False
                        return len(a_str) == len(p_str)
                    comp = [match_len(a, p) for a, p in zip(actual_at_pos[mask_valid_actual], pred_str_at_pos[mask_valid_actual])]
                    matches = int(sum(comp))
                    accuracy = matches / total

            else:  # digit modes
                k = digit_mode_map[mode]
                def pred_kth_from_cell(s: object) -> Optional[str]:
                    p = token_str_at_pos(s, pos)
                    if p is None or p == "" or len(p) < k:
                        return None
                    return p[k - 1]
                pred_kth_at_pos = df[col].apply(pred_kth_from_cell)
                mask_valid_actual = actual_at_pos.notna()
                total = int(mask_valid_actual.sum())
                if total == 0:
                    accuracy = np.nan
                    matches = 0
                else:
                    comp = [ (a == p) for a, p in zip(actual_at_pos[mask_valid_actual], pred_kth_at_pos[mask_valid_actual]) ]
                    matches = int(sum(1 for v in comp if v))
                    accuracy = matches / total

            results.append({
                "iter": iter_num,
                "accuracy": accuracy,
                "matches": matches,
                "total": total,
                "col": col
            })

        out_df = pd.DataFrame(results).sort_values('iter').reset_index(drop=True)
        results_by_pos[pos] = out_df

    # --- compute joint metric across all requested positions (pos 0) ---
    joint_results = []

    for col in pred_cols:
        iter_num = parse_iter_number(col)

        # Strict mode: joint metric == original whole-result exact-match (canonical sorted)
        if mode == "strict":
            mask_valid_whole = canonical_sorted_series.apply(lambda toks: len(toks) > 0)
            total_whole = int(mask_valid_whole.sum())
            if total_whole == 0:
                accuracy = np.nan
                matches = 0
            else:
                pred_lists = df[col].apply(extract_token_ints)
                matches = 0
                for valid_idx, canon in zip(canonical_sorted_series[mask_valid_whole].index, canonical_sorted_series[mask_valid_whole].values):
                    pred_list = pred_lists[valid_idx]
                    if pred_list == canon:
                        matches += 1
                accuracy = matches / total_whole

            joint_results.append({
                "iter": iter_num,
                "accuracy": accuracy,
                "matches": matches,
                "total": total_whole,
                "col": col
            })
            continue

        # digit modes
        if mode in digit_mode_map:
            k = digit_mode_map[mode]
            valid_indices = []
            for idx, toks in actual_strs_series.items():
                has_any = False
                for pos in positions:
                    if len(toks) >= pos and toks[pos - 1] != "" and len(toks[pos - 1]) >= k:
                        has_any = True
                        break
                if has_any:
                    valid_indices.append(idx)
            total_joint = len(valid_indices)

            if total_joint == 0:
                accuracy = np.nan
                matches = 0
                joint_results.append({
                    "iter": iter_num,
                    "accuracy": accuracy,
                    "matches": matches,
                    "total": total_joint,
                    "col": col
                })
                continue

            matches = 0
            for idx in valid_indices:
                all_ok = True
                for pos in positions:
                    a_str = token_str_at_pos(df.at[idx, 'actual'], pos)
                    if a_str is None or a_str == "" or len(a_str) < k:
                        continue
                    p_str = token_str_at_pos(df.at[idx, col], pos)
                    if p_str is None or len(p_str) < k:
                        all_ok = False
                        break
                    if a_str[k - 1] != p_str[k - 1]:
                        all_ok = False
                        break
                if all_ok:
                    matches += 1

            accuracy = matches / total_joint if total_joint > 0 else np.nan
            joint_results.append({
                "iter": iter_num,
                "accuracy": accuracy,
                "matches": matches,
                "total": total_joint,
                "col": col
            })
            continue

        # length mode
        if mode == "length":
            valid_indices = []
            for idx, toks in actual_strs_series.items():
                ok = True
                for pos in positions:
                    if len(toks) < pos:
                        ok = False
                        break
                if ok:
                    valid_indices.append(idx)
            total_joint = len(valid_indices)
            if total_joint == 0:
                accuracy = np.nan
                matches = 0
                joint_results.append({
                    "iter": iter_num,
                    "accuracy": accuracy,
                    "matches": matches,
                    "total": total_joint,
                    "col": col
                })
                continue

            matches = 0
            for idx in valid_indices:
                all_ok = True
                for pos in positions:
                    a_str = token_str_at_pos(df.at[idx, 'actual'], pos)
                    p_str = token_str_at_pos(df.at[idx, col], pos)
                    if a_str is None or p_str is None or len(a_str) != len(p_str):
                        all_ok = False
                        break
                if all_ok:
                    matches += 1
            accuracy = matches / total_joint if total_joint > 0 else np.nan
            joint_results.append({
                "iter": iter_num,
                "accuracy": accuracy,
                "matches": matches,
                "total": total_joint,
                "col": col
            })
            continue

    joint_df = pd.DataFrame(joint_results).sort_values('iter').reset_index(drop=True)
    results_by_pos[0] = joint_df

    return results_by_pos

def plot_joint_curves(joint_series_list: List[Dict], outpath: str,
                      show_plot: bool = False, max_iter: Optional[int] = None,
                      min_iter: Optional[int] = None, positions: List[int] = None,
                      plot_title: Optional[str] = None, legend_title: Optional[str] = None):
    """
    joint_series_list: list of dicts {"label": str, "df": DataFrame}
    Each df must have columns ['iter', 'accuracy', ...], accuracy in [0..1] or NaN.
    plot_title: optional override for the plot title
    legend_title: optional title for the legend box
    """
    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()

    if (min_iter is not None) and (max_iter is not None) and (min_iter > max_iter):
        print(f"Warning: --min-iter ({min_iter}) > --max-iter ({max_iter}). Nothing will be plotted.")
        return

    any_plotted = False

    # marker and linestyle pools for variety
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '*', 'P', 'X']
    linestyles = ['-', '--', '-.', ':']
    for idx, item in enumerate(joint_series_list):
        label = item["label"]
        dfp: pd.DataFrame = item["df"]
        df_plot = dfp
        if min_iter is not None:
            df_plot = df_plot[df_plot['iter'] >= min_iter]
        if max_iter is not None:
            df_plot = df_plot[df_plot['iter'] <= max_iter]

        if df_plot.empty:
            continue

        iters = df_plot['iter'].values
        accuracies = (df_plot['accuracy'].values * 100)  # percent
        m = markers[idx % len(markers)]
        ls = linestyles[idx % len(linestyles)]
        any_plotted = True
        ax.plot(iters, accuracies, marker=m, linestyle=ls, linewidth=2, label=label)

    if not any_plotted:
        print("Warning: no iterations to plot within the requested min/max iteration range.")
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    # title includes positions for clarity unless overridden
    if plot_title is not None:
        title = plot_title
    else:
        if positions:
            title = f"Joint accuracy vs iteration (positions {positions})"
        else:
            title = "Joint accuracy vs iteration"
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    # legend
    legend_title_to_use = legend_title if legend_title is not None else "CSV (mode)"
    ax.legend(title=legend_title_to_use, loc="best")
    # set x-limits if requested
    if (min_iter is not None) or (max_iter is not None):
        left = min_iter if min_iter is not None else 0
        if max_iter is not None:
            ax.set_xlim(left=left, right=max_iter)
        else:
            ax.set_xlim(left=left)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")
    if show_plot:
        plt.show()
    plt.close()

def parse_positions_arg(values: List[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        parts = [p.strip() for p in v.split(',') if p.strip() != ""]
        for p in parts:
            try:
                n = int(p)
                if n < 1:
                    raise ValueError("positions must be >= 1")
                out.append(n)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid position value: {p}")
    out = sorted(list(dict.fromkeys(out)))
    return out

def main():
    parser = argparse.ArgumentParser(description="Plot joint accuracy vs iteration from one or more CSVs (strict, length, or k-th digit)")
    parser.add_argument("--csv", "-c", required=True, nargs='+', help="Paths to input CSV files (one or more)")
    parser.add_argument("--positions", "-p", required=True, nargs='+',
                        help="Positions to evaluate. Provide as space-separated list or comma-separated string, e.g. 1 2 3 or 1,2,3")
    parser.add_argument("--mode", "-m", nargs='+', choices=["strict", "length", "first", "second", "third", "fourth"], default=["strict"],
                        help="Accuracy mode(s): strict, length, or first/second/third/fourth. Provide one per CSV, or a single mode to apply to all CSVs.")
    parser.add_argument("--out", "-o", default="joint_accuracy.png", help="Output image path (PNG)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--min-iter", type=int, default=None,
                        help="Minimum iteration value to draw on the x-axis (inclusive).")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Maximum iteration value to draw on the x-axis (inclusive).")
    parser.add_argument("--title", "-t", type=str, default=None,
                        help="Optional plot title (if omitted a default title including positions will be used).")
    parser.add_argument("--labels", nargs='+', default=None,
                        help="Optional legend labels — either a single label (applied to all CSVs) or one label per CSV in order.")
    args = parser.parse_args()

    csv_paths: List[str] = args.csv
    modes: List[str] = args.mode

    # broadcasting single mode to all csvs if necessary
    if len(modes) == 1 and len(csv_paths) > 1:
        modes = modes * len(csv_paths)

    if len(modes) != len(csv_paths):
        raise SystemExit("Number of --mode values must be 1 or equal to number of --csv files provided.")

    # handle labels
    provided_labels = None
    if args.labels is not None:
        provided_labels = args.labels
        if len(provided_labels) == 1 and len(csv_paths) > 1:
            provided_labels = provided_labels * len(csv_paths)
        if len(provided_labels) != len(csv_paths):
            raise SystemExit("Number of --labels values must be 1 or equal to number of --csv files provided.")

    positions = parse_positions_arg(args.positions)
    if not positions:
        raise SystemExit("No valid positions provided. Example: --positions 1 2 3 4 or --positions 1,2,3,4")

    joint_series_list = []

    # iterate with labels if available
    if provided_labels is not None:
        iter_triplet = zip(csv_paths, modes, provided_labels)
    else:
        # use None as placeholder for label; we'll compute default label inside
        iter_triplet = zip(csv_paths, modes, [None] * len(csv_paths))

    for csv_path, mode, user_label in iter_triplet:
        if not os.path.exists(csv_path):
            print(f"Warning: CSV path not found: {csv_path} -- skipping.")
            continue
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[''])
        accs_by_pos = compute_accuracies_for_positions(df, positions, mode=mode)
        joint_df = accs_by_pos[0]

        # if user supplied a label for this CSV, use it; otherwise create readable label: mode (basename)
        if user_label:
            label = user_label
        else:
            base = os.path.basename(csv_path)
            label = f"{mode} ({base})"

        joint_series_list.append({"label": label, "df": joint_df})

        # print joint metric summary for this CSV/mode
        if mode == "strict":
            label_name = "Exact whole-result match (canonical sorted target)"
        elif mode == "length":
            label_name = f"Joint-length accuracy (all positions {positions})"
        else:
            label_name = f"Joint-{mode}-digit accuracy (all positions {positions})"

        print(f"\n{label} - {label_name} results:")
        print(" Iteration | matches/total | accuracy(%)")
        for _, row in joint_df.iterrows():
            acc_pct = (row['accuracy'] * 100) if pd.notna(row['accuracy']) else float('nan')
            print(f" {int(row['iter']):8d} | {int(row['matches']):7d}/{int(row['total']):6d} | {acc_pct:8.3f}")

    if not joint_series_list:
        raise SystemExit("No valid CSV inputs were processed. Exiting.")

    # decide legend title: if user provided labels, use a simpler legend title
    legend_title = "Legend" if provided_labels is not None else None

    plot_joint_curves(joint_series_list, args.out, show_plot=args.show,
                      max_iter=args.max_iter, min_iter=args.min_iter,
                      positions=positions, plot_title=args.title, legend_title=legend_title)

if __name__ == "__main__":
    main()
