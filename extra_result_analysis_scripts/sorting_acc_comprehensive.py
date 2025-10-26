#!/usr/bin/env python3
"""
sorting_acc_comprehensive.py

Usage examples:
    # strict (default) numeric-equality per-position
    python sorting_acc_comprehensive.py --csv data.csv --positions 1,2,3,4 --mode strict --out acc.png

    # relaxed length-based: a prediction is correct if it has same number of digits
    python sorting_acc_comprehensive.py --csv data.csv --positions 1 2 3 4 --mode length --show

    # first/second/third/fourth-digit match: compare only the k-th digit at each position
    python sorting_acc_comprehensive.py --csv data.csv --positions 1 2 3 4 --mode third --show

    # only draw iterations up to 1000
    python sorting_acc_comprehensive.py --csv data.csv --positions 1 2 3 4 --mode strict --max-iter 1000
"""
from __future__ import annotations
import re
import argparse
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
    mode: "strict" -> numeric equality (int)
          "length" -> digit-length match (string length of digits)
          "first"/"second"/"third"/"fourth" -> kth-digit character match (from digit-only token, preserves leading zeros)
    Returns dict mapping pos -> DataFrame with columns ['iter','accuracy','matches','total','col'].
    Additionally computes pos==0 => joint-digit accuracy (the "joint-digit accuracy" metric described).
    For 'strict' mode the joint-digit accuracy equals original whole-result exact-match accuracy.
    The joint metric checks that ALL requested positions that actually have the k-th digit
    match simultaneously. An example is included in the denominator if at least one requested
    position has the required k-th digit (otherwise it's skipped).
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

    # compute per-position series (existing behavior + first/second/third/fourth modes)
    for pos in positions:
        if mode == "strict":
            # actual ints at pos
            actual_at_pos = actual_ints_series.apply(lambda toks: toks[pos - 1] if len(toks) >= pos else None)
        elif mode == "length":
            # actual digit-strings at pos (preserve leading zeros)
            actual_at_pos = actual_strs_series.apply(lambda toks: toks[pos - 1] if len(toks) >= pos else None)
        elif mode in digit_mode_map:
            k = digit_mode_map[mode]
            # actual k-th digit at pos if present
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
                        if a_str is None:
                            return False
                        if p_str is None:
                            return False
                        return len(a_str) == len(p_str)
                    comp = [match_len(a, p) for a, p in zip(actual_at_pos[mask_valid_actual], pred_str_at_pos[mask_valid_actual])]
                    matches = int(sum(comp))
                    accuracy = matches / total

            else:  # digit modes: first/second/third/fourth
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
            # only consider rows that have at least one token in canonical target
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

        # For digit modes we include an example in the joint denominator if AT LEAST ONE
        # of the requested positions actually has the k-th digit in the ground truth.
        # For that included example, we require that **every** position that DOES have
        # the k-th digit is predicted correctly (positions lacking the k-th digit are ignored).
        if mode in digit_mode_map:
            k = digit_mode_map[mode]
            valid_indices = []
            # determine which rows have at least one position with k-th digit
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

            # For each valid row, check every position that has k-th digit
            matches = 0
            for idx in valid_indices:
                all_ok = True
                for pos in positions:
                    a_str = token_str_at_pos(df.at[idx, 'actual'], pos)
                    # consider only positions where actual has k-th digit
                    if a_str is None or a_str == "" or len(a_str) < k:
                        continue  # ignore this position for this example
                    # now check predicted
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

        # For length mode (unchanged): require that all requested positions exist (as before)
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

def plot_multi_positions(accs_by_pos: Dict[int, pd.DataFrame], outpath: str, show_plot: bool = False,
                         mode: str = "strict", positions: List[int] = None, max_iter: Optional[int] = None):
    """Plot per-position and joint curves. If max_iter is provided, only plot iterations <= max_iter."""
    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()

    any_plotted = False
    for pos, dfp in sorted(accs_by_pos.items(), key=lambda kv: kv[0]):
        # filter by max_iter if provided
        if max_iter is not None:
            df_plot = dfp[dfp['iter'] <= max_iter]
        else:
            df_plot = dfp

        if df_plot.empty:
            # nothing to plot for this series within the requested range
            continue

        iters = df_plot['iter'].values
        accuracies = (df_plot['accuracy'].values * 100)  # percent

        any_plotted = True
        if pos == 0:
            # joint metric label depends on mode
            if mode == "strict":
                label = "exact match (whole result)"  # unchanged
            elif mode == "length":
                label = f"joint-length accuracy (all positions {positions})"
            else:
                label = f"joint-{mode}-digit accuracy (all positions {positions})"
            ax.plot(iters, accuracies, marker='s', linestyle='--', linewidth=2, label=label)
        else:
            if mode == "strict":
                match_mode = "numeric match"
            elif mode == "length":
                match_mode = "length match"
            else:
                match_mode = f"{mode}-digit match"
            label = f"pos {pos} ({match_mode})"
            ax.plot(iters, accuracies, marker='o', linestyle='-', label=label)

    if not any_plotted:
        print("Warning: no iterations to plot within the requested max_iter range.")
        # still create empty plot with labels
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    if mode == "strict":
        title = "Per-position accuracy (numeric match) vs iteration"
    elif mode == "length":
        title = "Per-position accuracy (length match) vs iteration"
    else:
        title = f"Per-position accuracy ({mode}-digit match) vs iteration"
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Metric", loc="best")
    if max_iter is not None:
        ax.set_xlim(left=0, right=max_iter)
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
    parser = argparse.ArgumentParser(description="Plot per-position accuracy vs iteration from CSV (strict, length, or k-th digit)")
    parser.add_argument("--csv", "-c", required=True, help="Path to input CSV file")
    parser.add_argument("--positions", "-p", required=True, nargs='+',
                        help="Positions to evaluate. Provide as space-separated list or comma-separated string, e.g. 1 2 3 or 1,2,3")
    parser.add_argument("--mode", "-m", choices=["strict", "length", "first", "second", "third", "fourth"], default="strict",
                        help="Accuracy mode: strict (numeric equality), length (digit-length match), or first/second/third/fourth (k-th digit match)")
    parser.add_argument("--out", "-o", default="pos_accuracy.png", help="Output image path (PNG)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Maximum iteration value to draw on the x-axis (inclusive). If omitted, draw all iterations collected.")
    args = parser.parse_args()

    positions = parse_positions_arg(args.positions)
    if not positions:
        raise SystemExit("No valid positions provided. Example: --positions 1 2 3 4 or --positions 1,2,3,4")

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, na_values=[''])
    accs_by_pos = compute_accuracies_for_positions(df, positions, mode=args.mode)

    # print table for each pos
    for pos in positions:
        print(f"\nPosition {pos} results (mode={args.mode}):")
        acc_df = accs_by_pos[pos]
        print(" Iteration | matches/total | accuracy(%)")
        for _, row in acc_df.iterrows():
            acc_pct = (row['accuracy'] * 100) if pd.notna(row['accuracy']) else float('nan')
            print(f" {int(row['iter']):8d} | {int(row['matches']):7d}/{int(row['total']):6d} | {acc_pct:8.3f}")

    # print joint metric (pos 0)
    joint_df = accs_by_pos[0]
    if args.mode == "strict":
        label_name = "Exact whole-result match (canonical sorted target)"
    elif args.mode == "length":
        label_name = f"Joint-length accuracy (all positions {positions})"
    else:
        label_name = f"Joint-{args.mode}-digit accuracy (all positions {positions})"

    print(f"\n{label_name} results:")
    print(" Iteration | matches/total | accuracy(%)")
    for _, row in joint_df.iterrows():
        acc_pct = (row['accuracy'] * 100) if pd.notna(row['accuracy']) else float('nan')
        print(f" {int(row['iter']):8d} | {int(row['matches']):7d}/{int(row['total']):6d} | {acc_pct:8.3f}")

    plot_multi_positions(accs_by_pos, args.out, show_plot=args.show, mode=args.mode, positions=positions, max_iter=args.max_iter)

if __name__ == "__main__":
    main()
