#!/usr/bin/env python3
"""
diff_histograms.py

Reads a CSV with an 'actual' column and columns named like 'pred_iter_8000',
'pred_iter_38000', ... . For each requested iteration it computes diff = actual - prediction,
counts how many rows produce each diff in the integer range diff_min..diff_max,
then outputs a CSV with two columns:
  - difference
  - first_phase  (the average count across the present iterations)

This script is robust to messy numeric cells such as "17+62" or " 154.0 ".
It will attempt to safely evaluate simple addition/subtraction expressions.
Invalid or missing numeric fields are treated as NaN and skipped for that iteration.
"""

import sys
import pandas as pd
import numpy as np
import ast
import math
from typing import Union

Input_file_path = '/Users/perfectpeter/Library/CloudStorage/GoogleDrive-llmfunexperiment@gmail.com/My Drive/addition/results/4_operands_0_to_999_uniform_wo_padding/reverse_out_with_complete_stats_measure/4_operands_0_to_999_uniform_wo_padding_with_complete_stats_measure_reverse_2/test_reverse_results.csv'
Output_file_path = 'difference_histograms_first_phase_single_iter_8K.csv'

# -------------------------
# Safe evaluator for simple arithmetic expressions (only + and - allowed)
# -------------------------
ALLOWED_BINOPS = (ast.Add, ast.Sub)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)
ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, getattr(ast, "Num", ()))  # ast.Num for py<3.8

def safe_eval_simple_expr(expr: str) -> Union[int, float, None]:
    """
    Safely evaluate a simple arithmetic expression containing numbers, + and - only.
    Returns numeric value (float) or None if evaluation is not allowed/failed.
    Treats NaN/Inf as invalid (returns None).
    """
    if expr is None:
        return None
    # handle already numeric values (numpy or python)
    if isinstance(expr, (int, float, np.integer, np.floating)):
        val = float(expr)
        if not math.isfinite(val):
            return None
        return val

    s = str(expr).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    try:
        tree = ast.parse(s, mode="eval")
    except Exception:
        return None

    # Validate nodes (walk the tree)
    for node in ast.walk(tree):
        # allow the small set of node types; if we encounter anything else, bail out
        if not isinstance(node, ALLOWED_NODES) and not isinstance(node, ast.BinOp) and not isinstance(node, ast.UnaryOp):
            return None
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, ALLOWED_BINOPS):
                return None
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ALLOWED_UNARYOPS):
                return None

    def _eval(node):
        # node expected to be ast.Expression or similar
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # older Python
            return node.n
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if left is None or right is None:
                raise ValueError("Invalid operand")
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            else:
                raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError("Unsupported unary op")
        raise ValueError("Unsupported AST node")

    try:
        val = _eval(tree)
        # convert bools and ints to float; treat non-finite as invalid
        if isinstance(val, bool):
            val = float(val)
        else:
            val = float(val)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None

# -------------------------
# Main routine (modified to output a single averaged column "first_phase")
# -------------------------
def build_histograms(
    infile: str,
    outfile: str = "difference_histograms_first_phase_single_iter_8K.csv",
    iter_start: int = 8000,
    iter_end: int = 8000,
    iter_step: int = 200,
    diff_min: int = -100,
    diff_max: int = 100,
    actual_col_name: str = "actual",
    pred_col_template: str = "pred_iter_{}",
    include_missing_counts: bool = False,
    avg_round: Union[int, None] = 2  # None -> keep full float, otherwise round to this many decimal places
):
    # read as str to let safe parser handle messy cells
    df = pd.read_csv(infile, dtype=str)

    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    if actual_col_name not in df.columns:
        raise ValueError(f"Actual column '{actual_col_name}' not found in CSV. Columns: {df.columns.tolist()}")

    iterations = list(range(iter_start, iter_end + 1, iter_step))
    present_iters = []
    for it in iterations:
        cname = pred_col_template.format(it)
        if cname in df.columns:
            present_iters.append((it, cname))
        else:
            print(f"Warning: column '{cname}' not found in CSV; skipping iteration {it}")

    if not present_iters:
        raise ValueError("No prediction columns found for the requested iterations. Nothing to average.")

    # Prepare index of differences
    diffs = list(range(diff_min, diff_max + 1))
    counts = pd.DataFrame(0, index=diffs, columns=[f"iter_{it}" for (it, _) in present_iters], dtype=int)

    # Optionally keep counters for how many predictions were missing or invalid
    invalid_counts = {f"iter_{it}": 0 for (it, _) in present_iters}
    missing_actual = 0
    total_rows = len(df)

    # Iterate rows
    for idx, row in df.iterrows():
        actual_raw = row[actual_col_name]
        actual_val = safe_eval_simple_expr(actual_raw)
        if actual_val is None:
            missing_actual += 1
            continue
        # use integer diffs (rounded)
        for it, cname in present_iters:
            pred_raw = row[cname]
            pred_val = safe_eval_simple_expr(pred_raw)
            if pred_val is None:
                invalid_counts[f"iter_{it}"] += 1
                continue
            diff = actual_val - pred_val
            # skip non-finite diffs (e.g., NaN or +/-inf)
            if not math.isfinite(diff):
                invalid_counts[f"iter_{it}"] += 1
                continue
            diff_int = int(round(diff))
            if diff_int < diff_min or diff_int > diff_max:
                continue
            counts.at[diff_int, f"iter_{it}"] += 1

    # Compute the average count across the present iterations for each difference
    mean_series = counts.mean(axis=1)  # float series, index = diffs
    if avg_round is not None:
        mean_series = mean_series.round(avg_round)

    # Build output DataFrame with two columns: difference, first_phase
    out_df = mean_series.reset_index()
    out_df.columns = ["difference", "avg_counts"]

    # Save output (difference as first column)
    out_df.to_csv(outfile, index=False)
    print(f"Saved averaged histogram to '{outfile}'")
    print(f"Processed {total_rows} rows. Missing actuals: {missing_actual}.")
    print("Invalid/missing predictions per iteration (counts skipped):")
    for k, v in invalid_counts.items():
        print(f"  {k}: {v}")

    # If user requested overflow bins, produce them too (optional)
    if include_missing_counts:
        below = {col: 0 for col in counts.columns}
        above = {col: 0 for col in counts.columns}
        for idx, row in df.iterrows():
            actual_val = safe_eval_simple_expr(row[actual_col_name])
            if actual_val is None:
                continue
            for it, cname in present_iters:
                pred_val = safe_eval_simple_expr(row[cname])
                if pred_val is None:
                    continue
                diff = actual_val - pred_val
                if not math.isfinite(diff):
                    continue
                diff_int = int(round(diff))
                if diff_int < diff_min:
                    below[f"iter_{it}"] += 1
                elif diff_int > diff_max:
                    above[f"iter_{it}"] += 1
        below_s = pd.Series(below).mean().round(avg_round) if avg_round is not None else pd.Series(below).mean()
        above_s = pd.Series(above).mean().round(avg_round) if avg_round is not None else pd.Series(above).mean()
        overflow_df = pd.DataFrame({
            "difference": [f"<{diff_min}", f">{diff_max}"],
            "first_phase": [below_s, above_s]
        })
        overflow_outfile = outfile.replace(".csv", "_with_overflow.csv")
        pd.concat([out_df, overflow_df], ignore_index=True).to_csv(overflow_outfile, index=False)
        print("Also saved counts with overflow bins to:", overflow_outfile)
        return out_df, overflow_df

    return out_df

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    # you can also call this script with two args: input.csv output.csv
    if len(sys.argv) >= 3:
        Input_file_path = sys.argv[1]
        Output_file_path = sys.argv[2]
    elif len(sys.argv) == 2:
        Input_file_path = sys.argv[1]

    build_histograms(Input_file_path, Output_file_path)
