#!/usr/bin/env python3
"""
leading_2_digit_error.py

Counts the error described by the user:
  "Does the leading-2-digits of the model's FIRST output token match the
   leading-2-digits of the 4-digit number in the ground-truth 'actual'?"

Usage:
    python leading_2_digit_error.py --csv data.csv --out errors.png --show

Outputs a PNG and prints a table of counts per iteration.

Notes:
 - Expects columns: "actual" and columns named "pred_iter_<N>" (N integer).
 - Uses digit extraction (keeps only digits) so messy cells like "104,700,5,510,51" are handled.
 - Rows without any 4-digit token in `actual` are skipped.
 - A predicted first token must have at least 2 digits to produce a leading-2 match;
   otherwise it's counted as "no 2-digit pred" (not considered an error).
"""
from __future__ import annotations
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

DIGIT_RE = re.compile(r'\d+')
PRED_PREFIX = "pred_iter_"

def digit_tokens(s: object) -> List[str]:
    """Return list of digit-only tokens found in the cell, in order.
    Example: "104,700,5,510,51" -> ["104","700","5","510","51"].
    """
    if pd.isna(s):
        return []
    return DIGIT_RE.findall(str(s))

def first_token_leading2_from_pred(cell: object) -> Optional[str]:
    """Return the leading-2-digit string of the first predicted token, or None if not present / too short."""
    toks = digit_tokens(cell)
    if not toks:
        return None
    first = toks[0]
    if len(first) < 2:
        return None
    return first[:2]

def actual_4digit_leading2(cell: object) -> Optional[str]:
    """Return leading-2-digit string of the actual 4-digit token (first token with length 4).
       If none found, return None.
    """
    toks = digit_tokens(cell)
    for t in toks:
        if len(t) == 4:
            return t[:2]
    return None

def find_pred_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith(PRED_PREFIX)]
    def iter_of(col: str) -> int:
        m = re.search(rf'{PRED_PREFIX}(\d+)', col)
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=iter_of)

def parse_iter_number(colname: str) -> int:
    m = re.search(rf'{PRED_PREFIX}(\d+)', colname)
    return int(m.group(1)) if m else -1

def compute_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns:
       iter, errors, total_examples_with_actual4, total_with_pred2, error_rate_all, error_rate_valid
    """
    pred_cols = find_pred_columns(df)
    if not pred_cols:
        raise SystemExit("No pred_iter_* columns found in CSV.")

    # find actual's leading-2 of 4-digit token (or None)
    actual_lead2 = df['actual'].apply(actual_4digit_leading2)

    # rows we can evaluate = those with an actual 4-digit token
    mask_valid_actual = actual_lead2.notna()
    total_examples = int(mask_valid_actual.sum())
    if total_examples == 0:
        raise SystemExit("No rows with a 4-digit token in 'actual' were found.")

    rows = []
    for col in pred_cols:
        iter_num = parse_iter_number(col)
        # predicted first token leading2 (or None)
        pred_lead2 = df[col].apply(first_token_leading2_from_pred)

        # count how many rows have predicted-first with >=2 digits among valid actual rows
        total_with_pred2 = int(pred_lead2[mask_valid_actual].notna().sum())

        # count errors: predicted-first leading2 equals actual 4-digit leading2
        # only rows with predicted-first leading2 present can be counted as match
        comparisons = (pred_lead2[mask_valid_actual] == actual_lead2[mask_valid_actual])
        errors = int(comparisons.sum())

        error_rate_all = errors / total_examples  # denom: all examples with actual4
        error_rate_valid = (errors / total_with_pred2) if total_with_pred2 > 0 else np.nan

        rows.append({
            "iter": iter_num,
            "errors": errors,
            "total_examples_with_actual4": total_examples,
            "total_with_pred2": total_with_pred2,
            "error_rate_all": error_rate_all,
            "error_rate_valid": error_rate_valid,
            "col": col
        })

    out = pd.DataFrame(rows).sort_values('iter').reset_index(drop=True)
    return out

def plot_error_rates(err_df: pd.DataFrame, outpath: str, show: bool = False):
    iters = err_df['iter'].values
    rates = err_df['error_rate_all'].values * 100  # percent
    plt.figure(figsize=(10,5))
    plt.plot(iters, rates, marker='o', linestyle='-')
    plt.ylim(-2, 102)
    plt.xlabel("Iteration")
    plt.ylabel("Error rate (%)  (leading-2 of first_pred == leading-2 of actual 4-digit)")
    plt.title("Leading-2-digit 'swap-to-4-digit' error rate vs iteration, 3000 test examples")
    plt.grid(True, linestyle='--', alpha=0.5)

    # annotate with errors/total and (valid) counts
    for x, y, err, tot, valid in zip(err_df['iter'], rates, err_df['errors'], err_df['total_examples_with_actual4'], err_df['total_with_pred2']):
        txt = f"{err}"
        plt.annotate(txt, (x,y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")
    if show:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot leading-2-digit error rate vs iteration")
    parser.add_argument("--csv", "-c", required=True, help="Path to input CSV file")
    parser.add_argument("--out", "-o", default="leading2_error_rate.png", help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, na_values=[''])
    if 'actual' not in df.columns:
        raise SystemExit("CSV must have an 'actual' column.")

    err_df = compute_error_rates(df)

    # print table
    print("\nIter | errors | total_examples_with_actual4 | total_with_pred2 | error_rate_all(%) | error_rate_valid(%)")
    for _, r in err_df.iterrows():
        era = r['error_rate_all'] * 100
        erv = (r['error_rate_valid'] * 100) if pd.notna(r['error_rate_valid']) else float('nan')
        print(f"{int(r['iter']):4d} | {int(r['errors']):6d} | {int(r['total_examples_with_actual4']):27d} | {int(r['total_with_pred2']):15d} | {era:16.3f} | {erv:18.3f}")

    plot_error_rates(err_df, args.out, args.show)

if __name__ == "__main__":
    main()
