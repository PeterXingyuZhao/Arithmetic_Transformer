#!/usr/bin/env python3
"""
leading_2_digit_fourth_number.py

Counts the error:
  "Does the leading-2-digits of the model's FOURTH output token equal the
   leading-2-digits of the LARGEST 3-digit number from the original operands?"

Usage:
    python leading_2_digit_fourth_number.py --csv data.csv --out fourth_leading2_error.png --show

Notes:
 - Expects columns: "operands" (original unsorted inputs) and columns named "pred_iter_<N>".
 - Uses digit extraction (keeps only digits) so messy cells are handled.
 - Rows without any 3-digit token in `operands` are skipped.
 - A predicted fourth token must have at least 2 digits to produce a leading-2 match;
   otherwise it's treated as "no 2-digit pred" (not an error).
"""
from __future__ import annotations
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

DIGIT_RE = re.compile(r'\d+')
PRED_PREFIX = "pred_iter_"

def digit_tokens(s: object) -> List[str]:
    """Return list of digit-only tokens found in the cell, in order."""
    if pd.isna(s):
        return []
    return DIGIT_RE.findall(str(s))

def largest_3digit_leading2_from_operands(cell: object) -> Optional[str]:
    """Return leading-2 of the largest numeric token that has length exactly 3 in operands.
       If none found, return None.
    """
    toks = digit_tokens(cell)
    three_digit_nums = [t for t in toks if len(t) == 3]
    if not three_digit_nums:
        return None
    # choose largest numerically (not lexicographically)
    largest = max(three_digit_nums, key=lambda x: int(x))
    return largest[:2]

def fourth_pred_leading2(cell: object) -> Optional[str]:
    """Return leading-2 of the predicted token at position 4 (1-based).
       If there's no 4th token or it's shorter than 2 digits, return None.
    """
    toks = digit_tokens(cell)
    if len(toks) < 4:
        return None
    tok4 = toks[3]
    if len(tok4) < 2:
        return None
    return tok4[:2]

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
    """
    Returns DataFrame with columns:
      iter, errors, total_examples_with_largest3, total_with_pred4_2digits, error_rate_all, error_rate_valid, col
    """
    pred_cols = find_pred_columns(df)
    if not pred_cols:
        raise SystemExit("No pred_iter_* columns found in CSV.")

    if 'operands' not in df.columns:
        raise SystemExit("CSV must have an 'operands' column (original, unsorted inputs).")

    # leading-2 of largest 3-digit in operands (or None)
    largest3_lead2 = df['operands'].apply(largest_3digit_leading2_from_operands)
    mask_valid_operands = largest3_lead2.notna()
    total_examples = int(mask_valid_operands.sum())
    if total_examples == 0:
        raise SystemExit("No rows with a 3-digit token in 'operands' were found.")

    rows = []
    for col in pred_cols:
        iter_num = parse_iter_number(col)
        pred4_lead2 = df[col].apply(fourth_pred_leading2)

        total_with_pred4_2digits = int(pred4_lead2[mask_valid_operands].notna().sum())
        comparisons = (pred4_lead2[mask_valid_operands] == largest3_lead2[mask_valid_operands])
        errors = int(comparisons.sum())

        error_rate_all = errors / total_examples
        error_rate_valid = (errors / total_with_pred4_2digits) if total_with_pred4_2digits > 0 else np.nan

        rows.append({
            "iter": iter_num,
            "errors": errors,
            "total_examples_with_largest3": total_examples,
            "total_with_pred4_2digits": total_with_pred4_2digits,
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
    plt.ylabel("Error rate (%)  (leading-2 of 4th_pred == leading-2 of largest 3-digit from operands)")
    plt.title("Leading-2-digit (largest-3 → 4th-output) error rate vs iteration")
    plt.grid(True, linestyle='--', alpha=0.5)

    for x, y, err, tot, valid in zip(err_df['iter'], rates, err_df['errors'], err_df['total_examples_with_largest3'], err_df['total_with_pred4_2digits']):
        txt = f"{err}"
        plt.annotate(txt, (x,y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")
    if show:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot leading-2-digit error rate for 4th output vs largest 3-digit in operands")
    parser.add_argument("--csv", "-c", required=True, help="Path to input CSV file (must contain 'operands' and pred_iter_* columns)")
    parser.add_argument("--out", "-o", default="fourth_leading2_error_rate.png", help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, na_values=[''])
    err_df = compute_error_rates(df)

    print("\nIter | errors | total_examples_with_largest3 | total_with_pred4_2digits | error_rate_all(%) | error_rate_valid(%)")
    for _, r in err_df.iterrows():
        era = r['error_rate_all'] * 100
        erv = (r['error_rate_valid'] * 100) if pd.notna(r['error_rate_valid']) else float('nan')
        print(f"{int(r['iter']):4d} | {int(r['errors']):6d} | {int(r['total_examples_with_largest3']):27d} | {int(r['total_with_pred4_2digits']):22d} | {era:16.3f} | {erv:18.3f}")

    plot_error_rates(err_df, args.out, args.show)

if __name__ == "__main__":
    main()
