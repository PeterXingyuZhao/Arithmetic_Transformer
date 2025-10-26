#!/usr/bin/env python3
"""
collect_incorrects.py

Collect incorrect examples under strict whole-result match (last evaluated iteration).

Usage:
    python collect_incorrects.py --csv results.csv --out incorrect.csv

Output:
    CSV with columns: operands, actual, model's output
"""
from __future__ import annotations
import argparse
import re
from typing import List
import pandas as pd

INT_TOKEN_RE = re.compile(r'-?\d+')

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

def find_pred_columns(df: pd.DataFrame) -> List[str]:
    pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]
    def iter_of(col: str) -> int:
        m = re.search(r'pred_iter_(\d+)', col)
        return int(m.group(1)) if m else -1
    return sorted(pred_cols, key=iter_of)

def parse_iter_number(colname: str) -> int:
    m = re.search(r'pred_iter_(\d+)', colname)
    return int(m.group(1)) if m else -1

def main():
    parser = argparse.ArgumentParser(description="Collect incorrect strict whole-result examples.")
    parser.add_argument("--csv", "-c", required=True, help="Input CSV with columns 'actual' and pred_iter_*")
    parser.add_argument("--out", "-o", default="incorrect.csv", help="Output CSV file for incorrect examples")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, na_values=[''])
    pred_cols = find_pred_columns(df)
    if not pred_cols:
        raise SystemExit("No pred_iter_* columns found in input CSV.")

    # choose last (largest iter) column
    last_col = pred_cols[-1]

    if 'actual' not in df.columns:
        raise SystemExit("Input CSV must contain an 'actual' column.")

    # canonical sorted target (list of ints)
    canonical_sorted = df['actual'].apply(extract_token_ints).apply(lambda toks: sorted(toks) if toks else [])

    # predicted lists (list of ints) from last iteration column
    pred_lists = df[last_col].apply(extract_token_ints)

    # only consider rows that have at least one token in canonical target
    mask_valid = canonical_sorted.apply(lambda toks: len(toks) > 0)

    # collect indices where prediction != canonical (strict whole-result)
    incorrect_indices = []
    for idx in mask_valid[mask_valid].index:
        if pred_lists.at[idx] != canonical_sorted.at[idx]:
            incorrect_indices.append(idx)

    # Prepare output dataframe with required columns
    # If 'operands' is not present in input, fall back to copying 'actual' into operands
    if 'operands' not in df.columns:
        operands_series = df['actual'].astype(str)
    else:
        operands_series = df['operands'].astype(str)

    out_df = pd.DataFrame({
        "operands": operands_series.loc[incorrect_indices].values,
        "actual": df['actual'].loc[incorrect_indices].values,
        "model's output": df[last_col].loc[incorrect_indices].values
    })

    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} incorrect examples to {args.out} (using prediction column '{last_col}').")

if __name__ == "__main__":
    main()
