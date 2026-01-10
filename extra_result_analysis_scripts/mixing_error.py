#!/usr/bin/env python3
"""
count_sorting_errors.py

Usage:
    python mixing_error.py [path_to_csv]

Description:
    Reads a CSV containing columns: operands, actual, pred_iter_0, pred_iter_50, ...
    Counts four types of errors on the two middle numbers:
      - swapping last digit
      - swapping last two digits
      - repeating last digit
      - repeating last two digits
    Plots counts vs iteration and saves 'errors_vs_iter.png'.
"""
import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers: parsing and checks
# -----------------------------
def detect_sep_and_read(path):
    """
    Attempt to read CSV/TSV. If file contains tabs, use '\t', else fallback to comma.
    Use pandas to read header-preserving.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        head = fh.read(4096)
    if '\t' in head:
        sep = '\t'
    else:
        sep = ','
    # Use pandas; allow quoting; sometimes values themselves contain commas, but sample shows tabs between columns.
    df = pd.read_csv(path, sep=sep, engine='python', dtype=str).fillna('')
    return df

def parse_4nums(cell):
    """
    Parse a cell string like "1000,2434,6131,9999" into list of 4 ints.
    Return None if can't parse exactly 4 ints.
    Trim spaces.
    """
    if not isinstance(cell, str):
        return None
    # remove whitespace
    s = cell.strip()
    if s == '':
        return None
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    # If there are more than 4 components because of stray commas, try to find 4-digit-looking tokens.
    if len(parts) != 4:
        # attempt to extract 4 groups of 1-4 digit numbers in order
        cand = re.findall(r'\d{1,4}', s)
        if len(cand) >= 4:
            parts = cand[:4]
        else:
            return None
    try:
        nums = [int(p) for p in parts]
        if any(n < 0 or n > 9999 for n in nums):
            # still accept but guard
            pass
        return nums
    except:
        return None

def prefix_n(n, count):
    """Return the prefix of n in digit-terms: count=3 -> first 3 digits (i.e. n//10)"""
    if count == 3:
        return n // 10
    elif count == 2:
        return n // 100
    elif count == 1:
        return n // 1000
    else:
        raise ValueError("unsupported prefix count")

# -----------------------------
# Error detectors
# -----------------------------
def is_swap_last_digit(A1, A2, P1, P2):
    # require first three digits match actual for both numbers
    if prefix_n(P1, 3) == prefix_n(A1, 3) and prefix_n(P2, 3) == prefix_n(A2, 3):
        # last digits swapped
        if (P1 % 10 == (A2 % 10)) and (P2 % 10 == (A1 % 10)):
            return True
    return False

def is_swap_last_two(A1, A2, P1, P2):
    # require first two digits match actual for both numbers
    if prefix_n(P1, 2) == prefix_n(A1, 2) and prefix_n(P2, 2) == prefix_n(A2, 2):
        # last two digits swapped
        if (P1 % 100 == (A2 % 100)) and (P2 % 100 == (A1 % 100)):
            return True
    return False

def is_repeat_last_digit(A1, A2, P1, P2):
    # require first three digits match actual for both numbers
    if prefix_n(P1, 3) == prefix_n(A1, 3) and prefix_n(P2, 3) == prefix_n(A2, 3):
        # check if P1 took the last digit of A2 while P2 unchanged:
        if (P1 % 10 == (A2 % 10)) and (P2 % 10 == (A2 % 10)):
            return True
        # Or symmetric: P2 took last digit of A1 while P1 unchanged:
        if (P2 % 10 == (A1 % 10)) and (P1 % 10 == (A1 % 10)):
            return True
    return False

def is_repeat_last_two(A1, A2, P1, P2):
    # require first two digits match actual for both numbers
    if prefix_n(P1, 2) == prefix_n(A1, 2) and prefix_n(P2, 2) == prefix_n(A2, 2):
        # P1 copied last-two from A2 while P2 unchanged
        if (P1 % 100 == (A2 % 100)) and (P2 % 100 == (A2 % 100)):
            return True
        # symmetric: P2 copied last-two from A1 while P1 unchanged
        if (P2 % 100 == (A1 % 100)) and (P1 % 100 == (A1 % 100)):
            return True
    return False

# -----------------------------
# Main
# -----------------------------
def main(csv_path='results.csv', out_png='errors_vs_iter.png'):
    if not os.path.exists(csv_path):
        print("File not found:", csv_path)
        print("Please provide correct path to your CSV file.")
        sys.exit(1)

    df = detect_sep_and_read(csv_path)

    # normalize column names to strings
    cols = list(df.columns)
    # find pred_iter_* columns
    pred_cols = [c for c in cols if isinstance(c, str) and c.strip().startswith('pred_iter')]
    if len(pred_cols) == 0:
        # try columns containing 'pred'
        pred_cols = [c for c in cols if isinstance(c, str) and ('pred' in c)]
    if len(pred_cols) == 0:
        raise RuntimeError("No pred_iter_* columns found. Columns: " + ", ".join(cols))

    # parse operands & actual into middle two numbers
    # expecting columns named 'operands' and 'actual'
    if 'operands' not in df.columns or 'actual' not in df.columns:
        # try close name matches
        low = [c.lower() for c in df.columns]
        op_idx = None
        act_idx = None
        for i,c in enumerate(low):
            if 'operand' in c:
                op_idx = i
            if c == 'actual' or 'actual' in c:
                act_idx = i
        if op_idx is None or act_idx is None:
            raise RuntimeError("Cannot find 'operands' and 'actual' columns. Found: " + ", ".join(df.columns))
        operands_col = df.columns[op_idx]
        actual_col = df.columns[act_idx]
    else:
        operands_col = 'operands'
        actual_col = 'actual'

    # build iteration list (extract numeric part from pred_iter_N)
    iter_values = []
    pred_col_to_iter = {}
    for c in pred_cols:
        m = re.search(r'pred_iter_(-?\d+)', c)
        if m:
            it = int(m.group(1))
        else:
            # fallback: find any number in column name
            m2 = re.search(r'(\d+)', c)
            it = int(m2.group(1)) if m2 else 0
        pred_col_to_iter[c] = it
        iter_values.append(it)

    # We'll sort columns by iteration value
    pred_cols_sorted = sorted(pred_cols, key=lambda c: pred_col_to_iter[c])
    iter_sorted = [pred_col_to_iter[c] for c in pred_cols_sorted]

    # initialize counters
    swap_last_counts = {it:0 for it in iter_sorted}
    swap_last2_counts = {it:0 for it in iter_sorted}
    repeat_last_counts = {it:0 for it in iter_sorted}
    repeat_last2_counts = {it:0 for it in iter_sorted}

    # iterate rows
    n_rows = df.shape[0]
    for idx, row in df.iterrows():
        # parse actual
        actual_cell = str(row[actual_col])
        actual_parsed = parse_4nums(actual_cell)
        if actual_parsed is None:
            # try to parse operands as actual fallback
            actual_parsed = parse_4nums(str(row[operands_col]))
            if actual_parsed is None:
                # skip this row if cannot parse actual
                continue
        # get the two middle numbers in sorted order from actual parsing
        A1 = actual_parsed[1]
        A2 = actual_parsed[2]

        # For each pred column, parse and check errors
        for c in pred_cols_sorted:
            pred_cell = str(row[c]) if c in row else ''
            pred_parsed = parse_4nums(pred_cell)
            if pred_parsed is None:
                # skip malformed/missing pred entries
                continue
            P1 = pred_parsed[1]
            P2 = pred_parsed[2]

            it = pred_col_to_iter[c]

            # Now check each error type (each is counted separately)
            if is_swap_last_digit(A1, A2, P1, P2):
                swap_last_counts[it] += 1
            if is_swap_last_two(A1, A2, P1, P2):
                swap_last2_counts[it] += 1
            if is_repeat_last_digit(A1, A2, P1, P2):
                repeat_last_counts[it] += 1
            if is_repeat_last_two(A1, A2, P1, P2):
                repeat_last2_counts[it] += 1

    # Prepare arrays for plotting (aligned with sorted iteration list)
    xs = iter_sorted
    swap_last_y = [swap_last_counts[it] for it in xs]
    swap_last2_y = [swap_last2_counts[it] for it in xs]
    repeat_last_y = [repeat_last_counts[it] for it in xs]
    repeat_last2_y = [repeat_last2_counts[it] for it in xs]

    # Create a DataFrame of counts for possible inspection / saving
    counts_df = pd.DataFrame({
        'iter': xs,
        'swap_last_digit': swap_last_y,
        'swap_last_two': swap_last2_y,
        'repeat_last_digit': repeat_last_y,
        'repeat_last_two': repeat_last2_y,
    })

    # Plot all four on a single figure
    plt.figure(figsize=(10,6))
    plt.plot(counts_df['iter'], counts_df['swap_last_digit'], marker='o', label='swap last digit')
    plt.plot(counts_df['iter'], counts_df['swap_last_two'], marker='o', label='swap last two')
    plt.plot(counts_df['iter'], counts_df['repeat_last_digit'], marker='o', label='repeat last digit')
    plt.plot(counts_df['iter'], counts_df['repeat_last_two'], marker='o', label='repeat last two')
    plt.xlabel('iteration')
    plt.ylabel('count')
    plt.title('Error counts vs iteration')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print("Saved plot to", out_png)

    # also print table
    print("\nCounts table:")
    print(counts_df.to_string(index=False))

    return counts_df, out_png

if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else 'results.csv'
    main(csv_path)
