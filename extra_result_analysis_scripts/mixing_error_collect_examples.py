#!/usr/bin/env python3
"""
mixing_error_collect_examples.py

Usage:
    python mixing_error_collect_examples.py [path_to_csv]

Description:
    Reads a CSV containing columns: operands, actual, pred_iter_0, pred_iter_50, ...
    Counts four types of errors on the two middle numbers:
      - swapping last digit
      - swapping last two digits
      - repeating last digit
      - repeating last two digits
    Plots counts vs iteration and saves 'errors_vs_iter.png'.
    Additionally collects examples that trigger each error type into four CSV files.
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
    df = pd.read_csv(path, sep=sep, engine='python', dtype=str).fillna('')
    return df

def parse_4nums(cell):
    """
    Parse a cell string like "1000,2434,6131,9999" into list of 4 ints.
    Return None if can't parse at least 4 ints.
    Trim spaces.
    """
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if s == '':
        return None
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if len(parts) != 4:
        # attempt to extract numeric tokens
        cand = re.findall(r'\d{1,4}', s)
        if len(cand) >= 4:
            parts = cand[:4]
        else:
            return None
    try:
        nums = [int(p) for p in parts]
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
# Error detectors (fixed)
# -----------------------------
def is_swap_last_digit(A1, A2, P1, P2):
    """
    Detect swapping of the last digit between the two middle numbers.
    Fixed: if the two actual numbers share the same last digit, do NOT count.
    Also do not count when prediction equals actual (correct).
    """
    # If prediction exactly matches actual, it's not an error
    if P1 == A1 and P2 == A2:
        return False
    # If actual last digits are identical, ambiguous -> skip
    if (A1 % 10) == (A2 % 10):
        return False
    # require first three digits (prefix of 3) match actuals
    if prefix_n(P1, 3) == prefix_n(A1, 3) and prefix_n(P2, 3) == prefix_n(A2, 3):
        # last digits swapped and they are different (we checked actual last digits differ)
        if (P1 % 10 == (A2 % 10)) and (P2 % 10 == (A1 % 10)):
            return True
    return False

def is_swap_last_two(A1, A2, P1, P2):
    """
    Detect swapping of the last two digits between the two middle numbers.
    Fixed: if the two actual numbers share the same last-two digits, do NOT count.
    Also do not count when prediction equals actual.
    """
    if P1 == A1 and P2 == A2:
        return False
    if (A1 % 100) == (A2 % 100):
        return False
    # require first two digits (prefix of 2) match actuals
    if prefix_n(P1, 2) == prefix_n(A1, 2) and prefix_n(P2, 2) == prefix_n(A2, 2):
        if (P1 % 100 == (A2 % 100)) and (P2 % 100 == (A1 % 100)):
            return True
    return False

def is_repeat_last_digit(A1, A2, P1, P2):
    """
    Detect repeating the last digit from one number into the other.
    Fixed: if the two actual numbers share the same last digit, do NOT count.
    Also do not count when prediction equals actual.
    """
    if P1 == A1 and P2 == A2:
        return False
    if (A1 % 10) == (A2 % 10):
        return False
    # require first three digits match actuals
    if prefix_n(P1, 3) == prefix_n(A1, 3) and prefix_n(P2, 3) == prefix_n(A2, 3):
        # check if P1 copied last digit of A2 while P2 unchanged to A2's last digit
        if (P1 % 10 == (A2 % 10)) and (P2 % 10 == (A2 % 10)):
            return True
        # symmetric: P2 copied last digit of A1
        if (P2 % 10 == (A1 % 10)) and (P1 % 10 == (A1 % 10)):
            return True
    return False

def is_repeat_last_two(A1, A2, P1, P2):
    """
    Detect repeating the last two digits from one number into the other.
    Fixed: if the two actual numbers share the same last-two digits, do NOT count.
    Also do not count when prediction equals actual.
    """
    if P1 == A1 and P2 == A2:
        return False
    if (A1 % 100) == (A2 % 100):
        return False
    # require first two digits match actuals
    if prefix_n(P1, 2) == prefix_n(A1, 2) and prefix_n(P2, 2) == prefix_n(A2, 2):
        # P1 copied last-two from A2 while P2 unchanged or symmetric
        if (P1 % 100 == (A2 % 100)) and (P2 % 100 == (A2 % 100)):
            return True
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

    # find pred_iter_* columns
    cols = list(df.columns)
    pred_cols = [c for c in cols if isinstance(c, str) and c.strip().startswith('pred_iter')]
    if len(pred_cols) == 0:
        pred_cols = [c for c in cols if isinstance(c, str) and ('pred' in c)]
    if len(pred_cols) == 0:
        raise RuntimeError("No pred_iter_* columns found. Columns: " + ", ".join(cols))

    # find operands and actual columns
    if 'operands' not in df.columns or 'actual' not in df.columns:
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

    # extract iteration numbers from pred column names
    pred_col_to_iter = {}
    for c in pred_cols:
        m = re.search(r'pred_iter_(-?\d+)', c)
        if m:
            it = int(m.group(1))
        else:
            m2 = re.search(r'(\d+)', c)
            it = int(m2.group(1)) if m2 else 0
        pred_col_to_iter[c] = it

    pred_cols_sorted = sorted(pred_cols, key=lambda c: pred_col_to_iter[c])
    iter_sorted = [pred_col_to_iter[c] for c in pred_cols_sorted]

    # initialize counters
    swap_last_counts = {it:0 for it in iter_sorted}
    swap_last2_counts = {it:0 for it in iter_sorted}
    repeat_last_counts = {it:0 for it in iter_sorted}
    repeat_last2_counts = {it:0 for it in iter_sorted}

    # collectors for examples
    swap_last_examples = []
    swap_last2_examples = []
    repeat_last_examples = []
    repeat_last2_examples = []

    # iterate rows
    for idx, row in df.iterrows():
        actual_cell = str(row[actual_col])
        actual_parsed = parse_4nums(actual_cell)
        if actual_parsed is None:
            actual_parsed = parse_4nums(str(row[operands_col]))
            if actual_parsed is None:
                continue
        A1 = actual_parsed[1]
        A2 = actual_parsed[2]

        for c in pred_cols_sorted:
            pred_cell = str(row[c]) if c in row else ''
            pred_parsed = parse_4nums(pred_cell)
            if pred_parsed is None:
                continue
            P1 = pred_parsed[1]
            P2 = pred_parsed[2]
            it = pred_col_to_iter[c]

            # check errors and increment + collect
            if is_swap_last_digit(A1, A2, P1, P2):
                swap_last_counts[it] += 1
                swap_last_examples.append({
                    'row_index': idx,
                    'operands': row.get(operands_col, ''),
                    'actual': row.get(actual_col, ''),
                    'pred_col': c,
                    'iter': it,
                    'pred': pred_cell,
                    'A1': A1, 'A2': A2, 'P1': P1, 'P2': P2,
                    'error_type': 'swap_last_digit'
                })
            if is_swap_last_two(A1, A2, P1, P2):
                swap_last2_counts[it] += 1
                swap_last2_examples.append({
                    'row_index': idx,
                    'operands': row.get(operands_col, ''),
                    'actual': row.get(actual_col, ''),
                    'pred_col': c,
                    'iter': it,
                    'pred': pred_cell,
                    'A1': A1, 'A2': A2, 'P1': P1, 'P2': P2,
                    'error_type': 'swap_last_two'
                })
            if is_repeat_last_digit(A1, A2, P1, P2):
                repeat_last_counts[it] += 1
                repeat_last_examples.append({
                    'row_index': idx,
                    'operands': row.get(operands_col, ''),
                    'actual': row.get(actual_col, ''),
                    'pred_col': c,
                    'iter': it,
                    'pred': pred_cell,
                    'A1': A1, 'A2': A2, 'P1': P1, 'P2': P2,
                    'error_type': 'repeat_last_digit'
                })
            if is_repeat_last_two(A1, A2, P1, P2):
                repeat_last2_counts[it] += 1
                repeat_last2_examples.append({
                    'row_index': idx,
                    'operands': row.get(operands_col, ''),
                    'actual': row.get(actual_col, ''),
                    'pred_col': c,
                    'iter': it,
                    'pred': pred_cell,
                    'A1': A1, 'A2': A2, 'P1': P1, 'P2': P2,
                    'error_type': 'repeat_last_two'
                })

    # Prepare arrays for plotting (aligned with sorted iteration list)
    xs = iter_sorted
    swap_last_y = [swap_last_counts[it] for it in xs]
    swap_last2_y = [swap_last2_counts[it] for it in xs]
    repeat_last_y = [repeat_last_counts[it] for it in xs]
    repeat_last2_y = [repeat_last2_counts[it] for it in xs]

    # Create a DataFrame of counts for inspection / saving
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

    # Prepare output folder for example CSVs
    out_folder = "error_examples"
    os.makedirs(out_folder, exist_ok=True)

    # Convert example collectors to DataFrames and save
    def save_examples(list_of_dicts, fname):
        if len(list_of_dicts) == 0:
            # create empty dataframe with columns
            cols = ['row_index','operands','actual','pred_col','iter','pred','A1','A2','P1','P2','error_type']
            pd.DataFrame(columns=cols).to_csv(os.path.join(out_folder, fname), index=False)
            return
        pdf = pd.DataFrame(list_of_dicts)
        # sort for readability (by iter then row_index)
        pdf = pdf.sort_values(['iter','row_index']).reset_index(drop=True)
        pdf.to_csv(os.path.join(out_folder, fname), index=False)

    save_examples(swap_last_examples, "swap_last_digit_examples.csv")
    save_examples(swap_last2_examples, "swap_last_two_examples.csv")
    save_examples(repeat_last_examples, "repeat_last_digit_examples.csv")
    save_examples(repeat_last2_examples, "repeat_last_two_examples.csv")

    print(f"Saved example CSVs into folder: {out_folder}")
    print("Files:")
    for nm in ["swap_last_digit_examples.csv","swap_last_two_examples.csv",
               "repeat_last_digit_examples.csv","repeat_last_two_examples.csv"]:
        print("  ", os.path.join(out_folder, nm))

    # also print table
    print("\nCounts table:")
    print(counts_df.to_string(index=False))

    # return data for programmatic use
    return {
        'counts_df': counts_df,
        'plot_path': out_png,
        'example_files': {
            'swap_last_digit': os.path.join(out_folder, "swap_last_digit_examples.csv"),
            'swap_last_two': os.path.join(out_folder, "swap_last_two_examples.csv"),
            'repeat_last_digit': os.path.join(out_folder, "repeat_last_digit_examples.csv"),
            'repeat_last_two': os.path.join(out_folder, "repeat_last_two_examples.csv"),
        }
    }

if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else 'results.csv'
    main(csv_path)
