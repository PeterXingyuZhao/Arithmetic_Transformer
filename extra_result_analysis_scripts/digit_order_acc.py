#!/usr/bin/env python3
"""
digit_order_acc.py

Compute and plot digit-place accuracies across multiple experiment result files
on a single graph (no markers).

Example usage:
    python digit_order_acc_multi.py \
        --files \
          "/path/to/digitwise_random_results.csv" \
          "/path/to/digitwise_thousand_results.csv" \
          "/path/to/digitwise_hundred_results.csv" \
          "/path/to/digitwise_ten_results.csv" \
        --places thousands hundreds tens units \
        --labels Thousand Hundred Ten Unit \
        --out combined_accuracy

This script:
 - reads each CSV (with columns: operands, actual, pred_iter_0, pred_iter_50, ...)
 - computes accuracy per iteration for the given digit place
 - plots all curves on the same graph (no markers)
 - writes combined_accuracy_accuracy_vs_iter.png
"""

from __future__ import annotations
import argparse
import re
import sys
from typing import List, Tuple, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
RE_4DIGIT = re.compile(r"\b([1-9][0-9]{3})\b")
RE_ANYINT = re.compile(r"\b([0-9]+)\b")

def extract_numbers(cell: str, require_4digit: bool = True) -> List[int]:
    if not isinstance(cell, str):
        return []
    cell = cell.strip()
    found = RE_4DIGIT.findall(cell)
    if found:
        return [int(x) for x in found]
    found_any = RE_ANYINT.findall(cell)
    if not found_any:
        return []
    ints = [int(x) for x in found_any]
    ints_4 = [x for x in ints if 1000 <= x <= 9999]
    return ints_4 or ints

def digit_for_place(n: int, place: str) -> int:
    if place == "thousands": return (n // 1000) % 10
    if place == "hundreds":  return (n // 100) % 10
    if place == "tens":      return (n // 10) % 10
    if place == "units":     return n % 10
    raise ValueError("place must be one of thousands, hundreds, tens, units")

# ---------- core ----------
def compute_accuracies(df: pd.DataFrame, place: str, pred_prefix: str = "pred_iter_") -> Tuple[List[int], List[float]]:
    pred_cols = [c for c in df.columns if c.startswith(pred_prefix)]
    if not pred_cols:
        raise ValueError(f"No columns starting with '{pred_prefix}' found in file.")
    def iter_from_col(col: str) -> int:
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else 0
    pred_cols_sorted = sorted(pred_cols, key=iter_from_col)
    iters = [iter_from_col(c) for c in pred_cols_sorted]
    correct_counts = [0] * len(pred_cols_sorted)
    actual_digits_list = []
    for _, row in df.iterrows():
        actual_nums = extract_numbers(row.get("actual", ""), require_4digit=True)
        if len(actual_nums) < 4:
            actual_nums = extract_numbers(row.get("actual", ""), require_4digit=False)
        if len(actual_nums) < 4:
            actual_digits_list.append(None)
            continue
        actual_digits_list.append([digit_for_place(x, place) for x in actual_nums[:4]])
    n_valid = sum(1 for a in actual_digits_list if a)
    for col_idx, col in enumerate(pred_cols_sorted):
        for i, row in df.iterrows():
            if not actual_digits_list[i]:
                continue
            pred_nums = extract_numbers(row.get(col, ""), require_4digit=True)
            if len(pred_nums) < 4:
                pred_nums = extract_numbers(row.get(col, ""), require_4digit=False)
            if len(pred_nums) < 4:
                continue
            pred_digits = [digit_for_place(x, place) for x in pred_nums[:4]]
            if pred_digits == actual_digits_list[i]:
                correct_counts[col_idx] += 1
    accuracies = [c / n_valid for c in correct_counts]
    return iters, accuracies

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Compute and plot multiple digit-place accuracies together.")
    parser.add_argument("--files", nargs="+", required=True, help="List of CSV result files.")
    parser.add_argument("--places", nargs="+", required=True, choices=["thousands", "hundreds", "tens", "units"],
                        help="Digit places corresponding to the files.")
    parser.add_argument("--labels", nargs="+", help="Optional curve labels for legend.")
    parser.add_argument("--out", default="combined_accuracy", help="Output file prefix for the combined plot.")
    parser.add_argument("--delimiter", default=None, help="Delimiter override for input files (default auto-detect).")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (for servers).")
    args = parser.parse_args()

    if len(args.files) != len(args.places):
        sys.exit("Error: number of --files must match number of --places.")
    labels = args.labels or [f.split('/')[-1].replace('.csv', '') for f in args.files]
    if len(labels) != len(args.files):
        sys.exit("Error: number of --labels must match number of --files (if provided).")

    plt.figure(figsize=(8, 5))

    for file, place, label in zip(args.files, args.places, labels):
        print(f"Processing {file} ({place}) ...", file=sys.stderr)
        read_kwargs = {"dtype": str, "keep_default_na": False}
        if args.delimiter is None:
            read_kwargs.update({"sep": None, "engine": "python"})
        else:
            read_kwargs.update({"sep": args.delimiter, "engine": "python"})
        try:
            df = pd.read_csv(file, **read_kwargs)
        except Exception:
            df = pd.read_csv(file, sep="\t", dtype=str, keep_default_na=False, engine="python")
        df.columns = [c.strip() for c in df.columns]
        iters, accs = compute_accuracies(df, place)
        plt.plot(iters, accs, label=label, linewidth=2)  # no marker

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Digit-place Accuracy vs Iteration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = f"{args.out}_accuracy_vs_iter.png"
    plt.savefig(out_png)
    print(f"Saved combined plot to {out_png}")
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
