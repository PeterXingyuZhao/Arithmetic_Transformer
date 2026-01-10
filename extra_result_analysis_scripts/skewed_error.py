#!/usr/bin/env python3
"""
leading_k_digit_errors.py

Extended version that computes digit-match errors for three test CSVs:
 - Test1: compare ONLY the single highest digit (k=1).
 - Test2: compare leading-2 digits (k=2).
 - Test3: compare leading-3 digits (k=3).

Produces one PNG per test and a combined PNG with all three plots.

Usage:
    python leading_k_digit_errors.py --csv1 test1.csv --csv2 test2.csv --csv3 test3.csv \
        --outdir ./out --show --titles "Title 1" "Title 2" "Title 3" \
        --ylabels "Y label 1" "Y label 2" "Y label 3"

Disable titles entirely:
    python leading_k_digit_errors.py ... --no-titles

CSV expectations:
 - Column 'actual' containing the ground-truth sorted line (e.g. "2774,524,996,875=524,875,996,2774$")
 - Prediction columns named pred_iter_<N> (e.g. pred_iter_0, pred_iter_1, ...).
"""
from __future__ import annotations
import re
import argparse
import os
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIGIT_RE = re.compile(r'\d+')
PRED_PREFIX = "pred_iter_"

def digit_tokens(s: object) -> List[str]:
    if pd.isna(s):
        return []
    return DIGIT_RE.findall(str(s))

def first_token_leading_k_from_pred(cell: object, k: int) -> Optional[str]:
    """Return the first k digits of the first digit-token in prediction, or None."""
    toks = digit_tokens(cell)
    if not toks:
        return None
    first = toks[0]
    if len(first) < k:
        return None
    return first[:k]

def actual_4digit_leading_k(cell: object, k: int) -> Optional[str]:
    """Return the first k digits of the first 4-digit token in actual, or None."""
    toks = digit_tokens(cell)
    for t in toks:
        if len(t) == 4:
            if k <= 4:
                return t[:k]
            else:
                return None
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

def compute_error_rates_for_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
     iter, matches, total_examples_with_actual4, total_with_pred_k, match_rate_all, match_rate_valid, col
    """
    pred_cols = find_pred_columns(df)
    if not pred_cols:
        raise SystemExit("No pred_iter_* columns found in CSV.")

    actual_k = df['actual'].apply(lambda c: actual_4digit_leading_k(c, k))
    mask_valid_actual = actual_k.notna()
    total_examples = int(mask_valid_actual.sum())
    if total_examples == 0:
        raise SystemExit("No rows with a 4-digit token in 'actual' were found.")

    rows = []
    for col in pred_cols:
        iter_num = parse_iter_number(col)
        pred_k = df[col].apply(lambda c: first_token_leading_k_from_pred(c, k))

        total_with_pred_k = int(pred_k[mask_valid_actual].notna().sum())
        comparisons = (pred_k[mask_valid_actual] == actual_k[mask_valid_actual])
        matches = int(comparisons.sum())

        match_rate_all = matches / total_examples
        match_rate_valid = (matches / total_with_pred_k) if total_with_pred_k > 0 else np.nan

        rows.append({
            "iter": iter_num,
            "matches": matches,
            "total_examples_with_actual4": total_examples,
            "total_with_pred_k": total_with_pred_k,
            "match_rate_all": match_rate_all,
            "match_rate_valid": match_rate_valid,
            "col": col
        })

    return pd.DataFrame(rows).sort_values('iter').reset_index(drop=True)

def compute_ever_exhibited_for_k(df: pd.DataFrame, k: int) -> Tuple[int,int,float]:
    pred_cols = find_pred_columns(df)
    actual_k = df['actual'].apply(lambda c: actual_4digit_leading_k(c, k))
    mask_valid_actual = actual_k.notna()
    total_examples = int(mask_valid_actual.sum())
    if total_examples == 0:
        return 0, 0, float('nan')

    preds = pd.DataFrame({col: df[col].apply(lambda c: first_token_leading_k_from_pred(c, k)) for col in pred_cols})
    matches_df = preds.eq(actual_k, axis=0)
    ever_exhibited = matches_df.any(axis=1)
    ever_count = int(ever_exhibited[mask_valid_actual].sum())
    prop = ever_count / total_examples if total_examples > 0 else float('nan')
    return ever_count, total_examples, prop

def plot_rates(df_rates: pd.DataFrame, k: int, title: Optional[str], outpath: str, ever_info: Tuple[int,int,float], show: bool=False, y_label: Optional[str]=None):
    iters = df_rates['iter'].values
    rates = df_rates['match_rate_all'].values * 100
    plt.figure(figsize=(8,4))
    plt.plot(iters, rates, marker='o', linestyle='-')
    plt.ylim(-2, 102)
    plt.xlabel("Iteration")
    ylabel_text = y_label if y_label is not None else f"Match rate (%) (leading-{k} of first_pred == leading-{k} of actual 4-digit)"
    plt.ylabel(ylabel_text)
    # Only set title when provided (and not explicitly empty)
    if title:
        plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)

    for x, y, matches in zip(df_rates['iter'], rates, df_rates['matches']):
        plt.annotate(f"{matches}", (x,y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)

    if ever_info is not None:
        ever_count, total_examples, prop = ever_info
        txt = f"Ever exhibited: {ever_count}/{total_examples} = {prop*100:.3f}%"
        plt.gcf().text(0.02, 0.95, txt, fontsize=9, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"Saved plot: {outpath}")
    if show:
        plt.show()
    plt.close()

def create_combined_plot(all_rate_dfs: Dict[str, pd.DataFrame], titles: Dict[str,Optional[str]], combined_outpath: str, ever_infos: Dict[str,Tuple[int,int,float]], ylabels: Dict[str,Optional[str]], show: bool=False):
    fig, axes = plt.subplots(1, len(all_rate_dfs), figsize=(5*len(all_rate_dfs), 4), sharey=False)
    if len(all_rate_dfs) == 1:
        axes = [axes]
    for ax, (key, df_rates) in zip(axes, all_rate_dfs.items()):
        iters = df_rates['iter'].values
        rates = df_rates['match_rate_all'].values * 100
        ax.plot(iters, rates, marker='o', linestyle='-')
        ax.set_xlabel("Iteration")
        ylabel_text = ylabels.get(key)
        if ylabel_text is None:
            ax.set_ylabel("Match rate (%)")
        else:
            # if user passed empty string intentionally, still set it; but we normally use None to mean default
            ax.set_ylabel(ylabel_text)
        title_text = titles.get(key)
        if title_text:
            ax.set_title(title_text)
        ax.grid(True, linestyle='--', alpha=0.4)
        for x, y, matches in zip(df_rates['iter'], rates, df_rates['matches']):
            ax.annotate(f"{matches}", (x,y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)
        ever_info = ever_infos.get(key)
        if ever_info:
            ever_count, total_examples, prop = ever_info
            txt = f"Ever: {ever_count}/{total_examples} = {prop*100:.3f}%"
            ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=8, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    fig.savefig(combined_outpath, dpi=150)
    print(f"Saved combined plot: {combined_outpath}")
    if show:
        plt.show()
    plt.close(fig)

def process_one(csvpath: str, k: int, label: str, title: Optional[str], y_label: Optional[str], outdir: str, show: bool=False):
    df = pd.read_csv(csvpath, dtype=str, keep_default_na=False, na_values=[''])
    if 'actual' not in df.columns:
        raise SystemExit(f"CSV {csvpath} must have an 'actual' column.")
    rates_df = compute_error_rates_for_k(df, k)
    ever_info = compute_ever_exhibited_for_k(df, k)
    print(f"\n=== Results for {label} (k={k}) on {os.path.basename(csvpath)} ===")
    print("Iter | matches | total_examples_with_actual4 | total_with_pred_k | match_rate_all(%) | match_rate_valid(%)")
    for _, r in rates_df.iterrows():
        m_all = r['match_rate_all'] * 100
        m_valid = (r['match_rate_valid'] * 100) if pd.notna(r['match_rate_valid']) else float('nan')
        print(f"{int(r['iter']):4d} | {int(r['matches']):7d} | {int(r['total_examples_with_actual4']):27d} | {int(r['total_with_pred_k']):15d} | {m_all:16.3f} | {m_valid:18.3f}")
    ever_count, total_examples, prop = ever_info
    print(f"Ever-exhibited match across ALL iterations: {ever_count}/{total_examples} = {prop*100:.3f}%")

    outpath = os.path.join(outdir, f"{label}_k{k}_error.png")
    plot_rates(rates_df, k, title, outpath, ever_info, show=show, y_label=y_label)
    return rates_df, ever_info

def main():
    parser = argparse.ArgumentParser(description="Compute leading-k digit matching errors for three test CSVs and plot per-iteration rates.")
    parser.add_argument("--csv1", required=True, help="CSV for test1 (NOW uses k=1)")
    parser.add_argument("--csv2", required=True, help="CSV for test2 (k=2)")
    parser.add_argument("--csv3", required=True, help="CSV for test3 (k=3)")
    parser.add_argument("--outdir", default=".", help="Output directory for PNGs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--titles", nargs=3, metavar=('T1','T2','T3'),
                        help="Optional titles for the three plots (in order: csv1 csv2 csv3)")
    parser.add_argument("--ylabels", nargs=3, metavar=('Y1','Y2','Y3'),
                        help="Optional y-axis labels for the three plots (in order: csv1 csv2 csv3)")
    parser.add_argument("--no-titles", dest="no_titles", action="store_true",
                        help="Disable titles on all plots (overrides --titles)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # defaults for titles and ylabels if not provided
    default_titles = [
        "test1 (leading-1 match)",
        "test2 (leading-2 match)",
        "test3 (leading-3 match)",
    ]
    default_ylabels = [
        "Match rate (%) (leading-1)",
        "Match rate (%) (leading-2)",
        "Match rate (%) (leading-3)",
    ]

    # choose titles: priority -> --no-titles (None), --titles (provided list), defaults
    if args.no_titles:
        provided_titles: List[Optional[str]] = [None, None, None]
    else:
        if args.titles is not None:
            provided_titles = list(args.titles)
        else:
            provided_titles = default_titles

    provided_ylabels = list(args.ylabels) if args.ylabels is not None else default_ylabels

    specs = [
        (args.csv1, 1, "test1", provided_titles[0], provided_ylabels[0]),
        (args.csv2, 2, "test2", provided_titles[1], provided_ylabels[1]),
        (args.csv3, 3, "test3", provided_titles[2], provided_ylabels[2]),
    ]

    all_rate_dfs: Dict[str, pd.DataFrame] = {}
    titles: Dict[str, Optional[str]] = {}
    ever_infos: Dict[str, Tuple[int,int,float]] = {}
    ylabels_map: Dict[str, Optional[str]] = {}
    for csvpath, k, label, title, y_label in specs:
        rates_df, ever_info = process_one(csvpath, k, label, title, y_label, args.outdir, show=args.show)
        all_rate_dfs[label] = rates_df
        titles[label] = title
        ever_infos[label] = ever_info
        ylabels_map[label] = y_label

    # combined plot
    combined_outpath = os.path.join(args.outdir, "combined_three_tests.png")
    create_combined_plot(all_rate_dfs, titles, combined_outpath, ever_infos, ylabels_map, show=args.show)

if __name__ == "__main__":
    main()
