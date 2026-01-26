#!/usr/bin/env python3
"""
sorting_acc_10_90_range.py

Compute JOINT accuracy per pred_iter_* column, then report:
  a = first iteration where accuracy >= lower (default 0.10)
  b = first iteration where accuracy >= upper (default 0.90)

Also draws a progress-bar plot (like plot_10_90_ranges):
- If [a,b] exists and (b-a) <= 50 -> draw a single dot
- Else draw a thick horizontal bar from a to b
- If missing a or b -> legend shows "none" with gray patch

Important behavior:
- Filters out iter < 0 (e.g., pred_iter_-1) by default.
- Starts from iter 0 by default.
- Saves PDF to the directory of the first CSV: accuracy_10_90_ranges.pdf
"""

from __future__ import annotations
import argparse
import os
import re
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

INT_TOKEN_RE = re.compile(r"-?\d+")
DIGIT_TOKEN_RE = re.compile(r"\d+")
DIGIT_MODE_MAP = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def extract_token_ints(s: object) -> List[int]:
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
    if pd.isna(s) or (isinstance(s, str) and s.strip() == ""):
        return []
    s = str(s)
    return DIGIT_TOKEN_RE.findall(s)


def token_str_at_pos(cell: object, pos: int) -> Optional[str]:
    toks = extract_token_strs(cell)
    if len(toks) >= pos:
        return toks[pos - 1]
    return None


def find_pred_columns(df: pd.DataFrame) -> List[str]:
    pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]

    def iter_of(col: str) -> int:
        m = re.search(r"pred_iter_(-?\d+)", col)
        return int(m.group(1)) if m else 10**18

    return sorted(pred_cols, key=iter_of)


def parse_iter_number(colname: str) -> int:
    m = re.search(r"pred_iter_(-?\d+)", colname)
    return int(m.group(1)) if m else -1


def parse_positions_arg(values: List[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip() != ""]
        for p in parts:
            try:
                n = int(p)
                if n < 1:
                    raise ValueError
                out.append(n)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid position value: {p}")
    return sorted(list(dict.fromkeys(out)))


def compute_joint_accuracy_curve(df: pd.DataFrame, positions: List[int], mode: str) -> pd.DataFrame:
    if "actual" not in df.columns:
        raise ValueError("CSV must contain an 'actual' column.")

    pred_cols = find_pred_columns(df)
    if not pred_cols:
        raise ValueError("No columns named pred_iter_* found in CSV.")

    actual_ints = df["actual"].apply(extract_token_ints)
    actual_strs = df["actual"].apply(extract_token_strs)
    canonical_sorted = actual_ints.apply(lambda toks: sorted(toks) if toks else [])

    rows: List[Dict] = []

    if mode == "strict":
        mask_valid = canonical_sorted.apply(lambda toks: len(toks) > 0)
        total = int(mask_valid.sum())
        for col in pred_cols:
            it = parse_iter_number(col)
            if total == 0:
                rows.append({"iter": it, "accuracy": np.nan, "matches": 0, "total": 0})
                continue
            pred_lists = df[col].apply(extract_token_ints)
            matches = 0
            for idx in canonical_sorted[mask_valid].index:
                if pred_lists[idx] == canonical_sorted[idx]:
                    matches += 1
            rows.append({"iter": it, "accuracy": matches / total, "matches": matches, "total": total})

    elif mode == "length":
        valid_indices: List[int] = []
        for idx, toks in actual_strs.items():
            if all(len(toks) >= pos for pos in positions):
                valid_indices.append(idx)

        total = len(valid_indices)
        for col in pred_cols:
            it = parse_iter_number(col)
            if total == 0:
                rows.append({"iter": it, "accuracy": np.nan, "matches": 0, "total": 0})
                continue
            matches = 0
            for idx in valid_indices:
                all_ok = True
                for pos in positions:
                    a_str = token_str_at_pos(df.at[idx, "actual"], pos)
                    p_str = token_str_at_pos(df.at[idx, col], pos)
                    if a_str is None or p_str is None or len(a_str) != len(p_str):
                        all_ok = False
                        break
                if all_ok:
                    matches += 1
            rows.append({"iter": it, "accuracy": matches / total, "matches": matches, "total": total})

    elif mode in DIGIT_MODE_MAP:
        k = DIGIT_MODE_MAP[mode]

        valid_indices: List[int] = []
        for idx, toks in actual_strs.items():
            has_any = any(len(toks) >= pos and toks[pos - 1] and len(toks[pos - 1]) >= k for pos in positions)
            if has_any:
                valid_indices.append(idx)

        total = len(valid_indices)
        for col in pred_cols:
            it = parse_iter_number(col)
            if total == 0:
                rows.append({"iter": it, "accuracy": np.nan, "matches": 0, "total": 0})
                continue
            matches = 0
            for idx in valid_indices:
                all_ok = True
                for pos in positions:
                    a_tok = token_str_at_pos(df.at[idx, "actual"], pos)
                    if a_tok is None or a_tok == "" or len(a_tok) < k:
                        continue  # ignore positions lacking kth digit in actual
                    p_tok = token_str_at_pos(df.at[idx, col], pos)
                    if p_tok is None or len(p_tok) < k or a_tok[k - 1] != p_tok[k - 1]:
                        all_ok = False
                        break
                if all_ok:
                    matches += 1
            rows.append({"iter": it, "accuracy": matches / total, "matches": matches, "total": total})

    else:
        raise ValueError("Unknown mode. Choose: strict, length, first, second, third, fourth.")

    # filter out iter < 0 (e.g., pred_iter_-1) and sort
    out = pd.DataFrame(rows)
    out = out[pd.to_numeric(out["iter"], errors="coerce").notna()]
    out["iter"] = out["iter"].astype(int)
    out = out[out["iter"] >= 0].sort_values("iter").reset_index(drop=True)
    return out


def first_iter_at_or_above(curve: pd.DataFrame, threshold: float) -> Optional[Tuple[int, float]]:
    if curve.empty:
        return None
    curve2 = curve.dropna(subset=["accuracy"]).sort_values("iter")
    for _, row in curve2.iterrows():
        if float(row["accuracy"]) >= threshold:
            return int(row["iter"]), float(row["accuracy"])
    return None


def _apply_legend_font_settings(legend_obj,
                               legend_fontsize: Optional[float],
                               legend_title_fontsize: Optional[float],
                               legend_title_bold: bool):
    if legend_obj is None:
        return
    if legend_fontsize is not None:
        for txt in legend_obj.get_texts():
            txt.set_fontsize(legend_fontsize)
    if legend_title_fontsize is not None:
        legend_obj.get_title().set_fontsize(legend_title_fontsize)
    if legend_title_bold:
        legend_obj.get_title().set_fontweight("bold")


def plot_10_90_progress(ranges_list: List[Dict],
                        pdf_outpath: str,
                        show_plot: bool = False,
                        min_iter: Optional[int] = 0,
                        max_iter: Optional[int] = None,
                        dot_threshold: int = 50,
                        legend_title: Optional[str] = None,
                        xlabel_fontsize: Optional[float] = None,
                        ylabel_fontsize: Optional[float] = None,
                        legend_fontsize: Optional[float] = None,
                        legend_title_fontsize: Optional[float] = None,
                        legend_title_bold: bool = False,
                        xtick_fontsize: Optional[float] = None):
    """
    Progress bar plot:
    - If a,b exist and (b-a) <= dot_threshold -> dot
    - Else bar from a to b
    - Else gray "none"

    Styling follows your plot_10_90_ranges function.
    """
    if not ranges_list:
        print("No ranges to plot.")
        return

    n_series = len(ranges_list)
    fig_h = max(2.0, 0.6 + n_series * 0.45)
    plt.figure(figsize=(10, fig_h))
    ax = plt.gca()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ys = np.arange(n_series)[::-1]
    ax.set_ylim(-0.5, n_series - 0.5)

    if xlabel_fontsize is not None:
        ax.set_xlabel("Training steps", fontsize=xlabel_fontsize)
    else:
        ax.set_xlabel("Training steps")

    if ylabel_fontsize is not None:
        ax.set_ylabel("Subskill learning order", fontsize=ylabel_fontsize)
    else:
        ax.set_ylabel("Subskill learning order")

    if xtick_fontsize is not None:
        ax.tick_params(axis="x", labelsize=xtick_fontsize)

    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    legend_handles: List = []
    legend_labels: List[str] = []

    for idx, item in enumerate(ranges_list):
        label_only = item.get("label", f"run_{idx}")
        a = item.get("a", None)
        b = item.get("b", None)

        color = colors[idx % len(colors)]
        y = ys[idx]

        if a is None or b is None:
            legend_handles.append(Patch(color="lightgray"))
            legend_labels.append(f"{label_only}: none")
            continue

        a = int(a)
        b = int(b)
        span = b - a

        if span <= dot_threshold:
            # draw a single dot; place at midpoint so it represents the short range
            x = int(round((a + b) / 2))
            ax.plot([x], [y], marker="o", markersize=8, color=color, label=None)

            handle = Line2D([], [], marker="o", linestyle="None",
                            markerfacecolor=color, markeredgecolor=color, markersize=8)
            legend_handles.append(handle)

            if a == b:
                legend_labels.append(f"{label_only}: {a}")
            else:
                legend_labels.append(f"{label_only}: {a}-{b}")
        else:
            ax.hlines(y=y, xmin=a, xmax=b, linewidth=8, color=color, alpha=0.95, label=None)

            handle = Line2D([], [], color=color, linestyle="-", linewidth=6)
            legend_handles.append(handle)
            legend_labels.append(f"{label_only}: {a}-{b}")

    lg = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title=legend_title if legend_title else "10% - 90% Range",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        prop={"size": legend_fontsize} if legend_fontsize is not None else None,
    )
    _apply_legend_font_settings(lg, legend_fontsize, legend_title_fontsize, legend_title_bold)

    # x-limits
    if (min_iter is not None) or (max_iter is not None):
        left = min_iter if min_iter is not None else 0
        if max_iter is not None:
            ax.set_xlim(left=left, right=max_iter)
        else:
            ax.set_xlim(left=left)
    else:
        x0, x1 = ax.get_xlim()
        pad = max(1, (x1 - x0) * 0.03)
        ax.set_xlim(left=max(0, x0 - pad), right=x1 + pad)

    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(pdf_outpath)), exist_ok=True)
        plt.savefig(pdf_outpath, bbox_inches="tight")
        print(f"Saved progress plot PDF to {pdf_outpath}")
    except Exception as e:
        print(f"Error saving progress plot PDF to {pdf_outpath}: {e}")

    if show_plot:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Report [a,b] where joint accuracy crosses 10% then 90% (per CSV), and plot ranges.")
    ap.add_argument("--csv", "-c", required=True, nargs="+", help="CSV paths (one or more).")
    ap.add_argument(
        "--mode",
        "-m",
        nargs="+",
        choices=["strict", "length", "first", "second", "third", "fourth"],
        default=["strict"],
        help="Provide one per CSV, or a single mode to apply to all CSVs.",
    )
    ap.add_argument(
        "--positions",
        "-p",
        required=True,
        nargs="+",
        help="Positions to evaluate (joint). Example: --positions 1,2,3,4 OR --positions 1 2 3 4",
    )

    ap.add_argument("--min-iter", type=int, default=0, help="Minimum iteration to consider (inclusive). Default: 0.")
    ap.add_argument("--max-iter", type=int, default=None, help="Maximum iteration to consider (inclusive).")
    ap.add_argument("--lower", type=float, default=0.10, help="Lower threshold (default 0.10).")
    ap.add_argument("--upper", type=float, default=0.90, help="Upper threshold (default 0.90).")

    ap.add_argument("--show", action="store_true", help="Show the plot interactively (optional).")

    args = ap.parse_args()

    csv_paths: List[str] = args.csv
    modes: List[str] = args.mode
    if len(modes) == 1 and len(csv_paths) > 1:
        modes = modes * len(csv_paths)
    if len(modes) != len(csv_paths):
        raise SystemExit("Number of --mode values must be 1 or equal to number of --csv files provided.")

    positions = parse_positions_arg(args.positions)
    if not positions:
        raise SystemExit("No valid positions provided.")

    # Build plot ranges list
    ranges_list: List[Dict] = []

    # If labels collide (same mode repeated), disambiguate
    label_counts: Dict[str, int] = {}

    for csv_path, mode in zip(csv_paths, modes):
        if not os.path.exists(csv_path):
            print(f"{csv_path} [{mode}] -> ERROR: file not found")
            # still include in plot as "none"
            label = mode
            label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts[label] > 1:
                label = f"{label}#{label_counts[label]}"
            ranges_list.append({"label": label, "a": None, "b": None})
            continue

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
        curve = compute_joint_accuracy_curve(df, positions, mode=mode)

        # apply min/max bounds (min defaults to 0)
        if args.min_iter is not None:
            curve = curve[curve["iter"] >= args.min_iter]
        if args.max_iter is not None:
            curve = curve[curve["iter"] <= args.max_iter]
        curve = curve.sort_values("iter").reset_index(drop=True)

        a_hit = first_iter_at_or_above(curve, args.lower)
        b_hit = first_iter_at_or_above(curve, args.upper)

        a_iter = a_hit[0] if a_hit else None
        b_iter = b_hit[0] if b_hit else None

        # choose label (default: mode)
        label = mode
        label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts[label] > 1:
            label = f"{label}#{label_counts[label]}"

        ranges_list.append({"label": label, "a": a_iter, "b": b_iter})

        a_str = "NA" if a_iter is None else str(a_iter)
        b_str = "NA" if b_iter is None else str(b_iter)

        print(f"\nCSV:  {csv_path}")
        print(f"mode: {mode}   positions: {positions}")
        print(f"range [{args.lower*100:.0f}%, {args.upper*100:.0f}%] -> [{a_str}, {b_str}]")

        if a_hit:
            print(f"  a: iter={a_hit[0]}  acc={a_hit[1]*100:.3f}%")
        else:
            print(f"  a: not reached (acc never >= {args.lower*100:.0f}%)")

        if b_hit:
            print(f"  b: iter={b_hit[0]}  acc={b_hit[1]*100:.3f}%")
        else:
            print(f"  b: not reached (acc never >= {args.upper*100:.0f}%)")

    # Save plot PDF to first CSV directory
    first_csv_dir = os.path.dirname(os.path.abspath(csv_paths[0])) if csv_paths else os.getcwd()
    pdf_outpath = os.path.join(first_csv_dir, "accuracy_10_90_ranges.pdf")

    plot_10_90_progress(
        ranges_list=ranges_list,
        pdf_outpath=pdf_outpath,
        show_plot=args.show,
        min_iter=args.min_iter,
        max_iter=args.max_iter,
        dot_threshold=50,  # per your requirement
        legend_title="10% - 90% Range",
    )


if __name__ == "__main__":
    main()
