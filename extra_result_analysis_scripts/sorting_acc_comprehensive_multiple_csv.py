#!/usr/bin/env python3
"""
sorting_acc_comprehensive.py (fixed)

Produces TWO separate PDF files:
 - <args.out with .pdf extension> : original accuracy vs training steps curves (saved at args.out base)
   AND a copy of that PDF is saved in the directory of the first CSV.
 - accuracy_10_90_ranges.pdf in the directory of the first CSV : horizontal bands (no faint curves)
"""
from __future__ import annotations
import re
import argparse
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# regexes
INT_TOKEN_RE = re.compile(r'-?\d+')
DIGIT_TOKEN_RE = re.compile(r'\d+')

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
    found = DIGIT_TOKEN_RE.findall(s)
    return found

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
    if 'actual' not in df.columns:
        raise ValueError("CSV must contain an 'actual' column with the true tokens.")

    pred_cols = find_pred_columns(df)
    if not pred_cols:
        raise ValueError("No columns named pred_iter_* found in CSV.")

    actual_ints_series = df['actual'].apply(extract_token_ints)
    actual_strs_series = df['actual'].apply(extract_token_strs)
    canonical_sorted_series = actual_ints_series.apply(lambda toks: sorted(toks) if toks else [])

    results_by_pos: Dict[int, pd.DataFrame] = {}
    digit_mode_map = {"first": 1, "second": 2, "third": 3, "fourth": 4}

    for pos in positions:
        if mode == "strict":
            actual_at_pos = actual_ints_series.apply(lambda toks: toks[pos - 1] if len(toks) >= pos else None)
        elif mode == "length":
            actual_at_pos = actual_strs_series.apply(lambda toks: toks[pos - 1] if len(toks) >= pos else None)
        elif mode in digit_mode_map:
            k = digit_mode_map[mode]
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
                        if a_str is None or p_str is None:
                            return False
                        return len(a_str) == len(p_str)
                    comp = [match_len(a, p) for a, p in zip(actual_at_pos[mask_valid_actual], pred_str_at_pos[mask_valid_actual])]
                    matches = int(sum(comp))
                    accuracy = matches / total

            else:  # digit modes
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

    # Joint metric (pos 0)
    joint_results = []
    for col in find_pred_columns(df):
        iter_num = parse_iter_number(col)
        if mode == "strict":
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

        if mode in digit_mode_map:
            k = digit_mode_map[mode]
            valid_indices = []
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

            matches = 0
            for idx in valid_indices:
                all_ok = True
                for pos in positions:
                    a_str = token_str_at_pos(df.at[idx, 'actual'], pos)
                    if a_str is None or a_str == "" or len(a_str) < k:
                        continue
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

def _find_longest_10_90_run(df_plot: pd.DataFrame) -> Optional[tuple]:
    if df_plot.empty:
        return None
    mask = df_plot['accuracy'].notna() & (df_plot['accuracy'] >= 0.10) & (df_plot['accuracy'] <= 0.90)
    if not mask.any():
        return None
    iters = df_plot['iter'].values
    mask_arr = mask.values.astype(bool)

    best_span = -1
    best_pair = None
    start_idx = None
    for i, val in enumerate(mask_arr):
        if val:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                s_iter = int(iters[start_idx])
                e_iter = int(iters[i - 1])
                span = e_iter - s_iter
                if span > best_span:
                    best_span = span
                    best_pair = (s_iter, e_iter)
                start_idx = None
    if start_idx is not None:
        s_iter = int(iters[start_idx])
        e_iter = int(iters[len(iters) - 1])
        span = e_iter - s_iter
        if span > best_span:
            best_pair = (s_iter, e_iter)
    return best_pair

def _apply_legend_font_settings(legend_obj, legend_fontsize: Optional[float], legend_title_fontsize: Optional[float], legend_title_bold: bool):
    """Utility to apply font size and bolding to a matplotlib Legend object."""
    if legend_obj is None:
        return
    if legend_fontsize is not None:
        # legend texts
        for txt in legend_obj.get_texts():
            txt.set_fontsize(legend_fontsize)
    if legend_title_fontsize is not None:
        legend_obj.get_title().set_fontsize(legend_title_fontsize)
    if legend_title_bold:
        legend_obj.get_title().set_fontweight('bold')


def plot_accuracy_curves(joint_series_list: List[Dict], pdf_outpath: str,
                         show_plot: bool = False, max_iter: Optional[int] = None,
                         min_iter: Optional[int] = None, positions: List[int] = None,
                         plot_title: Optional[str] = None, legend_title: Optional[str] = None,
                         save_png_path: Optional[str] = None,
                         xlabel_fontsize: Optional[float] = None, ylabel_fontsize: Optional[float] = None,
                         legend_fontsize: Optional[float] = None, legend_title_fontsize: Optional[float] = None,
                         legend_title_bold: bool = False, xtick_fontsize: Optional[float] = None,
                         legend_labels_override: Optional[List[str]] = None,
                         ytick_fontsize: Optional[float] = None,
                         force_line_thickness: Optional[float] = None):
    """
    Robust plotting of accuracy-vs-iteration curves.

    Default font sizes (used when corresponding argument is None) are set below.
    To nudge the legend, edit LEGEND_BBOX (axes fraction coordinates) or LEGEND_LOC.
    """
    # ---------- Small internal overrides you can edit ----------
    FORCE_MIN_ITER = 50             # set to None if you don't want to force min_iter
    LEGEND_LABELS_OVERRIDE = ["Training distribution", "Fourth digit", "Distinct lengths"]
    # Legend placement: (x, y) in axes fraction coordinates (0..1).
    # Increase x to move legend to the right; decrease to move left.
    LEGEND_LOC = "upper left"               # anchor location relative to bbox
    LEGEND_BBOX = (0.07, 0.75)              # (x, y) in axes fraction; tweak x to move right

    # Line thickness control: set to None to use the default thickness (3).
    # You can set this internal override, or pass force_line_thickness to the function.
    FORCE_LINE_THICKNESS = 4
    # ----------------------------------------------------------

    # Prefer function-argument override for legend labels
    if legend_labels_override is not None:
        LEGEND_LABELS_OVERRIDE = legend_labels_override

    # Prefer FORCE_MIN_ITER if set
    if FORCE_MIN_ITER is not None:
        min_iter = FORCE_MIN_ITER

    # Prefer function argument for line thickness, then internal override
    if force_line_thickness is not None:
        FORCE_LINE_THICKNESS = force_line_thickness

    # Defensive: coerce min/max to int if they are strings
    try:
        if min_iter is not None:
            min_iter = int(min_iter)
    except Exception:
        print(f"Warning: couldn't convert min_iter={min_iter!r} to int; ignoring min_iter.")
        min_iter = None
    try:
        if max_iter is not None:
            max_iter = int(max_iter)
    except Exception:
        print(f"Warning: couldn't convert max_iter={max_iter!r} to int; ignoring max_iter.")
        max_iter = None

    # Default font sizes (used only when caller passed None)
    if xlabel_fontsize is None:
        xlabel_fontsize = 20
    if ylabel_fontsize is None:
        ylabel_fontsize = 20
    if legend_fontsize is None:
        legend_fontsize = 18
    if legend_title_fontsize is None:
        legend_title_fontsize = 18
    # Respect caller value for legend_title_bold; do not override unless needed
    if xtick_fontsize is None:
        xtick_fontsize = 18
    if ytick_fontsize is None:
        ytick_fontsize = 18

    # resolved_linewidth will be used for plotting
    resolved_linewidth = FORCE_LINE_THICKNESS if FORCE_LINE_THICKNESS is not None else 3

    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    linestyles = ['-']  # force solid lines

    plotted_lines = []
    any_plotted = False
    for idx, item in enumerate(joint_series_list):
        # determine label (allow override from internal list)
        if LEGEND_LABELS_OVERRIDE and idx < len(LEGEND_LABELS_OVERRIDE):
            label = LEGEND_LABELS_OVERRIDE[idx]
        else:
            label = item.get("label") or f"run_{idx}"

        # make a safe copy of DataFrame and coerce iter column to numeric
        dfp: pd.DataFrame = item["df"].copy()
        if 'iter' not in dfp.columns:
            print(f"Warning: 'iter' column not found in item {idx}; skipping.")
            continue
        dfp['iter'] = pd.to_numeric(dfp['iter'], errors='coerce')  # invalid -> NaN
        dfp = dfp.dropna(subset=['iter'])

        df_plot = dfp
        if min_iter is not None:
            df_plot = df_plot[df_plot['iter'] >= min_iter]
        if max_iter is not None:
            df_plot = df_plot[df_plot['iter'] <= max_iter]
        if df_plot.empty:
            print(f"Info: after applying iter bounds, nothing to plot for label '{label}'.")
            continue

        iters = df_plot['iter'].values
        accuracies = (df_plot['accuracy'].values * 100)

        ls = linestyles[0]
        any_plotted = True
        line_objs = ax.plot(iters, accuracies, linestyle=ls, marker=None,
                            linewidth=resolved_linewidth, label=label)
        if line_objs:
            plotted_lines.append((line_objs[0], label))

    if not any_plotted:
        print("Warning: no iterations to plot within the requested min/max iteration range.")
    ax.set_ylim(-2, 102)

    # axis labels with the resolved font sizes
    ax.set_xlabel("Training steps", fontsize=xlabel_fontsize)
    ax.set_ylabel("Accuracy (%)", fontsize=ylabel_fontsize)
    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    ax.tick_params(axis='y', labelsize=ytick_fontsize)

    # no title by design
    ax.grid(True, linestyle='--', alpha=0.5)

    # Build legend from plotted lines
    legend_handles = [line for line, _ in plotted_lines]
    legend_labels = [lbl for _, lbl in plotted_lines]
    if legend_handles:
        # use bbox_to_anchor and axes transform so coordinates are in axes fraction
        lg = ax.legend(handles=legend_handles, labels=legend_labels,
                       title=legend_title if legend_title else "Tests",
                       loc=LEGEND_LOC,
                       bbox_to_anchor=LEGEND_BBOX,
                       bbox_transform=ax.transAxes,
                       prop={'size': legend_fontsize})
        # apply legend title fontsize and bold
        try:
            title_text = lg.get_title()
            title_text.set_fontsize(legend_title_fontsize)
            title_text.set_fontweight('bold' if legend_title_bold else 'normal')
        except Exception:
            pass
        _apply_legend_font_settings(lg, legend_fontsize, legend_title_fontsize, legend_title_bold)

    if (min_iter is not None) or (max_iter is not None):
        left = min_iter if min_iter is not None else 0
        if max_iter is not None:
            ax.set_xlim(left=left, right=max_iter)
        else:
            ax.set_xlim(left=left)

    plt.tight_layout()
    try:
        plt.savefig(pdf_outpath)
        print(f"Saved accuracy curves PDF to {pdf_outpath}")
    except Exception as e:
        print(f"Error saving accuracy curves PDF to {pdf_outpath}: {e}")
    if save_png_path is not None:
        try:
            plt.savefig(save_png_path, dpi=150)
            print(f"Saved accuracy curves image to {save_png_path}")
        except Exception as e:
            print(f"Warning: unable to save PNG to {save_png_path}: {e}")
    if show_plot:
        plt.show()
    plt.close()




def plot_10_90_ranges(joint_series_list: List[Dict], pdf_outpath: str,
                      show_plot: bool = False, max_iter: Optional[int] = None,
                      min_iter: Optional[int] = None, positions: List[int] = None,
                      plot_title: Optional[str] = None, legend_title: Optional[str] = None,
                      xlabel_fontsize: Optional[float] = None, ylabel_fontsize: Optional[float] = None,
                      legend_fontsize: Optional[float] = None, legend_title_fontsize: Optional[float] = None,
                      legend_title_bold: bool = False, xtick_fontsize: Optional[float] = None):
    """
    Plot the 10%-90% ranges as horizontal bars (non-overlapping bands).
    - No accuracy y-axis values; the y label describes experiment runs.
    - Legend entries match the plotted element: dot -> dot, bar -> thick line, none -> gray patch.
    """
    if not joint_series_list:
        print("No series to plot in 10%-90% ranges.")
        return

    n_series = len(joint_series_list)
    fig_h = max(2.0, 0.6 + n_series * 0.45)
    plt.figure(figsize=(10, fig_h))
    ax = plt.gca()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ys = np.arange(n_series)[::-1]  # top-to-bottom ordering
    ax.set_ylim(-0.5, n_series - 0.5)

    # axis labels with optional fontsize
    if xlabel_fontsize is not None:
        ax.set_xlabel("Training steps", fontsize=xlabel_fontsize)
    else:
        ax.set_xlabel("Training steps")
    if ylabel_fontsize is not None:
        ax.set_ylabel("Subskill learning order", fontsize=ylabel_fontsize)
    else:
        ax.set_ylabel("Subskill learning order")

    # x-axis tick label size
    if xtick_fontsize is not None:
        ax.tick_params(axis='x', labelsize=xtick_fontsize)

    ax.set_yticks([])  # hide numeric ticks
    ax.grid(axis='x', linestyle='--', alpha=0.35)

    # Build legend handles and labels that correspond to what is drawn.
    legend_handles: List = []
    legend_labels: List[str] = []

    for idx, item in enumerate(joint_series_list):
        label_only = item.get("label", "")
        dfp: pd.DataFrame = item["df"]
        df_plot = dfp
        if min_iter is not None:
            df_plot = df_plot[df_plot['iter'] >= min_iter]
        if max_iter is not None:
            df_plot = df_plot[df_plot['iter'] <= max_iter]

        color = colors[idx % len(colors)]
        y = ys[idx]

        if df_plot.empty:
            # no data -> gray patch
            legend_handles.append(Patch(color='lightgray'))
            legend_labels.append(f"{label_only}: none")
            continue

        run = _find_longest_10_90_run(df_plot)
        if run is None:
            legend_handles.append(Patch(color='lightgray'))
            legend_labels.append(f"{label_only}: none")
            continue

        s_iter, e_iter = run
        if s_iter == e_iter:
            # draw a single dot in the plot
            s_iter = s_iter - 10
            ax.plot([s_iter], [y], marker='o', markersize=8, color=color, label=None)
            # legend handle: marker-only Line2D
            handle = Line2D([], [], marker='o', linestyle='None',
                            markerfacecolor=color, markeredgecolor=color, markersize=8)
            legend_handles.append(handle)
            legend_labels.append(f"{label_only}: {s_iter + 10}")  # show original value in label
        else:
            # draw horizontal thick line in the plot
            ax.hlines(y=y, xmin=s_iter, xmax=e_iter, linewidth=8, color=color, alpha=0.95, label=None)
            # legend handle: short thick line (no marker)
            handle = Line2D([], [], color=color, linestyle='-', linewidth=6)
            legend_handles.append(handle)
            legend_labels.append(f"{label_only}: {s_iter}-{e_iter}")

    # place legend using the handles that match plotted elements
    if legend_fontsize is not None:
        lg = ax.legend(handles=legend_handles, labels=legend_labels, title=legend_title if legend_title else "10% - 90% Range",
                       loc="upper right", bbox_to_anchor=(0.98, 0.98), prop={'size': legend_fontsize},
                       frameon=True, fancybox=True, framealpha=0.9)
    else:
        lg = ax.legend(handles=legend_handles, labels=legend_labels, title=legend_title if legend_title else "10% - 90% Range",
                       loc="upper right", bbox_to_anchor=(0.98, 0.98),
                       frameon=True, fancybox=True, framealpha=0.9)
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
        plt.savefig(pdf_outpath, bbox_inches='tight')
        print(f"Saved 10%-90% ranges PDF to {pdf_outpath}")
    except Exception as e:
        print(f"Error saving 10%-90% ranges PDF to {pdf_outpath}: {e}")
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
    parser = argparse.ArgumentParser(description="Plot joint accuracy vs iteration from one or more CSVs")
    parser.add_argument("--csv", "-c", required=True, nargs='+', help="Paths to input CSV files (one or more)")
    parser.add_argument("--positions", "-p", required=True, nargs='+',
                        help="Positions to evaluate. Provide as space-separated list or comma-separated string, e.g. 1 2 3 or 1,2,3")
    parser.add_argument("--mode", "-m", nargs='+', choices=["strict", "length", "first", "second", "third", "fourth"], default=["strict"],
                        help="Accuracy mode(s): strict, length, or first/second/third/fourth. Provide one per CSV, or a single mode to apply to all CSVs.")
    parser.add_argument("--out", "-o", default="joint_accuracy.png", help="Output image path (PNG). A PDF with the same base name will also be written.")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--min-iter", type=int, default=None,
                        help="Minimum iteration value to draw on the x-axis (inclusive).")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Maximum iteration value to draw on the x-axis (inclusive).")
    parser.add_argument("--title", "-t", type=str, default=None,
                        help="Optional plot title (if omitted a default title including positions will be used).")
    parser.add_argument("--labels", nargs='+', default=None,
                        help="Optional legend labels — either a single label (applied to all CSVs) or one label per CSV in order.")

    # NEW: font-size controls
    parser.add_argument("--xlabel-fontsize", type=float, default=None, help="Font size for x-axis label (points).")
    parser.add_argument("--ylabel-fontsize", type=float, default=None, help="Font size for y-axis label (points).")
    parser.add_argument("--legend-fontsize", type=float, default=None, help="Font size for legend text (points).")
    parser.add_argument("--legend-title-fontsize", type=float, default=None, help="Font size for legend title (points).")
    parser.add_argument("--legend-title-bold", action="store_true", help="Make the legend title bold if provided.")

    # NEW: x-axis tick label font size
    parser.add_argument("--xtick-fontsize", type=float, default=None, help="Font size for x-axis tick labels (points).")

    args = parser.parse_args()

    csv_paths: List[str] = args.csv
    modes: List[str] = args.mode

    if len(modes) == 1 and len(csv_paths) > 1:
        modes = modes * len(csv_paths)
    if len(modes) != len(csv_paths):
        raise SystemExit("Number of --mode values must be 1 or equal to number of --csv files provided.")

    provided_labels = None
    if args.labels is not None:
        provided_labels = args.labels
        if len(provided_labels) == 1 and len(csv_paths) > 1:
            provided_labels = provided_labels * len(csv_paths)
        if len(provided_labels) != len(csv_paths):
            raise SystemExit("Number of --labels values must be 1 or equal to number of --csv files provided.")

    positions = parse_positions_arg(args.positions)
    if not positions:
        raise SystemExit("No valid positions provided. Example: --positions 1 2 3 4 or --positions 1,2,3,4")

    joint_series_list = []
    if provided_labels is not None:
        iter_triplet = zip(csv_paths, modes, provided_labels)
    else:
        iter_triplet = zip(csv_paths, modes, [None] * len(csv_paths))

    for csv_path, mode, user_label in iter_triplet:
        if not os.path.exists(csv_path):
            print(f"Warning: CSV path not found: {csv_path} -- skipping.")
            continue
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[''])
        accs_by_pos = compute_accuracies_for_positions(df, positions, mode=mode)
        joint_df = accs_by_pos[0]

        # label used in legends: user-supplied label if given, otherwise just the mode (no filename)
        if user_label:
            label = user_label
        else:
            label = mode

        joint_series_list.append({"label": label, "df": joint_df})

        if mode == "strict":
            label_name = "Exact whole-result match (canonical sorted target)"
        elif mode == "length":
            label_name = f"Joint-length accuracy (all positions {positions})"
        else:
            label_name = f"Joint-{mode}-digit accuracy (all positions {positions})"

        print(f"\n{label} - {label_name} results:")
        print(" Iteration | matches/total | accuracy(%)")
        for _, row in joint_df.iterrows():
            acc_pct = (row['accuracy'] * 100) if pd.notna(row['accuracy']) else float('nan')
            print(f" {int(row['iter']):8d} | {int(row['matches']):7d}/{int(row['total']):6d} | {acc_pct:8.3f}")

    if not joint_series_list:
        raise SystemExit("No valid CSV inputs were processed. Exiting.")

    legend_title = "Legend" if provided_labels is not None else None

    # prepare output PDF paths
    out_arg = args.out
    base_noext = os.path.splitext(out_arg)[0]
    pdf_for_curves = base_noext + ".pdf"

    first_csv_dir = os.path.dirname(os.path.abspath(csv_paths[0])) if csv_paths else os.getcwd()
    pdf_for_curves_in_first_dir = os.path.join(first_csv_dir, os.path.basename(base_noext) + ".pdf")
    pdf_for_ranges = os.path.join(first_csv_dir, "accuracy_10_90_ranges.pdf")

    # save the accuracy curves PDF (+ png)
    plot_accuracy_curves(
        joint_series_list, pdf_for_curves, show_plot=args.show,
        max_iter=args.max_iter, min_iter=args.min_iter,
        positions=positions, plot_title=args.title, legend_title=legend_title,
        save_png_path=args.out,
        xlabel_fontsize=args.xlabel_fontsize, ylabel_fontsize=args.ylabel_fontsize,
        legend_fontsize=args.legend_fontsize, legend_title_fontsize=args.legend_title_fontsize,
        legend_title_bold=args.legend_title_bold,
        xtick_fontsize=args.xtick_fontsize
    )

    # copy the curves PDF into the first CSV's directory unless it's the same path
    try:
        abs_src = os.path.abspath(pdf_for_curves)
        abs_dst = os.path.abspath(pdf_for_curves_in_first_dir)
        if abs_src != abs_dst:
            shutil.copyfile(abs_src, abs_dst)
            print(f"Copied curves PDF to {abs_dst}")
        else:
            print(f"Curves PDF already at {abs_dst}")
    except Exception as e:
        print(f"Warning: unable to copy curves PDF to first CSV directory: {e}")

    # save the 10%-90% ranges PDF (no faint curves)
    plot_10_90_ranges(
        joint_series_list, pdf_for_ranges, show_plot=args.show,
        max_iter=args.max_iter, min_iter=args.min_iter,
        positions=positions, plot_title=args.title, legend_title=legend_title,
        xlabel_fontsize=args.xlabel_fontsize, ylabel_fontsize=args.ylabel_fontsize,
        legend_fontsize=args.legend_fontsize, legend_title_fontsize=args.legend_title_fontsize,
        legend_title_bold=args.legend_title_bold,
        xtick_fontsize=args.xtick_fontsize
    )

if __name__ == "__main__":
    main()
