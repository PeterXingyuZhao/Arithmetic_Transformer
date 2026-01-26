#!/usr/bin/env python3
"""
fit_normal.py (combined)

1) Reads an input CSV with an 'actual' column and prediction columns like 'pred_iter_8000', ...
2) For iterations iter_start..iter_end (step iter_step), computes diff = actual - prediction,
   and builds a histogram of integer diffs in [diff_min, diff_max].
3) Saves that histogram CSV to:
      OUTDIR/{iter_start}_to_{iter_end}_diff_histogram.csv
   where OUTDIR is created under the input CSV's directory:
      <input_csv_dir>/{iter_start}_to_{iter_end}_fitted_normal/
4) Fits a weighted Normal distribution to the histogram counts and saves:
      - OUTDIR/{hist_stem}_fit_plot.png
      - OUTDIR/{hist_stem}_fit_plot.pdf
      - OUTDIR/{hist_stem}_fitted_params.txt
"""

import argparse
import ast
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt


# ============================================================
# Safe evaluator for simple arithmetic expressions (only + and -)
# ============================================================
ALLOWED_BINOPS = (ast.Add, ast.Sub)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)
ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    getattr(ast, "Num", ()),  # ast.Num for py<3.8
)


def safe_eval_simple_expr(expr: str) -> Union[float, None]:
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
        return val if math.isfinite(val) else None

    s = str(expr).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    try:
        tree = ast.parse(s, mode="eval")
    except Exception:
        return None

    # Validate nodes (walk the tree)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES) and not isinstance(node, (ast.BinOp, ast.UnaryOp)):
            return None
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ALLOWED_BINOPS):
            return None
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ALLOWED_UNARYOPS):
            return None

    def _eval(node):
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
            if isinstance(node.op, ast.Sub):
                return left - right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary op")
        raise ValueError("Unsupported AST node")

    try:
        val = float(_eval(tree))
        return val if math.isfinite(val) else None
    except Exception:
        return None


# ============================================================
# Part 1: Build diff histogram
# ============================================================
def build_histograms(
    infile: str,
    out_csv: str,
    iter_start: int,
    iter_end: int,
    iter_step: int,
    diff_min: int,
    diff_max: int,
    actual_col_name: str = "actual",
    pred_col_template: str = "pred_iter_{}",
    avg_round: Union[int, None] = 2,
) -> pd.DataFrame:
    # read as str to let safe parser handle messy cells
    df = pd.read_csv(infile, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if actual_col_name not in df.columns:
        raise ValueError(
            f"Actual column '{actual_col_name}' not found in CSV. Columns: {df.columns.tolist()}"
        )

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
    counts = pd.DataFrame(
        0,
        index=diffs,
        columns=[f"iter_{it}" for (it, _) in present_iters],
        dtype=int,
    )

    invalid_counts = {f"iter_{it}": 0 for (it, _) in present_iters}
    missing_actual = 0
    total_rows = len(df)

    for _, row in df.iterrows():
        actual_val = safe_eval_simple_expr(row[actual_col_name])
        if actual_val is None:
            missing_actual += 1
            continue

        for it, cname in present_iters:
            pred_val = safe_eval_simple_expr(row[cname])
            if pred_val is None:
                invalid_counts[f"iter_{it}"] += 1
                continue

            diff = actual_val - pred_val
            if not math.isfinite(diff):
                invalid_counts[f"iter_{it}"] += 1
                continue

            diff_int = int(round(diff))
            if diff_min <= diff_int <= diff_max:
                counts.at[diff_int, f"iter_{it}"] += 1

    # average across iterations for each diff
    mean_series = counts.mean(axis=1)
    if avg_round is not None:
        mean_series = mean_series.round(avg_round)

    out_df = mean_series.reset_index()
    out_df.columns = ["difference", "avg_counts"]
    out_df.to_csv(out_csv, index=False)

    print(f"\nSaved averaged histogram to: {out_csv}")
    print(f"Processed {total_rows} rows. Missing actuals: {missing_actual}.")
    print("Invalid/missing predictions per iteration (counts skipped):")
    for k, v in invalid_counts.items():
        print(f"  {k}: {v}")

    return out_df


# ============================================================
# Part 2: Fit normal to histogram counts + plot
# ============================================================
# --- plotting style config (tweak these) ---
FIGSIZE = (10, 6)
DPI = 200
AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 24
MARKER_SIZE = 6
MARKER_EDGEWIDTH = 1.0
POINT_ALPHA = 0.95
FIT_LINEWIDTH = 6
MU_LINEWIDTH = 4
GRID_LINEWIDTH = 0.9
LEGEND_FONT_SIZE = 26


def weighted_normal_fit(xs: np.ndarray, ws: np.ndarray) -> Tuple[float, float]:
    if len(xs) == 0:
        raise ValueError("Empty data")
    wsum = np.sum(ws)
    if wsum <= 0:
        raise ValueError("Sum of weights must be positive")
    mu = float(np.sum(ws * xs) / wsum)
    var = float(np.sum(ws * (xs - mu) ** 2) / wsum)
    sigma = math.sqrt(max(var, 0.0))
    return mu, sigma


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x, dtype=float)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    z = (x - mu) / sigma
    return coef * np.exp(-0.5 * z * z)


def nice_tick_step(span: float, target_intervals: int = 8) -> int:
    if span <= 0:
        return 1
    raw = float(span) / float(target_intervals)
    exp = math.floor(math.log10(raw))
    base = 10 ** exp
    for mult in (1, 2, 5, 10):
        step = base * mult
        if step >= raw - 1e-12:
            return int(max(1, round(step)))
    return int(base * 10)


def compute_goodness_of_fit(
    diffs: np.ndarray, counts: np.ndarray, mu: float, sigma: float
) -> Tuple[Dict[str, Any], np.ndarray]:
    def normal_cdf(x, m, s):
        return 0.5 * (1 + erf((x - m) / (s * sqrt(2))))

    total = counts.sum()
    expected = []
    for x in diffs:
        p = normal_cdf(x + 0.5, mu, sigma) - normal_cdf(x - 0.5, mu, sigma)
        expected.append(total * max(p, 1e-15))
    expected = np.array(expected, dtype=float)

    obs = counts.astype(float)
    mask = expected > 0
    chi2_stat = float(np.sum(((obs[mask] - expected[mask]) ** 2) / expected[mask])) if mask.sum() else float("nan")
    valid_bins = int(mask.sum())
    num_params = 2
    dfree = max(0, valid_bins - num_params)

    p_value = None
    try:
        from scipy.stats import chi2 as scipy_chi2
        if not math.isnan(chi2_stat) and dfree > 0:
            p_value = 1.0 - scipy_chi2.cdf(chi2_stat, dfree)
    except Exception:
        p_value = None

    rmse = float(np.sqrt(np.mean((obs - expected) ** 2)))
    mae = float(np.mean(np.abs(obs - expected)))
    try:
        r = float(np.corrcoef(obs, expected)[0, 1]) if np.std(obs) > 0 and np.std(expected) > 0 else float("nan")
    except Exception:
        r = float("nan")

    ss_res = float(np.sum((obs - expected) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        ll = float(np.sum(obs * np.log(expected) - expected))

    k = num_params
    try:
        aic = float(2 * k - 2 * ll)
        bic = float(k * math.log(max(1, len(obs))) - 2 * ll)
    except Exception:
        aic = float("nan")
        bic = float("nan")

    metrics = {
        "chi2": chi2_stat,
        "chi2_df": dfree,
        "chi2_pvalue": p_value,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": r,
        "r_squared": r_squared,
        "poisson_ll_omit_const": ll,
        "AIC_omit_const": aic,
        "BIC_omit_const": bic,
        "valid_bins": valid_bins,
        "total_count": int(total),
    }
    return metrics, expected


def _fmt_val(v):
    if v is None:
        return "None"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.6g}"
    return str(v)


def plot_and_save(
    diffs: np.ndarray,
    counts: np.ndarray,
    mu: float,
    sigma: float,
    outpng: Optional[str],
    outpdf: Optional[str],
    show: bool,
    r_squared: Optional[float],
) -> Dict[str, Optional[str]]:
    total = counts.sum()
    x_min, x_max = diffs.min(), diffs.max()
    xs_smooth = np.linspace(x_min - 1, x_max + 1, 800)
    pdf_vals = normal_pdf(xs_smooth, mu, sigma)
    scaled_pdf = pdf_vals * total

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        diffs,
        counts,
        marker="o",
        linestyle="None",
        label="Empirical counts",
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGEWIDTH,
        alpha=POINT_ALPHA,
    )
    ax.plot(xs_smooth, scaled_pdf, linestyle="-", linewidth=FIT_LINEWIDTH, label="Normal fit")
    ax.axvline(mu, color="gray", linestyle="--", linewidth=MU_LINEWIDTH)

    if r_squared is None or (isinstance(r_squared, float) and not np.isfinite(r_squared)):
        rstr = "nan"
    else:
        rstr = f"{r_squared:.3f}"

    ax.set_xlabel(
        f"difference (actual - predicted)\nμ={mu:.3f}   σ={sigma:.3f}   R²={rstr}",
        fontsize=AXIS_LABEL_SIZE,
    )
    # ax.set_ylabel("Empirical error counts", fontsize=AXIS_LABEL_SIZE)

    ax.grid(axis="y", linestyle=":", alpha=0.7, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)

    # compact legend — should now be narrow
    # ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True,
    #           handlelength=1.2, handletextpad=0.6, borderaxespad=0.5,
    #           loc='upper right', bbox_to_anchor=(1.0, 1.0))
    # ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True,
    #       handlelength=1.2, handletextpad=0.6, borderaxespad=0.5,
    #       loc='center', bbox_to_anchor=(0.5, 0.22))

    max_abs = max(abs(x_min), abs(x_max))
    if max_abs == 0:
        x_left, x_right = -1, 1
        ticks = [0]
    else:
        span = 2 * max_abs
        step = nice_tick_step(span, target_intervals=8)
        max_tick = int(math.ceil(max_abs / step) * step)
        x_left, x_right = -max_tick, max_tick
        ticks = list(range(x_left, x_right + 1, step))
        if 0 not in ticks:
            ticks.append(0)
            ticks = sorted(ticks)

    ax.set_xlim(x_left - 0.1, x_right + 0.1)
    ax.set_xticks(ticks)
    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()

    saved = {"png": None, "pdf": None}
    if outpng:
        try:
            fig.savefig(outpng, dpi=DPI)
            saved["png"] = outpng
        except Exception as e:
            print(f"Warning: failed to save PNG: {e}")
    if outpdf:
        try:
            fig.savefig(outpdf, format="pdf", bbox_inches="tight")
            saved["pdf"] = outpdf
        except Exception as e:
            print(f"Warning: failed to save PDF: {e}")

    if show:
        plt.show()
    plt.close(fig)
    return saved


def fit_normal_counts(
    infile: str,
    outdir: str,
    diffs_col: str = "difference",
    counts_col: str = "avg_counts",
    save_png: bool = True,
    save_pdf: bool = True,
    show_plot: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(infile).copy()
    df.columns = [c.strip() for c in df.columns]

    if diffs_col not in df.columns:
        raise ValueError(f"'{diffs_col}' column not found in {infile}")
    if counts_col not in df.columns:
        raise ValueError(f"'{counts_col}' column not found in {infile}")

    df[diffs_col] = pd.to_numeric(df[diffs_col], errors="coerce")
    df[counts_col] = pd.to_numeric(df[counts_col], errors="coerce").fillna(0).astype(float)
    df = df.dropna(subset=[diffs_col]).sort_values(diffs_col).reset_index(drop=True)

    diffs = df[diffs_col].to_numpy(dtype=float)
    counts = df[counts_col].to_numpy(dtype=float)

    mu, sigma = weighted_normal_fit(diffs, counts)
    metrics, _expected = compute_goodness_of_fit(diffs, counts, mu, sigma)

    os.makedirs(outdir, exist_ok=True)

    base = Path(infile).stem
    outpng = str(Path(outdir) / f"{base}_fit_plot.png") if save_png else None
    outpdf = str(Path(outdir) / f"{base}_fit_plot.pdf") if save_pdf else None

    saved = plot_and_save(
        diffs=diffs,
        counts=counts,
        mu=mu,
        sigma=sigma,
        outpng=outpng,
        outpdf=outpdf,
        show=show_plot,
        r_squared=metrics.get("r_squared"),
    )

    params_path = str(Path(outdir) / f"{base}_fitted_params.txt")
    try:
        with open(params_path, "w") as f:
            f.write(f"input_histogram\t{infile}\n")
            f.write(f"counts_column\t{counts_col}\n")
            f.write(f"mu\t{mu}\n")
            f.write(f"sigma\t{sigma}\n")
            f.write(f"total_count\t{int(counts.sum())}\n")
            f.write("\n# Goodness-of-fit metrics\n")
            for k, v in metrics.items():
                f.write(f"{k}\t{_fmt_val(v)}\n")
    except Exception as e:
        print(f"Warning: failed to write params file: {e}")

    print(f"\nNormal fit outputs saved under: {outdir}")
    print(f"  mu    = {mu:.6f}")
    print(f"  sigma = {sigma:.6f}")
    print(f"  R^2   = {_fmt_val(metrics.get('r_squared'))}")
    print(f"  plot files: {saved}")
    print(f"  params file: {params_path}")

    return {
        "mu": mu,
        "sigma": sigma,
        "metrics": metrics,
        "saved": saved,
        "params_path": params_path,
    }


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build avg diff histogram over pred_iter_* columns and fit a Normal distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Preferred: named args
    p.add_argument("--input", dest="input_file_path", help="Path to input CSV.")
    p.add_argument("--iter-start", type=int, help="First iteration (inclusive).")
    p.add_argument("--iter-end", type=int, help="Last iteration (inclusive).")
    p.add_argument("--iter-step", type=int, help="Iteration step size.")
    p.add_argument("--diff-min", type=int, help="Minimum difference value (inclusive).")
    p.add_argument("--diff-max", type=int, help="Maximum difference value (inclusive).")

    # Backward-compat: positional args (optional)
    # If provided, they fill in any missing named args.
    p.add_argument("pos_input_file_path", nargs="?", help="(positional legacy) Path to input CSV.")
    p.add_argument("pos_iter_start", nargs="?", type=int, help="(positional legacy) First iteration (inclusive).")
    p.add_argument("pos_iter_end", nargs="?", type=int, help="(positional legacy) Last iteration (inclusive).")
    p.add_argument("pos_iter_step", nargs="?", type=int, help="(positional legacy) Iteration step size.")
    p.add_argument("pos_diff_min", nargs="?", type=int, help="(positional legacy) Minimum difference (inclusive).")
    p.add_argument("pos_diff_max", nargs="?", type=int, help="(positional legacy) Maximum difference (inclusive).")

    p.add_argument("--show-plot", action="store_true", help="Show the plot window (also saves files).")
    p.add_argument("--no-png", action="store_true", help="Do not save PNG.")
    p.add_argument("--no-pdf", action="store_true", help="Do not save PDF.")

    args = p.parse_args()

    # Fill named args from positional args if named args not provided
    if args.input_file_path is None:
        args.input_file_path = args.pos_input_file_path
    if args.iter_start is None:
        args.iter_start = args.pos_iter_start
    if args.iter_end is None:
        args.iter_end = args.pos_iter_end
    if args.iter_step is None:
        args.iter_step = args.pos_iter_step
    if args.diff_min is None:
        args.diff_min = args.pos_diff_min
    if args.diff_max is None:
        args.diff_max = args.pos_diff_max

    # Validate required fields are present
    missing = [name for name in ("input_file_path", "iter_start", "iter_end", "iter_step", "diff_min", "diff_max")
               if getattr(args, name) is None]
    if missing:
        p.error("Missing required arguments: " + ", ".join(missing) +
                "\n\nUse named args, e.g.\n"
                "  python fit_normal.py --input FILE --iter-start 1000 --iter-end 1800 --iter-step 200 --diff-min -800 --diff-max 800")

    return args


def main() -> None:
    args = parse_args()

    if args.iter_step <= 0:
        raise ValueError("iter_step must be > 0")
    if args.diff_min > args.diff_max:
        raise ValueError("diff_min must be <= diff_max")
    if args.iter_start > args.iter_end:
        raise ValueError("iter_start must be <= iter_end")

    input_path = Path(args.input_file_path).expanduser().resolve()
    input_dir = input_path.parent

    # OUTDIR under same directory as input CSV: f"{iter_start}_to_{iter_end}_fitted_normal"
    outdir = input_dir / f"{args.iter_start}_to_{args.iter_end}_fitted_normal"
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram output path
    hist_csv = outdir / f"{args.iter_start}_to_{args.iter_end}_diff_histogram.csv"

    # 1) Build histogram CSV
    build_histograms(
        infile=str(input_path),
        out_csv=str(hist_csv),
        iter_start=args.iter_start,
        iter_end=args.iter_end,
        iter_step=args.iter_step,
        diff_min=args.diff_min,
        diff_max=args.diff_max,
    )

    # 2) Fit normal & save plot/params (CSV_PATH = histogram output)
    fit_normal_counts(
        infile=str(hist_csv),
        outdir=str(outdir),
        save_png=not args.no_png,
        save_pdf=not args.no_pdf,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
