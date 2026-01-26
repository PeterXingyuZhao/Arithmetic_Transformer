# result_analysis.py
import re
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------------------- USER CONFIG (edit in notebook) ------------------------
CSV_PATH   = Path('/Users/perfectpeter/Library/CloudStorage/GoogleDrive-llmfunexperiment@gmail.com/My Drive/addition/results/50_digit_times_1_digit/plain_out/50_digit_times_1_digit_plain/test_results.csv')   # <-- set this in your notebook/script
PLOT_INTERVAL = 100          # choose every ~100 steps (script finds the nearest available pred column)
OFFSET     = 0               # first iteration to consider
MAX_STEPS  = 3000            # maximum training step to draw
ACTUAL_COL = "actual"
PRED_REGEX = r"pred_iter_(\d+)"
OUT_FIG_SUFFIX = "_digitwise_error_rates.pdf"
SAVE_COUNTS_CSV = False
LABEL_STEP = 5               # show colorbar labels every LABEL_STEP digits (e.g., 5 -> 1st,6th,11th,...)
# ---------------------------------------------------------------------------

# styling similar to your sample
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors

mpl.rcParams.update({
    "lines.linewidth": 2.5,
    "axes.labelsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
})

PRED_REGEX_DEFAULT = PRED_REGEX

def _ordinal(n: int) -> str:
    """Return the ordinal string for a positive integer (1 -> '1st', 2 -> '2nd', ...)."""
    if n <= 0:
        return f"{n}th"
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def _place_names(count: int) -> List[str]:
    """Generate place names '0th', '1st', ... for internal use (0 = units)."""
    return [f"{i}th" for i in range(count)]

def _place_order_key(name: str) -> Tuple[int, str]:
    """Key function to order ordinal place names numerically, fallback to lexical order."""
    match = re.fullmatch(r"(\d+)(st|nd|rd|th)?", name)
    if match:
        return int(match.group(1)), name
    return float("inf"), name

def digit_error_tally(actuals, preds) -> Dict[str, int]:
    """
    Count digit-wise mismatches between two sequences of numbers (or numeric strings).
    Determine maximum width from actuals only (so long/wrong preds won't expand places).
    Returns dict mapping place names ('0th','1st',...) where '0th' = units.
    """
    str_actuals = []
    str_preds = []
    for a, p in zip(actuals, preds):
        try:
            str_actuals.append(str(int(a)))
        except Exception:
            str_actuals.append("")
        try:
            str_preds.append(str(int(p)))
        except Exception:
            str_preds.append(str(p) if p is not None else "")

    if str_actuals:
        actual_lengths = [len(s) for s in str_actuals if s != ""]
        max_width = max(actual_lengths) if actual_lengths else 1
    else:
        max_width = 1
    max_width = max(1, max_width)

    place_names = _place_names(max_width)  # '0th','1st',...
    counts = {place: 0 for place in place_names}

    for a_str, p_str in zip(str_actuals, str_preds):
        a_pad = a_str.zfill(max_width)
        p_right = p_str[-max_width:] if len(p_str) >= max_width else p_str
        p_pad = p_right.zfill(max_width)
        for i in range(max_width):
            if a_pad[i] != p_pad[i]:
                place_idx = max_width - 1 - i  # right-aligned: last char is units (0th)
                counts[place_names[place_idx]] += 1

    return counts

def _select_pred_columns(available_steps: List[int], interval: int, offset: int, max_steps: int) -> List[int]:
    """
    Given a sorted list of available pred steps, choose a subset that approximates every 'interval' steps
    between offset and max_steps by choosing the available column nearest to each target multiple.
    Returns a sorted list of chosen available steps (unique).
    """
    if interval <= 0:
        # if interval not positive, return all steps in range
        return [s for s in available_steps if offset <= s <= max_steps]

    targets = list(range(offset, max_steps + 1, interval))
    chosen = []
    avail = [s for s in available_steps if offset <= s <= max_steps]
    if not avail:
        return []
    used = set()
    for t in targets:
        best = min(avail, key=lambda s: (abs(s - t), s))
        if best not in used:
            chosen.append(best)
            used.add(best)
    chosen_sorted = sorted(chosen)
    return chosen_sorted

def analyze_csv(
    csv_path: Union[str, Path],
    plot_interval: int = PLOT_INTERVAL,
    offset: int = OFFSET,
    max_steps: int = MAX_STEPS,
    actual_col: str = ACTUAL_COL,
    pred_regex: str = PRED_REGEX_DEFAULT,
    save_fig: bool = True,
    fig_path: Optional[Union[str, Path]] = None,
    save_counts_csv: bool = SAVE_COUNTS_CSV,
    label_step: int = LABEL_STEP,
    cmap_name: str = "viridis",
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]]]:
    """
    Read csv_path, compute digit-wise error tallies for pred_iter_* columns,
    produce a PDF plot saved next to the CSV, and return the DataFrame and stats.
    Legend is shown as a colorbar with tick labels every `label_step` digits.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if actual_col not in df.columns:
        raise ValueError(f"Actual column '{actual_col}' not found in CSV.")

    # discover pred columns and their steps
    prog = re.compile(pred_regex)
    available = []
    col_by_step = {}
    for col in df.columns:
        m = prog.fullmatch(col)
        if not m:
            continue
        step = int(m.group(1))
        available.append(step)
        col_by_step[step] = col
    available_sorted = sorted(available)

    if not available_sorted:
        raise ValueError("No prediction columns found with the provided regex.")

    # choose which steps to plot (approx every plot_interval steps)
    chosen_steps = _select_pred_columns(available_sorted, plot_interval, offset, max_steps)
    if not chosen_steps:
        raise ValueError("No prediction columns selected after applying interval/offset/max_steps filters.")

    # collect counts
    place_counts_over_iters: Dict[int, Dict[str,int]] = {}
    for step in chosen_steps:
        col = col_by_step[step]
        stats = digit_error_tally(df[actual_col], df[col])
        place_counts_over_iters[step] = stats

    # union of all places
    all_places = set()
    for stats in place_counts_over_iters.values():
        all_places.update(stats.keys())

    ordered_places = sorted(all_places, key=_place_order_key)
    if not ordered_places:
        example_stats = next(iter(place_counts_over_iters.values()))
        ordered_places = list(example_stats.keys())

    # convert to series (counts -> ratios)
    n_examples = int(df[actual_col].notna().sum()) if actual_col in df.columns else len(df)
    iterations = sorted(place_counts_over_iters.keys())
    series_counts = {p: [place_counts_over_iters[it].get(p, 0) for it in iterations] for p in ordered_places}
    series_ratios = {}
    for p in ordered_places:
        if n_examples > 0:
            series_ratios[p] = [c / n_examples for c in series_counts[p]]
        else:
            series_ratios[p] = [0.0 for _ in series_counts[p]]

    # human-friendly labels: '1st digit' corresponds to units (0th)
    human_labels = []
    for p in ordered_places:
        match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", p)
        if match:
            idx = int(match.group(1))
            digit_number = idx + 1  # units (0th) -> 1st digit
            human_labels.append(f"{_ordinal(digit_number)} digit")
        else:
            human_labels.append(p)

    # ---------- PLOTTING (color-shaded legend via colorbar) ----------
    fig, ax = plt.subplots(figsize=(10, 6))

    n_places = len(ordered_places)
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max(0, n_places - 1))

    for idx, p in enumerate(ordered_places):
        color = cmap(norm(idx))
        ax.plot(iterations, series_ratios[p], color=color)

    # labels per your request
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Digit-wise error rates")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, linestyle="--", linewidth=0.5)

    # remove standard legend (we use colorbar instead)
    # ax.legend().set_visible(False)  # no-op if no legend created

    # x-axis: choose up to ~8 nice tick positions and show plain integers with commas
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

    # optional: rotate xtick labels slightly if crowded
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # --- create colorbar inside plot using explicit axes placement ---
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # get current axes position in figure coords
    pos = ax.get_position()  # Bbox(x0, y0, width, height) in figure coords

    # cax coords: [left, bottom, width, height] in figure coords
    # tweak these numbers (width, left offset, bottom margin) to taste
    cax_width = 0.03
    cax_left = pos.x0 + pos.width - cax_width - 0.08  # slightly inset from right edge of plot
    cax_bottom = pos.y0 + 0.08
    cax_height = pos.height * 0.95

    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")

    # compute tick indices: show first, then every label_step, ensure last is shown
    if n_places <= 0:
        tick_inds = []
    else:
        tick_inds = list(range(0, n_places, label_step))
        if (n_places - 1) not in tick_inds:
            tick_inds.append(n_places - 1)
    if tick_inds:
        cbar.set_ticks(tick_inds)
        cbar.set_ticklabels([human_labels[i] for i in tick_inds])
    # (optional) shorten tick label font size so they fit
    cax.tick_params(labelsize=16)

    # # colorbar as legend: ticks every label_step digits (0-based indices)
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])  # required for colorbar in some backends

    # # compute tick indices: show first, then every label_step, ensure last is shown
    # if n_places <= 0:
    #     tick_inds = []
    # else:
    #     tick_inds = list(range(0, n_places, label_step))
    #     if (n_places - 1) not in tick_inds:
    #         tick_inds.append(n_places - 1)

    # cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.06)
    # if tick_inds:
    #     cbar.set_ticks(tick_inds)
    #     cbar.set_ticklabels([human_labels[i] for i in tick_inds])
    # cbar.set_label("Digit place", rotation=270, labelpad=15)

    plt.tight_layout()

    if save_fig:
        if fig_path is None:
            fig_path = csv_path.with_name(csv_path.stem + OUT_FIG_SUFFIX)
        fig.set_size_inches(10, 6)
        fig.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    # optionally save counts per iteration
    if save_counts_csv:
        counts_rows = []
        for it in iterations:
            row = {"iter": it}
            for p in ordered_places:
                row[p] = place_counts_over_iters[it].get(p, 0)
            counts_rows.append(row)
        counts_df = pd.DataFrame(counts_rows)
        counts_csv_path = csv_path.with_name(csv_path.stem + "_digit_counts.csv")
        counts_df.to_csv(counts_csv_path, index=False)

    return df, place_counts_over_iters


# if run as script, accept CLI args
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a results CSV with pred_iter_* columns and plot digit-wise error rates."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file to analyze (e.g., test_results.csv).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum training step to draw (default: {MAX_STEPS}).",
    )

    args = parser.parse_args()

    df, stats = analyze_csv(
        args.csv_path,
        plot_interval=PLOT_INTERVAL,
        offset=OFFSET,
        max_steps=args.max_steps,
        actual_col=ACTUAL_COL,
        pred_regex=PRED_REGEX,
        save_fig=True,
        fig_path=None,
        save_counts_csv=SAVE_COUNTS_CSV,
        label_step=LABEL_STEP,
        cmap_name="viridis",
    )

    out_path = args.csv_path.with_name(args.csv_path.stem + OUT_FIG_SUFFIX)
    print("Done. Figure saved next to CSV:", out_path)
