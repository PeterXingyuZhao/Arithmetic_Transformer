# result_analysis.py
import re
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt


PRED_REGEX_DEFAULT = r"pred_iter_(\d+)"

def _ordinal(n: int) -> str:
    """Return the ordinal string ("0th", "1st", …) for a non-negative integer."""

    if n < 0:
        raise ValueError("Ordinal is only defined for non-negative integers in this context.")

    # Handle the "teens" which always end in "th"
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _place_names(count: int) -> List[str]:
    """Generate a list of ordinal place names up to ``count`` digits."""

    return [_ordinal(i) for i in range(count)]


def _place_order_key(name: str) -> Tuple[int, str]:
    """Key function to order ordinal place names numerically, fallback to lexical order."""

    match = re.fullmatch(r"(\d+)(st|nd|rd|th)", name)
    if match:
        return int(match.group(1)), name
    # Fallback: put any unrecognised names after numeric ones but keep deterministic order
    return float("inf"), name

def digit_error_tally(actuals, preds) -> Dict[str, int]:
    """
    Count digit-wise mismatches between two sequences of numbers (or numeric strings).
    IMPORTANT: determine the maximum width ONLY from the actuals column (so garbage/long model outputs
    won't increase the number of places we check). Predictions longer than the chosen width are
    truncated to the rightmost digits (so units/tens alignment is preserved).
    Returns dict mapping place names (0th, 1st, ...) to error counts.
    """
    str_actuals = []
    str_preds = []
    for a, p in zip(actuals, preds):
        # actual -> attempt to coerce to integer string; on failure produce empty string
        try:
            str_actuals.append(str(int(a)))
        except Exception:
            str_actuals.append("")
        # prediction -> keep raw string attempt to parse int; on failure keep raw string (we will later truncate)
        try:
            str_preds.append(str(int(p)))
        except Exception:
            # if prediction is non-int (contains letters, punctuation...), still keep string form
            # this avoids losing predictions entirely; they'll be truncated/padded below
            str_preds.append(str(p) if p is not None else "")

    # determine max width using only actuals (ignore predictions)
    if str_actuals:
        # consider lengths of those actuals that parsed to numbers; ignore empty strings
        actual_lengths = [len(s) for s in str_actuals if s != ""]
        max_width = max(actual_lengths) if actual_lengths else 1
    else:
        max_width = 1
    max_width = max(1, max_width)

    #base_places = ["units", "tens", "hundreds", "thousands",
    #               "ten-thousands", "hundred-thousands",
    #               "millions", "ten-millions", "hundred-millions"]
    #max_width = min(max_width, len(base_places))
    #place_names = base_places[:max_width]
    place_names = _place_names(max_width)

    counts = {place: 0 for place in place_names}

    # Compare digits aligned on the right (units). Predictions longer than max_width are truncated
    for a_str, p_str in zip(str_actuals, str_preds):
        a_pad = a_str.zfill(max_width)
        # take rightmost max_width digits of prediction, then zfill (so short preds are padded)
        p_right = p_str[-max_width:] if len(p_str) >= max_width else p_str
        p_pad = p_right.zfill(max_width)
        for i in range(max_width):
            if a_pad[i] != p_pad[i]:
                place_idx = max_width - 1 - i
                counts[place_names[place_idx]] += 1

    return counts


def analyze_csv(
    csv_path: str | Path,
    step_size: int = 5,
    offset: int = 0,
    max_steps: int = 800000,
    actual_col: str = "actual",
    pred_regex: str = PRED_REGEX_DEFAULT,
    save_fig: bool = True,
    fig_path: str | None = None,
    save_counts_csv: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]]]:
    """
    Read csv_path, compute digit-wise error tallies for matching pred_iter_* columns,
    plot the results and optionally save the figure.

    Returns:
      - df (pandas.DataFrame)
      - stats_by_iteration: dict mapping iteration -> {place: count, ...}
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # find relevant pred columns
    pred_cols: List[tuple[int, str]] = []
    prog = re.compile(pred_regex)
    for col in df.columns:
        m = prog.fullmatch(col)
        if not m:
            continue
        step = int(m.group(1))
        if step >= offset and (step - offset) % step_size == 0 and step <= max_steps:
            pred_cols.append((step, col))
    pred_cols.sort(key=lambda x: x[0])

    if not pred_cols:
        raise ValueError("No prediction columns found with the provided regex/filters.")

    iterations = []
    place_counts_over_iters = {}  # iteration -> counts dict

    # collect counts
    for step, col in pred_cols:
        # defensive: if actual_col missing, raise informative error
        if actual_col not in df.columns:
            raise ValueError(f"Actual column '{actual_col}' not found in CSV.")
        stats = digit_error_tally(df[actual_col], df[col])
        iterations.append(step)
        place_counts_over_iters[step] = stats

    # Build the union of all place names across iterations
    #base_places = ["units", "tens", "hundreds", "thousands",
    #               "ten-thousands", "hundred-thousands",
    #               "millions", "ten-millions", "hundred-millions"]
    #
    all_places = set()
    for stats in place_counts_over_iters.values():
        all_places.update(stats.keys())

    # Order places by base_places ordering if possible, then append any unknown places (sorted)
    #ordered_places = [p for p in base_places if p in all_places]
    #remaining = sorted(list(all_places - set(ordered_places)))
    #ordered_places.extend(remaining)
    # Order places numerically based on their ordinal name; fallback to lexical order for unknown names
    ordered_places = sorted(all_places, key=_place_order_key)

    # If something strange happened and ordered_places is empty, fall back to the first iteration's keys
    if not ordered_places:
        example_stats = next(iter(place_counts_over_iters.values()))
        ordered_places = list(example_stats.keys())

    # create series for plotting; fill missing entries with 0
    series = {p: [place_counts_over_iters[it].get(p, 0) for it in iterations] for p in ordered_places}

    # plotting
    plt.figure(figsize=(30, 18))
    for p in ordered_places:
        plt.plot(iterations, series[p], label=f"{p} errors")
    plt.title("Digit-wise error count vs. training iteration")
    plt.xlabel("Training iteration")
    plt.ylabel("# rows with an error at that digit")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_fig:
        if fig_path is None:
            fig_path = csv_path.with_suffix(".digit_errors.png")
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

    if save_counts_csv:
        counts_rows = []
        for it in iterations:
            row = {"iter": it}
            # ensure consistent column order / presence for all places
            for p in ordered_places:
                row[p] = place_counts_over_iters[it].get(p, 0)
            counts_rows.append(row)
        counts_df = pd.DataFrame(counts_rows)
        counts_csv_path = csv_path.with_name(csv_path.stem + "_digit_counts.csv")
        counts_df.to_csv(counts_csv_path, index=False)

    return df, place_counts_over_iters


# If called from the command line:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze digit-wise errors from predictions CSV.")
    parser.add_argument("csv_path", help="Path to CSV with columns 'actual' and pred_iter_<step> columns.")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=800000)
    parser.add_argument("--actual_col", type=str, default="actual")
    parser.add_argument("--no-save", action="store_true", help="Don't save figure; show it.")
    parser.add_argument("--save_counts_csv", action="store_true", help="Save counts per iteration as CSV.")
    args = parser.parse_args()

    analyze_csv(
        args.csv_path,
        step_size=args.step_size,
        offset=args.offset,
        max_steps=args.max_steps,
        actual_col=args.actual_col,
        save_fig=(not args.no_save),
        save_counts_csv=args.save_counts_csv,
    )
