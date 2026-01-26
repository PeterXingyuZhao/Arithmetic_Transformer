import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


mpl.rcParams.update({
    "lines.linewidth": 5,        # default line thickness
    # "axes.titlesize":   20,      # title font size
    "axes.labelsize":  24,      # x/y label font size
    "xtick.labelsize":  18,      # x-tick label size
    "ytick.labelsize":  18,      # y-tick label size
    "legend.fontsize":  22,      # legend text size
})


# ─────────────────────── CONSTANTS ───────────────────────
STEP_SIZE  = 5
OFFSET     = 0
MIN_STEPS  = 0
ACTUAL_COL = "actual"
PRED_REGEX = r"pred_iter_(\d+)"
# ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot digit-wise error rates from a CSV containing pred_iter_* columns."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=800000,
        help="Maximum training step to include (default: 800000)."
    )
    return parser.parse_args()


def digit_error_tally(actuals, preds) -> dict[str, int]:
    """
    Count digit-wise mismatches between two sequences of numbers (or numeric strings).
    actuals, preds: pandas Series or other iterables of numeric or string values.
    Returns a dict mapping place names (units, tens, …) to error counts.
    """
    # 1) Convert actuals/preds to *strings* (defensive: blank on failure)
    str_actuals = []
    str_preds   = []
    for a, p in zip(actuals, preds):
        try:
            str_actuals.append(str(int(a)))
        except Exception:
            str_actuals.append("")
        try:
            str_preds.append(str(int(p)))
        except Exception:
            str_preds.append("")

    # 2) Determine the widest number (so we know how many digit-places to check)
    max_width = max(
        max(len(s) for s in str_actuals),
        max(len(s) for s in str_preds),
    )

    # 3) Dynamically generate place names up to that width
    base_places = [
        "units", "tens", "hundreds", "thousands",
        "ten-thousands", "hundred-thousands",
        "millions", "ten-millions", "hundred-millions"
    ]
    max_width = min(max_width, len(base_places))
    place_names = base_places[:max_width]

    # 4) Initialize counters
    counts = {place: 0 for place in place_names}

    # 5) Zero-pad both sides and compare each digit
    for a_str, p_str in zip(str_actuals, str_preds):
        a_pad = a_str.zfill(max_width)
        p_pad = p_str.zfill(max_width)
        for i in range(max_width):
            if a_pad[i] != p_pad[i]:
                place_idx = max_width - 1 - i  # position from the *right*
                counts[place_names[place_idx]] += 1

    return counts


def main() -> None:
    args = parse_args()

    CSV_PATH = Path(args.csv_path).expanduser().resolve()
    MAX_STEPS = args.max_steps

    # directory containing the input CSV
    OUT_DIR = CSV_PATH.parent

    # optional: base figure name on CSV file name
    OUT_FIG = OUT_DIR / (CSV_PATH.stem + "_digitwise_error_rates.pdf")

    df = pd.read_csv(CSV_PATH)

    # find all pred_iter_* columns whose step = OFFSET + n*STEP_SIZE
    pred_cols = []
    for col in df.columns:
        m = re.fullmatch(PRED_REGEX, col)
        if not m:
            continue
        step = int(m.group(1))
        if (
            step >= OFFSET
            and (step - OFFSET) % STEP_SIZE == 0
            and step <= MAX_STEPS
            and step >= MIN_STEPS
        ):
            pred_cols.append((step, col))
    pred_cols.sort(key=lambda x: x[0])

    n_examples = len(df)

    print(pred_cols)

    iterations = []
    units = []
    tens = []
    hundreds = []
    thousands = []

    for step, col in pred_cols:
        stats = digit_error_tally(df[ACTUAL_COL], df[col])
        iterations.append(step)

        if n_examples > 0:
            units.append(stats.get("units", 0) / n_examples)
            tens.append(stats.get("tens", 0) / n_examples)
            hundreds.append(stats.get("hundreds", 0) / n_examples)
            thousands.append(stats.get("thousands", 0) / n_examples)
        else:
            units.append(0.0)
            tens.append(0.0)
            hundreds.append(0.0)
            thousands.append(0.0)

    # ──────────────────── PLOT ────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, units,     label="Units-place")
    plt.plot(iterations, tens,      label="Tens-place")
    plt.plot(iterations, hundreds,  label="Hundreds-place")
    plt.plot(iterations, thousands, label="Thousands-place")

    plt.xlabel("Training steps")
    plt.ylabel("Digit-wise error rate")
    plt.ylim(0.0, 1.0)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # --- format x-axis ticks as thousands with "K" suffix (e.g. 20000 -> 20K)
    def k_formatter(x, pos):
        try:
            if int(x) == 0:
                return "0"
        except Exception:
            pass
        return f"{int(x / 1000):d}K"

    plt.gca().xaxis.set_major_formatter(FuncFormatter(k_formatter))

    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(Path(OUT_FIG), format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
