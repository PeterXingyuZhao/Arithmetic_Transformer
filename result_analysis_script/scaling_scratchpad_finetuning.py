import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import string
from pathlib import Path
from typing import Optional, List, Dict, Any

# ----------------------------
# CONFIG: Appearance
# ----------------------------
AXIS_LABEL_FONT_SIZE = 16
TICKS_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
X_TICK_LABEL_FONT_SIZE = 13

# ----------------------------
# PATH / OUTPUT CONFIG
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_BASENAME = "scaling_scratchpad_finetuning_bar_graph"
DEFAULT_OUT_EXT = ".pdf"


# ----------------------------
# DATA LOGIC
# ----------------------------
PRED_REGEX = r"pred_iter_(\d+)"
ACTUAL_COL = "actual"


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v} (expected True/False)")


def normalize_input_path(p: str) -> Path:
    """
    Accept absolute paths, or relative paths.

    Resolution strategy:
      1) relative to current working directory (default Python behavior)
      2) if not found, relative to the script directory
    """
    raw = Path(p).expanduser()

    # Absolute path: just use it
    if raw.is_absolute():
        return raw

    # Relative to CWD
    cwd_candidate = (Path.cwd() / raw)
    if cwd_candidate.exists():
        return cwd_candidate

    # Fallback: relative to the script directory
    script_candidate = (SCRIPT_DIR / raw)
    if script_candidate.exists():
        return script_candidate

    # If it doesn't exist anywhere, return the CWD-relative form (so warnings look sensible)
    return cwd_candidate


def extract_number_from_reasoning(s: str) -> str:
    s = str(s)
    if "=" in s:
        parts = s.split("=")
        for part in reversed(parts):
            clean = part.strip()
            if clean.isdigit():
                return str(int(clean))
    m = re.search(r"(\d+)\D*$", s)
    if m:
        try:
            return str(int(m.group(1)))
        except Exception:
            pass
    return ""


def digit_error_tally(actuals, preds, reasoning=True) -> dict[str, int]:
    str_actuals = []
    str_preds = []
    for a, p in zip(actuals, preds):
        try:
            val = extract_number_from_reasoning(a) if reasoning else str(int(a))
            str_actuals.append(val)
        except Exception:
            str_actuals.append("")
        try:
            val = extract_number_from_reasoning(p) if reasoning else str(int(p))
            str_preds.append(val)
        except Exception:
            str_preds.append("")

    if not str_actuals or not str_preds:
        max_width = 0
    else:
        max_width = max(max(len(s) for s in str_actuals), max(len(s) for s in str_preds))

    base_places = [
        "units", "tens", "hundreds", "thousands",
        "ten-thousands", "hundred-thousands",
        "millions", "ten-millions", "hundred-millions"
    ]

    max_width = min(max_width, len(base_places))
    place_names = base_places[:max_width]
    counts = {place: 0 for place in place_names}

    for a_str, p_str in zip(str_actuals, str_preds):
        a_pad = a_str.zfill(max_width)
        p_pad = p_str.zfill(max_width)
        for i in range(max_width):
            if a_pad[i] != p_pad[i]:
                place_idx = max_width - 1 - i
                counts[place_names[place_idx]] += 1

    return counts


def compute_rates_for_step(csv_path: Path, target_step: int, reasoning: bool) -> dict[str, float]:
    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return {"Units": 0, "Tens": 0, "Hundreds": 0, "Thousands": 0}

    df = pd.read_csv(csv_path)

    target_col = f"pred_iter_{target_step}"
    if target_col not in df.columns:
        steps = []
        for col in df.columns:
            m = re.fullmatch(PRED_REGEX, col)
            if m:
                steps.append(int(m.group(1)))
        steps = sorted(steps)
        print(f"Warning: {target_col} not found in {csv_path.name}. Available (first 5): {steps[:5]}...")
        return {"Units": 0, "Tens": 0, "Hundreds": 0, "Thousands": 0}

    n_examples = len(df)
    stats = digit_error_tally(df[ACTUAL_COL], df[target_col], reasoning)

    rates = {
        "Units": stats.get("units", 0) / n_examples if n_examples else 0.0,
        "Tens": stats.get("tens", 0) / n_examples if n_examples else 0.0,
        "Hundreds": stats.get("hundreds", 0) / n_examples if n_examples else 0.0,
        "Thousands": stats.get("thousands", 0) / n_examples if n_examples else 0.0,
    }
    return rates


# ----------------------------
# DEFAULT INPUT CONFIGURATION
# ----------------------------
default_tests = [
    ("20M NanoGPT", Path("/content/drive/MyDrive/addition_1/4_operands_1M/20M_plain_out/20M_4_operands_0_to_999_uniform_plain_1/test_results.csv"), 20000, False),
    ("100M NanoGPT", Path("/content/drive/MyDrive/addition_1/4_operands_1M/100M_plain_out/100M_4_operands_0_to_999_uniform_plain_1/test_results.csv"), 50000, False),
    ("Pyhtia 1B", Path("/content/drive/MyDrive/addition_1/4_operands_0_to_999_uniform/pythia_out_plain/4_operands_pythia_1b_plain_3/test_results.csv"), 30000, False),
    ("Scratchpad D", Path("/content/drive/MyDrive/addition_1/4_operands_0_to_999_uniform_reasoning_plain_out/4_operands_0_to_999_uniform_reasoning_plain_4/test_transformed_1_results.csv"), 4500, True),
    ("Scratchpad A + D", Path("/content/drive/MyDrive/addition_1/4_operands_0_to_999_uniform_reasoning_plain_2_out/4_operands_0_to_999_uniform_reasoning_plain_2_1/test_transformed_2_results.csv"), 500, True),
]


# ----------------------------
# PLOTTING LOGIC
# ----------------------------
def plot_grouped_bars(
    data: List[dict],
    out_path: Optional[str] = None,
    show: bool = False,
    extra_out_paths: Optional[List[Path]] = None,
):
    metrics = ["Units", "Tens", "Hundreds", "Thousands"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    metric_to_color = {m: c for m, c in zip(metrics, colors)}

    labels = [d["label"] for d in data]
    n_models = len(data)
    bars_per_model = len(metrics)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.5), 3.0))

    group_positions = np.arange(n_models)
    group_width = 0.85
    bar_width = group_width / bars_per_model
    offsets = (np.arange(bars_per_model) - (bars_per_model - 1) / 2.0) * bar_width

    for i, metric in enumerate(metrics):
        values = [d["rates"][metric] for d in data]
        ax.bar(
            group_positions + offsets[i],
            values,
            width=bar_width * 0.95,
            color=metric_to_color[metric],
            label=metric,
        )

    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=X_TICK_LABEL_FONT_SIZE)

    ax.tick_params(axis="y", labelsize=TICKS_FONT_SIZE)
    ax.set_ylabel("Error Rate", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.5, n_models - 0.5)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    for i in range(1, n_models):
        ax.axvline(i - 0.5, linewidth=1, color="lightgray")

    handles = [plt.Rectangle((0, 0), 1, 1, color=metric_to_color[m]) for m in metrics]
    ax.legend(
        handles,
        metrics,
        ncol=len(metrics),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        columnspacing=1.0,
        borderaxespad=0.1,
        handletextpad=0.4,
    )

    fig.tight_layout()

    # Save primary output (old behavior)
    if out_path:
        out_path = str(out_path)
        if not out_path.lower().endswith(".pdf"):
            out_path += ".pdf"
        fig.savefig(out_path, bbox_inches="tight")  # PDF
        print(f"Saved plot to: {out_path}")

    # Save extra copies next to CSVs
    if extra_out_paths:
        for p in extra_out_paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(p), bbox_inches="tight")  # PDF
                print(f"Saved extra copy to: {p}")
            except Exception as e:
                print(f"Warning: failed to save extra copy to {p}: {e}")

    if show:
        plt.show()

    plt.close(fig)


def build_data_from_tests(tests: List[tuple]) -> List[Dict[str, Any]]:
    data = []
    letters = string.ascii_lowercase

    for i, (name, csv_path, step, reasoning) in enumerate(tests):
        csv_path = Path(csv_path)
        rates = compute_rates_for_step(csv_path, int(step), bool(reasoning))

        panel = letters[i] if i < len(letters) else f"x{i}"
        label = f"({panel}) {name}\n{int(step)}"

        data.append(
            {
                "name": name,
                "path": str(csv_path),
                "step": int(step),
                "reasoning": bool(reasoning),
                "rates": rates,
                "label": label,
            }
        )
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Grouped digit-error bar plot for multiple test CSVs.")
    parser.add_argument(
        "--test",
        action="append",
        nargs=4,
        metavar=("NAME", "CSV_PATH", "STEP", "REASONING"),
        help='Repeatable. Example: --test "20M" "results/test_results.csv" 20000 False',
    )
    parser.add_argument("--out", type=str, default=None, help="Output image path (e.g., plot.png).")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.test and len(args.test) > 0:
        tests = []
        for name, path, step, reasoning in args.test:
            csv_path = normalize_input_path(path)
            tests.append((name, csv_path, int(step), str2bool(reasoning)))
    else:
        tests = default_tests

    data = build_data_from_tests(tests)

    # Base name to use for per-CSV-dir saves
    if args.out:
        out_base = Path(args.out).stem     # ignore any extension user provides
    else:
        out_base = DEFAULT_OUT_BASENAME

    # Save a PDF in each unique CSV directory
    csv_dirs = sorted({Path(t[1]).parent for t in tests})
    extra_out_paths = [d / f"{out_base}{DEFAULT_OUT_EXT}" for d in csv_dirs]

    plot_grouped_bars(
        data,
        out_path=args.out,               # optional extra save (also coerced to .pdf)
        show=args.show,
        extra_out_paths=extra_out_paths
    )



if __name__ == "__main__":
    main()
