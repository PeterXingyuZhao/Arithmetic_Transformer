#!/usr/bin/env python3
"""
data_generate.py

Dispatches to individual generation scripts under /data_generation_script/individual_task_scripts/

Usage:
    python data_generate.py --task <task> --num_operands <n> --experiment_name <name> \
        [--train_size N] [--test_size N] [--val_size N] \
        [--train_eval] [--sample-size N] [--generate_reverse]

Example:
    python data_generate.py --task addition --num_operands 4 --experiment_name 4_operands_0_to_999_uniform \
        --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True

Notes:
 - --task (required): one of the supported tasks (addition, multiplication, sorting, ...).
 - --num_operands (optional, default 4): how many operands the generator should use (only passed to generators that accept it).
 - --experiment_name (required): name used to build the default output directory (unless --output_path is supplied).
 - --output_path / -o: optional explicit output directory (overrides default data/<experiment_name>).
 - --train_eval: if present/true, a train_eval file will be produced (sampling controlled by --sample-size).
 - --generate_reverse: if present/true, runs reverse_results.py on generated files at the end.

This script runs the target generator as a separate Python process and passes the chosen output path as --output_dir (same format as addition_gen.py).
If --output_path is omitted, default is parent_dir/data/{experiment_name}.
If --train_eval is given, runs sample.py to produce train_eval.txt from train.txt.
"""

import argparse
import os
import subprocess
import sys

# Defaults must match the generator scripts (addition_gen.py)
DEFAULT_NUM_OPERANDS = 4
DEFAULT_TRAIN_SIZE = 1_000_000
DEFAULT_TEST_SIZE = 10_000
DEFAULT_VAL_SIZE = 10_000
DEFAULT_SAMPLE_SIZE = 10_000

TASK_MAP = {
    "addition": {
        "file": os.path.join("data_generation_script", "individual_task_scripts", "addition", "addition_gen.py"),
        "accepts_num_operands": True,
        "generate_reverse": True,
    },
    "multiplication": {
        "file": os.path.join("data_generation_script", "individual_task_scripts", "multiplication", "multiplication_gen.py"),
        # multiplication_gen.py now accepts --num_operands
        "accepts_num_operands": True,
        "generate_reverse": True,
    },
    "sorting": {
        "file": os.path.join("data_generation_script", "individual_task_scripts", "sorting", "sorting_gen.py"),
        # set True/False depending on your sorting_gen implementation
        "accepts_num_operands": False,
        "generate_reverse": False,
    },
}

def default_data_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data")


def run_subprocess(cmd):
    """Run command and forward stdout/stderr. Raise RuntimeError on non-zero exit."""
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command exited with code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch data generation tasks (runs each generator in a separate Python process)."
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_MAP.keys()),
        required=True,
        help="Which task to generate data for. Supported: " + ", ".join(TASK_MAP.keys()),
    )
    parser.add_argument(
        "--num_operands",
        type=int,
        default=DEFAULT_NUM_OPERANDS,
        help="Number of operands (usually 2, 3, or 4).",
    )
    parser.add_argument(
        "--experiment_name",
        required=True,
        help=(
            "Name of the experiment. If --output_path is not specified, "
            "the default output directory will be /data/{experiment_name}."
        ),
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help=f"Number of training examples (default: {DEFAULT_TRAIN_SIZE})",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=f"Number of test examples (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help=f"Number of validation examples (default: {DEFAULT_VAL_SIZE})",
    )
    parser.add_argument(
        "--train_eval",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=False,
        help="Whether to create a train_eval file. Accepts True/False (case-insensitive).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of lines to sample for train_eval (default: {DEFAULT_SAMPLE_SIZE}). Only used if --train_eval is set.",
    )
    parser.add_argument(
        "--generate_reverse",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=False,
        help="Whether to run reverse_results.py at the end. Accepts True/False (case-insensitive).",
    )


    args = parser.parse_args()

    # Validate num_operands bounds: generators accept up to 6 operands for digit-based tasks.
    if args.num_operands is None:
        num_operands = DEFAULT_NUM_OPERANDS
    else:
        num_operands = args.num_operands

    if num_operands < 1:
        print("Error: --num_operands must be >= 1.", file=sys.stderr)
        sys.exit(2)

    if num_operands > 6:
        # Send message to user and exit with non-zero status.
        print(
            f"Error: --num_operands must be <= 6 (you provided {num_operands}). "
            "Please choose a value in the range 1..6.",
            file=sys.stderr,
        )
        sys.exit(2)

    task = args.task
    experiment_name = args.experiment_name

    output_path = os.path.abspath(os.path.join(default_data_dir(), experiment_name))

    # ensure output directory exists
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

    entry = TASK_MAP.get(task)
    if entry is None:
        print(f"Unknown task: {task}", file=sys.stderr)
        sys.exit(2)

    # path to the generator script (relative to this file)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), entry["file"])

    if not os.path.exists(file_path):
        print(f"Generator script not found: {file_path}", file=sys.stderr)
        sys.exit(3)

    # Always pass --output_dir and the size flags to child script (addition_gen.py expects these flags).
    gen_cmd = [
        sys.executable,
        file_path,
    ]

    if entry.get("accepts_num_operands", False):
        gen_cmd += ["--num_operands", str(num_operands)]

    # always pass the output dir and sizes
    gen_cmd += [
        "--output_dir", output_path,
        "--train_size", str(args.train_size),
        "--test_size", str(args.test_size),
        "--val_size", str(args.val_size),
    ]


    try:
        print(f"Running generator for task '{task}', experiment '{experiment_name}'")
        run_subprocess(gen_cmd)
    except RuntimeError as e:
        print(f"Generator failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Generation finished successfully.")

    # If requested, run sample.py to create train_eval.txt from train.txt
    if args.train_eval:
        sample_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation_script", "sample.py")
        if not os.path.exists(sample_script):
            print(f"sample.py not found at {sample_script}", file=sys.stderr)
            sys.exit(4)

        input_train = os.path.join(output_path, "train.txt")
        output_train_eval = os.path.join(output_path, "train_eval.txt")

        if not os.path.exists(input_train):
            print(f"Expected train file not found: {input_train}", file=sys.stderr)
            sys.exit(5)

        # argparse turns '--sample-size' into attribute 'sample_size'
        sample_size_arg = getattr(args, "sample_size", DEFAULT_SAMPLE_SIZE)

        sample_cmd = [
            sys.executable,
            sample_script,
            "--input", input_train,
            "--output", output_train_eval,
            "--sample-size", str(sample_size_arg),
        ]

        try:
            print(f"Sampling {sample_size_arg} lines to create train_eval at '{output_train_eval}'")
            run_subprocess(sample_cmd)
        except RuntimeError as e:
            print(f"Sampling failed: {e}", file=sys.stderr)
            sys.exit(1)

        print("Sampling finished successfully.")

    # If requested, run reverse_results.py on generated files
    if args.generate_reverse and entry.get("generate_reverse", False):
        reverse_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation_script", "reverse_results.py")
        if not os.path.exists(reverse_script):
            print(f"reverse_results.py not found at {reverse_script}", file=sys.stderr)
            sys.exit(6)

        # Build filenames in the order requested by you:
        # if train_eval: train.txt train_eval.txt test.txt val.txt
        # else:          train.txt test.txt val.txt
        filenames = ["train.txt"]
        if args.train_eval:
            filenames.append("train_eval.txt")
        filenames.extend(["test.txt", "val.txt"])

        # Verify input files exist before calling
        missing = [f for f in filenames if not os.path.exists(os.path.join(output_path, f))]
        if missing:
            print(f"Missing files required for reverse step: {missing}", file=sys.stderr)
            sys.exit(7)

        rev_cmd = [sys.executable, reverse_script] + filenames + ["--dir", output_path]
        try:
            print(f"Running reverse_results on: {', '.join(filenames)}")
            run_subprocess(rev_cmd)
        except RuntimeError as e:
            print(f"reverse_results failed: {e}", file=sys.stderr)
            sys.exit(1)

        print("reverse_results finished successfully.")


if __name__ == "__main__":
    main()
