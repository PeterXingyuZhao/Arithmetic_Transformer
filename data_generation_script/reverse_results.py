#!/usr/bin/env python3
"""
reverse_results.py

Example:
    python reverse_results.py train.txt train_eval.txt test.txt val.txt --dir ../data/4_operands_0_to_999_uniform
"""
import argparse
import os
from typing import List

def reverse_results(input_path: str, output_path: str) -> None:
    """
    Read each line from `input_path`, reverse the two-digit result before the '$',
    and write the modified line to `output_path`.
    """
    if not os.path.isfile(input_path):
        print(f"Error: '{input_path}' does not exist.")
        return

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")
            # Skip empty lines
            if not line.strip():
                continue

            # Expect format: <expression>=<two_digits>$
            if "=" in line and line.endswith("$"):
                left, right = line.split("=", 1)
                # right is like "42$" or "02$"
                digits = right[:-1]  # drop the trailing '$'
                # Reverse the digit string (e.g., "42" -> "24", "02" -> "20")
                reversed_digits = digits[::-1]
                new_line = f"{left}={reversed_digits}$"
            else:
                # If line doesn't match expected pattern, leave it unchanged
                new_line = line

            outfile.write(new_line + "\n")

    print(f"Processed '{input_path}' -> '{output_path}'.")


def make_input_output_paths(directory: str, filenames: List[str]) -> List[tuple]:
    """
    Given a directory and a list of filenames (like ['train.txt', 'test.txt']),
    return a list of (input_path, output_path) pairs where
      input_path = os.path.join(directory, filename)  (unless filename is absolute)
      output_path = same directory, filename becomes <stem>_reverse<ext>
    """
    pairs = []
    directory = os.path.expanduser(directory)
    for fname in filenames:
        # If fname is absolute or contains directory parts, keep as-is for input.
        if os.path.isabs(fname) or os.path.dirname(fname):
            input_path = os.path.abspath(os.path.expanduser(fname))
            out_dir = os.path.dirname(input_path) or directory
        else:
            input_path = os.path.abspath(os.path.join(directory, fname))
            out_dir = os.path.abspath(directory)

        stem, ext = os.path.splitext(os.path.basename(input_path))
        output_name = f"{stem}_reverse{ext or '.txt'}"
        output_path = os.path.join(out_dir, output_name)
        pairs.append((input_path, output_path))
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reverse the two-digit results before the '$' in each given file."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Filenames to reverse (e.g. train.txt train_eval.txt test.txt val.txt). "
             "If a filename contains a path or is absolute, that path is used; otherwise it's joined with --dir."
    )
    parser.add_argument(
        "--dir",
        "-d",
        default=".",
        help="Directory where the input files live (default: current directory). "
             "Ignored for filenames that include a path or are absolute."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = make_input_output_paths(args.dir, args.files)

    # Ensure output directories exist (create if needed)
    for _, out_path in pairs:
        out_dir = os.path.dirname(out_path) or "."
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    # Process each file
    for input_path, output_path in pairs:
        reverse_results(input_path, output_path)


if __name__ == "__main__":
    main()
