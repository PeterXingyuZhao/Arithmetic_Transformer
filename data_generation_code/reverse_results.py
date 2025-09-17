#!/usr/bin/env python3
"""
reverse_results.py

Reads a file "reverse_train.txt" containing lines of simple addition equations
with their (currently reversed) two-digit results ending with a dollar sign,
e.g.:

    7+2+7+8=42$
    1+2+8+9=02$
    ...

This script will reverse the digits of the result (dropping and re‐appending the '$')
so that "7+2+7+8=42$" becomes "7+2+7+8=24$", "1+2+8+9=02$" becomes "1+2+8+9=20$", etc.

Usage:
    python reverse_results.py

It reads from "reverse_train.txt" in the current directory and writes to
"reverse_train_reversed.txt" in the same directory.
"""

import os

INPUT_FILENAME = "./4_operands_0_to_999_10K_val.txt"
OUTPUT_FILENAME = "./4_operands_0_to_999_10K_val_reversed.txt"

def reverse_results(input_path: str, output_path: str) -> None:
    """
    Read each line from `input_path`, reverse the two‐digit result before the '$',
    and write the modified line to `output_path`.
    """
    if not os.path.isfile(input_path):
        print(f"Error: '{input_path}' does not exist.")
        return

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
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

    print(f"Processed '{input_path}' and wrote results to '{output_path}'.")


if __name__ == "__main__":
    reverse_results(INPUT_FILENAME, OUTPUT_FILENAME)
