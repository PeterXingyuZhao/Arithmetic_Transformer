"""
Sample N lines from a large training‐data file.

Training data is expected to be in the form of:
    927+812+774+113=2626$
    617+990+518+280=2405$

Usage:
    python sample.py \
    --input train.txt \
    --output train_eval.txt \
    --sample-size 10000
"""
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample N lines from a large training‐data file and strip off the ‘answer’ section."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the original training data text file (one expression per line)."
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path where the sampled & transformed lines will be written."
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=10000,
        help="Number of lines to sample (default: 10000)."
    )
    return parser.parse_args()

def transform_line(line: str) -> str:
    """
     Originally stripped off the answer.  Now just returns the full line.
    """
    return line.rstrip("\n")   # keep everything up to (and including) the answer


def main():
    args = parse_args()

    # Step 1: Read all lines into a list
    with open(args.input, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    print(f"Total lines in input file: {total_lines}")

    # Step 2: If file has fewer than sample_size, just sample all (or warn)
    if total_lines < args.sample_size:
        raise ValueError(
            f"Requested sample size ({args.sample_size}) is larger than total lines ({total_lines})."
        )

    # Step 3: Randomly sample indices (without replacement)
    sampled_indices = random.sample(range(total_lines), args.sample_size)
    # Sort indices if you want to keep original order; optional
    sampled_indices.sort()

    # Step 4: Transform and collect
    transformed_lines = []
    for idx in sampled_indices:
        original = all_lines[idx]
        transformed = transform_line(original)
        transformed_lines.append(transformed + "\n")

    # Step 5: Write out to the output file
    with open(args.output, "w", encoding="utf-8") as out_f:
        out_f.writelines(transformed_lines)

    print(f"Wrote {len(transformed_lines)} sampled lines to '{args.output}'.")

if __name__ == "__main__":
    main()
