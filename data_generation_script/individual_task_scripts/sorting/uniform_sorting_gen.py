#!/usr/bin/env python3
"""
Generate 4-operand sorting data files.

Each line example:
216,838,191,368=216,191,368,838$

Numbers are uniformly sampled from 0..999 (inclusive).

Creates: train.txt (1_000_000), val.txt (10_000), test.txt (10_000)
"""

import random
import argparse
import os
import time

def generate_file(path, n_examples, rng, batch_size=10000):
    """Generate n_examples lines and write to path. Uses rng (random.Random)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    start = time.time()
    with open(path, "w", encoding="utf-8") as f:
        written = 0
        for block_start in range(0, n_examples, batch_size):
            block_end = min(n_examples, block_start + batch_size)
            lines = []
            for _ in range(block_start, block_end):
                nums = [rng.randint(0, 9999) for _ in range(4)]
                unsorted = ",".join(str(x) for x in nums)
                sorted_part = ",".join(str(x) for x in sorted(nums))
                lines.append(f"{unsorted}={sorted_part}$\n")
            f.write("".join(lines))
            written += (block_end - block_start)
    elapsed = time.time() - start
    print(f"Wrote {written:,} examples to {path} in {elapsed:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Generate 4-operand sorting data.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("--train", type=int, default=100_000, help="Number of training examples")
    parser.add_argument("--val", type=int, default=5_000, help="Number of validation examples")
    parser.add_argument("--test", type=int, default=5_000, help="Number of test examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (optional)")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Batch size used when writing files")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    train_path = os.path.join(args.outdir, "train.txt")
    val_path = os.path.join(args.outdir, "val.txt")
    test_path = os.path.join(args.outdir, "test.txt")

    print(f"Generating data in '{os.path.abspath(args.outdir)}' with seed={args.seed}")
    generate_file(train_path, args.train, rng, batch_size=args.batch_size)
    generate_file(val_path, args.val, rng, batch_size=args.batch_size)
    generate_file(test_path, args.test, rng, batch_size=args.batch_size)
    print("All done.")

if __name__ == "__main__":
    main()
