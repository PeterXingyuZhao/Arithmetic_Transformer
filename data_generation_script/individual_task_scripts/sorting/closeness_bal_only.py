#!/usr/bin/env python3
"""
closeness_bal_only.py

Generate train/val/test files for the 4-number sorting task.

Each line has the form:
7752,3723,4472,1394=1394,3723,4472,7752$

Defaults:
 - train.txt : 100000 examples
 - val.txt   :  5000 examples
 - test.txt  :  5000 examples

Each example's closeness level is chosen uniformly from {1,2,3}.
"""

import random
import argparse
from typing import List

def gen_example(level: int) -> str:
    """Generate one example for a given level (1, 2, or 3)."""
    if level == 1:
        nums = [random.randint(1000, 9999) for _ in range(4)]
    elif level == 2:
        # same 1st digit (1-9), rest 3 digits 000..999
        first = random.randint(1, 9)
        nums = [int(f"{first}{random.randint(0, 999):03d}") for _ in range(4)]
    elif level == 3:
        # same first 2 digits (10-99), rest 2 digits 00..99
        prefix = random.randint(10, 99)
        nums = [int(f"{prefix}{random.randint(0, 99):02d}") for _ in range(4)]
    else:
        raise ValueError("level must be 1, 2 or 3")

    inputs_str = [f"{n:04d}" for n in nums]
    sorted_str = [f"{n:04d}" for n in sorted(nums)]
    return ",".join(inputs_str) + "=" + ",".join(sorted_str) + "$"

def generate_file(path: str, count: int) -> None:
    """Write `count` examples to `path` using the 1/3-level distribution."""
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(count):
            level = random.choice([1, 2, 3])
            f.write(gen_example(level) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test files for sorting task.")
    parser.add_argument("--train-count", type=int, default=100000, help="number of training examples (default 100000)")
    parser.add_argument("--val-count",   type=int, default=5000,   help="number of validation examples (default 5000)")
    parser.add_argument("--test-count",  type=int, default=5000,   help="number of test examples (default 5000)")
    parser.add_argument("--train-out", type=str, default="train.txt", help="train output filename")
    parser.add_argument("--val-out",   type=str, default="val.txt",   help="val output filename")
    parser.add_argument("--test-out",  type=str, default="test.txt",  help="test output filename")
    parser.add_argument("--seed", "-s", type=int, default=42, help="random seed (optional)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Generating {args.train_count} training examples -> {args.train_out}")
    generate_file(args.train_out, args.train_count)

    print(f"Generating {args.val_count} validation examples -> {args.val_out}")
    generate_file(args.val_out, args.val_count)

    print(f"Generating {args.test_count} test examples -> {args.test_out}")
    generate_file(args.test_out, args.test_count)

    print("Done.")

if __name__ == "__main__":
    main()
