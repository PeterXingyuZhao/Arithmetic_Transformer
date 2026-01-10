#!/usr/bin/env python3
"""
addition_permutation.py

Generate 4-operand addition examples in the format:
  349+102+382+907=1740$

The program accepts one required argument: a permutation of "1234" that
controls the digit-ordering of the 4-digit sum. Examples:
 - 1234 -> 1740$
 - 4321 -> 0471$
 - 2143 -> 7104$

Produces files:
 - train.txt (1_000_000 examples)
 - val.txt   (10_000 examples)
 - test.txt  (10_000 examples)
"""

import argparse
import random
import sys

def validate_perm(perm: str):
    if len(perm) != 4 or set(perm) != set("1234"):
        raise ValueError("Permutation must be a 4-character permutation of '1234', e.g. 1234, 4321, 2143")

def permute_sum_str(total: int, perm: str) -> str:
    """Return the zero-padded 4-digit total string permuted by perm."""
    s = f"{total:04d}"  # always 4 chars, leading zeros if needed
    # perm is like '2143' -> indices [1,0,3,2]
    return ''.join(s[int(p) - 1] for p in perm)

def generate_examples_to_file(path: str, n: int, perm: str, rng: random.Random):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            a = rng.randrange(0, 1000)  # 0..999
            b = rng.randrange(0, 1000)
            c = rng.randrange(0, 1000)
            d = rng.randrange(0, 1000)
            total = a + b + c + d
            permuted_total = permute_sum_str(total, perm)
            line = f"{a}+{b}+{c}+{d}={permuted_total}$\n"
            f.write(line)
            # optional: small progress printing for large files
            # (commented out to keep output clean)
            # if (i+1) % 100_000 == 0:
            #     print(f"Wrote {i+1} examples to {path}")

def main():
    parser = argparse.ArgumentParser(description="Generate 4-operand addition data with permuted sum digits.")
    parser.add_argument("perm", type=str, help="Permutation of '1234' controlling output order (e.g. 1234, 4321, 2143)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42) for reproducibility")
    parser.add_argument("--out-prefix", type=str, default="", help="Optional prefix for output files (default: none)")
    args = parser.parse_args()

    try:
        validate_perm(args.perm)
    except ValueError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)

    train_path = f"{args.out_prefix}train.txt"
    val_path   = f"{args.out_prefix}val.txt"
    test_path  = f"{args.out_prefix}test.txt"

    print(f"Generating files with permutation {args.perm} (seed={args.seed})...")
    generate_examples_to_file(train_path, 1_000_000, args.perm, rng)
    generate_examples_to_file(val_path, 10_000, args.perm, rng)
    generate_examples_to_file(test_path, 10_000, args.perm, rng)
    print("Done.")
    print(f"Files: {train_path} ({1_000_000} lines), {val_path} (10000 lines), {test_path} (10000 lines)")

if __name__ == "__main__":
    main()
