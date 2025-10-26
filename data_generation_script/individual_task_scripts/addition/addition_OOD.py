#!/usr/bin/env python3
"""
Generate three .txt files (train/val/test) containing addition examples.

For each output file you specify the number of 4-operand examples (N4).
The number of 3-operand examples is computed as round(N4 * 10 / 100)
and the number of 2-operand examples as round(N4 * 1 / 100) by default,
i.e. a 100:10:1 ratio. You can change the ratio with --ratio.

Example usage:
  python addition_OOD.py --train4 1000000 --val4 100000 --test4 10000 --seed 42

Output files (defaults):
  addition_train.txt
  addition_val.txt
  addition_test.txt
"""

import argparse
import random
import sys
from typing import Tuple, List, Optional

def parse_ratio(ratio_str: str) -> Tuple[int, int, int]:
    parts = ratio_str.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Ratio must be in format A:B:C (e.g. 100:10:1)")
    try:
        a, b, c = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Ratio parts must be integers")
    if a <= 0 or b < 0 or c < 0:
        raise argparse.ArgumentTypeError("Ratio parts must be non-negative and first part > 0")
    return a, b, c

def generate_examples(n_operands: int, count: int):
    """Yield `count` examples with `n_operands` operands each."""
    for _ in range(count):
        operands = [random.randint(0, 999) for _ in range(n_operands)]
        lhs = "+".join(str(x) for x in operands)
        rhs = str(sum(operands))
        yield f"{lhs}={rhs}$\n"

def compute_counts_from_ratio(n4: int, ratio: Tuple[int,int,int]) -> Tuple[int,int,int]:
    """Given n4 and ratio (r4,r3,r2) return (n4, n3, n2) with rounding."""
    r4, r3, r2 = ratio
    if r4 == 0:
        return 0, 0, 0
    factor3 = (r3 / r4)
    factor2 = (r2 / r4)
    n3 = int(round(n4 * factor3))
    n2 = int(round(n4 * factor2))
    return n4, n3, n2

def build_and_write_file(out_path: str, n4: int, ratio: Tuple[int,int,int], seed: Optional[int] = None):
    if seed is not None:
        # use a derived seed for each file if desired; keep deterministic overall
        random.seed(seed + hash(out_path) & 0xffffffff)

    n4c, n3c, n2c = compute_counts_from_ratio(n4, ratio)
    total = n4c + n3c + n2c

    print(f"Preparing {out_path}: 4-op={n4c}, 3-op={n3c}, 2-op={n2c}  (total={total})", file=sys.stderr)

    # Generate per operand-count and collect to list, batching to reduce intermediate overhead.
    examples: List[str] = []

    for n_operands, cnt in ((4, n4c), (3, n3c), (2, n2c)):
        if cnt <= 0:
            continue
        print(f"  Generating {cnt} examples with {n_operands} operands...", file=sys.stderr)
        batch: List[str] = []
        # generate in small batches to avoid too many small list appends
        batch_flush = 100_000
        i = 0
        for ex in generate_examples(n_operands, cnt):
            batch.append(ex)
            i += 1
            if i % batch_flush == 0:
                examples.extend(batch)
                batch = []
        if batch:
            examples.extend(batch)

    print("  Shuffling examples...", file=sys.stderr)
    random.shuffle(examples)

    print(f"  Writing {len(examples)} lines to {out_path} ...", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        chunk_size = 100_000
        for i in range(0, len(examples), chunk_size):
            f.writelines(examples[i:i+chunk_size])

    print(f"  Done writing {out_path} (wrote {len(examples)} lines).", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Generate addition examples split into train/val/test files.")
    parser.add_argument("--train4", type=int, default=1_000_000,
                        help="Number of 4-operand examples for the training file (default: 1_000_000)")
    parser.add_argument("--val4", type=int, default=100_000,
                        help="Number of 4-operand examples for the validation file (default: 100_000)")
    parser.add_argument("--test4", type=int, default=10_000,
                        help="Number of 4-operand examples for the test file (default: 10_000)")
    parser.add_argument("--train-out", default="train.txt",
                        help="Training output filename (default: train.txt)")
    parser.add_argument("--val-out", default="val.txt",
                        help="Validation output filename (default: val.txt)")
    parser.add_argument("--test-out", default="test.txt",
                        help="Test output filename (default: test.txt)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Optional global random seed for reproducibility")
    parser.add_argument("--ratio", default="100:10:1",
                        help="Ratio for (#4op : #3op : #2op), default 100:10:1")
    args = parser.parse_args()

    ratio = parse_ratio(args.ratio)
    # Validate inputs
    if args.train4 < 0 or args.val4 < 0 or args.test4 < 0:
        parser.error("Counts must be non-negative")

    if args.seed is not None:
        random.seed(args.seed)

    # Build each file separately (shuffled per file)
    build_and_write_file(args.train_out, args.train4, ratio, seed=args.seed)
    build_and_write_file(args.val_out, args.val4, ratio, seed=args.seed)
    build_and_write_file(args.test_out, args.test4, ratio, seed=args.seed)

    # Print summary
    t4, t3, t2 = compute_counts_from_ratio(args.train4, ratio)
    v4, v3, v2 = compute_counts_from_ratio(args.val4, ratio)
    te4, te3, te2 = compute_counts_from_ratio(args.test4, ratio)
    total_lines = (t4+t3+t2) + (v4+v3+v2) + (te4+te3+te2)
    print("\nSummary (counts per file):", file=sys.stderr)
    print(f"  Train: 4-op={t4}, 3-op={t3}, 2-op={t2}  (total={t4+t3+t2})", file=sys.stderr)
    print(f"  Val:   4-op={v4}, 3-op={v3}, 2-op={v2}  (total={v4+v3+v2})", file=sys.stderr)
    print(f"  Test:  4-op={te4}, 3-op={te3}, 2-op={te2}  (total={te4+te3+te2})", file=sys.stderr)
    print(f"  All files total lines: {total_lines}", file=sys.stderr)

if __name__ == "__main__":
    main()
