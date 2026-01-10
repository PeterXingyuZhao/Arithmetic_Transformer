#!/usr/bin/env python3
"""
generate_addition_datasets.py

Generates three files:
 - train.txt with 1_000_000 examples
 - val.txt   with 10_000 examples
 - test.txt  with 10_000 examples

Each line has the form:
    A+B+C+D=R$
where A..D are integers in 0..999, R is the decimal sum
with one digit (units/tens/hundreds/thousands) replaced by '?'.
An extra trailing '$' is appended to each line (kept from the original format).
Default mask position is 'tens' for backward compatibility.
"""

import argparse
import random
from pathlib import Path
from typing import Tuple

MASK_CHOICES = ("units", "tens", "hundreds", "thousands")
# mapping to position from right: units=0, tens=1, hundreds=2, thousands=3
MASK_POS_MAP = {"units": 0, "tens": 1, "hundreds": 2, "thousands": 3}

def replace_digit_with_mask(sum_val: int, pos_from_right: int, mask_char: str = "?") -> str:
    """
    Replace the digit at position `pos_from_right` (0 = units, 1 = tens, ...)
    with mask_char in the decimal representation of sum_val.

    If the number has fewer digits than pos_from_right+1, it is zero-left-padded
    to length pos_from_right+1 before masking.
    """
    s = str(sum_val)
    required_length = pos_from_right + 1
    if len(s) < required_length:
        s = s.zfill(required_length)
    # index to replace: -1 - pos_from_right
    idx = len(s) - 1 - pos_from_right
    # build masked string
    return s[:idx] + mask_char + s[idx+1:]

def generate_example(rng: random.Random, mask_pos: int) -> str:
    s = 0
    while s < 100:
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        c = rng.randint(0, 999)
        d = rng.randint(0, 999)
        s = a + b + c + d
    result_masked = replace_digit_with_mask(s, mask_pos, mask_char="?")
    # keep the trailing '$' as in the original format
    return f"{a}+{b}+{c}+{d}={result_masked}$\n"

def generate_file(path: Path, count: int, rng: random.Random, mask_pos: int, flush_every: int = 100_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(1, count + 1):
            f.write(generate_example(rng, mask_pos))
            # occasional flush to avoid big IO buffers for very large files
            if (i % flush_every) == 0:
                f.flush()
    print(f"Wrote {count} examples to {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-operand addition datasets with a chosen digit masked by '?'."
    )
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for train/val/test files.")
    parser.add_argument("--train", type=int, default=1_000_000, help="Number of training examples.")
    parser.add_argument("--val", type=int, default=10_000, help="Number of validation examples.")
    parser.add_argument("--test", type=int, default=10_000, help="Number of test examples.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    parser.add_argument(
        "--mask-digit", "-m",
        choices=MASK_CHOICES,
        default="tens",
        help="Which digit to mask (units, tens, hundreds, thousands). Default: tens."
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    mask_pos = MASK_POS_MAP[args.mask_digit]

    print(f"Generating datasets in {out_dir} with mask digit '{args.mask_digit}' (pos {mask_pos} from right).")

    generate_file(out_dir / "train.txt", args.train, rng, mask_pos)
    generate_file(out_dir / "val.txt", args.val, rng, mask_pos)
    generate_file(out_dir / "test.txt", args.test, rng, mask_pos)

if __name__ == "__main__":
    main()
