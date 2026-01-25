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
with one digit (units/tens/hundreds/thousands) replaced by a
random digit sampled uniformly for each example:
 - units/tens/hundreds: sample 0..9
 - thousands: sample 0..3

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


def replace_digit_with_mask(sum_val: int, pos_from_right: int, mask_char: str) -> str:
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
    return s[:idx] + mask_char + s[idx + 1 :]


def sample_replacement_digit(rng: random.Random, mask_pos: int) -> str:
    """
    Sample a replacement digit as a single-character string.
    For thousands (mask_pos == 3) draw from 0..3.
    For others draw from 0..9.
    """
    if mask_pos == 3:  # thousands
        return str(rng.randint(0, 3))
    else:
        return str(rng.randint(0, 9))


def generate_example(rng: random.Random, mask_pos: int) -> str:
    """
    Generate one example line "a+b+c+d=R$" where R has the digit at mask_pos
    replaced by a sampled digit (via sample_replacement_digit).
    """
    s = 0
    # ensure sum is of reasonable size (keeps behavior of original script)
    while s < 100:
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        c = rng.randint(0, 999)
        d = rng.randint(0, 999)
        s = a + b + c + d

    replacement = sample_replacement_digit(rng, mask_pos)
    result_masked = replace_digit_with_mask(s, mask_pos, mask_char=replacement)
    return f"{a}+{b}+{c}+{d}={result_masked}$\n"


def generate_file(path: Path, count: int, rng: random.Random, mask_pos: int, flush_every: int = 100_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(1, count + 1):
            f.write(generate_example(rng, mask_pos))
            if (i % flush_every) == 0:
                f.flush()
    print(f"Wrote {count} examples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-operand addition datasets with a chosen digit replaced by a uniformly sampled digit."
    )
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for train/val/test files.")
    parser.add_argument("--train_size", type=int, default=1_000_000, help="Number of training examples.")
    parser.add_argument("--val_size", type=int, default=10_000, help="Number of validation examples.")
    parser.add_argument("--test_size", type=int, default=10_000, help="Number of test examples.")
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
    print("Masking is randomized per-example: units/tens/hundreds -> uniform 0..9, thousands -> uniform 0..3.")

    generate_file(out_dir / "train.txt", args.train_size, rng, mask_pos)
    generate_file(out_dir / "val.txt", args.val_size, rng, mask_pos)
    generate_file(out_dir / "test.txt", args.test_size, rng, mask_pos)


if __name__ == "__main__":
    main()
