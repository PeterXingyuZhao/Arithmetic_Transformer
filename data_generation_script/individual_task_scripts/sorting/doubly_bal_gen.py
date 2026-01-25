#!/usr/bin/env python3
"""
doubly_bal_gen.py

Generates sorting task data files:
 - train.txt (100000 examples)
 - val.txt   (5000 examples)
 - test.txt  (5000 examples)

Each example has four numbers, each independently 3- or 4-digit
(50/50). After deciding digit-lengths, one of three modes is chosen:
  - totally random (1/3)
  - share the highest digit (1/3)
  - share the highest two digits (1/3)

Output format per line:
  6279,8238,4501,3780=3780,4501,6279,8238$
"""

import argparse
import random
from pathlib import Path
from typing import List

# --- Configurable parameters ---
TRAIN_N = 100_000
VAL_N = 5_000
TEST_N = 5_000

# You can set a seed for reproducibility. Set to None for different outputs each run.
RANDOM_SEED = None
# -------------------------------

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)


def choose_length() -> int:
    """Return 3 or 4 with equal probability."""
    return 4 if random.random() < 0.5 else 3


def gen_random_number_of_length(n: int) -> str:
    """Generate a fully random n-digit number as a string (n==3 or 4)."""
    if n == 3:
        return f"{random.randint(100, 999)}"
    if n == 4:
        return f"{random.randint(1000, 9999)}"
    raise ValueError("only support 3- or 4-digit numbers")


def gen_shared_mode_numbers(lengths: List[int], mode: str) -> List[str]:
    """
    Generate numbers that share highest digit(s) according to mode:
      mode == 'share1' -> share the highest 1 digit (1-9), other digits:
         - for 3-digit numbers: remaining two digits drawn from 00-99
         - for 4-digit numbers: remaining three digits drawn from 000-999
      mode == 'share2' -> share the highest 2 digits (10-99), other digits:
         - for 3-digit numbers: remaining one digit drawn from 0-9
         - for 4-digit numbers: remaining two digits drawn from 00-99
    """
    nums: List[str] = []
    if mode == "share1":
        common = random.randint(1, 9)  # single digit 1-9
        common_s = str(common)
        for n in lengths:
            if n == 3:
                suffix = random.randint(0, 99)  # two digits
                s = f"{common_s}{suffix:02d}"
            else:  # n == 4
                suffix = random.randint(0, 999)  # three digits
                s = f"{common_s}{suffix:03d}"
            nums.append(s)

    elif mode == "share2":
        common = random.randint(10, 99)  # two-digit prefix 10-99
        common_s = f"{common}"
        for n in lengths:
            if n == 3:
                suffix = random.randint(0, 9)  # single digit
                s = f"{common_s}{suffix}"
            else:  # n == 4
                suffix = random.randint(0, 99)  # two digits
                s = f"{common_s}{suffix:02d}"
            nums.append(s)
    else:
        raise ValueError("mode must be 'share1' or 'share2'")

    return nums


def generate_example() -> str:
    """
    Generate a single example line of the form:
      a,b,c,d=s1,s2,s3,s4$
    where the right side is the ascending numeric sort of the left side.
    """
    lengths = [choose_length() for _ in range(4)]

    r = random.random()
    if r < 1 / 3:
        mode = "random"
    elif r < 2 / 3:
        mode = "share1"
    else:
        mode = "share2"

    if mode == "random":
        nums = [gen_random_number_of_length(n) for n in lengths]
    else:
        nums = gen_shared_mode_numbers(lengths, mode)

    ints = [int(x) for x in nums]
    sorted_ints = sorted(ints)
    sorted_strs = [str(x) for x in sorted_ints]

    left = ",".join(nums)
    right = ",".join(sorted_strs)
    return f"{left}={right}$\n"


def write_file(path: Path, n_examples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n_examples):
            f.write(generate_example())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", "-o", default=".", help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating datasets...")
    write_file(outdir / "train.txt", TRAIN_N)
    write_file(outdir / "val.txt", VAL_N)
    write_file(outdir / "test.txt", TEST_N)
    print(
        f"Done. Files created in {outdir}: "
        f"train.txt ({TRAIN_N}), val.txt ({VAL_N}), test.txt ({TEST_N})"
    )


if __name__ == "__main__":
    main()
