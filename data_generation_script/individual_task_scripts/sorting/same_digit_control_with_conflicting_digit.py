#!/usr/bin/env python3
"""
Generate three test files (1000 examples each by default) for the 4-operand 4-digit sorting task.

Outputs:
 - b1_eq_b3diff.txt        : b1 == c1, b3 != c3
 - b3_eq_b1diff.txt        : b3 == c3, b1 != c1
 - b1c1_b3c3_bothdiff.txt  : b1 != c1, b3 != c3

Each example line has the format:
1000,<b1b2b3b4>,<c1c2c3c4>,9999=1000,<b1b2b3b4>,<c1c2c3c4>,9999$
"""
import random
from pathlib import Path
import argparse

NUM_EXAMPLES = 1000

def pick_strict_pair(low=0, high=9, less=True):
    """
    Return a pair (b, c) of integers in [low, high] such that
    b < c if less is True, else b > c. Strict inequality guaranteed.
    """
    if less:
        b = random.randint(low, high - 1)
        c = random.randint(b + 1, high)
    else:
        c = random.randint(low, high - 1)
        b = random.randint(c + 1, high)
    return b, c

def make_example_b1eq_b3diff():
    # b1 == c1 (1..9), b3 != c3 (0..9)
    d1 = random.randint(1, 9)
    b3 = random.randint(0, 9)
    c3 = random.choice([x for x in range(0, 10) if x != b3])

    # choose mode: either (b2<c2 and b4>c4) OR (b2>c2 and b4<c4)
    if random.random() < 0.5:
        b2, c2 = pick_strict_pair(0, 9, less=True)
        b4, c4 = pick_strict_pair(0, 9, less=False)
    else:
        b2, c2 = pick_strict_pair(0, 9, less=False)
        b4, c4 = pick_strict_pair(0, 9, less=True)

    b = f"{d1}{b2}{b3}{b4}"
    c = f"{d1}{c2}{c3}{c4}"
    return b, c

def make_example_b3eq_b1diff():
    # b3 == c3 (0..9), b1 != c1 (1..9)
    d3 = random.randint(0, 9)
    b1 = random.randint(1, 9)
    c1 = random.choice([x for x in range(1, 10) if x != b1])

    if random.random() < 0.5:
        b2, c2 = pick_strict_pair(0, 9, less=True)
        b4, c4 = pick_strict_pair(0, 9, less=False)
    else:
        b2, c2 = pick_strict_pair(0, 9, less=False)
        b4, c4 = pick_strict_pair(0, 9, less=True)

    b = f"{b1}{b2}{d3}{b4}"
    c = f"{c1}{c2}{d3}{c4}"
    return b, c

def make_example_bothdiff():
    # b1 != c1 (1..9), b3 != c3 (0..9)
    b1 = random.randint(1, 9)
    c1 = random.choice([x for x in range(1, 10) if x != b1])
    b3 = random.randint(0, 9)
    c3 = random.choice([x for x in range(0, 10) if x != b3])

    if random.random() < 0.5:
        b2, c2 = pick_strict_pair(0, 9, less=True)
        b4, c4 = pick_strict_pair(0, 9, less=False)
    else:
        b2, c2 = pick_strict_pair(0, 9, less=False)
        b4, c4 = pick_strict_pair(0, 9, less=True)

    b = f"{b1}{b2}{b3}{b4}"
    c = f"{c1}{c2}{c3}{c4}"
    return b, c

def write_file(path: Path, maker_func, n=NUM_EXAMPLES):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            b, c = maker_func()
            left = f"1000,{b},{c},9999"
            nums = [1000, int(b), int(c), 9999]
            nums_sorted = sorted(nums)
            right = ",".join(str(x) for x in nums_sorted)
            line = f"{left}={right}$\n"
            f.write(line)

def main(out_dir=".", seed=None, n=NUM_EXAMPLES):
    if seed is not None:
        random.seed(seed)

    out_dir = Path(out_dir)
    write_file(out_dir / "b1_eq_b3diff.txt", make_example_b1eq_b3diff, n)
    write_file(out_dir / "b3_eq_b1diff.txt", make_example_b3eq_b1diff, n)
    write_file(out_dir / "b1c1_b3c3_bothdiff.txt", make_example_bothdiff, n)
    print(f"Wrote {n} examples each to:")
    print(f" - {out_dir / 'b1_eq_b3diff.txt'}")
    print(f" - {out_dir / 'b3_eq_b1diff.txt'}")
    print(f" - {out_dir / 'b1c1_b3c3_bothdiff.txt'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate 3 test files with specified digit constraints.")
    p.add_argument("--outdir", default=".", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Optional random seed for reproducibility")
    p.add_argument("--n", type=int, default=NUM_EXAMPLES, help="Number of examples per file")
    args = p.parse_args()
    main(out_dir=args.outdir, seed=args.seed, n=args.n)
