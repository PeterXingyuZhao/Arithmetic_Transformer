#!/usr/bin/env python3
"""
Generate train/val/test files with 4-digit comparisons.

Each line format:
    a1a2a3a4,b1b2b3b4#>$

Groups (each chosen with probability 1/5 for each example):
  Group 1: no condition, sample each number uniformly from 1000..9999
  Group 2: share a1 (first digit); other digits random
  Group 3: share a1,a2
  Group 4: share a1,a2,a3
  Group 5: identical numbers (share all digits)

Usage:
    python bal_gen.py [--outdir out] [--seed 123]
"""
import random
import argparse
from pathlib import Path

def random_digit(first=False):
    """Return a digit int. If first True, return 1..9, else 0..9."""
    return random.randint(1, 9) if first else random.randint(0, 9)

def digits_to_int(digits):
    return digits[0]*1000 + digits[1]*100 + digits[2]*10 + digits[3]

def sample_group1():
    # sample whole numbers uniformly from 1000..9999 (as requested)
    a = random.randint(1000, 9999)
    b = random.randint(1000, 9999)
    return a, b

def sample_group2():
    # a1=b1, other digits independent
    d0 = random_digit(first=True)
    a_digits = [d0, random_digit(False), random_digit(False), random_digit(False)]
    b_digits = [d0, random_digit(False), random_digit(False), random_digit(False)]
    return digits_to_int(a_digits), digits_to_int(b_digits)

def sample_group3():
    # a1=b1, a2=b2
    d0 = random_digit(first=True)
    d1 = random_digit(False)
    a_digits = [d0, d1, random_digit(False), random_digit(False)]
    b_digits = [d0, d1, random_digit(False), random_digit(False)]
    return digits_to_int(a_digits), digits_to_int(b_digits)

def sample_group4():
    # a1=b1, a2=b2, a3=b3
    d0 = random_digit(first=True)
    d1 = random_digit(False)
    d2 = random_digit(False)
    a_digits = [d0, d1, d2, random_digit(False)]
    b_digits = [d0, d1, d2, random_digit(False)]
    return digits_to_int(a_digits), digits_to_int(b_digits)

def sample_group5():
    # identical numbers
    d0 = random_digit(first=True)
    d1 = random_digit(False)
    d2 = random_digit(False)
    d3 = random_digit(False)
    a_digits = [d0, d1, d2, d3]
    return digits_to_int(a_digits), digits_to_int(a_digits)

GROUP_SAMPLERS = [sample_group1, sample_group2, sample_group3, sample_group4, sample_group5]

def make_example():
    # choose group uniformly 1..5
    sampler = random.choice(GROUP_SAMPLERS)
    a, b = sampler()
    if a > b:
        comp = '>'
    elif a < b:
        comp = '<'
    else:
        comp = '='
    return f"{a},{b}#{comp}$"

def generate_file(path: Path, n_examples: int):
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n_examples):
            f.write(make_example() + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", "-o", default=".", help="output directory")
    p.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_path = outdir / "train.txt"
    val_path = outdir / "val.txt"
    test_path = outdir / "test.txt"

    print("Generating datasets...")
    generate_file(train_path, 45000)
    generate_file(val_path, 5000)
    generate_file(test_path, 5000)
    print(f"Saved: {train_path} ({train_path.stat().st_size} bytes)")
    print(f"Saved: {val_path}   ({val_path.stat().st_size} bytes)")
    print(f"Saved: {test_path}  ({test_path.stat().st_size} bytes)")
    # Optional brief counts of equality vs others
    def quick_stats(path):
        eq = 0
        gt = 0
        lt = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # format: a,b#c$
                try:
                    tail = line.split("#", 1)[1]
                except IndexError:
                    continue
                if tail.startswith("=$"):
                    eq += 1
                elif tail.startswith(">$"):
                    gt += 1
                elif tail.startswith("<$"):
                    lt += 1
        return eq, gt, lt

    for path in (train_path, val_path, test_path):
        eq, gt, lt = quick_stats(path)
        print(f"{path.name}: = {eq}, > {gt}, < {lt}")

if __name__ == "__main__":
    main()
