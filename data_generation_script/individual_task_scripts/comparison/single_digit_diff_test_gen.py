#!/usr/bin/env python3
"""
Generate 4 .txt files (1000 examples each) with specific digit-difference constraints.

Line format:
    a1a2a3a4,b1b2b3b4#>$

Files:
- thousands_diff_only.txt : a1 != b1, a2 = b2, a3 = b3, a4 = b4
- hundreds_diff_only.txt  : a1 = b1, a2 != b2, a3 = b3, a4 = b4
- tens_diff_only.txt      : a1 = b1, a2 = b2, a3 != b3, a4 = b4
- units_diff_only.txt     : a1 = b1, a2 = b2, a3 = b3, a4 != b4

All sampling is uniform conditioned on the given constraints.
"""
import random
import argparse
from pathlib import Path

def digits_to_int(d):
    return d[0]*1000 + d[1]*100 + d[2]*10 + d[3]

def rand_first():
    return random.randint(1, 9)

def rand_digit():
    return random.randint(0, 9)

def format_example(a, b):
    if a > b:
        cmp_sym = '>'
    elif a < b:
        cmp_sym = '<'
    else:
        cmp_sym = '='
    return f"{a},{b}#{cmp_sym}$"

def sample_thousands_diff_only():
    # a1 != b1, a2=b2, a3=b3, a4=b4
    a1 = rand_first()
    b1_choices = [d for d in range(1, 10) if d != a1]
    b1 = random.choice(b1_choices)
    shared_d2 = rand_digit()
    shared_d3 = rand_digit()
    shared_d4 = rand_digit()
    a = digits_to_int([a1, shared_d2, shared_d3, shared_d4])
    b = digits_to_int([b1, shared_d2, shared_d3, shared_d4])
    return a, b

def sample_hundreds_diff_only():
    # a1=b1, a2 != b2, a3=b3, a4=b4
    d0 = rand_first()
    a2 = rand_digit()
    b2_choices = [d for d in range(0, 10) if d != a2]
    b2 = random.choice(b2_choices)
    shared_d3 = rand_digit()
    shared_d4 = rand_digit()
    a = digits_to_int([d0, a2, shared_d3, shared_d4])
    b = digits_to_int([d0, b2, shared_d3, shared_d4])
    return a, b

def sample_tens_diff_only():
    # a1=b1, a2=b2, a3 != b3, a4=b4
    d0 = rand_first()
    d1 = rand_digit()
    a3 = rand_digit()
    b3_choices = [d for d in range(0, 10) if d != a3]
    b3 = random.choice(b3_choices)
    shared_d4 = rand_digit()
    a = digits_to_int([d0, d1, a3, shared_d4])
    b = digits_to_int([d0, d1, b3, shared_d4])
    return a, b

def sample_units_diff_only():
    # a1=b1, a2=b2, a3=b3, a4 != b4
    d0 = rand_first()
    d1 = rand_digit()
    d2 = rand_digit()
    a4 = rand_digit()
    b4_choices = [d for d in range(0, 10) if d != a4]
    b4 = random.choice(b4_choices)
    a = digits_to_int([d0, d1, d2, a4])
    b = digits_to_int([d0, d1, d2, b4])
    return a, b

SAMPLERS = {
    "thousands_diff_only.txt": sample_thousands_diff_only,
    "hundreds_diff_only.txt": sample_hundreds_diff_only,
    "tens_diff_only.txt": sample_tens_diff_only,
    "units_diff_only.txt": sample_units_diff_only,
}

def generate_file(path: Path, sampler, n=1000):
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            a, b = sampler()
            f.write(format_example(a, b) + "\n")

def quick_stats(path: Path):
    eq = gt = lt = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", "-o", default=".", help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    parser.add_argument("--n", type=int, default=1000, help="examples per file (default 1000)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fname, sampler in SAMPLERS.items():
        generate_file(outdir / fname, sampler, n=args.n)

    print("Generated files:")
    for fname in SAMPLERS.keys():
        eq, gt, lt = quick_stats(outdir / fname)
        print(f"  {fname}: = {eq}, > {gt}, < {lt}")

if __name__ == "__main__":
    main()
