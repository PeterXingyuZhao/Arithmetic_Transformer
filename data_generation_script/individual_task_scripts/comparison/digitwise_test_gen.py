#!/usr/bin/env python3
"""
Generate 9 .txt files, each containing 1000 comparison examples.

Line format:
    a1a2a3a4,b1b2b3b4#>$

Files and sampling rules:
- thousands.txt:         no condition, sample each number uniformly from 1000..9999
- thousands_strict.txt:  a1 != b1 (first digits different)
- hundreds.txt:          a1 = b1
- hundreds_strict.txt:   a1 = b1, a2 != b2
- tens.txt:              a1 = b1, a2 = b2
- tens_strict.txt:       a1 = b1, a2 = b2, a3 != b3
- units.txt:             a1 = b1, a2 = b2, a3 = b3 (last digits sampled freely)
- units_strict.txt:      a1 = b1, a2 = b2, a3 = b3, a4 != b4
- equal.txt:             identical numbers (all digits equal)

Usage:
    python digitwise_test_gen.py [--outdir out] [--seed 123]
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
        cmp = '>'
    elif a < b:
        cmp = '<'
    else:
        cmp = '='
    return f"{a},{b}#{cmp}$"

# Samplers for each file
def sample_thousands():
    a = random.randint(1000, 9999)
    b = random.randint(1000, 9999)
    return a, b

def sample_thousands_strict():
    # a1 != b1
    a1 = rand_first()
    b1_choices = [d for d in range(1, 10) if d != a1]
    b1 = random.choice(b1_choices)
    a = digits_to_int([a1, rand_digit(), rand_digit(), rand_digit()])
    b = digits_to_int([b1, rand_digit(), rand_digit(), rand_digit()])
    return a, b

def sample_hundreds():
    # a1 = b1
    d0 = rand_first()
    a = digits_to_int([d0, rand_digit(), rand_digit(), rand_digit()])
    b = digits_to_int([d0, rand_digit(), rand_digit(), rand_digit()])
    return a, b

def sample_hundreds_strict():
    # a1=b1, a2 != b2
    d0 = rand_first()
    a2 = rand_digit()
    b2_choices = [d for d in range(0, 10) if d != a2]
    b2 = random.choice(b2_choices)
    a = digits_to_int([d0, a2, rand_digit(), rand_digit()])
    b = digits_to_int([d0, b2, rand_digit(), rand_digit()])
    return a, b

def sample_tens():
    # a1=b1, a2=b2
    d0 = rand_first()
    d1 = rand_digit()
    a = digits_to_int([d0, d1, rand_digit(), rand_digit()])
    b = digits_to_int([d0, d1, rand_digit(), rand_digit()])
    return a, b

def sample_tens_strict():
    # a1=b1, a2=b2, a3 != b3
    d0 = rand_first()
    d1 = rand_digit()
    a3 = rand_digit()
    b3_choices = [d for d in range(0, 10) if d != a3]
    b3 = random.choice(b3_choices)
    a = digits_to_int([d0, d1, a3, rand_digit()])
    b = digits_to_int([d0, d1, b3, rand_digit()])
    return a, b

def sample_units():
    # a1=b1, a2=b2, a3=b3, last digits sampled freely (can be equal)
    d0 = rand_first()
    d1 = rand_digit()
    d2 = rand_digit()
    a = digits_to_int([d0, d1, d2, rand_digit()])
    b = digits_to_int([d0, d1, d2, rand_digit()])
    return a, b

def sample_units_strict():
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

def sample_equal():
    d0 = rand_first()
    d1 = rand_digit()
    d2 = rand_digit()
    d3 = rand_digit()
    a = digits_to_int([d0, d1, d2, d3])
    return a, a

SAMPLERS = {
    "thousands.txt": sample_thousands,
    "thousands_strict.txt": sample_thousands_strict,
    "hundreds.txt": sample_hundreds,
    "hundreds_strict.txt": sample_hundreds_strict,
    "tens.txt": sample_tens,
    "tens_strict.txt": sample_tens_strict,
    "units.txt": sample_units,
    "units_strict.txt": sample_units_strict,
    "equal.txt": sample_equal,
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
    parser.add_argument("--outdir", "-o", default=".", help="output directory for files")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating 9 files (1000 examples each)...")
    for fname, sampler in SAMPLERS.items():
        path = outdir / fname
        generate_file(path, sampler, n=1000)

    print("Done. Quick stats:")
    for fname in SAMPLERS.keys():
        path = outdir / fname
        eq, gt, lt = quick_stats(path)
        print(f"{fname}: = {eq}, > {gt}, < {lt}")

if __name__ == "__main__":
    main()
