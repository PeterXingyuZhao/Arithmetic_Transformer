#!/usr/bin/env python3
"""
Generate 9 mixture files from two source files:
 - conflicting_theta_1.txt  (the "conflicting equality" file)
 - conflicting_theta_0.txt  (the "same direction inequality" file)

Output:
 - combined_theta_0.1.txt, combined_theta_0.2.txt, ..., combined_theta_0.9.txt

Each output file contains `total` examples where
 total = min(len(file1), len(file2))
and for theta, we take round(theta * total) examples from file1 and the rest from file2.
"""
from pathlib import Path
import random
import argparse
import math

def read_lines_strip(path: Path):
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    return lines

def write_lines(path: Path, lines):
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def make_mixtures(file1_path, file0_path, thetas, out_dir=".", seed=None):
    file1 = Path(file1_path)
    file0 = Path(file0_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_lines = read_lines_strip(file1)
    b_lines = read_lines_strip(file0)

    if len(a_lines) == 0 or len(b_lines) == 0:
        raise ValueError("One of the input files is empty.")

    total = min(len(a_lines), len(b_lines))
    print(f"Found {len(a_lines)} lines in {file1.name}, {len(b_lines)} lines in {file0.name}.")
    print(f"Using total = {total} examples per mixture (min of the two files).")

    # Use a master seed for reproducibility of the whole run if requested
    if seed is not None:
        random.seed(seed)

    for theta in thetas:
        # determine counts (round to nearest integer, then adjust to ensure sum=total)
        k1 = int(round(theta * total))
        k0 = total - k1

        # If counts exceed available (shouldn't happen because total <= len(file)), guard:
        if k1 > len(a_lines):
            k1 = len(a_lines)
            k0 = total - k1
        if k0 > len(b_lines):
            k0 = len(b_lines)
            k1 = total - k0

        # Use per-theta deterministic seed if desired (keeps overall run reproducible and distinct per theta).
        # If a global seed is provided, we mix it with theta to get a per-theta variation:
        if seed is not None:
            # combine into an integer seed per theta
            per_theta_seed = seed + int(round(theta * 1000))
            rnd = random.Random(per_theta_seed)
            sample_a = rnd.sample(a_lines, k1)
            sample_b = rnd.sample(b_lines, k0)
            combined = sample_a + sample_b
            rnd.shuffle(combined)
        else:
            # nondeterministic sampling
            sample_a = random.sample(a_lines, k1)
            sample_b = random.sample(b_lines, k0)
            combined = sample_a + sample_b
            random.shuffle(combined)

        # write out with filename that shows theta with one decimal place
        out_name = f"combined_theta_{theta:.1f}.txt"
        out_path = out_dir / out_name
        write_lines(out_path, combined)
        print(f"Wrote {len(combined)} lines -> {out_path} (from file1: {k1}, file0: {k0})")

def parse_args():
    p = argparse.ArgumentParser(description="Create convex-combination mixture files from two source files.")
    p.add_argument("--file1", default="conflicting_theta_1.txt", help="Path to conflicting_theta_1.txt (first source).")
    p.add_argument("--file0", default="conflicting_theta_0.txt", help="Path to conflicting_theta_0.txt (second source).")
    p.add_argument("--outdir", default=".", help="Output directory for mixture files.")
    p.add_argument("--seed", type=int, default=42, help="Optional integer seed for reproducible sampling.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    thetas = [i / 10.0 for i in range(1, 10)]  # 0.1 .. 0.9
    make_mixtures(args.file1, args.file0, thetas, out_dir=args.outdir, seed=args.seed)
