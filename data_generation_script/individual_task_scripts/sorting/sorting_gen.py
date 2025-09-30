#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

# defaults (same as your original constants)
DEFAULT_TRAIN_SIZE = 1_000_000
DEFAULT_TEST_SIZE  = 10_000
DEFAULT_VAL_SIZE   = 10_000
DEFAULT_OUTPUT_DIR = "."
SEED = 42

def sample_number():
    """
    Sample one number whose length is chosen uniformly from {1,2,3,4}.
    For length L:
      L=1 -> range 0..9
      L=2 -> range 10..99
      L=3 -> range 100..999
      L=4 -> range 1000..9999
    """
    L = random.choice([1, 2, 3, 4])
    if L == 1:
        low, high = 0, 9
    else:
        low, high = 10**(L-1), 10**L - 1
    return random.randint(low, high)

def fmt_sort(a: int, b: int, c: int, d: int) -> str:
    """Return a line of the form: 'a,b,c,d=sa,sb,sc,sd$' where s* are ascending-sorted values."""
    sorted_vals = sorted([a, b, c, d])
    sorted_str = ",".join(str(x) for x in sorted_vals)
    return f"{a},{b},{c},{d}={sorted_str}$"

def parse_args():
    p = argparse.ArgumentParser(description="Generate sorting dataset (4 numbers -> sorted).")
    p.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE,
                   help=f"Number of training examples (default: {DEFAULT_TRAIN_SIZE})")
    p.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE,
                   help=f"Number of test examples (default: {DEFAULT_TEST_SIZE})")
    p.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE,
                   help=f"Number of validation examples (default: {DEFAULT_VAL_SIZE})")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Directory to write train.txt, test.txt, val.txt (default: current directory).")
    return p.parse_args()

def main():
    args = parse_args()

    TRAIN_SIZE = args.train_size
    TEST_SIZE  = args.test_size
    VAL_SIZE   = args.val_size
    OUT_DIR    = Path(args.output_dir)

    # basic validation
    if TRAIN_SIZE < 0 or TEST_SIZE < 0 or VAL_SIZE < 0:
        raise ValueError("Sizes must be non-negative integers.")

    total_needed = TRAIN_SIZE + TEST_SIZE + VAL_SIZE

    # ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    samples = []
    for _ in range(total_needed):
        a = sample_number()
        b = sample_number()
        c = sample_number()
        d = sample_number()
        samples.append((a, b, c, d))

    train_samples = samples[:TRAIN_SIZE]
    test_samples  = samples[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    val_samples   = samples[TRAIN_SIZE + TEST_SIZE:]

    train_path = OUT_DIR / "train.txt"
    test_path  = OUT_DIR / "test.txt"
    val_path   = OUT_DIR / "val.txt"

    with train_path.open("w") as f_train, \
         test_path.open("w")  as f_test, \
         val_path.open("w")   as f_val:

        for a, b, c, d in train_samples:
            f_train.write(fmt_sort(a, b, c, d) + "\n")

        for a, b, c, d in test_samples:
            f_test.write(fmt_sort(a, b, c, d) + "\n")

        for a, b, c, d in val_samples:
            f_val.write(fmt_sort(a, b, c, d) + "\n")

    print(f"Wrote {TRAIN_SIZE} train -> {train_path}, {TEST_SIZE} test -> {test_path}, {VAL_SIZE} val -> {val_path}.")

if __name__ == "__main__":
    main()
