#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

# defaults (same as your original constants)
DEFAULT_TRAIN_SIZE = 1_000_000
DEFAULT_TEST_SIZE  = 10_000
DEFAULT_VAL_SIZE   = 10_000
DEFAULT_NUM_OPERANDS = 4
SEED = 42

def decode_idx(idx: int, num_operands: int):
    """Turn an integer 0 <= idx < 1000**num_operands into num_operands 3-digit numbers.
    Returns a tuple of length num_operands (each in 0..999)."""
    if num_operands < 1:
        raise ValueError("num_operands must be >= 1")
    parts = []
    for _ in range(num_operands - 1):
        parts.append(idx % 1000)
        idx //= 1000
    parts.append(idx)  # remaining most-significant part
    parts.reverse()
    return tuple(parts)

def fmt(operands):
    """Format sequence of operands into 'a+b+...=sum$'"""
    s = "+".join(str(x) for x in operands)
    return f"{s}={sum(operands)}$"

def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset files of 'a+b+...=sum$' examples.")
    p.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE, help="Number of training examples")
    p.add_argument("--test_size",  type=int, default=DEFAULT_TEST_SIZE,  help="Number of test examples")
    p.add_argument("--val_size",   type=int, default=DEFAULT_VAL_SIZE,   help="Number of validation examples")
    p.add_argument("--num_operands", type=int, default=DEFAULT_NUM_OPERANDS,
                   help="Number of operands (each operand is a 3-digit number in 0..999)")
    p.add_argument("--output_dir", type=str, default=".", help="Directory to write train.txt, test.txt, val.txt")
    return p.parse_args()

def main():
    args = parse_args()

    TRAIN_SIZE = args.train_size
    TEST_SIZE  = args.test_size
    VAL_SIZE   = args.val_size
    NUM_OPERANDS = args.num_operands
    OUT_DIR    = Path(args.output_dir)

    # basic validation
    if TRAIN_SIZE < 0 or TEST_SIZE < 0 or VAL_SIZE < 0:
        raise ValueError("Sizes must be non-negative integers.")
    if NUM_OPERANDS < 1:
        raise ValueError("num_operands must be at least 1.")

    total_needed = TRAIN_SIZE + TEST_SIZE + VAL_SIZE
    max_available = pow(1000, NUM_OPERANDS)  # 1000**num_operands unique indices
    if total_needed > max_available:
        raise ValueError(f"Requested total {total_needed} exceeds available unique indices {max_available} for {NUM_OPERANDS} operands.")

    # ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    # Sample unique indices in [0, max_available)
    sampled = random.sample(range(max_available), total_needed)

    train_idx = sampled[:TRAIN_SIZE]
    test_idx  = sampled[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    val_idx   = sampled[TRAIN_SIZE+TEST_SIZE:]

    train_path = OUT_DIR / "train.txt"
    test_path  = OUT_DIR / "test.txt"
    val_path   = OUT_DIR / "val.txt"

    with train_path.open("w") as f_train, \
         test_path.open("w")  as f_test, \
         val_path.open("w")   as f_val:

        for idx in train_idx:
            f_train.write(fmt(decode_idx(idx, NUM_OPERANDS)) + "\n")

        for idx in test_idx:
            f_test.write(fmt(decode_idx(idx, NUM_OPERANDS)) + "\n")

        for idx in val_idx:
            f_val.write(fmt(decode_idx(idx, NUM_OPERANDS)) + "\n")

    print(f"Wrote {TRAIN_SIZE} train -> {train_path}, {TEST_SIZE} test -> {test_path}, {VAL_SIZE} val -> {val_path} (num_operands={NUM_OPERANDS}).")

if __name__ == "__main__":
    main()
