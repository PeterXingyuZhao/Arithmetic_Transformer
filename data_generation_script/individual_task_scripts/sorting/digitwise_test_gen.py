#!/usr/bin/env python3
"""
Generate sorting test examples in several variants.

Line format:
3060,4901,9514,2577=2577,3060,4901,9514$

Variants:
  random                      : each number uniform from 1000..9999
  thousands                   : all 4 share thousands digit X (1..9) -> X000..X999
  thousands_hundreds          : share thousands and hundreds -> XY00..XY99
  thousands_hundreds_tens     : share thousands, hundreds, tens -> XYZ0..XYZ9

Usage examples:
  python gen_sort_examples.py --variant thousands -n 3000 --out sorting_thousands.txt
  python gen_sort_examples.py --variant random    -n 3000 --distinct
"""
import random
import argparse
from typing import Tuple

def block_range_for_variant(variant: str) -> Tuple[int, int]:
    """
    Return a tuple (min_block_size, max_block_size) for basic validation.
    Not strictly required by main logic — kept for clarity.
    """
    if variant == "random":
        return 1000, 9999  # unused in code but informative
    # others handled in make_example_variant
    return 0, 0

def make_example_variant(variant: str = "random", allow_duplicates: bool = True) -> str:
    """
    Create one example string according to variant.
    variant in {"random", "thousands", "thousands_hundreds", "thousands_hundreds_tens"}
    allow_duplicates: if False, numbers in the example are distinct (sample without replacement)
    """
    if variant == "random":
        lo, hi = 1000, 9999
    else:
        # pick digits according to variant
        thousands = random.randint(1, 9)  # equal chance 1..9
        if variant == "thousands":
            lo = thousands * 1000
            hi = lo + 999
        elif variant == "thousands_hundreds":
            hundreds = random.randint(0, 9)
            lo = thousands * 1000 + hundreds * 100
            hi = lo + 99
        elif variant == "thousands_hundreds_tens":
            hundreds = random.randint(0, 9)
            tens = random.randint(0, 9)
            lo = thousands * 1000 + hundreds * 100 + tens * 10
            hi = lo + 9
        else:
            raise ValueError(f"Unknown variant: {variant}")

    block_size = hi - lo + 1
    if not allow_duplicates and block_size < 4:
        raise ValueError(f"Block size {block_size} too small to sample 4 distinct numbers for variant '{variant}'")

    if allow_duplicates:
        nums = [random.randint(lo, hi) for _ in range(4)]
    else:
        # sample without replacement
        nums = random.sample(range(lo, hi + 1), 4)

    left = ",".join(str(n) for n in nums)
    right = ",".join(str(n) for n in sorted(nums))
    return f"{left}={right}$"

def generate_examples(n: int = 3000, variant: str = "random", filename: str = "sorting_examples.txt",
                      seed: int = None, allow_duplicates: bool = True) -> None:
    if seed is not None:
        random.seed(seed)

    valid_variants = {"random", "thousands", "thousands_hundreds", "thousands_hundreds_tens"}
    if variant not in valid_variants:
        raise ValueError(f"variant must be one of {sorted(valid_variants)}")

    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(make_example_variant(variant=variant, allow_duplicates=allow_duplicates) + "\n")

    print(f"Wrote {n} examples (variant='{variant}', distinct={not allow_duplicates}) to '{filename}'")

def main():
    parser = argparse.ArgumentParser(description="Generate sorting test examples (multiple variants).")
    parser.add_argument("--variant", type=str, default="random",
                        choices=["random", "thousands", "thousands_hundreds", "thousands_hundreds_tens"],
                        help="which variant to generate (default: random)")
    parser.add_argument("-n", type=int, default=3000, help="number of examples (default 3000)")
    parser.add_argument("--distinct", action="store_true",
                        help="force the 4 numbers in each example to be distinct (no duplicates)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--out", type=str, default="sorting_examples.txt",
                        help="output filename (default sorting_examples.txt)")
    args = parser.parse_args()

    generate_examples(n=args.n, variant=args.variant, filename=args.out,
                      seed=args.seed, allow_duplicates=not args.distinct)

if __name__ == "__main__":
    main()
