#!/usr/bin/env python3
"""
multiplication_permutation.py

Generate (a * b) examples in the format:
  a*b=<permuted 41-digit product>$

Changes vs the original:
- The left operand `a` is now sampled with a *balanced* (triangular) digit-length
  distribution over 1..max_digits (default max_digits=40), like Code 2:
    P(len=k) ∝ k
  Then a is sampled uniformly among integers with that many digits.
- The right operand `b` remains 1-digit (0..9 by default; 1..9 with --no-zero-multiplier).

Permutation behavior preserved:
- The product is formatted as a zero-padded 41-digit string, then permuted by a
  user-supplied permutation of 1..41.

To recover the old behavior (always 40-digit a):
- Use --fixed-a-digits 40
"""

import argparse
import random
import re
import sys
from typing import List, Optional, Tuple


MAX_SUPPORTED_DIGITS = 128


def parse_perm(raw: str) -> List[int]:
    """
    Accept permutations in forms like:
      - "1,2,3,...,41" (commas)
      - "1 2 3 ... 41" (spaces)
      - "1-2-3-...-41" (other non-digits)
      - "123...41" ONLY if unambiguous as 1..41 tokens (we tokenize greedily)
    Returns a list of ints (1-based positions).
    """
    s = raw.strip()

    # If it contains any non-digit separators, split on non-digits.
    if re.search(r"\D", s):
        toks = [t for t in re.split(r"\D+", s) if t]
        if not toks:
            raise ValueError("Permutation could not be parsed.")
        return [int(t) for t in toks]

    # Otherwise it's digits only. Tokenize greedily into 1..41 (10..41 preferred).
    perm: List[int] = []
    i = 0
    while i < len(s):
        if i + 1 < len(s):
            two = int(s[i : i + 2])
            if 10 <= two <= 41:
                perm.append(two)
                i += 2
                continue
        perm.append(int(s[i]))
        i += 1
    return perm


def validate_perm(perm: List[int], width: int = 41) -> None:
    if len(perm) != width:
        raise ValueError(f"Permutation must contain exactly {width} positions (got {len(perm)}).")
    needed = set(range(1, width + 1))
    got = set(perm)
    if got != needed:
        missing = sorted(needed - got)
        extra = sorted(got - needed)
        raise ValueError(
            f"Permutation must be a permutation of 1..{width}.\n"
            f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"Extra:   {extra[:10]}{'...' if len(extra) > 10 else ''}"
        )


def permute_fixed_width_number(n: int, perm: List[int], width: int = 41) -> str:
    """Zero-pad n to `width` digits, then permute digits by 1-based `perm`."""
    s = f"{n:0{width}d}"
    return "".join(s[p - 1] for p in perm)


# ---------------- Balanced sampling (triangular digit-length distribution) ----------------

def _triangular_sample_length(rng: random.Random, max_digits: int) -> int:
    """
    Sample length in 1..max_digits with weights 1,2,3,...,max_digits.
    Equivalent to Code 2's probabilities_from_max_digits.
    """
    total = max_digits * (max_digits + 1) // 2
    draw = rng.randint(1, total)
    acc = 0
    for length in range(1, max_digits + 1):
        acc += length
        if draw <= acc:
            return length
    return max_digits  # defensive fallback


def _sample_value_with_length(rng: random.Random, length: int) -> int:
    """Uniformly sample an integer with exactly `length` digits (1-digit includes 0..9)."""
    if length <= 0:
        raise ValueError("length must be positive")
    low = 0 if length == 1 else 10 ** (length - 1)
    high = 10 ** length - 1
    return rng.randint(low, high)


def sample_a_value(
    rng: random.Random,
    *,
    max_digits: int,
    fixed_digits: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Return (len_a, a) where len_a is the chosen digit length.
    - If fixed_digits is provided, always use that digit length.
    - Else, sample len_a from triangular distribution 1..max_digits.
    """
    if fixed_digits is not None:
        if fixed_digits <= 0 or fixed_digits > MAX_SUPPORTED_DIGITS:
            raise ValueError(f"--fixed-a-digits must be in 1..{MAX_SUPPORTED_DIGITS}")
        length = fixed_digits
    else:
        if max_digits <= 0 or max_digits > MAX_SUPPORTED_DIGITS:
            raise ValueError(f"--max-digits must be in 1..{MAX_SUPPORTED_DIGITS}")
        length = _triangular_sample_length(rng, max_digits)
    return length, _sample_value_with_length(rng, length)


def _format_length_counts(counts: List[int]) -> str:
    total = sum(counts)
    if total == 0:
        return "no samples"
    parts = []
    for idx, c in enumerate(counts, start=1):
        if c:
            parts.append(f"{idx}-digit: {c} ({100.0*c/total:.2f}%)")
    return ", ".join(parts)


def generate_examples_to_file(
    path: str,
    n: int,
    perm: List[int],
    rng: random.Random,
    allow_zero_multiplier: bool,
    *,
    a_max_digits: int = 40,
    fixed_a_digits: Optional[int] = None,
    report_counts: bool = False,
) -> Optional[List[int]]:
    """
    Generate `n` examples to `path`.

    a sampling:
      - balanced digit-length distribution over 1..a_max_digits (triangular),
        unless fixed_a_digits is set.

    b sampling:
      - 0..9 if allow_zero_multiplier else 1..9

    Output:
      - product padded to 41 digits, permuted by `perm`.
    """
    counts_a = [0] * (fixed_a_digits if fixed_a_digits is not None else a_max_digits)

    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            len_a, a = sample_a_value(rng, max_digits=a_max_digits, fixed_digits=fixed_a_digits)

            if allow_zero_multiplier:
                b = rng.randrange(0, 10)  # 0..9
            else:
                b = rng.randrange(1, 10)  # 1..9

            product = a * b
            permuted_product = permute_fixed_width_number(product, perm, width=41)
            f.write(f"{a}*{b}={permuted_product}$\n")

            # counts indexing: length starts at 1
            if 1 <= len_a <= len(counts_a):
                counts_a[len_a - 1] += 1

    if report_counts:
        return counts_a
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multiplication data with permuted (41-digit padded) products and balanced digit-length sampling for operand a."
    )
    parser.add_argument(
        "perm",
        type=str,
        help=(
            "Permutation of positions 1..41. Examples: "
            "'1 2 3 ... 41', '1,2,3,...,41', or digits-only '123...41' (must parse to 41 tokens)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42) for reproducibility")
    parser.add_argument("--out-prefix", type=str, default="", help="Optional prefix for output files (default: none)")
    parser.add_argument(
        "--no-zero-multiplier",
        action="store_true",
        help="If set, multiplier b is restricted to 1..9 (default allows 0..9).",
    )

    # NEW: balanced sampling controls (Code 2 style)
    parser.add_argument(
        "--max-digits",
        type=int,
        default=40,
        help=(
            "Max digit length for operand a in balanced (triangular) sampling over 1..max_digits "
            "(default: 40). Ignored if --fixed-a-digits is set."
        ),
    )
    parser.add_argument(
        "--fixed-a-digits",
        type=int,
        default=None,
        help="If set, always generate operand a with exactly this many digits (e.g. 40 to match old behavior).",
    )

    args = parser.parse_args()

    try:
        perm = parse_perm(args.perm)
        validate_perm(perm, width=41)
    except ValueError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)

    train_path = f"{args.out_prefix}train.txt"
    val_path = f"{args.out_prefix}val.txt"
    test_path = f"{args.out_prefix}test.txt"

    allow_zero = not args.no_zero_multiplier

    print(
        f"Generating files with permutation (1..41), seed={args.seed}, "
        f"b={'0..9' if allow_zero else '1..9'}, "
        f"a_sampling={'fixed ' + str(args.fixed_a_digits) if args.fixed_a_digits is not None else 'balanced 1..' + str(args.max_digits)}..."
    )

    train_counts = generate_examples_to_file(
        train_path, 1_000_000, perm, rng, allow_zero,
        a_max_digits=args.max_digits,
        fixed_a_digits=args.fixed_a_digits,
        report_counts=True
    )
    val_counts = generate_examples_to_file(
        val_path, 10_000, perm, rng, allow_zero,
        a_max_digits=args.max_digits,
        fixed_a_digits=args.fixed_a_digits,
        report_counts=True
    )
    test_counts = generate_examples_to_file(
        test_path, 10_000, perm, rng, allow_zero,
        a_max_digits=args.max_digits,
        fixed_a_digits=args.fixed_a_digits,
        report_counts=True
    )

    print("Done.")
    print(f"Files: {train_path} (1000000 lines), {val_path} (10000 lines), {test_path} (10000 lines)")
    if train_counts is not None:
        print("Train a-length distribution:", _format_length_counts(train_counts))
    if val_counts is not None:
        print("Val a-length distribution:", _format_length_counts(val_counts))
    if test_counts is not None:
        print("Test a-length distribution:", _format_length_counts(test_counts))


if __name__ == "__main__":
    main()
