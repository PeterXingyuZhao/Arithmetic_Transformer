#!/usr/bin/env python3
"""Utility for generating multiplication datasets with controllable digit lengths.

Examples are written in the form ``a*b=c$``.  The digit-length for each operand
is sampled independently.  Users can supply explicit probability lists
(``--a_probabilities``/``--b_probabilities``) or request triangular
probabilities derived from maximum digit lengths
(``--a_max_digits``/``--b_max_digits``).  Once a digit-length has been selected,
its value is sampled uniformly from the integers that contain that many digits.

The script mirrors the interface of the other generators in this project: it
accepts train/validation/test sizes, a random seed, and an output directory. If
no probability or max-digit configuration is provided for an operand, it
defaults to sampling single-digit numbers via ``_probabilities_from_max_digits``.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DEFAULT_TRAIN_SIZE = 10_000
DEFAULT_VAL_SIZE = 3_000
DEFAULT_TEST_SIZE = 3_000
DEFAULT_OUTPUT_DIR = "."
DEFAULT_SEED = 42
MAX_SUPPORTED_DIGITS = 128


class ProbabilityParseError(ValueError):
    """Raised when a probability specification is invalid."""


@dataclass(frozen=True)
class DigitLengthDistribution:
    """Distribution over digit lengths 1..N with cached sampling metadata."""

    probabilities: Tuple[float, ...]
    cumulative: Tuple[float, ...]
    ranges: Tuple[Tuple[int, int], ...]

    @property
    def num_lengths(self) -> int:
        return len(self.probabilities)

    def sample_length(self, rng: random.Random) -> int:
        """Draw a digit length in ``1..num_lengths`` according to the distribution."""

        draw = rng.random()
        for idx, threshold in enumerate(self.cumulative):
            # Guard against floating point round-off by returning the last bucket.
            if draw < threshold or idx == self.num_lengths - 1:
                return idx + 1
        return self.num_lengths

    def sample_value(self, rng: random.Random) -> Tuple[int, int]:
        """Sample a value and report the digit-length used for bookkeeping."""

        length = self.sample_length(rng)
        low, high = self.ranges[length - 1]
        value = rng.randint(low, high)
        return length, value


def _normalize_probabilities(values: Sequence[float]) -> List[float]:
    if not values:
        raise ProbabilityParseError("Probability list cannot be empty.")
    if len(values) > MAX_SUPPORTED_DIGITS:
        raise ProbabilityParseError(
            f"At most {MAX_SUPPORTED_DIGITS} probabilities are supported; received {len(values)}."
        )
    if any(v < 0 for v in values):
        raise ProbabilityParseError("Probabilities must be non-negative numbers.")

    total = sum(values)
    if total <= 0:
        raise ProbabilityParseError("Probability list must contain at least one positive value.")

    return [v / total for v in values]


def _build_distribution(probabilities: Sequence[float]) -> DigitLengthDistribution:
    probs = tuple(_normalize_probabilities(probabilities))
    cumulative: List[float] = []
    running = 0.0
    for p in probs:
        running += p
        cumulative.append(running)

    ranges: List[Tuple[int, int]] = []
    for idx in range(len(probs)):
        digits = idx + 1
        low = 0 if digits == 1 else 10 ** (digits - 1)
        high = 10 ** digits - 1
        ranges.append((low, high))

    return DigitLengthDistribution(
        probabilities=probs,
        cumulative=tuple(cumulative),
        ranges=tuple(ranges),
    )


def _parse_probability_string(raw: str, *, flag_name: str) -> List[float]:
    if raw is None:
        raise ProbabilityParseError(f"Missing probability specification for {flag_name}.")

    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not pieces:
        raise ProbabilityParseError(f"No probabilities provided for {flag_name}.")

    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ProbabilityParseError(f"Failed to parse probabilities for {flag_name}: {exc}") from exc

    return values


def _probabilities_from_max_digits(max_digits: int) -> List[float]:
    if max_digits is None:
        raise ProbabilityParseError("max_digits must be provided.")
    if max_digits <= 0:
        raise ProbabilityParseError("max_digits must be a positive integer.")
    if max_digits > MAX_SUPPORTED_DIGITS:
        raise ProbabilityParseError(
            f"max_digits cannot exceed {MAX_SUPPORTED_DIGITS}; received {max_digits}."
        )

    weights = list(range(1, max_digits + 1))
    total = max_digits * (max_digits + 1) / 2
    return [weight / total for weight in weights]


def _resolve_operand_probabilities(
    probabilities: Optional[Sequence[float]],
    max_digits: Optional[int],
    *,
    operand_name: str,
    default_max_digits: int,
) -> Sequence[float]:
    if probabilities is not None and max_digits is not None:
        raise ProbabilityParseError(
            f"Provide either --{operand_name}_probabilities or --{operand_name}_max_digits, not both."
        )

    if probabilities is not None:
        return probabilities

    if max_digits is None:
        max_digits = default_max_digits

    return _probabilities_from_max_digits(max_digits)


def _resolve_probability_sources(
    a_probabilities: Optional[Sequence[float]],
    b_probabilities: Optional[Sequence[float]],
    a_max_digits: Optional[int],
    b_max_digits: Optional[int],
    *,
    default_a_max_digits: int,
    default_b_max_digits: int,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Determine the operand distributions based on the provided inputs."""

    resolved_a = _resolve_operand_probabilities(
        a_probabilities,
        a_max_digits,
        operand_name="a",
        default_max_digits=default_a_max_digits,
    )
    resolved_b = _resolve_operand_probabilities(
        b_probabilities,
        b_max_digits,
        operand_name="b",
        default_max_digits=default_b_max_digits,
    )
    return resolved_a, resolved_b


def _format_length_counts(counts: Sequence[int]) -> str:
    total = sum(counts)
    if total == 0:
        return "no samples"

    parts = []
    for idx, count in enumerate(counts, start=1):
        if count == 0:
            continue
        percentage = 100.0 * count / total
        parts.append(f"{idx}-digit: {count} ({percentage:.2f}%)")
    return ", ".join(parts)


def _write_dataset(
    path: Path,
    num_examples: int,
    rng: random.Random,
    a_dist: DigitLengthDistribution,
    b_dist: DigitLengthDistribution,
) -> Tuple[List[int], List[int]]:
    counts_a = [0] * a_dist.num_lengths
    counts_b = [0] * b_dist.num_lengths

    with path.open("w", encoding="utf-8") as handle:
        for _ in range(num_examples):
            len_a, a_value = a_dist.sample_value(rng)
            len_b, b_value = b_dist.sample_value(rng)
            product = a_value * b_value

            handle.write(f"{a_value}*{b_value}={product}$\n")

            counts_a[len_a - 1] += 1
            counts_b[len_b - 1] += 1

    return counts_a, counts_b


def make_dataset(
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    output_dir: str,
    seed: int,
    a_probabilities: Optional[Sequence[float]] = None,
    b_probabilities: Optional[Sequence[float]] = None,
    a_max_digits: Optional[int] = None,
    b_max_digits: Optional[int] = None,
    max_digits: Optional[int] = None,
) -> None:
    rng = random.Random(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if max_digits is not None:
        if a_probabilities is None and a_max_digits is None:
            a_max_digits = max_digits
        if b_probabilities is None and b_max_digits is None:
            b_max_digits = max_digits

    resolved_a, resolved_b = _resolve_probability_sources(
        a_probabilities,
        b_probabilities,
        a_max_digits,
        b_max_digits,
        default_a_max_digits=1,
        default_b_max_digits=1,
    )

    a_distribution = _build_distribution(resolved_a)
    b_distribution = _build_distribution(resolved_b)

    print("Generating multiplication dataset...")
    print(f"  Train size: {train_size}")
    print(f"  Validation size: {val_size}")
    print(f"  Test size: {test_size}")
    print(f"  Output directory: {out_path.resolve()}")
    print(f"  Seed: {seed}")
    print(f"  'a' digit lengths supported: {a_distribution.num_lengths} (1..{a_distribution.num_lengths})")
    print(f"  'b' digit lengths supported: {b_distribution.num_lengths} (1..{b_distribution.num_lengths})")

    train_counts_a, train_counts_b = _write_dataset(
        out_path / "train.txt", train_size, rng, a_distribution, b_distribution
    )
    val_counts_a, val_counts_b = _write_dataset(
        out_path / "val.txt", val_size, rng, a_distribution, b_distribution
    )
    test_counts_a, test_counts_b = _write_dataset(
        out_path / "test.txt", test_size, rng, a_distribution, b_distribution
    )

    print("Finished writing datasets.")
    print("  Train 'a' length distribution:", _format_length_counts(train_counts_a))
    print("  Train 'b' length distribution:", _format_length_counts(train_counts_b))
    print("  Val 'a' length distribution:", _format_length_counts(val_counts_a))
    print("  Val 'b' length distribution:", _format_length_counts(val_counts_b))
    print("  Test 'a' length distribution:", _format_length_counts(test_counts_a))
    print("  Test 'b' length distribution:", _format_length_counts(test_counts_b))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiplication data where operand digit lengths follow user-specified distributions. "
            "Provide the distributions as comma-separated probability lists."
        )
    )
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE, help="Number of training examples.")
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE, help="Number of validation examples.")
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE, help="Number of test examples.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to write the dataset files.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used for sampling.")
    parser.add_argument(
        "--a_probabilities",
        "--a_length_probs",
        dest="a_probabilities",
        type=str,
        default=None,
        help=(
            "Comma-separated probabilities for the digit length of operand 'a'. "
            "Index 0 corresponds to 1-digit numbers, index 1 to 2-digit numbers, and so on."
        ),
    )
    parser.add_argument(
        "--b_probabilities",
        "--b_length_probs",
        dest="b_probabilities",
        type=str,
        default=None,
        help=(
            "Comma-separated probabilities for the digit length of operand 'b'. "
            "Each position corresponds to the length starting at one digit."
        ),
    )
    parser.add_argument(
        "--max_digits",
        type=int,
        default=None,
        help=(
            "Positive integer specifying the largest digit length to sample. "
            "Used as a fallback for --a_max_digits/--b_max_digits when those flags are omitted "
            "and no explicit probability distributions are supplied."
        ),
    )
    parser.add_argument(
        "--a_max_digits",
        type=int,
        default=None,
        help=(
            "Positive integer specifying the largest digit length to sample for operand 'a' when "
            "--a_probabilities is not provided. Defaults to 1 when omitted."
        ),
    )
    parser.add_argument(
        "--b_max_digits",
        type=int,
        default=None,
        help=(
            "Positive integer specifying the largest digit length to sample for operand 'b' when "
            "--b_probabilities is not provided. Defaults to 1 when omitted."
        ),
    )
    # Accept --num_operands for compatibility with the dispatcher script.  The flag is ignored.
    parser.add_argument(
        "--num_operands",
        type=int,
        default=None,
        help="Unused legacy flag accepted for compatibility with data_generate.py.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        a_probs = (
            _parse_probability_string(
                args.a_probabilities, flag_name="--a_probabilities/--a_length_probs"
            )
            if args.a_probabilities is not None
            else None
        )
        b_probs = (
            _parse_probability_string(
                args.b_probabilities, flag_name="--b_probabilities/--b_length_probs"
            )
            if args.b_probabilities is not None
            else None
        )
    except ProbabilityParseError as exc:
        parser.error(str(exc))

    setattr(args, "a_probabilities", a_probs)
    setattr(args, "b_probabilities", b_probs)
    return args


def main() -> None:
    args = parse_args()
    try:
        make_dataset(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            output_dir=args.output_dir,
            seed=args.seed,
            a_probabilities=args.a_probabilities,
            b_probabilities=args.b_probabilities,
            a_max_digits=args.a_max_digits,
            b_max_digits=args.b_max_digits,
            max_digits=args.max_digits,
        )
    except ProbabilityParseError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
