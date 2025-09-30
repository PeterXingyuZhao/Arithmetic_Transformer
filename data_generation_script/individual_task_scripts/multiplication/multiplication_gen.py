#!/usr/bin/env python3
"""
Generate multiplication dataset for (a * b) where:
 - a in [0, 999_999] (0..6-digit)
 - b in [0, 9] (1-digit)
Training examples are drawn from 6 buckets partitioned by digit-length of `a`.
Train-bucket relative ratios (default): (100, 200, 400, 800, 1500, 7000)
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Set

# default values matching your original script
DEFAULT_TRAIN_SIZE = 10_000   # total training examples (will be split proportionally)
DEFAULT_TEST_SIZE  = 3_000
DEFAULT_VAL_SIZE   = 3_000
DEFAULT_OUT_DIR    = "."
DEFAULT_SEED       = 42

# bucket ratios and population sizes (a ranges)
BUCKET_RATIOS = [100, 200, 400, 800, 1500, 7000]  # ones, twos, threes, fours, fives, sixes
# For 'a' digit buckets the counts of a-values are:
# ones: 0..9 (10 values), twos: 10..99 (90), threes:100..999 (900),
# four:1000..9999 (9000), five:10000..99999 (90000), six:100000..999999 (900000)
A_BUCKET_RANGES = [
    (0, 10),       # ones: a in [0,9]    -> count_a = 10
    (10, 100),     # twos: a in [10,99]  -> count_a = 90
    (100, 1000),   # threes: [100,999]   -> 900
    (1000, 10000), # fours: [1000,9999]  -> 9000
    (10000, 100000),# fives: [10000,99999]-> 90000
    (100000, 1000000) # sixes: [100000,999999] -> 900000
]

def compute_bucket_counts(train_total: int, ratios: List[int]) -> List[int]:
    total_ratio = sum(ratios)
    # initial integer allocation (floor)
    counts = [train_total * r // total_ratio for r in ratios]
    # distribute remainder to last buckets (to make sum == train_total)
    remainder = train_total - sum(counts)
    i = len(counts) - 1
    while remainder > 0:
        counts[i] += 1
        remainder -= 1
        i -= 1
        if i < 0:
            i = len(counts) - 1
    return counts

def bucket_population_sizes() -> List[int]:
    # Each a-value pairs with 10 b-values, so population per bucket is (count_a * 10)
    pops = []
    for a0, a1 in A_BUCKET_RANGES:
        count_a = a1 - a0
        pops.append(count_a * 10)
    return pops

def sample_pairs_from_bucket(bucket_idx: int, k: int, rng: random.Random) -> List[Tuple[int,int]]:
    """
    Sample k unique (a,b) pairs from bucket bucket_idx.
    Sampling uses integer indexes to avoid materializing all pairs when the population is large.
    """
    a0, a1 = A_BUCKET_RANGES[bucket_idx]
    count_a = a1 - a0
    pop = count_a * 10  # total pairs in this bucket
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []

    if k > pop:
        raise ValueError(f"Requested {k} samples from bucket {bucket_idx} but bucket only has {pop} pairs.")

    # sample unique indices in [0, pop)
    chosen_indices = rng.sample(range(pop), k)
    pairs = []
    for idx in chosen_indices:
        a = a0 + (idx // 10)
        b = idx % 10
        pairs.append((a, b))
    return pairs

def global_index(a: int, b: int) -> int:
    """Map pair (a,b) to unique global index in [0, 10_000_000)."""
    return a * 10 + b

def make_dataset(
    train_total: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    validation_size: int = DEFAULT_VAL_SIZE,
    out_dir: str = DEFAULT_OUT_DIR,
    seed: int = DEFAULT_SEED
):
    rng = random.Random(seed)
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # compute train counts per bucket
    bucket_counts = compute_bucket_counts(train_total, BUCKET_RATIOS)
    pops = bucket_population_sizes()

    # sanity: ensure requested per-bucket counts are available
    for i, (req, p) in enumerate(zip(bucket_counts, pops)):
        if req > p:
            raise ValueError(f"Train bucket {i} requested {req} samples but only {p} available. "
                             f"Reduce train_total or adjust distribution.")

    # 1) Sample training pairs per bucket (disjoint by construction)
    training_pairs: List[Tuple[int,int]] = []
    training_global_indices: Set[int] = set()

    for i, k in enumerate(bucket_counts):
        if k == 0:
            continue
        pairs = sample_pairs_from_bucket(i, k, rng)
        for a, b in pairs:
            idx = global_index(a, b)
            training_pairs.append((a, b))
            training_global_indices.add(idx)

    # shuffle training
    rng.shuffle(training_pairs)

    # 2) Sample testing set (no answers) and validation (with answers) from remaining global space
    TOTAL_PAIRS = 1_000_000 * 10  # a in 0..999999, b in 0..9 -> 10_000_000
    # helper to sample unique global indices not in training
    def sample_from_remaining(k: int, forbidden: Set[int]) -> List[Tuple[int,int]]:
        chosen: Set[int] = set()
        # If k is small relative to remaining space, simple rejection sampling is fine
        rem_space = TOTAL_PAIRS - len(forbidden)
        if k > rem_space:
            raise ValueError(f"Requested {k} samples but only {rem_space} remaining available.")
        while len(chosen) < k:
            r = rng.randrange(TOTAL_PAIRS)
            if r in forbidden or r in chosen:
                continue
            chosen.add(r)
        # map back to (a,b)
        result = []
        for idx in chosen:
            a = idx // 10
            b = idx % 10
            result.append((a, b))
        return result

    testing_pairs = sample_from_remaining(test_size, training_global_indices)
    # add testing indices to forbidden set to avoid picking them for validation
    testing_global = {global_index(a, b) for a, b in testing_pairs}
    forbidden_for_val = training_global_indices.union(testing_global)
    validation_pairs = sample_from_remaining(validation_size, forbidden_for_val)

    # shuffle testing & validation
    rng.shuffle(testing_pairs)
    rng.shuffle(validation_pairs)

    # 3) Write out files
    train_path = OUT_DIR / "train.txt"
    test_path  = OUT_DIR / "test.txt"
    val_path   = OUT_DIR / "val.txt"

    with train_path.open("w") as f_tr:
        for a, b in training_pairs:
            f_tr.write(f"{a}*{b}={a*b}$\n")

    with test_path.open("w") as f_te:
        # tests: no answers (as your original print stated)
        for a, b in testing_pairs:
            f_te.write(f"{a}*{b}={a*b}\n")

    with val_path.open("w") as f_val:
        for a, b in validation_pairs:
            f_val.write(f"{a}*{b}={a*b}$\n")

    print(f"Wrote {len(training_pairs)} shuffled lines (with answers) to '{train_path}'")
    print(f"Wrote {len(testing_pairs)} shuffled lines (no answers) to '{test_path}'")
    print(f"Wrote {len(validation_pairs)} shuffled lines (with answers) to '{val_path}'")

def parse_args():
    p = argparse.ArgumentParser(description="Generate balanced multiplication dataset (a * b).")
    p.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE,
                   help="Total number of training examples (will be split across digit-buckets by ratio).")
    p.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE, help="Number of test examples (no answers).")
    p.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE, help="Number of validation examples (with answers).")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to write train.txt, test.txt, val.txt")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make_dataset(train_total=args.train_size, test_size=args.test_size,
                 validation_size=args.val_size, out_dir=args.output_dir, seed=args.seed)
