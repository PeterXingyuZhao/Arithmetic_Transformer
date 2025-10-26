#!/usr/bin/env python3
"""
Generate multiplication dataset for (a * b) where:
 - a in [0, 999_999] (0..6-digit)
 - b in [0, 9] (1-digit)
Training examples are drawn from the first N buckets partitioned by digit-length of `a`.
Train-bucket relative ratios (default full list): (100, 200, 400, 800, 1500, 7000)
Use --num_operands to pick a prefix of the above list (1..6).
Example: num_operands=4 -> ratios [100,200,400,800] (ones..fours)
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
DEFAULT_NUM_OPERANDS = 6  # default use all six buckets

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

def bucket_population_sizes(ranges: List[Tuple[int,int]] = None) -> List[int]:
    # Each a-value pairs with 10 b-values, so population per bucket is (count_a * 10)
    if ranges is None:
        ranges = A_BUCKET_RANGES
    pops = []
    for a0, a1 in ranges:
        count_a = a1 - a0
        pops.append(count_a * 10)
    return pops

def sample_pairs_from_bucket(bucket_idx: int, k: int, rng: random.Random) -> List[Tuple[int,int]]:
    """
    Sample k unique (a,b) pairs from bucket bucket_idx.
    bucket_idx is the index into the global A_BUCKET_RANGES (0..5).
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

def _redistribute_surplus_to_higher(
    requested: List[int],
    ratios: List[int],
    start_idx: int,
    surplus: int
) -> None:
    """
    Add `surplus` to requested counts of buckets > start_idx according to ratios.
    Uses floor allocation with largest-remainder tie-breaking.
    Modifies requested in-place.
    """
    n = len(requested)
    available_idxs = [j for j in range(start_idx + 1, n)]
    if not available_idxs:
        raise ValueError("No higher buckets available to receive surplus.")

    # sum of ratios for higher buckets
    sum_r = sum(ratios[j] for j in available_idxs)
    if sum_r <= 0:
        raise ValueError("Higher buckets have zero total ratio; cannot redistribute.")

    # compute raw allocations
    raw_allocs = []
    for j in available_idxs:
        raw = surplus * (ratios[j] / sum_r)
        raw_allocs.append(raw)

    floor_allocs = [int(x) for x in raw_allocs]
    allocated = sum(floor_allocs)
    remainder = surplus - allocated

    # fractional parts for largest-remainder method
    fracs = [(i, raw_allocs[i] - floor_allocs[i]) for i in range(len(floor_allocs))]
    # sort indexes by fractional part descending
    fracs.sort(key=lambda x: x[1], reverse=True)

    # distribute remainder one-by-one to buckets with largest fractional parts
    extra = [0] * len(floor_allocs)
    for idx_in_list, frac in fracs:
        if remainder <= 0:
            break
        extra[idx_in_list] += 1
        remainder -= 1

    # apply allocations to requested
    for local_i, j in enumerate(available_idxs):
        add_amount = floor_allocs[local_i] + extra[local_i]
        if add_amount:
            requested[j] += add_amount

def rebalance_requested_counts(
    requested: List[int],
    pops: List[int],
    ratios: List[int]
) -> List[int]:
    """
    Given initial requested counts per bucket, cap buckets at their population and
    redistribute surplus to higher-digit buckets according to original ratios.
    Iteratively handles cascading overflows. Returns final assigned counts (sum should
    equal sum(requested_initial) unless total population < requested_total, in which
    case a ValueError is raised.
    """
    n = len(requested)
    requested = requested[:]  # local copy (we will mutate it)
    total_requested = sum(requested)
    total_population = sum(pops)
    if total_population < total_requested:
        raise ValueError(f"Total available pairs ({total_population}) < requested train_total ({total_requested}). "
                         "Cannot satisfy request even after redistribution.")

    # iterate left-to-right and cap+redistribute surplus
    for i in range(n):
        if requested[i] <= pops[i]:
            continue
        surplus = requested[i] - pops[i]
        requested[i] = pops[i]  # cap
        # redistribute surplus to higher buckets (may cause further overflows handled later)
        try:
            _redistribute_surplus_to_higher(requested, ratios, i, surplus)
        except Exception as e:
            # propagate as ValueError for clarity
            raise ValueError(f"Failed to redistribute surplus from bucket {i}: {e}")

    # final sanity: ensure no bucket exceeds its pop
    for i in range(n):
        if requested[i] > pops[i]:
            # this should not happen because we iterated left->right and redistributed forward,
            # but guard defensively in case of rounding / numeric issues.
            overflow = requested[i] - pops[i]
            raise ValueError(f"After redistribution, bucket {i} still overflows by {overflow} (requested {requested[i]}, pop {pops[i]}).")

    # sum check
    if sum(requested) != total_requested:
        # It's possible (due to rounding) to be off by 1 etc. but redistribution preserves integer sum.
        # If this happens, fix by adjusting last bucket that still has headroom.
        diff = total_requested - sum(requested)
        if diff != 0:
            # try to distribute diff to rightmost buckets with headroom
            for j in range(n - 1, -1, -1):
                headroom = pops[j] - requested[j]
                if headroom <= 0:
                    continue
                give = min(headroom, diff)
                requested[j] += give
                diff -= give
                if diff == 0:
                    break
            if diff != 0:
                raise ValueError("Unable to reconcile rounding difference after redistribution.")
    return requested

def make_dataset(
    train_total: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    validation_size: int = DEFAULT_VAL_SIZE,
    out_dir: str = DEFAULT_OUT_DIR,
    seed: int = DEFAULT_SEED,
    num_operands: int = DEFAULT_NUM_OPERANDS
):
    if not (1 <= num_operands <= 6):
        raise ValueError("num_operands must be between 1 and 6 inclusive.")
    rng = random.Random(seed)
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # select the prefix of ratios and corresponding bucket ranges
    ratios_subset = BUCKET_RATIOS[:num_operands]
    ranges_subset = A_BUCKET_RANGES[:num_operands]

    # compute initial requested train counts per selected bucket
    initial_bucket_counts = compute_bucket_counts(train_total, ratios_subset)
    pops = bucket_population_sizes(ranges_subset)

    # If total available < requested_total, error immediately (can't satisfy)
    total_pop = sum(pops)
    if total_pop < train_total:
        raise ValueError(f"Total available pairs in selected buckets ({total_pop}) < requested train_total ({train_total}). "
                         "Reduce train_total or increase num_operands.")

    # If any bucket requests more than its pop, rebalance by pushing surplus to higher buckets
    need_rebalance = any(req > p for req, p in zip(initial_bucket_counts, pops))
    final_bucket_counts = initial_bucket_counts
    if need_rebalance:
        print("Some buckets request more samples than available. Rebalancing surplus to higher-digit buckets.")
        print("Initial requested per-bucket counts:", initial_bucket_counts)
        try:
            final_bucket_counts = rebalance_requested_counts(initial_bucket_counts, pops, ratios_subset)
        except ValueError as e:
            raise ValueError(f"Failed to redistribute overflow across buckets: {e}")
        # Print final generated counts and their percentages
        print("Final per-bucket counts after rebalancing:", final_bucket_counts)
        total_final = sum(final_bucket_counts)
        print("Final per-bucket percentages of train_total:")
        for i, cnt in enumerate(final_bucket_counts):
            pct = 100.0 * cnt / total_final if total_final else 0.0
            print(f"  Bucket {i}: {cnt} ({pct:.2f}%)")
    else:
        print("No bucket overflow detected; using initial per-bucket allocation.")
        print("Per-bucket counts:", initial_bucket_counts)

    # sanity: ensure requested per-bucket counts are available
    for i_local, (req, p) in enumerate(zip(final_bucket_counts, pops)):
        if req > p:
            # show global bucket index for clarity
            global_idx = i_local  # since we took prefix, local maps directly to global
            raise ValueError(f"Train bucket {global_idx} requested {req} samples but only {p} available. "
                             f"Reduce train_total or adjust distribution.")

    # print selection info
    print("Selected ratios (first num_operands):", ratios_subset)
    for i_local, (rng_range, pop) in enumerate(zip(ranges_subset, pops)):
        a0, a1 = rng_range
        print(f"Bucket {i_local} a-range {a0}-{a1-1}, population pairs: {pop}")

    # 1) Sample training pairs per selected bucket (disjoint by construction)
    training_pairs: List[Tuple[int,int]] = []
    training_global_indices: Set[int] = set()

    for i_local, k in enumerate(final_bucket_counts):
        if k == 0:
            continue
        # i_local corresponds to the global bucket index since we use a prefix
        global_bucket_idx = i_local
        pairs = sample_pairs_from_bucket(global_bucket_idx, k, rng)
        for a, b in pairs:
            idx = global_index(a, b)
            training_pairs.append((a, b))
            training_global_indices.add(idx)

    # shuffle training
    rng.shuffle(training_pairs)

    # 2) Sample testing set and validation set uniformly from selected a-range x {0..9}.
    #    Sampling is WITH REPLACEMENT and NOT required to be disjoint from training.
    #    If num_operands = N, max_a is ranges_subset[-1][1] - 1 (inclusive).
    max_a_inclusive = ranges_subset[-1][1] - 1

    def sample_uniform_pairs(k: int, max_a_incl: int, rng: random.Random) -> List[Tuple[int,int]]:
        if k <= 0:
            return []
        pairs: List[Tuple[int,int]] = []
        for _ in range(k):
            a = rng.randint(0, max_a_incl)
            b = rng.randint(0, 9)
            pairs.append((a, b))
        return pairs

    testing_pairs = sample_uniform_pairs(test_size, max_a_inclusive, rng)
    validation_pairs = sample_uniform_pairs(validation_size, max_a_inclusive, rng)

    # shuffle testing & validation (optional, but keep behavior similar)
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
        # tests: no answers (kept consistent with original code comment)
        for a, b in testing_pairs:
            f_te.write(f"{a}*{b}={a*b}$\n")

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
    p.add_argument("--num_operands", type=int, default=DEFAULT_NUM_OPERANDS,
                   help="Number of operand-digit-buckets to use (prefix of [100,200,400,800,1500,7000]). Range 1..6.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make_dataset(train_total=args.train_size, test_size=args.test_size,
                 validation_size=args.val_size, out_dir=args.output_dir,
                 seed=args.seed, num_operands=args.num_operands)
