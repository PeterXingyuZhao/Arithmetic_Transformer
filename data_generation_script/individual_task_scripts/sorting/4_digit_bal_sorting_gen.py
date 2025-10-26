#!/usr/bin/env python3
"""
4_digit_bal_sorting_gen.py

Generate train/val/test text files of "4-number" sorting examples with these distributions:
 - 90%: 4 numbers uniformly from 1000-9999
 - 9%: share the thousands digit (1-9), last 3 digits uniformly 000-999
 - 0.9%: share thousands+hundreds (prefix 10-99), last 2 digits 00-99
 - 0.1%: share thousands+hundreds+tens (prefix 100-999), last digit 0-9

Example line format:
3216,2238,2141,8674=2141,2238,3216,8674$
"""
import argparse
import random
import sys

def format_example(nums):
    orig_str = ",".join(f"{n:04d}" for n in nums)
    sorted_str = ",".join(f"{n:04d}" for n in sorted(nums))
    return f"{orig_str}={sorted_str}$\n"

def gen_uniform(count, rnd):
    for _ in range(count):
        yield [rnd.randint(1000, 9999) for _ in range(4)]

def gen_share_thousands(count, rnd):
    for _ in range(count):
        t = rnd.randint(1, 9)
        yield [t * 1000 + rnd.randint(0, 999) for _ in range(4)]

def gen_share_thousands_hundreds(count, rnd):
    for _ in range(count):
        prefix = rnd.randint(10, 99)
        yield [prefix * 100 + rnd.randint(0, 99) for _ in range(4)]

def gen_share_three_digits(count, rnd):
    for _ in range(count):
        prefix = rnd.randint(100, 999)
        yield [prefix * 10 + rnd.randint(0, 9) for _ in range(4)]

def compute_counts_for_total(total):
    # proportions 0.9, 0.09, 0.009, 0.001; use integer arithmetic and put remainder
    c0 = int(total * 0.25)
    c1 = int(total * 0.25)
    c2 = int(total * 0.25)
    c3 = total - (c0 + c1 + c2)
    return [
        ("uniform", c0, gen_uniform),
        ("share_thousands", c1, gen_share_thousands),
        ("share_thousands_hundreds", c2, gen_share_thousands_hundreds),
        ("share_three_digits", c3, gen_share_three_digits),
    ]

def generate_examples_for_counts(counts, rnd, report_every=100_000):
    examples = []
    generated = 0
    total = sum(c for _, c, _ in counts)
    for name, cnt, gen_func in counts:
        gen = gen_func(cnt, rnd)
        for nums in gen:
            examples.append(format_example(nums))
            generated += 1
            if report_every and generated % report_every == 0:
                print(f"Generated {generated:,} / {total:,} examples...", file=sys.stderr)
    return examples

def write_buffered(filename, examples, buffer_size, report_every=100_000):
    with open(filename, "w", encoding="utf-8", newline="") as fout:
        buf = []
        written = 0
        total = len(examples)
        for ex in examples:
            buf.append(ex)
            if len(buf) >= buffer_size:
                fout.writelines(buf)
                written += len(buf)
                buf = []
                if report_every and written % report_every == 0:
                    print(f"Written {written:,} / {total:,} lines to {filename}...", file=sys.stderr)
        if buf:
            fout.writelines(buf)
            written += len(buf)
    print(f"Done. Wrote {written:,} examples to {filename}", file=sys.stderr)

def make_split(name, total_examples, master_rand, out_filename, buffer_size, report_every):
    # derive a seed for this split so it is reproducible but independent
    split_seed = master_rand.randrange(0, 2**63)
    split_rnd = random.Random(split_seed)
    counts = compute_counts_for_total(total_examples)
    print(f"Generating {name}: total={total_examples:,}, counts={[ (n,c) for n,c,_ in counts ]}, seed={split_seed}", file=sys.stderr)
    examples = generate_examples_for_counts(counts, split_rnd, report_every=report_every)
    print(f"Shuffling {name} (seed={split_seed})...", file=sys.stderr)
    split_rnd.shuffle(examples)
    write_buffered(out_filename, examples, buffer_size, report_every=report_every)
    return split_seed

def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test sorting files.")
    parser.add_argument("--out_train", default="train.txt", help="train output filename (default: train.txt)")
    parser.add_argument("--out_val", default="val.txt", help="val output filename (default: val.txt)")
    parser.add_argument("--out_test", default="test.txt", help="test output filename (default: test.txt)")
    parser.add_argument("--seed", type=int, default=None, help="master seed for reproducibility (optional)")
    parser.add_argument("--buffer", type=int, default=20000, help="write buffer size in lines (default 20000)")
    parser.add_argument("--report_every", type=int, default=100_000, help="progress report interval (lines)")
    parser.add_argument("--train_size", type=int, default=1_000_000, help="number of train examples (default 1,000,000)")
    parser.add_argument("--val_size", type=int, default=10_000, help="number of val examples (default 10,000)")
    parser.add_argument("--test_size", type=int, default=10_000, help="number of test examples (default 10,000)")
    args = parser.parse_args()

    master_rand = random.Random(args.seed)
    if args.seed is None:
        # report the actual master seed used (drawn randomly) so user can reproduce if desired
        reported_master_seed = master_rand.randrange(0, 2**63)
        # re-seed master_rand with that drawn seed to ensure determinism for split derivation
        master_rand.seed(reported_master_seed)
        print(f"No --seed provided. using generated master seed: {reported_master_seed}", file=sys.stderr)
    else:
        print(f"Using master seed: {args.seed}", file=sys.stderr)

    # Make train/val/test splits (each shuffled independently)
    train_seed = make_split("train", args.train_size, master_rand, args.out_train, args.buffer, args.report_every)
    val_seed   = make_split("val",   args.val_size,   master_rand, args.out_val,   args.buffer, args.report_every)
    test_seed  = make_split("test",  args.test_size,  master_rand, args.out_test,  args.buffer, args.report_every)

    print("Finished all splits.", file=sys.stderr)
    print(f"Split seeds: train={train_seed}, val={val_seed}, test={test_seed}", file=sys.stderr)

if __name__ == "__main__":
    main()
