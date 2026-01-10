#!/usr/bin/env python3
"""
make_digitwise_tests.py

Generates four test files (1000 examples each):
 - digitwise_thousands_MSDD.txt
 - digitwise_hundreds_MSDD.txt
 - digitwise_tens_MSDD.txt
 - digitwise_units_MSDD.txt

Each line example format:
  a,b,c,d=s1,s2,s3,s4$

Rules (all numbers are 4-digit):
 - thousands:   a1,b1,c1,d1 distinct (1-9); other digits iid 0-9
 - hundreds:    a1=b1=c1=d1 (1-9); a2,b2,c2,d2 distinct (0-9); other digits iid 0-9
 - tens:        a1=a2=b1=b2=c1=c2=d1=d2 (first two digits same); a3..d3 distinct (0-9); last digit iid 0-9
 - units:       first three digits common; last digits a4..d4 distinct (0-9)
"""
import random
from typing import List

# Config
N_EXAMPLES = 1000
RANDOM_SEED = None  # set to int for reproducible output

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def digits_to_str(digs: List[int]) -> str:
    return "".join(str(d) for d in digs)

def make_example_thousands() -> str:
    # first digits distinct from 1-9
    firsts = random.sample(range(1, 10), 4)
    nums = []
    for i in range(4):
        d1 = firsts[i]
        d2 = random.randint(0, 9)
        d3 = random.randint(0, 9)
        d4 = random.randint(0, 9)
        nums.append(digits_to_str([d1, d2, d3, d4]))
    return format_example(nums)

def make_example_hundreds() -> str:
    # common first digit 1-9; second digits distinct 0-9
    common_first = random.randint(1, 9)
    seconds = random.sample(range(0, 10), 4)
    nums = []
    for i in range(4):
        d1 = common_first
        d2 = seconds[i]
        d3 = random.randint(0, 9)
        d4 = random.randint(0, 9)
        nums.append(digits_to_str([d1, d2, d3, d4]))
    return format_example(nums)

def make_example_tens() -> str:
    # common first two digits; third digits distinct; last digit iid
    common_first = random.randint(1, 9)
    common_second = random.randint(0, 9)
    thirds = random.sample(range(0, 10), 4)
    nums = []
    for i in range(4):
        d1 = common_first
        d2 = common_second
        d3 = thirds[i]
        d4 = random.randint(0, 9)
        nums.append(digits_to_str([d1, d2, d3, d4]))
    return format_example(nums)

def make_example_units() -> str:
    # common first three digits; last digits distinct
    common_first = random.randint(1, 9)
    common_second = random.randint(0, 9)
    common_third = random.randint(0, 9)
    lasts = random.sample(range(0, 10), 4)
    nums = []
    for i in range(4):
        d1 = common_first
        d2 = common_second
        d3 = common_third
        d4 = lasts[i]
        nums.append(digits_to_str([d1, d2, d3, d4]))
    return format_example(nums)

def format_example(nums: List[str]) -> str:
    # left side exactly the generated 4 numbers, right side sorted numerically (no leading zeros)
    left = ",".join(nums)
    sorted_nums = sorted(nums, key=lambda s: int(s))
    right = ",".join(str(int(x)) for x in sorted_nums)
    return f"{left}={right}$\n"

def write_file(filename: str, gen_fn, n: int) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(gen_fn())

def main():
    write_file("digitwise_thousands_MSDD.txt", make_example_thousands, N_EXAMPLES)
    write_file("digitwise_hundreds_MSDD.txt",  make_example_hundreds,  N_EXAMPLES)
    write_file("digitwise_tens_MSDD.txt",     make_example_tens,      N_EXAMPLES)
    write_file("digitwise_units_MSDD.txt",    make_example_units,     N_EXAMPLES)
    print("Created 4 files with", N_EXAMPLES, "examples each.")

if __name__ == "__main__":
    main()
