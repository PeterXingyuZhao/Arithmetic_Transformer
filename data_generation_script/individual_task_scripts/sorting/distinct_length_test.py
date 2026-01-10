#!/usr/bin/env python3
"""
Generate 4-operand sorting data with distinct digit lengths.

Each line has the format:
a,b,c,d=sorted_a,sorted_b,sorted_c,sorted_d$

Where:
- one 1-digit number:   0–9
- one 2-digit number:   10–99
- one 3-digit number:   100–999
- one 4-digit number:   1000–9999
"""

import random

NUM_EXAMPLES = 3000
OUTPUT_FILE = "distinct_length_test.txt"

def generate_example(rng):
    nums = [
        rng.randint(0, 9),          # 1-digit
        rng.randint(10, 99),        # 2-digit
        rng.randint(100, 999),      # 3-digit
        rng.randint(1000, 9999),    # 4-digit
    ]
    rng.shuffle(nums)
    unsorted_part = ",".join(str(x) for x in nums)
    sorted_part = ",".join(str(x) for x in sorted(nums))
    return f"{unsorted_part}={sorted_part}$\n"

def main():
    rng = random.Random()  # optionally set a seed here
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for _ in range(NUM_EXAMPLES):
            f.write(generate_example(rng))
    print(f"Wrote {NUM_EXAMPLES} examples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
