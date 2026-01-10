#!/usr/bin/env python3
"""
Generate two test files for the 4-operand 4-digit sorting task.

Each line has the format:
1000,<b1b2b3b4>,<c1c2c3c4>,9999=1000,<b1b2b3b4>,<c1c2c3c4>,9999$

This version sorts the four numbers after '=' in ascending numeric order
and does NOT pad numbers with leading zeros.
"""
import random

NUM_EXAMPLES = 1000
OUT1 = "1_3_same_2_4_conflicting.txt"
OUT2 = "1_3_same_2_4_agreeing.txt"

def make_digits_file1():
    d1 = random.randint(1, 9)   # b1 == c1 in 1..9 (no leading zero)
    d3 = random.randint(0, 9)   # b3 == c3 in 0..9

    mode = random.getrandbits(1)
    if mode == 0:
        b2 = random.randint(0, 8)
        c2 = random.randint(b2 + 1, 9)
        c4 = random.randint(0, 8)
        b4 = random.randint(c4 + 1, 9)
    else:
        c2 = random.randint(0, 8)
        b2 = random.randint(c2 + 1, 9)
        b4 = random.randint(0, 8)
        c4 = random.randint(b4 + 1, 9)

    b = f"{d1}{b2}{d3}{b4}"
    c = f"{d1}{c2}{d3}{c4}"
    return b, c

def make_digits_file2():
    d1 = random.randint(1, 9)
    d3 = random.randint(0, 9)

    mode = random.getrandbits(1)
    if mode == 0:
        b2 = random.randint(0, 8)
        c2 = random.randint(b2 + 1, 9)
        b4 = random.randint(0, 8)
        c4 = random.randint(b4 + 1, 9)
    else:
        c2 = random.randint(0, 8)
        b2 = random.randint(c2 + 1, 9)
        c4 = random.randint(0, 8)
        b4 = random.randint(c4 + 1, 9)

    b = f"{d1}{b2}{d3}{b4}"
    c = f"{d1}{c2}{d3}{c4}"
    return b, c

def write_examples(filename, maker_func, n):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(n):
            b, c = maker_func()
            left = f"1000,{b},{c},9999"
            # numeric sort, then convert to plain strings (no zero padding)
            nums = [1000, int(b), int(c), 9999]
            nums_sorted = sorted(nums)
            right = ",".join(str(x) for x in nums_sorted)
            line = f"{left}={right}$\n"
            f.write(line)

def main():
    random.seed()  # default seeding (system time)
    write_examples(OUT1, make_digits_file1, NUM_EXAMPLES)
    write_examples(OUT2, make_digits_file2, NUM_EXAMPLES)
    print(f"Wrote {NUM_EXAMPLES} examples to '{OUT1}' and '{OUT2}'.")

if __name__ == "__main__":
    main()
