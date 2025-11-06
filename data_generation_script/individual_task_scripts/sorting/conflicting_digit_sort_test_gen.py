#!/usr/bin/env python3
"""
Generate 6 test files (test1.txt .. test6.txt), each with 1000 examples.
Each example format:
  left1,left2,left3,left4=right1,right2,right3,right4$

Right side is the four numbers sorted ascending.
Left side contains two uniformly random 4-digit numbers (1000-9999)
and two constructed 4-digit numbers whose individual digits meet the rules.

Original three:
  test1: a1 == b1 (1-9), digits 2 and 3 conflict, a4/b4 random 0-9
  test2: a2 == b2 (0-9), digits 1 and 3 conflict (a1/b1 in 1-9), a4/b4 random
  test3: a3 == b3 (0-9), digits 1 and 2 conflict (a1/b1 in 1-9), a4/b4 random

New three:
  test4: a2 == b2 and a3 == b3, digits 1 and 4 conflict (a1/b1 in 1-9, a4/b4 in 0-9)
  test5: a1 == b1 and a3 == b3, digits 2 and 4 conflict (a1/b1 in 1-9)
  test6: a1 == b1 and a2 == b2, digits 3 and 4 conflict (a1/b1 in 1-9)
"""

import random

NUM_EXAMPLES = 1000

def two_distinct_digits(low, high):
    """Return two distinct integers in [low, high]."""
    d1 = random.randint(low, high)
    d2 = random.randint(low, high)
    while d2 == d1:
        d2 = random.randint(low, high)
    return d1, d2

def make_4digit_from_digits(d1, d2, d3, d4):
    """Assemble 4 digits into a 4-digit integer (d1 is thousands place)."""
    return d1*1000 + d2*100 + d3*10 + d4

# --- original generators (kept and slightly reformatted) ---
def gen_file1_line():
    # test1: a1 == b1 (1-9), digits 2 and 3 conflict, a4/b4 random
    r1 = random.randint(1000, 9999)
    r2 = random.randint(1000, 9999)

    a1 = random.randint(1, 9)
    b1 = a1

    d2a, d2b = two_distinct_digits(0, 9)
    d3a, d3b = two_distinct_digits(0, 9)

    if random.choice([True, False]):
        a2, b2 = min(d2a, d2b), max(d2a, d2b)   # a2 < b2
        a3, b3 = max(d3a, d3b), min(d3a, d3b)   # a3 > b3
    else:
        a2, b2 = max(d2a, d2b), min(d2a, d2b)   # a2 > b2
        a3, b3 = min(d3a, d3b), max(d3a, d3b)   # a3 < b3

    a4 = random.randint(0, 9)
    b4 = random.randint(0, 9)

    a = make_4digit_from_digits(a1, a2, a3, a4)
    b = make_4digit_from_digits(b1, b2, b3, b4)

    arr = [r1, r2, a, b]
    random.shuffle(arr)
    left = ",".join(str(x) for x in arr)
    right = ",".join(str(x) for x in sorted(arr))
    return f"{left}={right}$"

def gen_file2_line():
    # test2: a2 == b2 (0-9), digits 1 and 3 conflict, a4 random
    r1 = random.randint(1000, 9999)
    r2 = random.randint(1000, 9999)

    a2 = random.randint(0, 9)
    b2 = a2

    d1a, d1b = two_distinct_digits(1, 9)   # thousands digit must be 1-9
    d3a, d3b = two_distinct_digits(0, 9)

    if random.choice([True, False]):
        a1, b1 = min(d1a, d1b), max(d1a, d1b)   # a1 < b1
        a3, b3 = max(d3a, d3b), min(d3a, d3b)   # a3 > b3
    else:
        a1, b1 = max(d1a, d1b), min(d1a, d1b)   # a1 > b1
        a3, b3 = min(d3a, d3b), max(d3a, d3b)   # a3 < b3

    a4 = random.randint(0, 9)
    b4 = random.randint(0, 9)

    a = make_4digit_from_digits(a1, a2, a3, a4)
    b = make_4digit_from_digits(b1, b2, b3, b4)

    arr = [r1, r2, a, b]
    random.shuffle(arr)
    left = ",".join(str(x) for x in arr)
    right = ",".join(str(x) for x in sorted(arr))
    return f"{left}={right}$"

def gen_file3_line():
    # test3: a3 == b3 (0-9), digits 1 and 2 conflict, a4 random
    r1 = random.randint(1000, 9999)
    r2 = random.randint(1000, 9999)

    a3 = random.randint(0, 9)
    b3 = a3

    d1a, d1b = two_distinct_digits(1, 9)
    d2a, d2b = two_distinct_digits(0, 9)

    if random.choice([True, False]):
        a1, b1 = min(d1a, d1b), max(d1a, d1b)   # a1 < b1
        a2, b2 = max(d2a, d2b), min(d2a, d2b)   # a2 > b2
    else:
        a1, b1 = max(d1a, d1b), min(d1a, d1b)   # a1 > b1
        a2, b2 = min(d2a, d2b), max(d2a, d2b)   # a2 < b2

    a4 = random.randint(0, 9)
    b4 = random.randint(0, 9)

    a = make_4digit_from_digits(a1, a2, a3, a4)
    b = make_4digit_from_digits(b1, b2, b3, b4)

    arr = [r1, r2, a, b]
    random.shuffle(arr)
    left = ",".join(str(x) for x in arr)
    right = ",".join(str(x) for x in sorted(arr))
    return f"{left}={right}$"

# --- new generators for the three additional files ---
def gen_file4_line():
    # test4: a2 == b2 and a3 == b3, digits 1 and 4 conflict (a1/b1 in 1-9, a4/b4 in 0-9)
    r1 = random.randint(1000, 9999)
    r2 = random.randint(1000, 9999)

    # a2 == b2, a3 == b3
    a2 = random.randint(0, 9)
    b2 = a2
    a3 = random.randint(0, 9)
    b3 = a3

    # pick distinct thousands digits (1-9) and distinct units digits (0-9)
    d1a, d1b = two_distinct_digits(1, 9)
    d4a, d4b = two_distinct_digits(0, 9)

    if random.choice([True, False]):
        a1, b1 = min(d1a, d1b), max(d1a, d1b)   # a1 < b1
        a4, b4 = max(d4a, d4b), min(d4a, d4b)   # a4 > b4
    else:
        a1, b1 = max(d1a, d1b), min(d1a, d1b)   # a1 > b1
        a4, b4 = min(d4a, d4b), max(d4a, d4b)   # a4 < b4

    a = make_4digit_from_digits(a1, a2, a3, a4)
    b = make_4digit_from_digits(b1, b2, b3, b4)

    arr = [r1, r2, a, b]
    random.shuffle(arr)
    left = ",".join(str(x) for x in arr)
    right = ",".join(str(x) for x in sorted(arr))
    return f"{left}={right}$"

def gen_file5_line():
    # test5: a1 == b1 and a3 == b3, digits 2 and 4 conflict
    r1 = random.randint(1000, 9999)
    r2 = random.randint(1000, 9999)

    # a1 == b1 (must be 1-9 to ensure 4-digit), a3 == b3 (0-9)
    a1 = random.randint(1, 9)
    b1 = a1
    a3 = random.randint(0, 9)
    b3 = a3

    # pick distinct digits for pos2 (0-9) and pos4 (0-9)
    d2a, d2b = two_distinct_digits(0, 9)
    d4a, d4b = two_distinct_digits(0, 9)

    if random.choice([True, False]):
        a2, b2 = min(d2a, d2b), max(d2a, d2b)   # a2 < b2
        a4, b4 = max(d4a, d4b), min(d4a, d4b)   # a4 > b4
    else:
        a2, b2 = max(d2a, d2b), min(d2a, d2b)   # a2 > b2
        a4, b4 = min(d4a, d4b), max(d4a, d4b)   # a4 < b4

    a = make_4digit_from_digits(a1, a2, a3, a4)
    b = make_4digit_from_digits(b1, b2, b3, b4)

    arr = [r1, r2, a, b]
    random.shuffle(arr)
    left = ",".join(str(x) for x in arr)
    right = ",".join(str(x) for x in sorted(arr))
    return f"{left}={right}$"

def gen_file6_line():
    # test6: a1 == b1 and a2 == b2, digits 3 and 4 conflict
    r1 = random.randint(1000, 9999)
    r2 = random.randint(1000, 9999)

    # a1 == b1 (1-9), a2 == b2 (0-9)
    a1 = random.randint(1, 9)
    b1 = a1
    a2 = random.randint(0, 9)
    b2 = a2

    # pick distinct digits for pos3 (0-9) and pos4 (0-9)
    d3a, d3b = two_distinct_digits(0, 9)
    d4a, d4b = two_distinct_digits(0, 9)

    if random.choice([True, False]):
        a3, b3 = min(d3a, d3b), max(d3a, d3b)   # a3 < b3
        a4, b4 = max(d4a, d4b), min(d4a, d4b)   # a4 > b4
    else:
        a3, b3 = max(d3a, d3b), min(d3a, d3b)   # a3 > b3
        a4, b4 = min(d4a, d4b), max(d4a, d4b)   # a4 < b4

    a = make_4digit_from_digits(a1, a2, a3, a4)
    b = make_4digit_from_digits(b1, b2, b3, b4)

    arr = [r1, r2, a, b]
    random.shuffle(arr)
    left = ",".join(str(x) for x in arr)
    right = ",".join(str(x) for x in sorted(arr))
    return f"{left}={right}$"

def write_examples(filename, generator_func, n=NUM_EXAMPLES):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(generator_func() + "\n")

if __name__ == "__main__":
    # Set seed for reproducibility if you want:
    # random.seed(12345)
    random.seed()

    write_examples("test1.txt", gen_file1_line)
    write_examples("test2.txt", gen_file2_line)
    write_examples("test3.txt", gen_file3_line)
    write_examples("test4.txt", gen_file4_line)  # new
    write_examples("test5.txt", gen_file5_line)  # new
    write_examples("test6.txt", gen_file6_line)  # new

    print("Wrote test1.txt .. test6.txt (each with", NUM_EXAMPLES, "examples).")
