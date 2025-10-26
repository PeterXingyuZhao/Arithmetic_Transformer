#!/usr/bin/env python3
"""
generate_skewed_tests.py

Creates three skewed 4-number sorting test files (3000 examples each).
Output files:
  - test1.txt  (first specification)
  - test2.txt  (second specification)
  - test3.txt  (third specification)

Line format:
  unsorted_numbers_comma_separated=sorted_numbers_comma_separated$

Example:
  2774,524,996,875=524,875,996,2774$
"""

import random

NUM_EXAMPLES = 3000

def write_lines(filename, lines):
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def format_example(nums):
    """Return string 'a,b,c,d=sorted(...)$' with numbers as integers."""
    left_order = nums[:]  # a copy
    random.shuffle(left_order)
    left_str = ",".join(str(n) for n in left_order)
    right = sorted(nums)
    right_str = ",".join(str(n) for n in right)
    return f"{left_str}={right_str}$"

def gen_test1(n):
    """
    Test 1:
    - 3 numbers are 3-digit with hundreds digit in 5-9.
    - 1 number is 4-digit with thousands digit in 1-4.
    - All unspecified digits random.
    """
    lines = []
    for _ in range(n):
        three = []
        for _ in range(3):
            h = random.randint(5, 9)   # hundreds digit for 3-digit number
            t = random.randint(0, 9)
            u = random.randint(0, 9)
            three.append(100*h + 10*t + u)
        # 4-digit number
        th = random.randint(1, 4)  # thousands-place
        h2 = random.randint(0, 9)
        t2 = random.randint(0, 9)
        u2 = random.randint(0, 9)
        four = 1000*th + 100*h2 + 10*t2 + u2
        nums = [four] + three
        lines.append(format_example(nums))
    return lines

def gen_test2(n):
    """
    Test 2:
    - All four numbers share same highest digit (leading digit).
      (For 3-digit numbers that's the hundreds digit; for 4-digit, the thousands digit.)
    - For each 3-digit number: tens digit randomly from 5-9.
    - For the 4-digit number: hundreds digit randomly from 0-4.
    - The three 3-digit numbers must be pairwise distinct.
    """
    lines = []
    for _ in range(n):
        leading = random.randint(1, 9)  # shared highest digit
        three_set = set()
        three = []
        # build three distinct 3-digit numbers
        attempts = 0
        while len(three) < 3:
            attempts += 1
            if attempts > 1000:
                # extremely unlikely, restart with a new leading digit
                leading = random.randint(1, 9)
                three_set.clear()
                three.clear()
                attempts = 0
                continue
            tens = random.randint(5, 9)
            units = random.randint(0, 9)
            num = 100*leading + 10*tens + units
            if num not in three_set:
                three_set.add(num)
                three.append(num)
        # 4-digit number with same leading digit; hundreds digit in 0-4
        hundreds_4 = random.randint(0, 4)
        tens_4 = random.randint(0, 9)
        units_4 = random.randint(0, 9)
        four = 1000*leading + 100*hundreds_4 + 10*tens_4 + units_4
        nums = three + [four]
        lines.append(format_example(nums))
    return lines

def gen_test3(n):
    """
    Test 3:
    - All four numbers share the same highest digit and second-highest digit.
      (For 3-digit numbers: hundreds and tens. For 4-digit: thousands and hundreds.)
    - Units digit of the 3-digit numbers randomly from 5-9 and pairwise distinct.
    - For the 4-digit number: tens digit randomly from 0-4.
    """
    lines = []
    for _ in range(n):
        d1 = random.randint(1, 9)    # shared highest digit
        d2 = random.randint(0, 9)    # shared second-highest digit
        # choose 3 distinct units from 5-9
        units_choices = random.sample(range(5, 10), 3)  # ensures uniqueness
        three = [100*d1 + 10*d2 + u for u in units_choices]
        # build 4-digit number: thousands=d1, hundreds=d2, tens in 0-4, units random 0-9
        tens_4 = random.randint(0, 4)
        units_4 = random.randint(0, 9)
        four = 1000*d1 + 100*d2 + 10*tens_4 + units_4
        nums = three + [four]
        lines.append(format_example(nums))
    return lines

def main():
    random.seed()  # system time or OS entropy; set an int for reproducibility if desired

    print("Generating test1.txt ...")
    lines1 = gen_test1(NUM_EXAMPLES)
    write_lines("test1.txt", lines1)
    print("Wrote test1.txt (%d lines)" % len(lines1))

    print("Generating test2.txt ...")
    lines2 = gen_test2(NUM_EXAMPLES)
    write_lines("test2.txt", lines2)
    print("Wrote test2.txt (%d lines)" % len(lines2))

    print("Generating test3.txt ...")
    lines3 = gen_test3(NUM_EXAMPLES)
    write_lines("test3.txt", lines3)
    print("Wrote test3.txt (%d lines)" % len(lines3))

    print("Done.")

if __name__ == "__main__":
    main()
