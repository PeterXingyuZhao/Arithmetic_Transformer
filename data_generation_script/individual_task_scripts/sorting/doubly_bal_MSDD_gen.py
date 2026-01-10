#!/usr/bin/env python3
"""
doubly_bal_gen.py

Generates sorting task data files:
 - train.txt (TRAIN_N examples)
 - val.txt   (VAL_N examples)
 - test.txt  (TEST_N examples)

Each example has four numbers, each independently 3- or 4-digit (50/50).
After deciding digit-lengths, one of three levels is chosen (equal prob):
  - level1: first digits all distinct; remaining digits drawn iid 0-9
  - level2: all share first digit; second digits all distinct; remaining digits iid 0-9
  - level3: all share first and second digit; third digits all distinct; remaining digits iid 0-9

Output format per line:
  627,8238,4501,378=378,627,4501,8238$
"""
import random
from typing import List

# --- Configurable parameters ---
TRAIN_N = 100_000
VAL_N = 5_000
TEST_N = 5_000

# You can set a seed for reproducibility. Set to None for different outputs each run.
RANDOM_SEED = None
# -------------------------------

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def choose_length() -> int:
    """Return 3 or 4 with equal probability."""
    return 4 if random.random() < 0.5 else 3

def _digits_to_str(digits: List[int]) -> str:
    """Convert a list of digit ints to a string number (no leading zeros because first digit >0)."""
    return "".join(str(d) for d in digits)

def gen_numbers_by_level(lengths: List[int], level: int) -> List[str]:
    """
    Generate 4 numbers (strings) following the specified level:
      level == 1:
        - first digits a1,b1,c1,d1 are distinct (1-9)
        - remaining digits drawn uniformly 0-9 independently per position
      level == 2:
        - all share first digit (1-9)
        - second digits a2..d2 are distinct (0-9)
        - remaining digits drawn uniformly 0-9
      level == 3:
        - all share first digit (1-9) and second digit (0-9)
        - third digits a3..d3 are distinct (0-9)
        - remaining digits drawn uniformly 0-9
    lengths is a list of four ints each 3 or 4.
    """
    if level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3")

    nums: List[str] = []

    if level == 1:
        # choose 4 distinct first digits from 1-9
        first_digits = random.sample(range(1, 10), 4)
        for i, l in enumerate(lengths):
            digits = [first_digits[i]]
            # remaining l-1 digits uniform 0-9
            for _ in range(l - 1):
                digits.append(random.randint(0, 9))
            nums.append(_digits_to_str(digits))

    elif level == 2:
        # choose a common first digit 1-9
        common_first = random.randint(1, 9)
        # choose 4 distinct second digits from 0-9
        second_digits = random.sample(range(0, 10), 4)
        for i, l in enumerate(lengths):
            digits = [common_first, second_digits[i]]
            # remaining digits after position 2
            for _ in range(l - 2):
                digits.append(random.randint(0, 9))
            nums.append(_digits_to_str(digits))

    else:  # level == 3
        # choose common first (1-9) and common second (0-9)
        common_first = random.randint(1, 9)
        common_second = random.randint(0, 9)
        # choose 4 distinct third digits from 0-9
        third_digits = random.sample(range(0, 10), 4)
        for i, l in enumerate(lengths):
            digits = [common_first, common_second, third_digits[i]]
            # remaining digits after position 3 (only one more possible for 4-digit numbers)
            for _ in range(l - 3):
                digits.append(random.randint(0, 9))
            nums.append(_digits_to_str(digits))

    return nums

def generate_example() -> str:
    """
    Generate a single example line of the form:
      a,b,c,d=s1,s2,s3,s4$
    where the right side is the ascending numeric sort of the left side.
    """
    lengths = [choose_length() for _ in range(4)]
    # Choose level uniformly among 1,2,3
    level = random.choice([1, 2, 3])

    nums = gen_numbers_by_level(lengths, level)

    # Convert to ints for numeric sort, but keep original string forms on left.
    ints = [int(x) for x in nums]
    sorted_ints = sorted(ints)
    # Format sorted numbers as plain integers (no extra leading zeros)
    sorted_strs = [str(x) for x in sorted_ints]

    left = ",".join(nums)
    right = ",".join(sorted_strs)
    return f"{left}={right}$\n"

def write_file(filename: str, n_examples: int) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        for _ in range(n_examples):
            f.write(generate_example())

def main():
    print("Generating datasets...")
    write_file("train.txt", TRAIN_N)
    write_file("val.txt", VAL_N)
    write_file("test.txt", TEST_N)
    print(f"Done. Files created: train.txt ({TRAIN_N}), val.txt ({VAL_N}), test.txt ({TEST_N})")

if __name__ == "__main__":
    main()
