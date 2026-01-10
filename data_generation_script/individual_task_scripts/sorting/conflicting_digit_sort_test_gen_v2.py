#!/usr/bin/env python3
"""
generate_sorting_tests_v2.py

Generates 24 text files (1000 examples each) for 4-number sorting tests.
Each line format:
  1000,b1b2b3b4,c1c2c3c4,9999=1000,****,****,9999$

- First 12 files follow your original three categories (b1=b1&b2=b2; b1=b1&b3=b3; b2=b2&b3=b3).
- Additional 12 files follow the new three categories you requested:
    * keep b1=c1 and compare b2 vs c2 and b3 vs c3
    * keep b2=c2 and compare b1 vs c1 and b3 vs c3
    * keep b3=c3 and compare b1 vs c1 and b2 vs c2
- Unspecified digits (including the 4th digit when unspecified) are drawn from 0..9 (pos1 from 1..9).
"""
import random
import os

# You can change seed for reproducibility or set to None for different runs.
random.seed(42)

OUT_DIR = "output_tests"
os.makedirs(OUT_DIR, exist_ok=True)
NUM_PER_FILE = 1000

def make_pair_for_relation(pos, relation):
    """
    Return (b_digit, c_digit) for given 1-based position and relation in {'=', '>', '<'}.
    For pos==1 ensure digits in 1..9 (no leading zero). Other positions 0..9.
    """
    if relation == '=':
        if pos == 1:
            d = random.randint(1, 9)
        else:
            d = random.randint(0, 9)
        return d, d

    if pos == 1:
        # both in 1..9
        if relation == '>':
            c = random.randint(1, 8)
            b = random.randint(c + 1, 9)
        else:  # '<'
            b = random.randint(1, 8)
            c = random.randint(b + 1, 9)
        return b, c
    else:
        # digits 0..9
        if relation == '>':
            c = random.randint(0, 8)
            b = random.randint(c + 1, 9)
        else:  # '<'
            b = random.randint(0, 8)
            c = random.randint(b + 1, 9)
        return b, c

def digits_to_int(digits):
    """Convert list of 4 digits to its integer value."""
    return digits[0]*1000 + digits[1]*100 + digits[2]*10 + digits[3]

def fmt4(n):
    return f"{n:04d}"

def gen_example(equal_positions, comparisons):
    """
    equal_positions: set of positions (1..4) where b[pos]==c[pos]
    comparisons: dict position -> relation ('>' or '<') for positions to compare
    Returns: (b_digits_list, c_digits_list)
    """
    b = [None]*4
    c = [None]*4

    # assign equal positions
    for pos in equal_positions:
        if pos == 1:
            d = random.randint(1, 9)
        else:
            d = random.randint(0, 9)
        b[pos-1] = d
        c[pos-1] = d

    # assign compared positions
    for pos, rel in comparisons.items():
        bd, cd = make_pair_for_relation(pos, rel)
        b[pos-1] = bd
        c[pos-1] = cd

    # fill remaining positions (unspecified digits) -- pos1 must be 1..9; others 0..9
    for i in range(4):
        if b[i] is None:
            b[i] = random.randint(1,9) if i == 0 else random.randint(0,9)
        if c[i] is None:
            c[i] = random.randint(1,9) if i == 0 else random.randint(0,9)

    # safety: avoid identical full numbers; tweak a non-equal position if needed
    if digits_to_int(b) == digits_to_int(c):
        for i in range(3, -1, -1):
            pos = i + 1
            if pos not in equal_positions:
                # change c[i] slightly (preserve digit range)
                old = c[i]
                if i == 0:
                    # ensure 1..9
                    c[i] = 1 + ((old) % 9)
                else:
                    c[i] = (old + 1) % 10
                break
        # final fallback if still equal (very unlikely)
        if digits_to_int(b) == digits_to_int(c):
            c[-1] = (c[-1] + 1) % 10
            if c[0] == 0:
                c[0] = 1

    return b, c

# --- Original 12 categories (first batch) ---
original_categories = [
    # (filename_prefix, equal_positions_set, [compare_posA, compare_posB])
    ("cat1_equal_b1b2", {1,2}, [3,4]),  # keep b1=c1, b2=c2; compare positions 3 and 4
    ("cat2_equal_b1b3", {1,3}, [2,4]),  # keep b1=c1, b3=c3; compare positions 2 and 4
    ("cat3_equal_b2b3", {2,3}, [1,4]),  # keep b2=c2, b3=c3; compare positions 1 and 4
]

# patterns for the two compared positions in order (A,B)
comp_patterns = [
    ('>', '<'),
    ('<', '>'),
    ('<', '<'),
    ('>', '>'),
]

# --- Additional 12 categories (second batch) as requested ---
additional_categories = [
    # First additional category: keep b1=c1; compare b2 vs c2 and b3 vs c3
    ("equal_b1", {1}, [2,3]),
    # Second additional category: keep b2=c2; compare b1 vs c1 and b3 vs c3
    ("equal_b2", {2}, [1,3]),
    # Third additional category: keep b3=c3; compare b1 vs c1 and b2 vs c2
    ("equal_b3", {3}, [1,2]),
]

all_created = []

def write_files_for_categories(cat_list):
    for cat_name, equal_pos, comp_pos in cat_list:
        for idx, (relA, relB) in enumerate(comp_patterns, start=1):
            fname = f"{cat_name}_file{idx}.txt"
            path = os.path.join(OUT_DIR, fname)
            with open(path, "w") as fh:
                written = 0
                # generate NUM_PER_FILE valid examples
                while written < NUM_PER_FILE:
                    comparisons = { comp_pos[0]: relA, comp_pos[1]: relB }
                    b_digits, c_digits = gen_example(equal_pos, comparisons)
                    b_val = digits_to_int(b_digits)
                    c_val = digits_to_int(c_digits)
                    if b_val == c_val:
                        # shouldn't normally happen after gen_example's safety, but skip if it does
                        continue
                    middle1 = min(b_val, c_val)
                    middle2 = max(b_val, c_val)
                    line = f"1000,{fmt4(b_val)},{fmt4(c_val)},9999=1000,{fmt4(middle1)},{fmt4(middle2)},9999$\n"
                    fh.write(line)
                    written += 1
            all_created.append(path)

# generate original 12
write_files_for_categories(original_categories)
# generate additional 12 per the new specification
write_files_for_categories(additional_categories)

# Summary output
print(f"Created {len(all_created)} files in '{OUT_DIR}' (each with {NUM_PER_FILE} examples):")
for p in all_created:
    print("  ", p)

# show a few sample lines from the first few files (quick spot-check)
print("\nSample lines from a couple of files:")
for p in all_created[:3]:
    print(f"\n--- {os.path.basename(p)} ---")
    with open(p, "r") as fh:
        for _ in range(5):
            print("  ", fh.readline().strip())

# End of script
