#!/usr/bin/env python3
"""
analyze_4op_conditional_independence.py

Usage:
    python analyze_4op_conditional_independence.py input.txt out_dir

Description:
    Input file should contain lines like:
        123+5+401+501=1030$
        811+856+239+313=2219$
        267+469+214+295=1245$
    (first addend is 'a', output is the number right of '=')

    This script tests whether a3 (units digit of the first addend)
    and o4 (units digit of the sum) are independent given o3 (tens digit).
    It writes per-o3 joint count CSVs and a summary CSV in out_dir.
"""
import argparse
import math
import os
from pathlib import Path
import re
import csv

def parse_line_get_digits(line):
    """
    Return (a3, o3, o4) or None on parse failure.
    - a3: units digit of the first addend (int 0..9)
    - o3: tens digit of the output (int 0..9)
    - o4: units digit of the output (int 0..9)
    """
    # find all integers on the line
    nums = re.findall(r'\d+', line)
    # we expect at least 5 numbers: a,b,c,d,sum  -> using first and last
    if len(nums) < 2:
        return None
    try:
        a = int(nums[0])
        out = int(nums[-1])
    except ValueError:
        return None
    a3 = a % 10
    o4 = out % 10
    o3 = (out // 10) % 10
    return a3, o3, o4

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_file', help='Path to input txt file (1M lines)')
    p.add_argument('out_dir', help='Directory to save CSVs and summary')
    p.add_argument('--report-every', type=int, default=100_000,
                   help='print progress every N lines (default 100000)')
    args = p.parse_args()

    input_path = Path(args.input_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # counts[o3, a3, o4]
    counts = [[[0 for _ in range(10)] for _ in range(10)] for _ in range(10)]
    total_lines = 0
    parsed_lines = 0
    bad_lines = 0

    print(f"Streaming file: {input_path}")
    with input_path.open('r', encoding='utf-8', errors='replace') as fh:
        for i, line in enumerate(fh, start=1):
            total_lines += 1
            parsed = parse_line_get_digits(line)
            if parsed is None:
                bad_lines += 1
            else:
                a3, o3, o4 = parsed
                counts[o3][a3][o4] += 1
                parsed_lines += 1
            if args.report_every and i % args.report_every == 0:
                print(f"Processed {i:,} lines (parsed {parsed_lines:,}, bad {bad_lines:,})")

    print(f"Done streaming. Total lines = {total_lines:,}, parsed = {parsed_lines:,}, bad = {bad_lines:,}")

    # compute per-o3 stats
    summary_rows = []
    overall_total = parsed_lines
    if overall_total == 0:
        raise SystemExit("No valid lines parsed — check input format.")

    # helper to compute MI, max abs dev, chi2
    def compute_stats_for_o3(joint_counts):
        # joint_counts: 10x10 list-of-lists for a3 (rows) x o4 (cols)
        total = sum(sum(row) for row in joint_counts)
        if total == 0:
            return {'total': 0, 'mi_bits': 0.0, 'max_abs_dev': 0.0, 'chi2': 0.0}
        # convert to probabilities
        P = [[c / total for c in row] for row in joint_counts]
        # marginals
        P_row = [sum(P[i][j] for j in range(10)) for i in range(10)]  # P(a3)
        P_col = [sum(P[i][j] for i in range(10)) for j in range(10)]  # P(o4)
        # mutual information
        mi = 0.0
        for i in range(10):
            for j in range(10):
                p = P[i][j]
                if p > 0 and P_row[i] > 0 and P_col[j] > 0:
                    mi += p * math.log2(p / (P_row[i] * P_col[j]))
        # max abs deviation between joint and product of marginals
        max_abs = 0.0
        for i in range(10):
            for j in range(10):
                prod = P_row[i] * P_col[j]
                max_abs = max(max_abs, abs(P[i][j] - prod))
        # chi-square statistic (sum over cells where expected > 0)
        chi2 = 0.0
        for i in range(10):
            for j in range(10):
                expected = (P_row[i] * P_col[j]) * total
                observed = joint_counts[i][j]
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected
        return {'total': total, 'mi_bits': mi, 'max_abs_dev': max_abs, 'chi2': chi2}

    # process each o3, write joint CSVs and collect summary
    for o3 in range(10):
        jc = counts[o3]  # 10x10
        stats = compute_stats_for_o3(jc)
        # write joint-count CSV (rows a3=0..9, cols o4=0..9)
        csv_path = out_dir / f'joint_counts_o3_{o3}.csv'
        with csv_path.open('w', newline='') as csvf:
            writer = csv.writer(csvf)
            header = ['a3\\o4'] + [str(j) for j in range(10)]
            writer.writerow(header)
            for i in range(10):
                writer.writerow([str(i)] + [str(jc[i][j]) for j in range(10)])
        summary_rows.append({
            'o3': o3,
            'count': stats['total'],
            'count_frac': stats['total'] / overall_total,
            'I_bits_given_o3': stats['mi_bits'],
            'max_abs_dev': stats['max_abs_dev'],
            'chi2_stat': stats['chi2'],
            'csv': str(csv_path.name)
        })

    # overall conditional mutual information: sum_z P(o3=z) * I(a3; o4 | o3=z)
    overall_cmi = sum((row['count'] / overall_total) * row['I_bits_given_o3'] for row in summary_rows)

    # write summary CSV
    summary_path = out_dir / 'summary_by_o3.csv'
    with summary_path.open('w', newline='') as csvf:
        fieldnames = ['o3', 'count', 'count_frac', 'I_bits_given_o3', 'max_abs_dev', 'chi2_stat', 'csv']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # print human-readable summary
    print("\nPer-o3 summary (first lines):")
    print(f"{'o3':>3} {'count':>10} {'frac':>8} {'I_bits':>8} {'max_abs_dev':>12} {'chi2':>10}")
    for r in summary_rows:
        print(f"{r['o3']:3d} {r['count']:10d} {r['count_frac']:8.6f} {r['I_bits_given_o3']:8.6f} {r['max_abs_dev']:12.6f} {r['chi2_stat']:10.3f}")
    print(f"\nOverall conditional mutual information I(a3; o4 | o3) = {overall_cmi:.6f} bits")

    print(f"\nAll per-o3 CSVs and summary saved in: {out_dir.resolve()}")
    if bad_lines:
        print(f"Warning: {bad_lines} lines failed to parse and were skipped.")

if __name__ == '__main__':
    main()
