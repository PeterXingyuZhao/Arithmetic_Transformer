#!/usr/bin/env python3
"""
actual_column_correction.py

Reads a CSV, uses column 0 ("operands") to compute a sorted string,
and writes that sorted string into column 1 ("actual"), leaving all
other columns unchanged.

- Detects delimiter automatically (comma, tab, etc).
- Splits the operands field on commas (operands are expected to be comma-separated).
- For each token, tries to extract the first integer-looking substring (e.g. "1070", "81116" from "81116<pad>").
  If no integer substring exists, the raw token is kept and sorted lexicographically after numeric tokens.
- Preserves header and all other columns exactly.

Usage:
    python actual_column_correction.py input.csv [output.csv] [--inplace]

"""
import argparse
import csv
import re
import sys
from pathlib import Path

def detect_delimiter(sample: str) -> str:
    """Try to detect delimiter using csv.Sniffer; fallback to comma."""
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        return ','

_int_re = re.compile(r'-?\d+')

def token_to_key(token: str):
    """
    Returns a tuple to use as a sort key:
      - (0, integer_value) for tokens that contain a detectable integer
      - (1, token_lower) for tokens without integers (so they sort after numeric tokens)
    """
    token_stripped = token.strip()
    matches = _int_re.findall(token_stripped)
    if matches:
        try:
            # use the first integer-like match
            val = int(matches[0])
            return (0, val)
        except Exception:
            pass
    # no integer found — sort after numbers using case-insensitive token
    return (1, token_stripped.lower())

def process_row(row, num_columns):
    """
    Given a row (list of strings), use row[0] (operands) to produce a sorted
    comma-joined string and place it into row[1] (actual). If row has fewer
    than 2 columns, it will be extended with empty strings as needed.
    """
    # Ensure row has at least num_columns capacity (we preserve whatever columns exist)
    if len(row) < 2:
        # extend row to ensure index 1 exists
        row = row + [''] * (2 - len(row))

    operands_field = row[0] if len(row) >= 1 else ""
    # split by commas (operands are comma-separated)
    # keep empty tokens if present? Here we drop empty tokens to avoid empty entries.
    tokens = [t.strip() for t in re.split(r',', operands_field) if t.strip() != ""]

    # If there's nothing parseable, set actual to empty
    if not tokens:
        row[1] = ""
        return row

    # Create sorted list using token_to_key
    sorted_tokens = sorted(tokens, key=token_to_key)

    # Join back with commas (no spaces)
    row[1] = ",".join(sorted_tokens)
    return row

def main():
    parser = argparse.ArgumentParser(description="Replace 'actual' column with sorted 'operands' column.")
    parser.add_argument("input", type=Path, help="Input CSV file path")
    parser.add_argument("output", nargs="?", type=Path, default=None, help="Output CSV file path (optional)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file {input_path} not found", file=sys.stderr)
        sys.exit(2)

    # Read a small sample to detect delimiter
    sample = input_path.read_text(errors='ignore')[:4096]
    delimiter = detect_delimiter(sample)

    # If user requested inplace, set output accordingly
    if args.inplace:
        output_path = input_path
    else:
        if args.output:
            output_path = args.output
        else:
            output_path = input_path.with_name(input_path.stem + "_sorted_actual" + input_path.suffix)

    # Read all rows, process, and write out preserving delimiter and quoting
    with input_path.open(newline='', encoding='utf-8', errors='ignore') as fin:
        reader = csv.reader(fin, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        print("Input file is empty. Nothing to do.", file=sys.stderr)
        sys.exit(0)

    # We'll preserve the header (first row) as-is, and treat subsequent rows as data.
    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []

    # Process each data row
    processed_rows = [header]
    for r in data_rows:
        processed_rows.append(process_row(list(r), len(r)))

    # Write with same delimiter
    with output_path.open('w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for r in processed_rows:
            writer.writerow(r)

    print(f"Wrote {len(processed_rows)-1} data rows to: {output_path}")

if __name__ == "__main__":
    main()
