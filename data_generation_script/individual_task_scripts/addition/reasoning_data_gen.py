import argparse
import os

def resolve_output_path(input_file_path: str, output_file_path: str) -> str:
    """
    If output_file_path is a directory, write into that directory using the input basename.
    Otherwise treat output_file_path as a full file path.
    """
    if os.path.isdir(output_file_path):
        return os.path.join(output_file_path, os.path.basename(input_file_path))
    return output_file_path

def create_reasoning_data_1(file_path, output_path):
    """
    Transforms raw reasoning data into a format suitable for training.

    Args:
        file_path: Path to the raw data file
        output_path: Path to save the transformed data
    """
    with open(file_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            clean_line = line.strip().replace('$', '')
            if '=' not in clean_line:
                continue
            left_side, result = clean_line.split('=', 1)
            operands = left_side.split('+')

            hundreds, tens, units = [], [], []
            for number in operands:
                n = int(number)
                h = (n // 100) % 10
                t = (n // 10) % 10
                u = n % 10
                hundreds.append(str(h))
                tens.append(str(t))
                units.append(str(u))

            breakdown = (
                f"100({'+'.join(hundreds)})+"
                f"10({'+'.join(tens)})+"
                f"1({'+'.join(units)})"
            )
            transformed_line = f"{left_side}={breakdown}={result}$"
            fout.write(transformed_line + "\n")
    print(f"Transformed data saved to {output_path}")


def create_reasoning_data_2(file_path, output_path):
    with open(file_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            clean_line = line.strip().replace('$', '')
            if '=' not in clean_line:
                continue
            left_side, result = clean_line.split('=', 1)
            operands = left_side.split('+')

            hundreds, tens, units = [], [], []
            for number in operands:
                n = int(number)
                h = (n // 100) % 10
                t = (n // 10) % 10
                u = n % 10
                hundreds.append(str(h))
                tens.append(str(t))
                units.append(str(u))

            breakdown = (
                f"100({'+'.join(hundreds)})+"
                f"10({'+'.join(tens)})+"
                f"1({'+'.join(units)})"
            )

            breakdown_2 = (
                f"100({sum(int(h) for h in hundreds)})+"
                f"10({sum(int(t) for t in tens)})+"
                f"1({sum(int(u) for u in units)})"
            )

            transformed_line = f"{left_side}={breakdown}={breakdown_2}={result}$"
            fout.write(transformed_line + "\n")
    print(f"Transformed data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", required=True)
    parser.add_argument("--output_file_path", required=True)
    parser.add_argument("--mode", default="plain")
    parsed_args = parser.parse_args()

    in_path = os.path.abspath(parsed_args.input_file_path)
    out_path = os.path.abspath(parsed_args.output_file_path)

    if in_path == out_path:
        raise SystemExit(
            f"Error: output_file_path must be different from input_file_path (got {out_path}). "
            "Writing to the same file will truncate it."
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if parsed_args.mode == "plain":
        create_reasoning_data_1(in_path, out_path)
    elif parsed_args.mode == "plain_v2":
        create_reasoning_data_2(in_path, out_path)
    else:
        raise SystemExit(f"Error: Unknown --mode {parsed_args.mode}")

if __name__ == "__main__":
    main()