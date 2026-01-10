from main_utilities import *
from tqdm.auto import tqdm
import torch
import numpy as np
import random
import math
import os
import pandas as pd
import csv


def get_abc_new(abc: str, data_format="plain", mode: str = "compute_gold"):
    """Unified parser: mode='compute_gold' computes the groudtruth on the fly;
       mode='read_gold_as_str' reads the groundtruth from the evaluation files (testing, validation) to do string matching.
    Returns either
      (operands_str, result_int, operation)            # compute_gold
    or
      (operands_str, result_str)                      # read_gold_as_str
    """

    def _is_permutation_of_1_to_n(perm: str, n: int) -> bool:
        """Return True if perm is a permutation of '1'..str(n)."""
        if len(perm) != n:
            return False
        try:
            expected = {str(i) for i in range(1, n + 1)}
            return set(perm) == expected
        except Exception:
            return False

    # Split the input string into parts
    parts = abc.split('=')
    if len(parts) != 2:
        print(f'Invalid format, expected "a+b+c...=result", got: {abc}')
        return None, None, None

    # Get the operands part (before =)
    operands_str = parts[0]
    if operands_str and operands_str[0] == '$':
        operands_str = operands_str[1:]
    if operands_str.startswith('Input:\n'):
        operands_str = operands_str.split('Input:\n')[-1]
    if 'Target' in operands_str:
        operands_str = operands_str.split('\nTarget')[0]

    # version 1: compute the result
    if mode == "compute_gold":
        if '+' in abc:
            operation = '+'
        elif '-' in abc:
            operation = '-'
        elif '*' in abc:
            operation = '*'
        else:
            print(f'operation not found, abc: {abc}')
            return None, None, None
        # Split into individual operands
        operands = [op.strip() for op in operands_str.split(operation)]

        # Clean up operands
        operands = [op.replace(' ', '') for op in operands]

        if operation == '+':
            result = sum(int(op) for op in operands)
        elif operation == '-':
            result = int(operands[0]) - sum(int(op) for op in operands[1:])
        elif operation == '*':
            result = 1
            for op in operands:
                result *= int(op)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return operands_str, result, operation
    
    # version 2: read the groundtruth from the evaluation files
    if mode == "read_gold_as_str":
        # parts[1] is the result part, which may contain a trailing '$' or newline
        result_str = parts[1].strip()
        if result_str.endswith('\n'):
            result_str = result_str[:-1].strip()
        if result_str.endswith('$'):
            result_str = result_str[:-1].strip()
        if data_format == "reverse":
            sign = ''
            if result_str.startswith('-') or result_str.startswith('+'):
                sign = result_str[0]
                result_str = result_str[1:]
            result_str = sign + result_str[::-1]  # reverse the result string if needed

    # version 2: read the groundtruth from the evaluation files
    if mode == "read_gold_as_str":
        # parts[1] is the result part, which may contain a trailing '$' or newline
        result_str = parts[1].strip()
        if result_str.endswith('\n'):
            result_str = result_str[:-1].strip()
        if result_str.endswith('$'):
            result_str = result_str[:-1].strip()

        # Preserve and strip sign if present
        sign = ''
        if result_str and (result_str[0] == '-' or result_str[0] == '+'):
            sign = result_str[0]
            core = result_str[1:]
        else:
            core = result_str

        # normalize data_format naming
        df = (data_format or "").lower()
        if df in ("plain", "normal", "1234"):
            # no change
            normalized = core
        elif df in ("reverse", "reversed", "4321"):
            # reverse digits back to canonical order
            normalized = core[::-1]
        else:
            # try to interpret data_format as a permutation like "2143"
            # Only attempt if permutation length matches core length and it's a permutation of 1..n
            perm = data_format.strip()
            n = len(core)
            if _is_permutation_of_1_to_n(perm, n):
                # inverse-permute: given output[j] = s[perm[j]-1], we recover s by:
                s_list = [''] * n
                for j, ch in enumerate(core):
                    target_idx = int(perm[j]) - 1  # where this char belongs in the original string
                    if 0 <= target_idx < n:
                        s_list[target_idx] = ch
                    else:
                        # malformed perm, fallback to core unchanged
                        s_list = None
                        break
                if s_list is None or '' in s_list:
                    # fallback: treat as plain if something went wrong
                    normalized = core
                else:
                    normalized = ''.join(s_list)
            else:
                # unknown format: fallback to treating as plain (no-op)
                normalized = core

        # reattach sign if present
        result_str_canonical = (sign + normalized) if sign else normalized

        return operands_str, result_str_canonical

    # If an unknown mode is passed
    print(f"Unknown mode: {mode}")
    return None, None, None

_precomputed_batches = {}
def prepare_addition_batches(config, encode, num_digit=3, data_type='binary', 
                             operator='+', data_format='plain', add_space=False, mode: str = "compute_gold", batch_method: str = "per_example"):
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    print(f"Preparing batches from: {start}")
    
    if start.startswith('FILE:'): # start is just the test file path
        with open(start[5:], 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
    else:
        lines = start.splitlines()

    total = len(lines)
    print(f'Preparing batches for {total} examples from: {start}')
    
    # Process all lines and group by prompt length
    prompt_dict = {}
    for line in lines:
        # split off gold answer
        # e.g. line = "123+456=579"
        if batch_method == 'per_example':
            prompt_str = line.split('=')[0] + '='  # keep the '=' at the end
        else:
            prompt_str = '$' + line.split('=')[0] + '='      # "123+456="
        prompt_ids = encode(prompt_str)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
        prompt_length = x.size(1)

        # parse out gold for evaluation later
        operands, result= get_abc_new(
            line,
            data_format=data_format,
            mode=mode
        )

        entry = (x, operands, result)
        prompt_dict.setdefault(prompt_length, []).append(entry)

    # Construct batches of prompts
    batch_list = []
    for prompt_length in prompt_dict.keys():
        input_tuple_list = prompt_dict[prompt_length]
        for batch_idx in range(math.ceil(len(input_tuple_list)/test_batch_size)):
            batch_list.append(input_tuple_list[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size])

    print(f'Created {len(batch_list)} batches')
    
    # Cache the batches using a hash of the configuration
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{data_format}_{add_space}"
    _precomputed_batches[batch_key] = (batch_list, total)
    
    return batch_list, total

# Modified evaluation function that uses pre-created batches
def evaluate_addition_precomputed(config, model, ctx, decode, batch_list, total,
                                  verbose=False, num_digit=3, data_format='plain', add_space=False, 
                                  operator='+', verbose_correct=False, analyze=False, mode: str = "compute_gold", randomize=None):
    """
    randomize: None | "units" | "tens" | "hundreds" | "thousands"
    If randomize is None: behave exactly as before.
    Otherwise: ignore the specified place when checking correctness (units = last digit).
    """
    model.eval()
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    
    # base values from config
    temperature = config.get('temperature', 0.8)
    top_k = config.get('top_k', 200)

    # if greedy flag is set, force deterministic greedy decoding
    if config.get('greedy', False):
        top_k = 1
        temperature = 0.0

    if add_space:
        max_new_tokens = 2 * num_digit + 3

    correct = 0

    op = operator
    correct_examples = []
    incorrect_examples = []

    # helper to extract "digit-like" characters (keep digits and '?', which you use as mask)
    def extract_digits_allow_mask(s: str):
        s = str(s)
        # keep digits and '?' as placeholders; remove everything else (signs, spaces, punctuation)
        return ''.join(ch for ch in s if ch.isdigit() or ch == '?')

    # map randomize name to index-from-right (0-based)
    place_to_offset = {
        "units": 0,
        "tens": 1,
        "hundreds": 2,
        "thousands": 3
    }

    # --- NEW helper functions for data_format handling ---
    def reverse_string(s: str) -> str:
        return s[::-1]

    def is_permutation_of_1_to_n(perm: str, n: int) -> bool:
        if len(perm) != n:
            return False
        expected = {str(i) for i in range(1, n + 1)}
        return set(perm) == expected

    def invert_permutation(core: str, perm: str) -> str:
        """
        Given core (string without sign) and perm like '2143' produced by generator
        where permuted[j] = original[perm[j]-1], recover original.
        If len(core) < len(perm), left-pad core with zeros to match length.
        If perm malformed, return core unchanged.
        """
        n = len(perm)
        c = core
        if len(c) < n:
            # left-pad with zeros to support permutations that expect fixed width (e.g. 4 digits)
            c = c.rjust(n, '0')
        if len(c) != n:
            # can't invert if lengths don't match after padding
            return core

        if not is_permutation_of_1_to_n(perm, n):
            return core

        s_list = [''] * n
        for j, ch in enumerate(c):
            try:
                target_idx = int(perm[j]) - 1
            except Exception:
                return core
            if 0 <= target_idx < n:
                s_list[target_idx] = ch
            else:
                return core
        # If anything left blank, fallback
        if '' in s_list:
            return core
        return ''.join(s_list)

    def normalize_by_data_format(raw: str, data_format: str) -> str:
        """
        Given raw predicted result (possibly with sign), convert it into canonical (unpermuted) representation.
        Handles 'plain'/'normal', 'reverse' (or 'reversed'), and explicit permutations like '2143'.
        """
        if raw is None:
            return raw
        s = raw.strip()
        # remove any trailing newlines/spaces already done outside, but be defensive
        # preserve sign if present
        sign = ''
        if s.startswith('-') or s.startswith('+'):
            sign = s[0]
            core = s[1:]
        else:
            core = s

        # remove any internal whitespace that might have been introduced
        core = core.replace(' ', '')

        df = (data_format or "").lower()

        if df in ("plain", "normal", "1234", ""):
            normalized_core = core
        elif df in ("reverse", "reversed", "4321"):
            normalized_core = core[::-1]
        else:
            # try to treat data_format as a permutation like '2143'
            perm = data_format.strip()
            # only attempt if perm is all digits
            if perm.isdigit():
                normalized_core = invert_permutation(core, perm)
            else:
                # unknown format -> fallback to plain
                normalized_core = core

        # reattach sign if present
        return (sign + normalized_core) if sign else normalized_core

    # --- end new helpers ---

    for batch_idx in tqdm(range(len(batch_list))):
        batch = batch_list[batch_idx]
        x_list = [input_tuple[0] for input_tuple in batch]
        x = torch.cat(x_list, dim=0)

        # Run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                outcome_list = [decode(y_i.tolist()) for y_i in y]

                for i, outcome in enumerate(outcome_list):
                    _, operands, result = batch[i]

                    if mode == "compute_gold":
                        c_hat = outcome.split('=')[1].split('$')[0].strip()

                        # plain addition: take first line if multiline
                        c_hat = c_hat.split('\n')[0]

                        # normalize according to data_format (reverse, permutation, etc.)
                        c_hat = normalize_by_data_format(c_hat, data_format)

                        if add_space:
                            c_hat = c_hat.replace(' ', '')

                        if is_number(c_hat):
                            if '.' in c_hat:
                                c_hat = float(c_hat)
                            else:
                                c_hat = int(c_hat)
                        else:  # c_hat is not a number; keep result as string for mismatch reporting
                            result = str(result)

                    if mode == "read_gold_as_str":
                        c_hat = outcome.split('=')[1].split('$')[0].strip()

                        # normalize according to data_format (reverse, permutation, etc.)
                        c_hat = normalize_by_data_format(c_hat, data_format)

                    # --- New: masking/randomize-aware correctness check ---
                    is_correct = False
                    if randomize is None:
                        # preserve original semantics: compare raw values
                        is_correct = (result == c_hat)
                    else:
                        # compare digit-by-digit while ignoring one place
                        # convert both to digit sequences (keeping '?' as placeholder if present)
                        res_digits = extract_digits_allow_mask(result)
                        pred_digits = extract_digits_allow_mask(c_hat)

                        # right-align both by left-padding with zeros so positions match
                        L = max(len(res_digits), len(pred_digits))
                        # ensure length covers the place being ignored (so thousands can be ignored even when len < 4)
                        if randomize in place_to_offset:
                            needed = place_to_offset[randomize] + 1
                            L = max(L, needed)

                        res_padded = res_digits.rjust(L, '0')
                        pred_padded = pred_digits.rjust(L, '0')

                        # index to ignore (from left, 0-based)
                        ignore_idx = L - 1 - place_to_offset.get(randomize, 0)

                        # compare all positions except ignore_idx
                        all_match = True
                        for idx in range(L):
                            if idx == ignore_idx:
                                continue
                            if res_padded[idx] != pred_padded[idx]:
                                all_match = False
                                break
                        is_correct = all_match

                    # record example
                    if is_correct:
                        correct += 1
                        correct_examples.append((operands, result, outcome, c_hat))
                        if verbose_correct:
                            print('outputs(o): ', outcome)
                            print(f'correct: {operands}={result}')
                    else:
                        incorrect_examples.append((operands, result, outcome, c_hat))
                        if verbose:
                            print('outputs(x): ', outcome)
                            print(f'wrong  : {operands}={c_hat}')
                            print(f'correct: {operands}={result}')
                    

    accuracy = correct / total * 100

    model.train()
    return accuracy, correct_examples, incorrect_examples


# Keep the original function for backward compatibility, but make it use the new functions
def evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3,
                          data_type='binary', operator='+', data_format='plain', add_space=False, verbose_correct=False,
                          analyze=False, mode: str = "compute_gold", batch_method: str = "per_example", randomize=None):
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{data_format}_{add_space}"
    
    if batch_key in _precomputed_batches:
        print("Using precomputed batches")
        batch_list, total = _precomputed_batches[batch_key]
    else:
        print("Creating new batches")
        batch_list, total = prepare_addition_batches(
            config, encode, num_digit=num_digit, 
            data_type=data_type, operator=operator, data_format=data_format, add_space=add_space, mode=mode, batch_method=batch_method
        )

    # Evaluate using the batches
    return evaluate_addition_precomputed(
        config, model, ctx, decode, batch_list, total, verbose=verbose,
        num_digit=num_digit, data_format=data_format,
        add_space=add_space, operator=operator, verbose_correct=verbose_correct, analyze=analyze, mode=mode, randomize=randomize
    )

def evaluate_multiple_files(config, model, ctx, encode, decode, test_files, iter_num, result_dir,
                          verbose=False, num_digit=3, data_type='binary', operator='+', 
                          data_format='plain', analyze=False, mode: str = "compute_gold", batch_method: str = "per_example", randomize=None):
    """
    Evaluate model on multiple test files and store results.
    Args:
        test_files: List of test file paths
        iter_num: Current iteration number
        result_dir: Directory to store results
    Returns:
        dict: Dictionary containing accuracies for each test file
    """

    test_names = []
    accuracy_multiple_files = {}
    correct_multiple_files = {}
    incorrect_multiple_files = {}

    for test_file in test_files:
    
        # Get test file name without path and extension
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        test_names.append(test_name)
        
        # Set the current test file as start
        config['start'] = f"FILE:{test_file}"
        
        # Run evaluation
        accuracy, correct, incorrect = evaluate_addition_batch(
            config, model, ctx, encode=encode, decode=decode,
            verbose=verbose, num_digit=num_digit,
            data_type=data_type, operator=operator,
            data_format=data_format, analyze=analyze, mode=mode, batch_method=batch_method, randomize=randomize
        )

        accuracy_multiple_files[test_name] = accuracy
        correct_multiple_files[test_name] = correct
        incorrect_multiple_files[test_name] = incorrect
        
        # Path for this test file's results
        results_file = os.path.join(result_dir, f'{test_name}_results.csv')
        
        # Combine correct and incorrect examples and sort by operands to maintain consistent order
        all_examples = correct + incorrect
        all_examples.sort(key=lambda x: x[0])  # Sort by operands
        
        # Create new DataFrame with operands and actual results
        new_df = pd.DataFrame({
            'operands': [ex[0] for ex in all_examples],
            'actual': [ex[1] for ex in all_examples],
            f'pred_iter_{iter_num}': [ex[3] for ex in all_examples]
        })
        
            # --- before merging: ensure consistent types and remove duplicates ---
        if os.path.exists(results_file):
            old_df = pd.read_csv(results_file, dtype={'operands': str, 'actual': str}, low_memory=False)
            
            # normalize strings
            for df in (old_df, new_df):
                df['operands'] = df['operands'].astype(str).str.strip()
                df['actual']   = df['actual'].fillna('').astype(str).str.strip()

            # drop exact duplicate rows on key columns to avoid many-to-many merges
            old_df = old_df.drop_duplicates(subset=['operands', 'actual'])
            new_df = new_df.drop_duplicates(subset=['operands', 'actual'])

            # set multi-index and join (this avoids Cartesian duplication)
            old_idx = old_df.set_index(['operands', 'actual'])
            new_idx = new_df.set_index(['operands', 'actual'])

            # do the join; new columns will be added, existing columns preserved
            merged_idx = old_idx.join(new_idx, how='outer')

            # optional: sanity check that the join didn't blow up
            if len(merged_idx) > len(old_idx) + len(new_idx):
                # this is a conservative check; it triggers if many-to-many occurred
                print(f"Warning: merged size {len(merged_idx)} > old+new ({len(old_idx)}+{len(new_idx)}) — check duplicate keys!")

            merged_df = merged_idx.reset_index()
        else:
            merged_df = new_df

        
        # Save results
        merged_df.to_csv(results_file, index=False)
        
        # Save accuracy separately in a summary file
        accuracy_file = os.path.join(result_dir, f'{test_name}_accuracy.csv')
        if os.path.exists(accuracy_file):
            acc_df = pd.read_csv(accuracy_file)
        else:
            acc_df = pd.DataFrame(columns=['iteration', 'accuracy'])
        
        # Add new accuracy
        new_row = pd.DataFrame({'iteration': [iter_num], 'accuracy': [accuracy]})
        acc_df = pd.concat([acc_df, new_row], ignore_index=True)
        acc_df.to_csv(accuracy_file, index=False)
    
    return test_names, accuracy_multiple_files, correct_multiple_files, incorrect_multiple_files