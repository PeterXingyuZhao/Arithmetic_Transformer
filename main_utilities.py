
import string
import pickle
import numpy as np
import random
import tiktoken
import math
import os
import copy
import torch
from pathlib import Path
from typing import Tuple, List, Union

def create_meta_file(vocabulary, input_data_str=None, tokenizer='char'):
    operators_str = string.punctuation
    if vocabulary == 'custom_input_data' and input_data_str:
        print(f"Input file {input_data_str[:100]} specified. Reading data from file...")
        data = input_data_str
        print(f"length of dataset in characters: {len(data):,}")
        vocabulary = 'custom_input_data'
    elif vocabulary == 'numbers_only':
        print(f"Creating meta file for numbers only...")
        data = string.digits + operators_str + ' \n'
    elif vocabulary == 'all_ascii_chars':
        print(f"Creating meta file for all reasonable characters...")
        data = string.ascii_lowercase + string.ascii_uppercase + string.digits + operators_str + ' \n'
    else:
        raise ValueError(f"Vocabulary {vocabulary} not supported!")

    if tokenizer == 'char':
        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        def data_encoder(s):
            data_ids = [stoi[c] for c in s] # encoder: take a string, output a list of integers
            print(f"data has {len(data_ids):,} tokens")
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids
        def data_decoder(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # data_ids = data_encoder(data)
        # print(f"data has {len(data_ids):,} tokens")
        # # convert to np array for efficiency
        # data_ids = np.array(data_ids, dtype=np.uint16)

        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        meta_path = f'meta_{vocabulary}.pkl'

    elif tokenizer == 'gpt2':
        print("Ignore all above messages about the meta file!!!")
        print(f"Tokenizer specified as {tokenizer}. Loading it from tiktoken")
        enc = tiktoken.get_encoding("gpt2")
        # karpathy uses enc.encode_ordinary(), but since there is no decode_ordinary(), I'm switching to .encode()
        def data_encoder(s):
            data_ids = enc.encode(s, allowed_special={"<|endoftext|>"}) # encoder: take a string, output a list of integers
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids

        def data_decoder(l):
            return enc.decode(l) # decoder: take a list of integers, output a string


        meta = {
            'vocab_size': enc.n_vocab,
        }
        meta_path = f'meta_pretrained_gpt2_tokenizer.pkl'

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    return meta, meta_path, data_encoder, data_decoder


def get_encode_decode(meta_path=None, tokenizer='char'):
    import pickle, tiktoken
    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if meta_path and tokenizer == 'char':
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    elif tokenizer:
        print(f"Trying to load tiktoken's openAI {tokenizer} tokenizer")
        enc = tiktoken.get_encoding(f"{tokenizer}")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode


def get_results_dir(config):
    results_dir = config['out_dir']+'/'
    # results_dir += config['dataset']+'_'
    if config['exp_name'] == 'default_exp_name':
        config['exp_name'] = config['wandb_run_name']

    results_dir += config['exp_name']

    if os.path.exists(results_dir):
        print(f"WARNING: results directory {results_dir} already exists, overwriting...")
        id = 1
        while os.path.exists(results_dir+'_'+str(id)):
            id += 1
        results_dir += '_'+str(id)

    os.makedirs(results_dir, exist_ok=True)

    return results_dir

def convert_to_binary(num):
    return bin(num).replace("0b", "")

def remove_zero_pad(a: str):
    assert(all([i=='0' for i in a[::2]]))
    return a[1::2]

def reverse_string(a: str) -> str:
    a = str(a)
    return a[::-1]

def get_num_digits(a: str):
    if a == '':
        return 0
    else:
        if '.' in a: # if a contains a decimal point
            return len(a) - 1
        else:
            return len(str(int(a)))

def numCarryOps(a, b, binary=False):
    def digitSum(n):
        return sum(map(int,str(n)))
    if b == '':
        return 0

    if not binary:
        a,b=int(a),int(b)
        # assert(a >= 0); assert(b >= 0);
        return int((digitSum(a) + digitSum(b) - digitSum(a+b)) / 9)
    else:
        c = int(a,2) + int(b,2)
        return int((digitSum(a) + digitSum(b) - digitSum(convert_to_binary(c))) )
def is_number(s):
    # handle "xey" case (e.g. 1.2e-3) - we do not use this notation in our dataset
    if 'e' in s:
        return False
    elif 'E' in s:
        return False
    elif 'inf' in s or "INF" in s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def gather_test_files(test_file_path: Union[str, Path], main_test_name: str) -> Tuple[List[Path], str]:
    """
    Given test_file_path (file or directory), return a list of Path objects:
      - If test_file_path points to a single .txt file, return [that_file].
      - If it points to a directory, return every file directly under that directory.
    Raises FileNotFoundError if path doesn't exist,
    ValueError if a single-file path is provided but it's not a .txt file.
    """
    p = Path(test_file_path)

    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    if p.is_file():
        if p.suffix.lower() != ".txt":
            raise ValueError(f"Expected a .txt file but got: {p.name}")
        return [p.resolve()], p.stem

    if p.is_dir():
        # list only files (not subdirectories), sorted for deterministic order
        files = sorted([f.resolve() for f in p.iterdir() if f.is_file()])
        return files, main_test_name

    # fallback (e.g., special file types)
    raise ValueError(f"Unsupported path type: {p}")

# --- normalize relative paths in config to absoßlute paths (project root = cwd) ---
def abs_if_rel(p):
    """If p is a non-empty relative path string, return absolute path relative to cwd.
    Otherwise return p unchanged (handles None, empty string, and absolute paths)."""
    if p is None:
        return p
    p = str(p)
    if p == "":
        return p
    if os.path.isabs(p):
        return p
    # treat current working directory as the project's main directory
    return os.path.abspath(os.path.join(os.getcwd(), p))

def concat_strip_dollar(path: Union[str, Path]) -> str:
    """
    Read a text file and return a single string made by concatenating each line.
    If a line ends with '$', that trailing '$' is removed. Each original line
    in the result ends with a newline character '\\n'.
    """
    parts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # remove only newline characters, keep other trailing whitespace
            s = line.rstrip('\r\n')
            # if the last character (before newline) is '$', drop it
            if s.endswith('$'):
                s = s[:-1]
            s = "$" + s + "$"
            parts.append(s)
    return '\n'.join(parts) + ('\n' if parts else '')

def create_meta_for_addition(data, batch_method='per_example'):
    """Create metadata for addition data."""
    # Define the vocabulary for addition problems
    # This includes digits, operators, equals sign, and newline
    if batch_method == 'slicing':
        operators_str = string.punctuation
        data = string.ascii_lowercase + string.ascii_uppercase + string.digits + operators_str + ' \n'

    chars = sorted(list(set(data)))

    # ensure special eos/pad tokens exist
    if '$' not in chars:
        chars.append('$')
    if '<pad>' not in chars:
        chars.append('<pad>')
    chars = sorted(chars)

    # make a readable escaped form for each char
    readable = [c.encode('unicode_escape').decode('ascii') for c in chars]

    vocab_size = len(chars)
    print(f"vocab size: {vocab_size:,}")
    print("all the unique characters:", ", ".join(readable))
    # Create encoder and decoder dictionaries
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def data_encoder(s):
            data_ids = [stoi[c] for c in s] # encoder: take a string, output a list of integers
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids
    def data_decoder(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    
    meta = {
        'vocab_size': vocab_size,
        'vocab': chars,
        'stoi': stoi,
        'itos': itos
    }
    return meta, data_encoder, data_decoder

def create_meta_for_llama(model_name="meta-llama/Meta-Llama-3.1-8B"):
    """
    Create metadata for LLaMA tokenizer.

    This function wraps the LLaMA tokenizer to provide the same interface
    as create_meta_for_addition, making it compatible with the existing
    training and evaluation pipeline.

    Args:
        model_name: HuggingFace model identifier for LLaMA

    Returns:
        tuple: (meta, data_encoder, data_decoder)
            - meta: Dictionary with vocab_size, tokenizer reference, and placeholder stoi/itos
            - data_encoder: Function that encodes text to numpy array of token IDs
            - data_decoder: Function that decodes token IDs to text
    """
    from llama_tokenizer import LlamaTokenizerWrapper

    # Initialize tokenizer wrapper
    wrapper = LlamaTokenizerWrapper(model_name)
    meta = wrapper.create_meta()

    def data_encoder(s):
        """Encode text to token IDs (as numpy array)."""
        token_ids = wrapper.encode(s)
        # Convert to numpy array for consistency with char-level encoder
        return np.array(token_ids, dtype=np.int32)

    def data_decoder(token_ids):
        """Decode token IDs to text."""
        return wrapper.decode(token_ids)

    return meta, data_encoder, data_decoder