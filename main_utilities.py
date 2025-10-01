
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

def generate_data_str(data_list, operator='+', format='plain', train=True, shuffle=True, fewshot=False, prompt=None, add_space=False, simple=False, random_A=False, random_C=False):

    if format == 'algo_reasoning' and add_space:
        # TODO: add_space=True will add a space between each numbers, but not yet supported for algo_reasoning
        raise ValueError("add_space=True not supported for algo_reasoning format!")

    if shuffle:
        random.shuffle(data_list)

    if fewshot:
        with open(prompt, 'r') as f:
            prompt = f.read()

    # for idx, (x1, x2, y) in enumerate(data_list):
    for idx, data_tuple in enumerate(data_list):
        operator = data_tuple[-1]
        if operator in ['+', '-', '*']:
            x1, x2, y = data_tuple[0], data_tuple[1], data_tuple[2]
            if train:
            # create training data (x1+x2=y)
                if format == 'plain':
                    output_str = f"{x1}{operator}{x2}={y}\n"
                elif format == 'plain2':
                    output_str = f"${x1}{operator}{x2}={y}$\n"
                elif format == 'reverse':
                    output_str = f"${x1}{operator}{x2}={str(y)[::-1]}$\n"
                elif format == 'reverse2':
                    output_str = f"{x1}{operator}{x2}={str(y)[::-1]}\n"
                elif format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x1, x2, operator=operator, train=train, simple=simple, random_A=random_A, random_C=random_C)
            else:
                # create test data (x1+x2=)
                if format == 'plain':
                    output_str = f"{x1}{operator}{x2}=\n"
                elif format == 'plain2':
                    output_str = f"${x1}{operator}{x2}=\n"
                elif format == 'reverse':
                    output_str = f"${x1}{operator}{x2}=\n"
                elif format == 'reverse2':
                    output_str = f"{x1}{operator}{x2}=\n"
                elif format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x1, x2, operator=operator, train=train, simple=simple, random_A=random_A, random_C=random_C)
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str

        elif operator in ['sin', 'sqrt']:
            x, y = data_tuple[0], data_tuple[1]
        # for idx, (x, y) in enumerate(data_list):
            if train:
                if format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x, operator=operator, train=train)
                else:
                    output_str = f"{operator}({x})={y}\n"
            else:
                if format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x, operator=operator, train=train)
                else:
                    output_str = f"{operator}({x})=\n"
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str

        elif operator in ['text']:
            output_str = data_tuple[0]
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str+'\n\n'
            else:
                data_str += output_str+'\n\n'

    return data_str

def get_data_list(filename=None, operator='+', delim=None):
    import re
    data_list = []
    if filename: # read data from file
        if operator in ['text']:
            with open(filename, 'r') as f:
                data = f.read()
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            with open(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # if first char is $, assume it's a delimiter
                if line[0] == '$':
                    delim = '$'
                if delim:
                    # remove delim from line
                    line = line.replace(delim, '')
                # x1, x2 = line.strip().split(operator)
                if operator in ['+', '-', '*']:
                    x1, x2 = re.split(r'[+\-\*]', line.strip())
                    x2, y = x2.split("=")
                    if operator == '+':
                        y2 = int(x1) + int(x2)
                    elif operator == '-':
                        y2 = int(x1) - int(x2)
                    elif operator == '*':
                        y2 = int(x1) * int(x2)

                    data_list.append((int(x1), int(x2), int(y2), operator))
    else: # generate random data
        
        for _ in range(1000):
            if operator in ['+', '-', '*']:
                x1, x2 = random.randint(0, 999), random.randint(0, 999)
                if operator == '+':
                    y = x1 + x2
                elif operator == '-':
                    y = x1 - x2
                elif operator == '*':
                    y = x1 * x2
                data_list.append((int(x1), int(x2), int(y), operator))
                if operator == 'sin':
                    x = random.uniform(-math.pi/2, math.pi/2)
                    x = math.floor(x * 10000) / 10000
                    y = math.sin(x)
                elif operator == 'sqrt':
                    x = random.uniform(0, 10)
                    x = math.floor(x * 10000) / 10000
                    y = math.sqrt(x)

                y = math.floor(y * 10000) / 10000

                data_list.append((float(x), float(y), operator))

    return data_list

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

def get_batch(split):
    data = train_data if split == 'train' else val_data
    if train_both:
        data2 = train_data2 if split == 'train' else val_data2
        batch_size2 = int(batch_size*data_ratio)
        ix = torch.randint(len(data) - block_size, (batch_size-batch_size2,))
        ix2 = torch.randint(len(data2) - block_size, (batch_size2,))
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if train_both:
        x2 = torch.stack([torch.from_numpy((data2[i:i+block_size]).astype(np.int64)) for i in ix2])
        y2 = torch.stack([torch.from_numpy((data2[i+1:i+1+block_size]).astype(np.int64)) for i in ix2])
        x = torch.cat([x,x2])
        y = torch.cat([y,y2])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

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