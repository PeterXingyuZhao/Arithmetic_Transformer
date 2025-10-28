import os
import pickle
import requests
import numpy as np
import random
from tqdm import tqdm
import copy
import time

from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import wandb
import torch.nn.functional as F
import math

from model import GPTConfig, GPT
from main_utilities import *
from evaluation import *
from statistical_measurements import *

import re
from pathlib import Path as _Path
import argparse as _argparse
import result_analysis
# I/O

out_dir = '/drive/MyDrive/addition/plain_no_pad/out'
resume_dir = None
resume_iter = False # if True, resume from saved iter_num, otherwise resume from iter_num 0
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_entity = 'ssdd'
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
exp_name = 'default_exp_name'

# data
dataset = 'bal'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
test_batch_size = 128
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
data_dir = 'data/4_operands_0_to_999_uniform/'
train_data_name = 'train.txt'
train_data_test_name = "train_eval.txt"
val_data_name = 'val.txt'
test_file_name = 'test.txt'
main_test_name = "test" # if test_file_path is a directory, this is the name of the test file to be displayed in wandb
multi_digit = False
num_digit = 3
max_new_tokens = 5
binary = False

eval_addition = False # if True compute test accuracy of "a+b="

eval_other = False # use this to evaluate other operations (ex. train on operator '-' but evaluate on other_operator '+')
other_operator = '+'
eval_addition_train = False
zero_pad = False
algo_reason = False
add_space = False

# model
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
ckpt_path_name = 'ckpt.pt'
save_final = True

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = None # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
use_flash = True
data_type = 'binary' # 'binary' by default, can be 'text'
operator = '+' # can be '+', '-', '*', 'sin', 'sqrt'
data_shuffle = True
data_format = 'plain' # 'plain' or 'reverse' or 'algo_reasoning'
vocabulary = 'all_ascii_chars' # can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
meta_path_specified = True # use saved meta_file (False if data_type='text')
eps = 0
tokenizer = 'char' # by default, use char level tokenizer. but for pretrained models, use openai tokenizer eg: 'gpt2'

simple=False
random_A=False
random_C=False

use_lora = False # use lora (from minLoRA)
print_interval = 2  # if we're using gpt-2 model, I want to see it prompted on text

mode = "compute_gold"  # Mode for evaluation: "compute_gold" or "read_gold_as_str"

more_early_eval1 = False # if True, do early, more frequent eval on train and val data
early_eval_interval1 = 25
early_eval_border1 = 1000

more_early_eval2 = False # if True, do even earlier, even more frequent eval on train and val data
early_eval_interval2 = 5
early_eval_border2 = 500

# additional statistical measurements
mi_measurement = False # whether to do mutual information measurement
early_mi_measure_border = 20000 # border for early mutual information measurement
early_mi_measure_interval = 1000 # interval for early mutual information measurement
final_mi_measure_interval = 5000 # interval for final mutual information measurement
stats_measurement_data_name = '4_operand_addition_stats_measurement_data_reversed.txt'

drop_leading_digit = False

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]

# ---- begin: accept either a config file positional OR --task/--experiment_name flags ----

# parse only the two special options here, leave the rest for configurator.py
_cfg_parser = _argparse.ArgumentParser(add_help=False)
_cfg_parser.add_argument('config_path', nargs='?', help='Optional path to a config file (positional).')
_cfg_parser.add_argument('--task', choices=['addition', 'multiplication', 'sorting'], help='If provided, load a default config for this task.')
_cfg_parser.add_argument('--experiment_name', help='If provided with --task, override experiment/data directories with this experiment name.')
# NEW: select batch preparation method: 'per_example' (default) or 'slicing'
_cfg_parser.add_argument('--batch', choices=['per_example','slicing'], default='per_example',
                        help="Batch preparation method: per_example uses Dataset+DataLoader; slicing uses random-token slices.")
_known_args, _remaining_argv = _arg_parser = _cfg_parser.parse_known_args()


cfg_file_to_load = None
if _known_args.config_path:
    # user supplied explicit config file path as first positional argument
    config_path = "./configuration_files/" + _known_args.config_path
    cfg_file_to_load = os.path.abspath(config_path)
elif _known_args.task:
    # map task -> default config filename (adjust filenames/locations if yours live elsewhere)
    DEFAULT_CFG_MAP = {
        'addition': '4_operands_addition_plain.txt',
        'multiplication': '2_operands_mul_plain.txt',
        'sorting': '4_operands_sorting.txt',
    }
    candidate = DEFAULT_CFG_MAP.get(_known_args.task)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "configuration_files", candidate)
    if os.path.exists(cfg_path):
        cfg_file_to_load = cfg_path
    else:
        # not found; warn and continue with built-in defaults
        print(f"Warning: default config for task '{_known_args.task}' not found at '{cfg_path}'. Using built-in defaults.", file=sys.stderr)
        cfg_file_to_load = None

# If we found a config file to load, exec it into globals (overrides default variables defined above)
if cfg_file_to_load:
    print(f"Loading config file: {cfg_file_to_load}")
    with open(cfg_file_to_load) as f:
        print(f.read())
    exec(open(cfg_file_to_load).read())

    # If user provided --experiment_name, patch data_dir and out_dir to use the new experiment name.
    if _known_args.experiment_name:
        new_name = _known_args.experiment_name
        # attempt to detect the existing 'experiment base' in data_dir (most robust), fallback to out_dir basename
        old_base = None
        if 'data_dir' in globals() and isinstance(data_dir, str):
            old_base = os.path.basename(os.path.normpath(data_dir))
            if old_base:
                data_dir = data_dir.replace(old_base, new_name)
        if 'out_dir' in globals() and isinstance(out_dir, str):
            if old_base is None:
                old_base = os.path.basename(os.path.normpath(out_dir))
            if old_base:
                out_dir = out_dir.replace(old_base, new_name)
        print(f"Overrode experiment name -> data_dir='{data_dir}', out_dir='{out_dir}'")

# ---- end: alternate-loading logic ----

# build config dict for logging as before
config = {k: globals()[k] for k in config_keys}

train_data_path = os.path.join(data_dir, train_data_name)
train_data_test_path = os.path.join(data_dir, train_data_test_name) if eval_addition_train else None
val_data_path = os.path.join(data_dir, val_data_name)
test_file_path = os.path.join(data_dir, test_file_name)
stats_measurement_data_file_path = os.path.join(data_dir, stats_measurement_data_name) if mi_measurement else None


# Apply normalization to the known path variables (if they exist in globals)
for varname in (
    "out_dir",
    "train_data_path",
    "train_data_test_path",
    "val_data_path",
    "test_file_path",
    "stats_measurement_data_file_path",
):
    if varname in globals():
        globals()[varname] = abs_if_rel(globals()[varname])

mi_measure_iters = set(
    list(range(0,  early_mi_measure_border, early_mi_measure_interval)) +    # every 20 steps before 200
    # list(range(100000, 100000, 20)) +   # every 50 steps from 200 up to 1500
    list(range(early_mi_measure_border, max_iters+1, final_mi_measure_interval))  # every 100 steps thereafter
)

def encode_addition(text, meta):
    """Encode text to tensor using the metadata."""
    return torch.tensor([meta['stoi'][c] for c in text], dtype=torch.long)

def decode_addition(tensor, meta):
    """Decode tensor to text using the metadata."""
    if isinstance(tensor, torch.Tensor):
        return ''.join([meta['itos'][i.item()] for i in tensor])
    else:
        return ''.join([meta['itos'][i] for i in tensor])
    
def pad_sequence(x: torch.Tensor, length: int, pad_value: int):
    if x.size(0) < length:
        padding = torch.full((length - x.size(0),), pad_value, dtype=torch.long)
        return torch.cat([x, padding], dim=0)
    else:
        return x

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# function to set seed for all random number generators
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # to make sure GPU runs are deterministic even if they are slower set this to True
    torch.backends.cudnn.deterministic = False
    # warning: this causes the code to vary across runs
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))

if min_lr == None:
    min_lr = learning_rate/10
master_process = True
seed_offset = 0
if master_process:
  os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True # cudnn auto-tuner
torch.backends.cudnn.deterministic = False # cudnn auto-tuner
# this is probably overkill but seed everything again
set_seed(1337 + seed_offset)

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



train_data_str = concat_strip_dollar(train_data_path)
val_data_str = concat_strip_dollar(val_data_path)

# Create metadata from the combined data
meta, data_encoder, data_decoder = create_meta_for_addition(train_data_str)

# Decide batch method from CLI
batch_method = getattr(_known_args, 'batch', 'per_example')
print(f"Using batch preparation method: {batch_method}")

def get_infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

class AdditionDataset(Dataset):
    def __init__(self, file_path, meta):
        self.meta = meta
        # Read the text file
        with open(file_path, 'r') as f:
            raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        # Remove any empty lines and strip whitespace
        self.lines = [line if line.endswith('$') else line + '$' for line in raw_lines]
        self.block_size = block_size  # from your config
        # pad/eos ids from meta
        self.pad_id = self.meta['stoi'].get('<pad>')
        if self.pad_id is None:
            raise ValueError("pad token '<pad>' not found in meta['stoi']")
        self.eos_id = self.meta['stoi'].get('$')
        if self.eos_id is None:
            raise ValueError("eos token '$' not found in meta['stoi']")
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line_with_trailing_eos = self.lines[idx]
        # Convert the line to tensor using our encoder
        raw = encode_addition(line_with_trailing_eos, self.meta)  # raw ends with eos_id
        x = pad_sequence(raw[:-1], block_size, pad_value=self.pad_id)
        y = pad_sequence(raw[1:],  block_size, pad_value=self.pad_id)  # -100 is ignore_index
        return x, y

if batch_method == 'per_example':
    # per-example: use AdditionDataset and DataLoader
    train_dataset = AdditionDataset(train_data_path, meta)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device_type=='cuda')
    )
    val_dataset = AdditionDataset(val_data_path, meta)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device_type=='cuda')
    )
    train_loader_iter = get_infinite_dataloader(train_loader)

    # Ensure estimate_loss uses dataloaders below (we'll check it later)

else:
    # slicing: convert full text into 1D token tensor(s)
    # If train_data_str/val_data_str not created, load files (safe fallback)
    if 'train_data_str' not in globals():
        train_data_str = concat_strip_dollar(train_data_path)  # or open/read
    if 'val_data_str' not in globals():
        val_data_str = concat_strip_dollar(val_data_path)

    train_data = data_encoder(train_data_str)  # 1D tensor / numpy -> convert below if needed
    val_data = data_encoder(val_data_str)

    # Ensure 1D torch tensors
    if isinstance(train_data, np.ndarray):
        train_data = torch.from_numpy(train_data)
    if isinstance(val_data, np.ndarray):
        val_data = torch.from_numpy(val_data)

    # slicing get_batch used in your second file
    def get_batch(split='train'):
        data = train_data if split == 'train' else val_data
        assert data.dim() == 1, "slicing expects 1D sequence of token ids"
        max_start = data.size(0) - block_size - 1
        if max_start <= 0:
            raise ValueError("Dataset shorter than block_size")
        ix = torch.randint(0, max_start, (batch_size,), dtype=torch.long)
        x = torch.stack([data[i:i+block_size].long() for i in ix], dim=0)
        y = torch.stack([data[i+1:i+1+block_size].long() for i in ix], dim=0)

        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y



# meta already created above
pad_id = meta['stoi']['<pad>']
eos_id = meta['stoi']['$']
# expose these in config for other code to use
config['pad_id'] = pad_id
config['eos_id'] = eos_id

meta_vocab_size = meta['vocab_size']
print(f"Using vocabulary size: {meta_vocab_size}")


if mi_measurement:
    with open(stats_measurement_data_file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]

    if drop_leading_digit:
            S = num_digit
    else:
        S = num_digit + 1
    # a simple way to parse test strings
    padded_lines = [] # add 0 padding, remove $; an example padded_lines[6] is '932+084+230+349=5951'
    for i in range(len(lines)):
        numbers = re.split(r'[+=]', lines[i])
        numbers[-1] = numbers[-1][:-1]
        for k, number in enumerate(numbers[:-1]):
            numbers[k] = '0' * (3-len(number)) + number
        numbers[-1] = numbers[-1] + '0' * (S-len(numbers[-1]))
        padded_lines.append("+".join(numbers[:-1]) + "=" + numbers[-1])

    stats_measurement_data = torch.cat([encode_addition(padded_lines[i], meta).unsqueeze(0) for i in range(len(padded_lines))], dim=0)

# # get 16 different datasets (including the base dataset) by randomizing input/output integers of the base dataset
# stats_measurement_dataset_list = gen_randomized_datasets(
#     stats_measurement_data,
#     meta,
#     digits_per_num=num_digit,
#     base_seed=2005
# )

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_perplexity = 1e9 # on text data
best_accuracy = -1 # on addition data

try:
    test_files, main_test_name = gather_test_files(test_file_path, main_test_name)
except Exception as e:
    print("Error:", e)
    sys.exit(1)

print("Collected test files:")
for tp in test_files:
    print(tp)


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, use_flash=use_flash) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_id)
    model.to(device)
elif init_from == 'resume':
    if resume_dir:
        print(f"Resuming training from {resume_dir}")
        checkpoint = torch.load(resume_dir, map_location=device)
    else:
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, ckpt_path_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num'] if resume_iter else 0
    max_iters += iter_num
    best_val_loss = checkpoint['best_val_loss']
    if 'best_perplexity' in checkpoint.keys():
        best_perplexity = checkpoint['best_perplexity']
    if 'best_accuracy' in checkpoint.keys():
        best_accuracy = checkpoint['best_accuracy']

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if batch_method == 'per_example':
            dataloader = train_loader if split == 'train' else val_loader
            dataloader_iter = iter(dataloader)
            for k in range(eval_iters):
                try:
                    X, Y = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    X, Y = next(dataloader_iter)
                with ctx:
                    X, Y = X.to(device), Y.to(device)
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
        else:
            # slicing mode uses get_batch
            for k in range(eval_iters):
                X, Y = get_batch(split if split == 'train' else 'val')
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



def get_lr_for_iter(iter_num):
    """Calculate learning rate based on iteration number using cosine decay with warmup."""
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters
    
    if iter_num >= lr_decay_iters:
        return min_lr
    
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, dir = out_dir)
    wandb.define_metric("lr", step_metric="iter")
    wandb.define_metric("iter",   step_metric="iter")
    wandb.define_metric("test/accuracy", step_metric="iter")
    wandb.define_metric("train/loss", step_metric="iter")
    wandb.define_metric("train/accuracy", step_metric="iter")
    wandb.define_metric("val/loss",   step_metric="iter")
    

# encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer)

# Initialize result_dict with basic metrics
result_dict = {
    'iter': [],
    'train_loss': [],
    'val_loss': [],
    'test_acc': [],
    'train_acc': []
}

# Initialize test accuracy keys for all test files
result_dict[f'test_acc'] = []

result_dir = get_results_dir(config)
config['result_dir'] = result_dir
with open(os.path.join(result_dir, "config.yaml"), "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)


# # build a dict of open file handles, one per dataset
# csv_writers = {}
# for dataset in stats_measurement_dataset_list:
#     name = dataset['name']
#     path = os.path.join(result_dir, f"{name}_stats.csv")
#     f = open(path, 'w', newline='')
#     writer = csv.DictWriter(f, fieldnames=[
#         'iter',
#         'ave_correct_probs',
#         'ave_correct_preds',
#         'ave_diff_probs_L1',
#         'ave_diff_probs_L2',
#         'ave_diff_probs_kl',
#         'ave_diff_logits_L1',
#         'ave_diff_logits_L2',
#         'ave_diff_preds',
#     ])
#     writer.writeheader()
#     csv_writers[name] = writer


# Initialize additional metrics for statistical measurements
stats_oo = [] # output-output mutual information
stats_io = [] # input-output mutual information


import time
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0  
iter_num = 0  

max_iters = config.get('max_iters', 10000)
 # number of epochs to warm up learning rate

# Initialize tracking variables
iter_num = 0  ## NOTE: redundant line (defined a few lines above)
best_val_loss = 1e9
best_accuracy = -1
running_mfu = -1.0 ## NOTE: redundant line (defined a few lines above)

# Create infinite data loader
def get_infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if 'max_new_tokens' in config.keys():
    print(f"max_new_tokens: {config['max_new_tokens']}")
else:
    print(f"max_new_tokens used: {num_digit+2}")

#### ADDED
DIGIT_PLACES_LIST = [('units', 'units', 'tens', 'units'), 
                     ('tens', 'tens', 'hundreds', 'tens'),
                     ('hundreds', 'hundreds', 'thousands', 'hundreds'),
                     ('hundreds', 'thousands', 'thousands', 'thousands')]

num_operands = int(ckpt_path_name.split('_operands')[0][-1])
reverse = 'reverse' in ckpt_path_name
mi_lines = gen_stats_test(num_operands, reverse=reverse)
xyz_mi_list = find_xyz_dataset_mi(meta, mi_lines, digit_places_list=DIGIT_PLACES_LIST, reverse=reverse)
#### End of ADDED

# Training loop - iteration based
while iter_num < max_iters:
    model.train()
    
    # Get learning rate for current iteration
    if decay_lr:
        lr = get_lr_for_iter(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Get next batch
    if batch_method == 'per_example':
        X, Y = next(train_loader_iter)
        X, Y = X.to(device), Y.to(device)
    else:
        X, Y = get_batch('train')  # already moved to device in get_batch
    
    # Forward pass
    with ctx:
        logits, loss = model(X, Y)
    
    # Backward pass
    scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # REMOVED: Do additional statistical measurements 
    #if mi_measurement:
    #    if iter_num in mi_measure_iters:
    #        model.eval()
    #        
    #        with torch.no_grad():
    #            # eval_res = eval_model(model, meta, stats_measurement_dataset_list, digits_per_num=num_digit, batch_size=test_batch_size)
    #            mi_stats = calc_model_dataset_mi(
    #                model = model,
    #                metadata = meta,
    #                data = stats_measurement_data,
    #                digits_per_num = num_digit,
    #                batch_size = test_batch_size,
    #                drop_leading_digit = drop_leading_digit
    #            )
    #
    #        # for name, stats in eval_res.items():
    #        #     if name == "model_embeddings":
    #        #         continue
    #        #     if name == 'base':
    #        #         row = {
    #        #             'iter': iter_num,
    #        #             'ave_correct_probs': stats['ave_correct_probs'],
    #        #             'ave_correct_preds': stats['ave_correct_preds'],
    #        #         }
    #        #     else:
    #        #         row = {
    #        #             'iter': iter_num,
    #        #             'ave_correct_probs': stats['ave_correct_probs'],
    #        #             'ave_correct_preds': stats['ave_correct_preds'],
    #        #             'ave_diff_probs_L1': stats['ave_diff_probs_L1'],
    #        #             'ave_diff_probs_L2': stats['ave_diff_probs_L2'],
    #        #             'ave_diff_probs_kl': stats['ave_diff_probs_kl'],
    #        #             'ave_diff_logits_L1': stats['ave_diff_logits_L1'],
    #        #             'ave_diff_logits_L2': stats['ave_diff_logits_L2'],
    #        #             'ave_diff_preds': stats['ave_diff_preds'],
    #        #         }
    #        #     # Write to the CSV file for this dataset
    #        #     csv_writers[name].writerow(row)
    #
    #        
    #        # Calculate output-output mutual information
    #        mi_mat = mi_stats['output-output']['mutual_info']
    #        nmi_mat = mi_stats['output-output']['normalized_mutual_info']
    #        for i in range(mi_mat.shape[0]):
    #            for j in range(i, mi_mat.shape[1]):
    #                stats_oo.append({
    #                    'iter': iter_num,
    #                    'i': i,
    #                    'j': j,
    #                    'mi': mi_mat[i, j].item(),
    #                    'nmi': nmi_mat[i, j].item()
    #                })
    #
    #        # also calculate input-output mutual information
    #        mi_mat_io = mi_stats['input-output']['mutual_info']
    #        nmi_mat_io = mi_stats['input-output']['normalized_mutual_info']
    #        for i in range(mi_mat_io.shape[0]):
    #            for j in range(mi_mat_io.shape[1]):
    #                stats_io.append({
    #                    'iter': iter_num,
    #                    'i': i,
    #                    'j': j,
    #                    'mi': mi_mat_io[i, j].item(),
    #                    'nmi': nmi_mat_io[i, j].item()
    #                })
    #
    #        # **NOW write out the two MI CSVs immediately:**
    #        stats_oo_df = pd.DataFrame(stats_oo)
    #        stats_oo_df.to_csv(os.path.join(result_dir, 'output_output_mi.csv'), index=False)
    #
    #        stats_io_df = pd.DataFrame(stats_io)
    #        stats_io_df.to_csv(os.path.join(result_dir, 'input_output_mi.csv'), index=False)
    #
    #        model.train()
        
    # Evaluation
    if iter_num % eval_interval == 0 or (more_early_eval1 and iter_num <= early_eval_border1 and iter_num % early_eval_interval1 == 0) or (more_early_eval2 and iter_num <= early_eval_border2 and iter_num % early_eval_interval2 == 0):
        losses = estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Initialize wandb_dict for this iteration
        wandb_dict = {
            "iter": iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
        }

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
        
        # Regular test evaluation
        test_accuracy = None
        if eval_addition:
            test_names, accuracy_multiple_file, correct_examples_multiple_file, incorrect_examples_multiple_file = evaluate_multiple_files(
                config, model, ctx,
                encode=lambda x: encode_addition(x, meta),
                decode=lambda x: decode_addition(x, meta),
                test_files=test_files,
                iter_num=iter_num,
                result_dir=result_dir,
                verbose=False,
                num_digit=num_digit,
                zero_pad=zero_pad,
                data_type=data_type,
                operator=operator,
                data_format=data_format,
                analyze=True,
                mode=mode, 
                batch_method=batch_method
            )

            test_accuracy = accuracy_multiple_file.get(main_test_name, None)
            total_num = len(correct_examples_multiple_file.get(main_test_name)) + len(incorrect_examples_multiple_file.get(main_test_name))
            correct_num = len(correct_examples_multiple_file.get(main_test_name))
            # Log results
            print("\nTest Results:")
            print(f"{main_test_name}.txt, {total_num} examples: {correct_num}/{total_num}  ({test_accuracy:.2f}%)")

            print()
            
            # Add test accuracy to wandb_dict
            wandb_dict["test/accuracy"] = test_accuracy
            
            if test_accuracy > best_accuracy and iter_num % 5 == 0:
                best_accuracy = test_accuracy
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'best_accuracy': best_accuracy,
                    'config': config,
                    'meta': meta,
                }
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_iter_{iter_num}_acc.pt'))
        
        # Training data evaluation
        train_accuracy = None
        if eval_addition_train:
            config['start'] = f"FILE:{train_data_test_path}"
            train_accuracy, correct, incorrect = evaluate_addition_batch(
                config, model, ctx, 
                encode=lambda x: encode_addition(x, meta),
                decode=lambda x: decode_addition(x, meta), 
                verbose=False, 
                num_digit=num_digit, 
                zero_pad=zero_pad,
                data_type=data_type, 
                operator=operator, 
                data_format=data_format,
                mode=mode,
                batch_method=batch_method
            )
            
            # Add train accuracy to wandb_dict
            wandb_dict["train/accuracy"] = train_accuracy

            #### ADDED
            if mi_measurement:
                if iter_num == 0:
                    mi_record_dict = {}
                model.eval()
                with torch.no_grad():
                    mi_stats = calc_model_dataset_mi_v2(
                        model = model,
                        meta = meta,
                        lines = mi_lines,
                        xyz_mi_list = xyz_mi_list,
                        reverse = reverse,
                        batch_size = test_batch_size,
                        padding_token = train_dataset.pad_id
                    )    
                mi_record_dict[f"iter_{iter_num}"] = mi_stats
                wandb_dict["mi/units"] = mi_stats[0][2][0]
                wandb_dict["mi/tens"] = mi_stats[1][2][0]
                wandb_dict["mi/hundreds"] = mi_stats[2][2][0]
                wandb_dict["mi/thousands"] = mi_stats[3][0][0]
                wandb_dict["mi/units-base"] = xyz_mi_list[0]['mi'][2][0]
                wandb_dict["mi/tens-base"] = xyz_mi_list[1]['mi'][2][0]
                wandb_dict["mi/hundreds-base"] = xyz_mi_list[2]['mi'][2][0]
                wandb_dict["mi/thousands-base"] = xyz_mi_list[3]['mi'][0][0]
                model.train()
            #### End of ADDED
      
        # Update and save basic metrics
        result_dict['iter'].append(iter_num)
        result_dict['train_loss'].append(losses['train'].item())
        result_dict['val_loss'].append(losses['val'].item())
        result_dict['test_acc'].append(test_accuracy)
        result_dict['train_acc'].append(train_accuracy)
        
        # Save results to CSV after each evaluation
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(os.path.join(result_dir, 'training_metrics.csv'), index=False)
        
        # Single wandb log per iteration with all metrics
        if wandb_log:
            wandb.log(wandb_dict, step=iter_num)

    
    iter_num += 1

# Save final checkpoint
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'best_accuracy': best_accuracy,
    'config': config,
    'meta': meta,
}
torch.save(checkpoint, os.path.join(out_dir, f'ckpt_final.pt'))


losses = estimate_loss()

for test_file in test_files:
    test_name = os.path.splitext(os.path.basename(test_file))[0]
    if test_name == main_test_name:
        standard_test_file_path = test_file
        break

if eval_addition:
    config['start'] = f"FILE:{standard_test_file_path}"
    test_accuracy, correct, incorrect = evaluate_addition_batch(
        config, model, ctx, 
        encode=lambda x: encode_addition(x, meta),
        decode=lambda x: decode_addition(x, meta), 
        verbose=False, 
        num_digit=num_digit, 
        zero_pad=zero_pad,
        data_type=data_type, 
        operator=operator, 
        data_format=data_format, 
        analyze=True,
        mode=mode,
        batch_method=batch_method
    )
    import csv
    # Save correct examples
    correct_path = os.path.join(result_dir, 'correct_examples.csv')
    with open(correct_path, 'w', newline='') as csvfile:
        fieldnames = ['operands', 'result', 'outcome', 'c_hat2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, nums in enumerate(correct):
            operands, result, outcome, c_hat2 = nums
            writer.writerow({'operands': operands, 'result': result, 'outcome': outcome, 'c_hat2': c_hat2})
    
    # Save incorrect examples
    incorrect_path = os.path.join(result_dir, 'incorrect_examples.csv')
    with open(incorrect_path, 'w', newline='') as csvfile:
        fieldnames = ['operands', 'result', 'outcome', 'c_hat2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, nums in enumerate(incorrect):
            operands, result, outcome, c_hat2 = nums
            writer.writerow({'operands': operands, 'result': result, 'outcome': outcome, 'c_hat2': c_hat2})

if eval_addition_train:
    config['start'] = f"FILE:{train_data_test_path}"
    train_accuracy, correct, incorrect = evaluate_addition_batch(
        config, model, ctx, 
        encode=lambda x: encode_addition(x, meta),
        decode=lambda x: decode_addition(x, meta), 
        verbose=False, 
        num_digit=num_digit, 
        zero_pad=zero_pad,
        data_type=data_type, 
        operator=operator, 
        data_format=data_format,
        mode=mode,
        batch_method=batch_method
    )
    
    
test_names, accuracy_multiple_file, correct_examples_multiple_file, incorrect_examples_multiple_file = evaluate_multiple_files(
    config, model, ctx,
    encode=lambda x: encode_addition(x, meta),
    decode=lambda x: decode_addition(x, meta),
    test_files=test_files,
    iter_num='final',
    result_dir=result_dir,
    verbose=False,
    num_digit=num_digit,
    zero_pad=zero_pad,
    data_type=data_type,
    operator=operator,
    data_format=data_format,
    analyze=True,
    mode=mode,
    batch_method=batch_method
)

test_accuracy = accuracy_multiple_file.get(main_test_name, None)
total_num = len(correct_examples_multiple_file.get(main_test_name)) + len(incorrect_examples_multiple_file.get(main_test_name))
correct_num = len(correct_examples_multiple_file.get(main_test_name))
print("\nFinal Test Results:")
print(f"{main_test_name}.txt, {total_num} examples: {correct_num}/{total_num}  ({test_accuracy:.2f}%)")
print()



# Final wandb logging
if wandb_log:
    final_dict = {
        "train/loss": losses['train'],
        "val/loss": losses['val'],
        "lr": lr,
        "iter": iter_num,
        "test/accuracy": test_accuracy if eval_addition else None,
        "train/accuracy": train_accuracy if eval_addition_train else None,
    }
    final_dict[f"final_test/accuracy"] = accuracy_multiple_file[main_test_name]
    wandb.log(final_dict)

# Save final DataFrame
result_df = pd.DataFrame(result_dict)
result_df.to_csv(os.path.join(result_dir, 'training_metrics.csv'), index=False)

csv_path = os.path.join(result_dir, f"{main_test_name}_results.csv")
# or whatever path your code wrote (adjust)
if os.path.exists(csv_path):
    if operator == '+' or operator == '-' or operator == '*':
        # save a figure into result_dir
        fig_path = os.path.join(result_dir, "digitwise_errors.png")
        print("Running result analysis on", csv_path)
        df, counts_by_iter = result_analysis.analyze_csv(
            csv_path,
            step_size=5,
            offset=0,
            max_steps=800000,
            actual_col="actual",
            save_fig=True,
            fig_path=fig_path,
            save_counts_csv=True
        )
        print("Saved digit error plot to", fig_path)
else:
    print("Result CSV not found at:", csv_path)
