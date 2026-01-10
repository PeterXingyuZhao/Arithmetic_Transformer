# eval_ckpt.py
import argparse
import os
import torch
from contextlib import nullcontext
import yaml

# import your project modules (adjust path / names if needed)
from model import GPTConfig, GPT
from main_utilities import abs_if_rel, get_results_dir, concat_strip_dollar, create_meta_for_addition, gather_test_files
from evaluation_ckpt import evaluate_multiple_files, evaluate_addition_batch  # whichever you prefer to call
# NOTE: the names above must match your codebase imports

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt

def build_model_from_ckpt(ckpt, device, compile_model=False):
    model_args = ckpt['model_args']
    # ensure model_args contains vocab_size, n_layer, n_head, n_embd, block_size, bias etc.
    gptconf = GPTConfig(**model_args)
    # try constructing in the two ways your repo has used:
    try:
        model = GPT(gptconf, ckpt['meta']['stoi'].get('<pad>'))
    except TypeError:
        model = GPT(gptconf)
    # load state dict and clean prefixes if needed
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            print("Compile failed, continuing without compile:", e)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='path to checkpoint (ckpt.pt/ckpt_final.pt/ckpt_iter_xxx_acc.pt)')
    ap.add_argument('--testfile', required=True, help='path to your test .txt file (single file).')
    ap.add_argument('--batch', choices=['per_example','slicing'], default='per_example',
                    help='batch preparation method used when training; must match checkpoint/data handling')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--result_dir', default=None, help='where to write results (default: same dir as ckpt)')
    ap.add_argument('--num_digit', type=int, default=3)
    ap.add_argument('--analyze', action='store_true', help='if your evaluate functions support analysis option')
    args = ap.parse_args()

    device = args.device
    ckpt = load_checkpoint(args.ckpt, device)
    meta = ckpt.get('meta')
    if meta is None:
        raise RuntimeError("Checkpoint does not contain 'meta'. You need the meta (stoi/itos) saved in checkpoint.")

    # load model
    model = build_model_from_ckpt(ckpt, device, compile_model=False)

    # safe device/dtype selection (avoid calling CUDA helpers when torch not built with cuda)
    # 'device' should be set from args earlier: device = args.device
    use_cuda = False
    try:
        use_cuda = (device is not None) and ('cuda' in str(device).lower()) and torch.cuda.is_available()
    except Exception:
        use_cuda = False

    if use_cuda:
        # Only call CUDA-specific checks inside try/except
        try:
            bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        except Exception:
            bf16_ok = False
        dtype = 'bfloat16' if bf16_ok else 'float16'
        device_type = 'cuda'
    else:
        dtype = 'float32'
        device_type = 'cpu'

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    # reproduce minimal config used by evaluate_multiple_files / evaluate_addition_batch
    result_dir = args.result_dir or os.path.dirname(args.ckpt)
    config = {
        'pad_id': meta['stoi'].get('<pad>'),
        'eos_id': meta['stoi'].get('$'),
        'result_dir': result_dir,
        'start': f"FILE:{args.testfile}",
        'batch_method': args.batch,
        # required by prepare_addition_batches / evaluate_addition_batch:
        'device': device,                 # e.g. 'cpu' or 'cuda'
        'test_batch_size': 128,           # adjust if you want larger/smaller eval batches
        'max_new_tokens': 10,  # sensible default for generation
        'top_k': 1,                       # greedy by default for deterministic eval
        'temperature': 0.0,               # deterministic sampling (0.0 -> greedy)
    }


    # define encode/decode helpers just like in your main script:
    encode = lambda s: torch.tensor([meta['stoi'][c] for c in s], dtype=torch.long)
    decode = lambda t: ''.join([meta['itos'][i] for i in (t.tolist() if isinstance(t, torch.Tensor) else t)])

    # gather test files list (some helpers expect this)
    test_files = [args.testfile]

    # call the same evaluation function used in training:
    test_names, accuracy_multiple_file, correct_examples_multiple_file, incorrect_examples_multiple_file = evaluate_multiple_files(
        config, model, ctx,
        encode=encode,
        decode=decode,
        test_files = test_files,
        iter_num = 'eval_from_ckpt',
        result_dir = result_dir,
        verbose = True,
        num_digit = args.num_digit,
        zero_pad = False,
        data_type = 'binary',
        operator = '+',
        data_format = 'reverse',
        analyze = args.analyze,
        mode = 'read_gold_as_str',
        batch_method = args.batch,
        randomize = None
    )

    # print summary
    for name, acc in accuracy_multiple_file.items():
        print(f"Test file: {name}, accuracy: {acc:.2f}%")
    

if __name__ == '__main__':
    main()
