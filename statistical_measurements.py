import os
import pickle
import numpy as np
import random
from tqdm import tqdm
import copy

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import sys

def encode_addition(text, meta):
    """Encode text to tensor using the metadata."""
    return torch.tensor([meta['stoi'][c] for c in text], dtype=torch.long)

def decode_addition(tensor, meta):
    """Decode tensor to text using the metadata."""
    if isinstance(tensor, torch.Tensor):
        return ''.join([meta['itos'][i.item()] for i in tensor])
    else:
        return ''.join([meta['itos'][i] for i in tensor])

def token_to_numeric(tensor, meta):
    """Convert tensor to numeric digits."""
    # Build lookup tensor
    lookup_tensor = torch.empty(len(meta["vocab"]), dtype=torch.long)
    for i, s in enumerate(meta["vocab"]):
        if s.isdigit():
            lookup_tensor[i] = int(s)
    return lookup_tensor[tensor]  # Same shape as tensor


def calc_embed_scores(model, diffs=list(range(5))):
    """
    calc_embed_scores uses the model's embedding and unembedding weight matrices to calculate scores,
        which measures the variance of cosine similarity between two embedding vetors E(n), E(m) when n-m is a constant
    """
    W0 = model.transformer.wte.weight.to("cpu") # (vocab_size, n_embd)
    W_out = model.lm_head.weight.to("cpu") # (vocab_size, n_embd)
    G0 = F.normalize(W0, dim=-1) @ F.normalize(W0, dim=-1).T
    G_out = F.normalize(W_out, dim=-1) @ F.normalize(W_out, dim=-1).T

    out = []
    for G in [G0, G_out]:
        score = torch.zeros(len(diffs))
        for k, diag_id in enumerate(diffs):
            score[k] = torch.var(torch.diag(G, diagonal=diag_id))
        out.append(score.mean().item())
    return out

def randomize_test_data(data: torch.Tensor, metadata, digits_per_num=3, randomize_digit_place=[0,1], seed=2025,
                        randomize="input", valid_carry=False, reverse_input=False, reverse_output=False) -> torch.Tensor:
    """
    randomize_test_data randomizes a part of the test data by keeping some digits and randomizing the other digits
    Arguments:
        data is a 2-order tensor of shape (sample_size, seq_len), representing tokenized inputs (padded right to the same length) such as
        '437+357+579+984=7532' and '932+084+230+349=5951' (reverse output)
        digits_per_num is the number of digits in a number
        randomize_digit_place is a list indicating which digits are to be randomized. [0, 1] means the least two digits are to be randomized
        randomize: if "input" then the input numbers are randomized, if "output" then the output number is randomized
        valid_carry is a boolean indicating whether randomization keeps carry valid (carry operation before randomization remains so)
    """
    assert isinstance(randomize_digit_place, list)
    L = len(randomize_digit_place)
    n, T = data.shape
    S = digits_per_num + 1
    assert (T - S) % S == 0, "data format not conform to expectation, e.g., '437+357+579+984=7532'. "
    assert randomize in ["input", "output"], "randomize is either `input` or `output`."
    num_op = (T - S) // S
    torch.manual_seed(seed)

    ids0 = [digits_per_num-1-id for id in randomize_digit_place] if not reverse_input else randomize_digit_place
    ids1 = [digits_per_num-id for id in randomize_digit_place] if not reverse_output else randomize_digit_place
    ids_rand_input = torch.cat([torch.arange(num_op).long() * S + j for j in ids0])
    ids_rand_output = torch.tensor(ids1).long() + S*num_op
    new_data = copy.deepcopy(data)
    ids2 = []
    if randomize == "output":
        for col_id in ids_rand_output:
            new_data[:,col_id] = data[torch.randperm(n),col_id]
        return new_data
    if valid_carry: # if control for valid carry
        if 0 in randomize_digit_place: # if least significant digit is randomized
            J = max(randomize_digit_place) if reverse_input else digits_per_num-1-max(randomize_digit_place) #
            ids2 = torch.arange(num_op).long() * S + J
            all_carry = token_to_numeric(data[:,ids2], meta=metadata).sum(dim=1) // 10
            unique_carry = torch.unique(all_carry)
            for carry in unique_carry:
                ids_rand = (all_carry == carry)
                n_rand = ids_rand.sum().item()
                subset_ids = ids_rand.nonzero(as_tuple=True)[0]
                subset_data = new_data[ids_rand, :][:, ids2]
                subset_data = subset_data[torch.randperm(n_rand), :]
                ii, jj = torch.meshgrid(subset_ids, ids2, indexing='ij')
                new_data[ii, jj] = subset_data
        # randomize other digits independently
        for col_id in ids_rand_input:
            if col_id not in ids2:
                new_data[:,col_id] = data[torch.randperm(n),col_id]
    else: # if disregard carry
        for col_id in ids_rand_input:
            new_data[:,col_id] = data[torch.randperm(n),col_id]
    return new_data

def _model_forward(model, metadata, data, digits_per_num=3, batch_size=128):
    n, T = data.shape
    vocab_size = len(metadata["vocab"])
    device = next(model.parameters()).device
    res = {"logits": torch.empty(n, digits_per_num+1, vocab_size, dtype=torch.float),
           "probs": torch.empty(n, digits_per_num+1, vocab_size, dtype=torch.float),
           "pred_ids": torch.empty(n, digits_per_num+1, dtype=torch.long)}
    num_batches = np.ceil(n / batch_size).astype(int)
    with torch.no_grad():
        for b in range(num_batches):
            if b < num_batches - 1:
                samp_ids = list(range(b*batch_size, b*batch_size+batch_size))
            else:
                samp_ids = list(range(b*batch_size, n))
            input_ids, targets = data[samp_ids, :-1].to(device), data[samp_ids, 1:].to(device)
            logits, _ = model(input_ids, targets)  # (batch_size, T-1, vocab_size)
            logits = logits[:, -(digits_per_num+1):, :] # (batch_size, digits_per_num+1, vocab_size)
            probs = torch.softmax(logits, dim=-1) # (batch_size, digits_per_num+1, vocab_size)
            pred_ids = torch.argmax(probs, dim=-1) # (batch_size, digits_per_num+1)
            res["logits"][samp_ids], res["probs"][samp_ids], res["pred_ids"][samp_ids] = logits.to("cpu"), probs.to("cpu"), pred_ids.to("cpu")
    return res

def gen_randomized_datasets(base_data, metadata, digits_per_num=3, base_seed=2005, reverse_input=False, reverse_output=False):
    """Generate a list of randomized datasets"""
    base_dataset = {"name": "base", "data": base_data}
    dataset_list = [base_dataset]
    # generate different datasets by randomizing input integers of the base dataset
    for is_carry in [True, False]:
        for increasing in [True, False]:
            for k in range(digits_per_num):
                randomize_digit_place = list(range(0,k+1)) if increasing else list(range(digits_per_num-1-k,digits_per_num))
                seed = base_seed+k+is_carry*100+increasing*10+k
                name = f"carry_{is_carry}_" + "_".join(map(str, randomize_digit_place))
                data = randomize_test_data(base_data, metadata, digits_per_num, randomize_digit_place, seed,
                        "input", is_carry, reverse_input, reverse_output)
                dataset = {"name": name, "data": data, "is_carry": is_carry, "randomize_digit_place": randomize_digit_place, "randomize": "input"}
                dataset_list.append(dataset)

    # generate different datasets by randomizing output integers of the base dataset
    for k in range(digits_per_num):
        randomize_digit_place = list(range(0,k+1))
        seed = base_seed + k
        name = "output_randomize_" + "_".join(map(str, randomize_digit_place))
        data = randomize_test_data(base_data, metadata, digits_per_num, randomize_digit_place, seed,
                "output", False, reverse_input, reverse_output)
        dataset = {"name": name, "data": data, "is_carry": None, "randomize_digit_place": randomize_digit_place, "randomize": "output"}
        dataset_list.append(dataset)
    return dataset_list


def eval_model(model, metadata, dataset_list, digits_per_num=3, batch_size=128):
    """
    eval_model evaluates a model on a list of datasets, including the baseset (testset) and randomized datasets
    Returns:
        eval_res: a dictionary of evaluation results with dataset names as keys, and values are again a dictionary of different evaluation metrics
    """

    dataset_names = [dataset["name"] for dataset in dataset_list]
    k0 = dataset_names.index("base")
    base_data = dataset_list[k0]["data"]
    n = base_data.shape[0]
    S = digits_per_num + 1
    vocab_size = len(metadata["vocab"])
    eval_res = {}

    base_res = _model_forward(model, metadata, base_data, digits_per_num=digits_per_num, batch_size=batch_size)
    batch_idx = torch.arange(n).unsqueeze(1)  # shape: (batch_size, 1)
    seq_idx = torch.arange(S).unsqueeze(0)       # shape: (1, S)
    eval_res["base"] = {}
    eval_res["base"]["ave_correct_probs"] = base_res["probs"][batch_idx, seq_idx, base_data[:, -(digits_per_num+1):]].mean(0).tolist()
    eval_res["base"]["ave_correct_preds"] = torch.mean((base_res["pred_ids"] == base_data[:, -(digits_per_num+1):]).float(), dim=0).tolist()

    for k, dataset in tqdm(enumerate(dataset_list)):
        if k == k0:
            continue
        eval_res[dataset["name"]] = {}
        res = _model_forward(model, metadata, dataset["data"], digits_per_num=digits_per_num, batch_size=batch_size)
        eval_res[dataset["name"]]["ave_correct_probs"] = res["probs"][batch_idx, seq_idx, base_data[:, -(digits_per_num+1):]].mean(0).tolist()
        eval_res[dataset["name"]]["ave_correct_preds"] = torch.mean((res["pred_ids"] == base_data[:, -(digits_per_num+1):]).float(), dim=0).tolist()
        eval_res[dataset["name"]]["ave_diff_probs_L1"] = torch.sum(torch.abs(res["probs"] - base_res["probs"]), dim=-1).mean(0).tolist()
        eval_res[dataset["name"]]["ave_diff_probs_L2"] = torch.sum((res["probs"] - base_res["probs"])**2, dim=-1).sqrt().mean(0).tolist()
        eval_res[dataset["name"]]["ave_diff_probs_kl"] = F.kl_div(F.log_softmax(res["logits"], dim=-1), base_res["probs"], reduction="none").sum(-1).mean(0).tolist()
        eval_res[dataset["name"]]["ave_diff_logits_L1"] = torch.sum(torch.abs(res["logits"] - base_res["logits"]), dim=-1).mean(0).tolist()
        eval_res[dataset["name"]]["ave_diff_logits_L2"] = torch.sum((res["logits"] - base_res["logits"])**2, dim=-1).sqrt().mean(0).tolist()
        eval_res[dataset["name"]]["ave_diff_preds"] = torch.mean((res["pred_ids"] == base_res["pred_ids"]).float(), dim=0).tolist()

    eval_res["model_embeddings"] = calc_embed_scores(model)
    return eval_res

def calc_mi_x_p(x, py_given_x):
    """
    Estimate mutual information I(X; Y) from:
    - x: 1D tensor of n samples from X (discrete values)
    - py_given_x: 2D tensor of shape (n, k), where each row is p(y | x_i)

    Assumes:
    - Each row of py_given_x is a valid probability distribution (sums to 1)
    - y takes k possible values
    """
    n, k = py_given_x.shape
    assert x.shape[0] == n, "x and py_given_x must have the same number of samples"

    # Compute empirical p(x)
    x_unique, x_counts = torch.unique(x, return_counts=True)
    px_dict = dict(zip(x_unique.tolist(), (x_counts / n).tolist()))

    # Aggregate by unique x values
    # Create mapping from each unique x to its p(y | x)
    py_given_x_dict = {}
    for val in x_unique:
        mask = (x == val)
        py_given_x_dict[val.item()] = py_given_x[mask].mean(dim=0)

    # Compute marginal p(y)
    py = sum(px_dict[val] * py_given_x_dict[val] for val in px_dict)
    py = py / py.sum()  # normalize for safety
    E_y = -torch.sum(py * torch.log(py))

    # Compute KL divergence for each unique x
    mi = 0.0
    for val in px_dict:
        pxy = py_given_x_dict[val]
        kl = (pxy * (pxy / py).log()).sum()
        mi += px_dict[val] * kl

    return {"mutual_info": float(mi), "normalized_mutual_info": float(mi/E_y)}


def calc_model_dataset_mi(model, metadata, data, digits_per_num=3, batch_size=128, drop_leading_digit=False):
    """
    calc_model_dataset_mi estimate mutual information I(X; Y) for various pairs of digits X and Y from both data and model prediction probs
    res1---X is taken to be one of the digits in the first number, Y is one of the digits in the output number
    res2---both X and Y are taken to be one of the digits in the output number
    """
    n, T = data.shape
    vocab_size = len(metadata["vocab"])
    device = next(model.parameters()).device
    if drop_leading_digit:
        S = digits_per_num
    else:
        S = digits_per_num + 1
    S = int(S)
    if drop_leading_digit:
        assert (T - S) % (S + 1) == 0, "data format not conform to expectation, e.g., '437+357+579+984=753'. "
    else:
        assert (T - S) % S == 0, "data format not conform to expectation, e.g., '437+357+579+984=7532'. "
    num_op = (T - S) // S
    num_batches = np.ceil(n / batch_size).astype(int)
    #
    res1 = {"mutual_info": np.empty((digits_per_num, S)),
            "normalized_mutual_info": np.empty((digits_per_num, S))}
    probs = torch.empty(n, S, vocab_size, dtype=torch.float)
    with torch.no_grad():
        for b in range(num_batches):
            if b < num_batches - 1:
                samp_ids = list(range(b*batch_size, b*batch_size+batch_size))
            else:
                samp_ids = list(range(b*batch_size, n))
            input_ids, targets = data[samp_ids, :-1].to(device), data[samp_ids, 1:].to(device)
            logits, _ = model(input_ids, targets)  # (batch_size, T-1, vocab_size)
            logits = logits[:, -S:, :] # (batch_size, digits_per_num+1, vocab_size)
            probs[samp_ids] = torch.softmax(logits, dim=-1).to("cpu") # (batch_size, digits_per_num+1, vocab_size)
    for i1 in range(digits_per_num):
        for i2 in range(S):
            out = calc_mi_x_p(data[:,i1], probs[:,i2])
            res1["mutual_info"][i1, i2] = out["mutual_info"]
            res1["normalized_mutual_info"][i1, i2] = out["normalized_mutual_info"]
    #
    res2 = {"mutual_info": np.empty((S-1, S-1)),
            "normalized_mutual_info": np.empty((S-1, S-1))}

    # first calculate joint probs of every pair (X, Y) of output digits, each joint probs is of size (vocab_size, vocab_size)
    # where X is one of the first digits_per_num digits in output number
    # and Y is one of the last digits_per_num digits in output number
    joint_probs_all = torch.empty(S-1, S-1, vocab_size, vocab_size, dtype=torch.float)
    with torch.no_grad():
        for i3 in range(S-1):
            for z in tqdm(range(vocab_size)):
                new_data = copy.deepcopy(data)
                new_data[:, T-S+i3] = z # replace the digit X at index T-S+i3 with a fixed value z
                probs3 = torch.empty(n, S-1, vocab_size, dtype=torch.float)
                for b in range(num_batches):
                    if b < num_batches - 1:
                        samp_ids = list(range(b*batch_size, b*batch_size+batch_size))
                    else:
                        samp_ids = list(range(b*batch_size, n))
                    input_ids, targets = new_data[samp_ids, :-1].to(device), new_data[samp_ids, 1:].to(device)
                    logits, _ = model(input_ids, targets)  # (batch_size, T-1, vocab_size)
                    logits = logits[:, -(S-1):, :] # (batch_size, digits_per_num, vocab_size)
                    probs3[samp_ids] = torch.softmax(logits, dim=-1).to("cpu") # (batch_size, digits_per_num, vocab_size) representing Pr(Y|X=z)
                joint_probs_all[i3,:,z,:] = torch.mean(probs3 * probs[:,i3,z].unsqueeze(1).unsqueeze(1), dim=0) # Pr(Y|X=z) * Pr(X=z)

    for j1 in range(S-1):
        for j2 in range(S-1):
            if j1 > j2:
                res2["mutual_info"][j1, j2] = None
                res2["normalized_mutual_info"][j1, j2] = None
                continue
            pxy = joint_probs_all[j1, j2] / joint_probs_all[j1, j2].sum() # P(x, y), normalized for safety
            px = pxy.sum(dim=1, keepdim=True)  # shape (num_x, 1)
            py = pxy.sum(dim=0, keepdim=True)  # shape (1, num_y)
            E_y = -torch.sum(py * torch.log(py))
            px_py = px @ py  # outer product, shape (num_x, num_y)
            mask = pxy > 0
            mi = torch.sum(pxy[mask] * torch.log(pxy[mask] / px_py[mask]))
            res2["mutual_info"][j1, j2] = float(mi)
            res2["normalized_mutual_info"][j1, j2] = float(mi/E_y)

    return {"input-output":res1, "output-output":res2}