from __future__ import annotations

from torch import Tensor
import os
import pickle
import numpy as np
import random
from tqdm import tqdm
import copy
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import sys
from typing import Iterable, List, Optional, Tuple, Union

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

def calc_mi_x_p_old(x, py_given_x):
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


########### OpenAI's Codex helped me with the following code ################


_EPS = 1e-12


def calc_mi_x_p(x: Tensor, py_given_x: Tensor) -> dict[str, float]:
    """Estimate ``I(X; Y)`` from conditional model predictions.

    Args:
        x: A 1D integer tensor with ``n`` samples drawn from ``X``.
        py_given_x: A 2D tensor of shape ``(n, k)`` where each row is the
            conditional distribution :math:`p(y \mid x_i)`.

    Returns:
        A mapping with keys ``"mutual_info"`` and ``"normalized_mutual_info"``.
    """

    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if py_given_x.ndim != 2:
        raise ValueError("py_given_x must be a 2D tensor")
    n, k = py_given_x.shape
    if x.shape[0] != n:
        raise ValueError("x and py_given_x must have the same number of samples")

    device = py_given_x.device
    dtype = py_given_x.dtype

    unique_x, inverse_indices, counts = torch.unique(
        x.to(device), return_inverse=True, return_counts=True
    )
    counts = counts.to(py_given_x.dtype)
    px = counts / float(n)

    # Aggregate conditional probabilities per unique x using scatter_add.
    probs_sum = torch.zeros(unique_x.numel(), k, dtype=dtype, device=device)
    probs_sum.scatter_add_(
        0, inverse_indices.unsqueeze(-1).expand_as(py_given_x), py_given_x
    )
    py_given_x_avg = probs_sum / counts.unsqueeze(-1)

    # Marginal p(y) and entropy H(Y).
    py = torch.matmul(px.unsqueeze(0), py_given_x_avg).squeeze(0)
    py = torch.clamp(py, min=_EPS)
    entropy_y = -(py * py.log()).sum()

    # Mutual information via KL divergence of p(y|x) to p(y).
    ratio = torch.clamp(py_given_x_avg / py, min=_EPS)
    kl_terms = torch.sum(py_given_x_avg * ratio.log(), dim=1)
    mi = torch.sum(px * kl_terms)

    if entropy_y <= _EPS:
        normalized_mi = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        normalized_mi = mi / entropy_y

    return {
        "mutual_info": float(mi.detach().cpu()),
        "normalized_mutual_info": float(normalized_mi.detach().cpu()),
    }


def calc_mi_x_y(x: Tensor, y: Tensor) -> dict[str, float]:
    """Estimate ``I(X; Y)`` empirically from paired observations ``(x, y)``.

    Args:
        x: A 1D integer tensor with ``n`` samples drawn from ``X``.
        y: A 1D integer tensor with ``n`` samples drawn from ``Y``.

    Returns:
        A mapping with keys ``"mutual_info"`` and ``"normalized_mutual_info"``.
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D tensors")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    n = x.shape[0]
    device = x.device
    dtype = torch.float32

    unique_x, inverse_x, counts_x = torch.unique(
        x, return_inverse=True, return_counts=True
    )
    unique_y, inverse_y, counts_y = torch.unique(
        y, return_inverse=True, return_counts=True
    )

    px = counts_x.to(dtype) / float(n)
    py = counts_y.to(dtype) / float(n)

    # Joint probability p(x, y).
    joint_indices = inverse_x * unique_y.numel() + inverse_y
    joint_counts = torch.bincount(
        joint_indices,
        minlength=unique_x.numel() * unique_y.numel(),
    ).reshape(unique_x.numel(), unique_y.numel())
    pxy = joint_counts.to(dtype) / float(n)

    entropy_y = -(py * torch.clamp(py, min=_EPS).log()).sum()

    px_py = px.unsqueeze(1) * py.unsqueeze(0)
    mask = pxy > 0
    log_ratio = torch.zeros_like(pxy)
    log_ratio[mask] = torch.log(torch.clamp(pxy[mask] / px_py[mask], min=_EPS))
    mi = torch.sum(pxy[mask] * log_ratio[mask])

    if entropy_y <= _EPS:
        normalized_mi = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        normalized_mi = mi / entropy_y

    return {
        "mutual_info": float(mi.detach().cpu()),
        "normalized_mutual_info": float(normalized_mi.detach().cpu()),
    }
    
    
def calc_mi_x_y_z(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """Estimate empirical conditional mutual information I(X; Y | Z).

    Args:
        x: 1D integer tensor of samples from X.
        y: 1D integer tensor of samples from Y (aligned with ``x``).
        z: 1D integer tensor of samples from Z (aligned with ``x`` and ``y``).

    Returns:
        A dictionary with keys ``mutual_info`` and ``normalized_mutual_info``.
        ``normalized_mutual_info`` divides I(X;Y|Z) by H(Y|Z).
    """

    assert x.dim() == y.dim() == z.dim() == 1, "Inputs must be 1D tensors"
    n = x.shape[0]
    assert y.shape[0] == n and z.shape[0] == n, "All inputs must have the same length"

    x = x.to(torch.long)
    y = y.to(torch.long)
    z = z.to(torch.long)

    total_mi = 0.0
    total_cond_entropy = 0.0

    z_vals, z_counts = torch.unique(z, return_counts=True)
    for z_val, count in zip(z_vals, z_counts):
        mask = (z == z_val)
        p_z = count.item() / n

        x_subset = x[mask]
        y_subset = y[mask]
        n_z = x_subset.shape[0]

        x_vals = torch.unique(x_subset, sorted=True)
        y_vals = torch.unique(y_subset, sorted=True)

        x_index = {int(val.item()): idx for idx, val in enumerate(x_vals)}
        y_index = {int(val.item()): idx for idx, val in enumerate(y_vals)}

        joint = torch.zeros(len(x_vals), len(y_vals), dtype=torch.double)
        pairs = torch.stack((x_subset, y_subset), dim=1)
        unique_pairs, pair_counts = torch.unique(pairs, dim=0, return_counts=True)
        for pair, cnt in zip(unique_pairs, pair_counts):
            i = x_index[int(pair[0].item())]
            j = y_index[int(pair[1].item())]
            joint[i, j] = cnt.double() / n_z

        px_given_z = joint.sum(dim=1)  # shape (num_x,)
        py_given_z = joint.sum(dim=0)  # shape (num_y,)

        mask_entropy = py_given_z > 0
        H_y_given_z = -torch.sum(py_given_z[mask_entropy] * torch.log(py_given_z[mask_entropy]))
        total_cond_entropy += p_z * H_y_given_z.item()

        px_py = px_given_z.unsqueeze(1) * py_given_z.unsqueeze(0)
        mask_joint = joint > 0
        mi_z = torch.sum(joint[mask_joint] * torch.log(joint[mask_joint] / px_py[mask_joint]))
        total_mi += p_z * mi_z.item()

    normalized_mi = total_mi / total_cond_entropy if total_cond_entropy > 0 else 0.0

    return {"mutual_info": float(total_mi), "normalized_mutual_info": float(normalized_mi)}


def calc_mi_x_p_z(x: torch.Tensor, py_given_xz: torch.Tensor, z: torch.Tensor):
    """Estimate I(X; Y | Z) using conditional probabilities p(y | x, z).

    Args:
        x: 1D integer tensor of samples from X.
        py_given_xz: 2D tensor of shape (n, m) with conditional distributions p(y | x_i, z_i).
        z: 1D integer tensor of samples from Z (aligned with ``x``).

    Returns:
        A dictionary with keys ``mutual_info`` and ``normalized_mutual_info``.
        ``normalized_mutual_info`` divides I(X;Y|Z) by H(Y|Z).
    """

    assert x.dim() == z.dim() == 1, "x and z must be 1D tensors"
    n = x.shape[0]
    assert z.shape[0] == n, "x and z must have the same number of samples"
    assert py_given_xz.shape[0] == n, "py_given_xz must align with samples"

    x = x.to(torch.long)
    z = z.to(torch.long)
    py_given_xz = py_given_xz.to(torch.double)

    total_mi = 0.0
    total_cond_entropy = 0.0

    z_vals, z_counts = torch.unique(z, return_counts=True)
    for z_val, count in zip(z_vals, z_counts):
        mask_z = (z == z_val)
        p_z = count.item() / n

        x_subset = x[mask_z]
        py_subset = py_given_xz[mask_z]

        x_vals, x_counts = torch.unique(x_subset, return_counts=True)
        p_x_given_z = x_counts.double() / x_counts.sum()

        # Aggregate conditional probabilities for each unique x
        py_given_x_dict = {}
        for x_val in x_vals:
            mask_x = (x_subset == x_val)
            py_given_x_dict[int(x_val.item())] = py_subset[mask_x].mean(dim=0)

        # Compute p(y | z)
        vocab_size = py_subset.shape[1]
        py_given_z = torch.zeros(vocab_size, dtype=torch.double)
        for idx, x_val in enumerate(x_vals):
            py_given_z += p_x_given_z[idx].item() * py_given_x_dict[int(x_val.item())]
        py_given_z = py_given_z / py_given_z.sum()

        mask_entropy = py_given_z > 0
        H_y_given_z = -torch.sum(py_given_z[mask_entropy] * torch.log(py_given_z[mask_entropy]))
        total_cond_entropy += p_z * H_y_given_z.item()

        # Compute I(X; Y | Z = z)
        mi_z = 0.0
        for idx, x_val in enumerate(x_vals):
            p_x_z = p_x_given_z[idx].item()
            if p_x_z == 0:
                continue
            py_given_x = py_given_x_dict[int(x_val.item())]
            mask_valid = (py_given_x > 0) & (py_given_z > 0)
            if mask_valid.any():
                kl = torch.sum(py_given_x[mask_valid] * torch.log(py_given_x[mask_valid] / py_given_z[mask_valid]))
                mi_z += p_x_z * kl.item()
        total_mi += p_z * mi_z

    normalized_mi = total_mi / total_cond_entropy if total_cond_entropy > 0 else 0.0

    return {"mutual_info": float(total_mi), "normalized_mutual_info": float(normalized_mi)}


## parsing

_PLACE_TO_OFFSET = {
    "unit": 0,
    "tens": 1,
    "hundreds": 2,
    "thousands": 3,
}


def _resolve_digit_index(
    number: str, place: str, start_index: int, *, reversed_digits: bool = False
) -> int:
    """Return the index of the requested place value digit within the expression.

    Args:
        number: Substring containing only the digits of the number of interest.
        place: One of ``{"unit", "tens", "hundreds", "thousands"}`` describing
            the desired place value.
        start_index: The index within the full expression where ``number`` starts.
        reversed_digits: If ``True``, ``number`` represents the digits of the
            value in reverse order.

    Returns:
        The 0-based index in the full expression pointing to the requested digit.

    Raises:
        ValueError: If ``place`` is not recognised or the number does not have the
            requested place value digit.
    """

    if place not in _PLACE_TO_OFFSET:
        raise ValueError(f"Unsupported place specification: {place!r}")

    offset = _PLACE_TO_OFFSET[place]
    if offset >= len(number):
        #raise ValueError(
        #    f"Number {number!r} does not contain a digit in the {place} place"
        #)
        return float('nan')

    if reversed_digits:
        digit_index = offset
    else:
        digit_index = len(number) - 1 - offset
    return start_index + digit_index

def find_indices(lines, x_place, y_place, z_place="unit", reverse=False):
    """Locate digit indices for the specified place values within expressions.

    Args:
        lines: Iterable of addition expressions such as ``"932+84+230+349=1595"``.
        x_place: Place value to extract from the first operand.
        y_place: Place value to extract from the result (right-hand side).
        z_place: Place value to extract from the result (right-hand side).
            Defaults to ``"unit"``.
        reverse: Whether the digits of the result appear in reverse order.

    Returns:
        A list of tuples ``(x_idx, y_idx, z_idx)`` containing the 0-based indices
        of the requested digits within each corresponding expression string.

    Raises:
        ValueError: If an expression is malformed or does not contain the
            requested place value digits.
    """

    indices = []
    for expr in lines:
        expr = expr.strip().split('$')[0]
        if "=" not in expr:
            raise ValueError(f"Expression missing '=': {expr!r}")

        lhs, rhs = expr.split("=", maxsplit=1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not rhs:
            raise ValueError(f"Expression missing result after '=': {expr!r}")

        # The first operand is always the first number on the LHS.
        first_operand = lhs.split("+", maxsplit=1)[0]
        if not first_operand.isdigit():
            raise ValueError(
                f"First operand must be a positive integer, got {first_operand!r}"
            )

        result = rhs
        if not result.isdigit():
            raise ValueError(f"Result must be a positive integer, got {result!r}")

        x_idx = _resolve_digit_index(first_operand, x_place, 0)
        equals_index = expr.index("=")
        result_start = equals_index + 1
        y_idx = _resolve_digit_index(
            result, y_place, result_start, reversed_digits=reverse
        )
        z_idx = _resolve_digit_index(
            result, z_place, result_start, reversed_digits=reverse
        )
        indices.append((x_idx, y_idx, z_idx))

    return indices

## parsing v2


def _normalize_place(place: Union[str, int, None]) -> Union[str, int, None]:
    """Normalize textual place names to a canonical lowercase form."""

    if place is None or isinstance(place, int):
        return place

    normalized = str(place).replace("-", " ").replace("_", " ").lower().strip()
    normalized = " ".join(normalized.split())  # collapse repeated whitespace
    return normalized


def _place_to_offset(place: Union[str, int, None]) -> Optional[int]:
    """Convert a place specifier (e.g. "unit", 0) into a digit offset."""

    if place is None:
        return None

    if isinstance(place, int):
        if place < 0:
            raise ValueError("Digit offsets must be non-negative integers.")
        return place

    normalized = _normalize_place(place)

    place_mapping = {
        "unit": 0,
        "units": 0,
        "ones": 0,
        "one": 0,
        "tens": 1,
        "ten": 1,
        "hundreds": 2,
        "hundred": 2,
        "thousands": 3,
        "thousand": 3,
        "ten thousands": 4,
        "ten thousand": 4,
        "hundred thousands": 5,
        "hundred thousand": 5,
    }

    if normalized not in place_mapping:
        raise ValueError(f"Unsupported place specifier: {place}")

    return place_mapping[normalized]


def _extract_numbers_with_positions(line: str) -> List[Tuple[str, int]]:
    """Extract digit sequences and their start indices from ``line``."""

    numbers: List[Tuple[str, int]] = []
    current_start: Optional[int] = None

    for idx, ch in enumerate(line):
        if ch.isdigit():
            if current_start is None:
                current_start = idx
        else:
            if current_start is not None:
                numbers.append((line[current_start:idx], current_start))
                current_start = None

    if current_start is not None:
        numbers.append((line[current_start:], current_start))

    return numbers

def _digit_index(start_idx: int, number: str, place_offset: int, reverse: bool) -> Optional[int]:
    """Return the absolute index for a requested place within ``number``."""

    if place_offset >= len(number):
        return float('nan')

    if reverse:
        char_offset = place_offset
    else:
        char_offset = len(number) - 1 - place_offset

    return start_idx + char_offset



def new_find_indices(
    lines: Iterable[str],
    x_place: Union[str, int],
    y_place: Union[str, int],
    z_place: Union[str, int] = "unit",
    carry_place: Optional[Union[str, int]] = None,
    reverse: bool = False,
):
    """Return indices of requested place values for operands and result.

    ``x_place`` refers to the first operand on the left-hand side, while both
    ``y_place`` and ``z_place`` refer to the result on the right-hand side.
    """

    x_offset = _place_to_offset(x_place)
    y_offset = _place_to_offset(y_place)
    z_offset = _place_to_offset(z_place)
    carry_offset = _place_to_offset(carry_place)

    x_indices: List[Optional[int]] = []
    y_indices: List[Optional[int]] = []
    z_indices: List[Optional[int]] = []
    carry_indices: List[Tuple[Optional[int], ...]] = [] if carry_offset is not None else None

    for line in lines:
        numbers = _extract_numbers_with_positions(line)

        if len(numbers) < 3:
            raise ValueError(
                "Each line must contain at least two operands and one result in the format 'a+b= c'."
            )

        operands = numbers[:-1]
        result = numbers[-1]

        if len(operands) < 2:
            raise ValueError("Need at least two operands to extract x and y indices.")

        first_operand = operands[0]
        result_number = result

        def _lookup(entry: Tuple[str, int], offset: Optional[int], reverse: bool) -> Optional[int]:
            if offset is None:
                return None
            digits, start_idx = entry
            return _digit_index(start_idx, digits, offset, reverse)

        x_indices.append(_lookup(first_operand, x_offset, False))
        y_indices.append(_lookup(result_number, y_offset, reverse))
        z_indices.append(_lookup(result_number, z_offset, reverse))

        if carry_offset is not None and carry_indices is not None:
            carry_indices.append(tuple(_lookup(operand, carry_offset, False) for operand in operands))

    if carry_indices is not None:
        return x_indices, y_indices, z_indices, carry_indices
    return x_indices, y_indices, z_indices

## calculate probabilities from model 

def get_token_probabilities_at_indices(
    model,
    metadata,
    lines,
    y_indices,
    batch_size=128,
    padding_token=0,
):
    """Return model probabilities for specific token positions.

    Args:
        model: Character-level transformer model.
        metadata: Dictionary containing ``stoi``/``itos`` mappings and ``vocab``.
        lines: List of ``n`` strings. Each string will be encoded into tokens.
        y_indices: Iterable of length ``n`` with the (0-based) token indices of
            the targets whose probabilities should be returned. The index is the
            position of the token itself (not the following token).
        batch_size: Optional mini-batch size used for the forward pass.
        padding_token: Token ID used to right-pad sequences to a uniform length
            when the encoded lines have different lengths.

    Returns:
        ``torch.Tensor`` of shape ``(n, vocab_size)`` with probability
        distributions for the requested tokens.
    """

    if len(lines) == 0:
        return torch.empty(0, len(metadata["vocab"]), dtype=torch.float)

    token_seqs = [encode_addition(line, metadata) for line in lines]
    seq_lengths = torch.tensor([seq.numel() for seq in token_seqs], dtype=torch.long)
    max_len = seq_lengths.max().item()

    tokens = torch.full((len(token_seqs), max_len), padding_token, dtype=torch.long)
    for idx, seq in enumerate(token_seqs):
        tokens[idx, : seq.numel()] = seq

    y_indices = torch.as_tensor(y_indices, dtype=torch.long)
    if y_indices.numel() != tokens.size(0):
        raise ValueError("y_indices must have the same length as lines.")

    if torch.any(y_indices < 1):
        raise ValueError("y_indices must be at least 1 to compute probabilities.")

    if torch.any(y_indices >= seq_lengths):
        raise ValueError("Each y_index must be within the length of its corresponding line.")

    vocab_size = len(metadata["vocab"])
    n = tokens.size(0)
    device = next(model.parameters()).device
    probs_out = torch.empty(n, vocab_size, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_tokens = tokens[start:end].to(device)
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]
            logits, _ = model(inputs, targets)

            if logits.dim() == 2:
                logits = logits.view(inputs.size(0), inputs.size(1), -1)

            batch_probs = torch.softmax(logits, dim=-1)

            batch_y_indices = (y_indices[start:end] - 1).to(device)
            sample_ids = torch.arange(batch_tokens.size(0), device=device)
            probs_out[start:end] = batch_probs[sample_ids, batch_y_indices, :].to("cpu")

    return probs_out

#############################################################################

def gen_stats_test(num_operands, size=50_000, min=0, max=999, reverse=False, digits_per_num=None, mask=False):
    """
    gen_tats_test generates a dataset (a list of addition expression strings)
    Arguments:
        size: the size of dataset, namely the number of lines (list elements)
        min, max: minimum and maximum integers of the range to be sampled
        reverse: bool indicating whether the result is reversed
        digits_per_num: the number of digits in each operand; if specified, max will be overwritten
        mask: bool indicating whether small results are filtered and removed from dataset
    Returns:
        lines: a list of strings that represent addition math expressions
    """
    if digits_per_num is not None:
        assert isinstance(digits_per_num, int)
        warnings.warn("digits_per_num specified, overwriting max!")
        max = (10 ** digits_per_num) - 1
    operands = np.empty((size,num_operands), dtype=int)
    for k in range(num_operands):
        operands[:,k] = np.random.randint(min,max,size)
    results = operands.sum(axis=1)
    if mask:
        mask = (results >= 10 ** (int(np.log10(max))))
        operands = operands[mask]
        results = results[mask]
        size = operands.shape[0]

    lines = []
    for j in range(size):
        if reverse:
            line = "+".join([f"{str(operands[j,k])}" for k in range(num_operands)]) + f"={str(results[j])[::-1]}$"
        else:
            line = "+".join([f"{str(operands[j,k])}" for k in range(num_operands)]) + f"={str(results[j])}$"
        lines.append(line)
    return lines

def find_xyz_dataset_mi(meta, lines, reverse=False, digit_places_list=None):
    """
    find_indices_dataset_mi extracts the tokens from data (lines of strings) as realized values of X, Y, Z, and calculates the mutual information I(X; Y) and conditional mutual information I(X; Y| Z)
    Arguments:
        meta: metadata that contains tokenizer
        lines: a list of lines, where each line is a string with format f"{str(operand1[j])}+{str(operand2[j])}={str(result[j])}$" (not reverse)
        reverse: bool indicating if result is reversed or not
        digit_places_list: a list of tuples, each tuple indicates the digit places of X, Y, Z taken from the operand and result of each expression. We always use the first operand for X, and the result for Y and Z
    Returns:
        xyz_mi_list: a list of dictionaries, where each dictionary contains the realized values X, Y, Z, carries extracted from lines according to the digit places specified in digit_places_list, as well as various mutual information metrics such as I(X;Y) and I(X;Y|Z).
    """
    import math
    N = len(lines)
    if digit_places_list is None:
        digit_places_list = [('hundreds', 'hundreds', 'thousands', 'hundreds')]
    
    M = len(digit_places_list) # number of MI measurements, each measurement corresponds to a set of (X, Y, Z, carry)
    xyz_mi_list = []
    for k, places in enumerate(digit_places_list):
        x_indices, y_indices, z_indices, carry_indices = new_find_indices(lines, places[0], places[1], places[2], carry_place=places[3], reverse=reverse)
        x = torch.empty(N, dtype=torch.long)
        y = torch.empty(N, dtype=torch.long)
        z = torch.empty(N, dtype=torch.long)
        carries = torch.zeros(N, dtype=torch.long)
        y_indices_cleaned = []
        for j in range(N):
            x[j] = meta['stoi'][lines[j][x_indices[j]]] if not np.isnan(x_indices[j]) else meta['stoi']['0'] # find token for hidden '0' if len(operand_1) < 3
            y[j] = meta['stoi'][lines[j][y_indices[j]]] if not np.isnan(y_indices[j]) else meta['stoi']['$'] # only works if reverse is True
            z[j] = meta['stoi'][lines[j][z_indices[j]]] if not np.isnan(z_indices[j]) else meta['stoi']['$'] # only works if reverse is True
            for k, item in enumerate(carry_indices[j]):
                if math.isnan(item):
                    continue
                carries[j] += int(lines[j][item])
            carries[j] = carries[j] // 10
            tmp = y_indices[j] if not np.isnan(y_indices[j]) else len(lines[j])-1
            y_indices_cleaned.append(tmp)
        o0 = calc_mi_x_y(x,y)
        o2 = calc_mi_x_y_z(x,y,z)
        o4 = calc_mi_x_y_z(x,y,carries)
        mi = [[o['mutual_info'], o['normalized_mutual_info']] for o in [o0, o2, o4]]
        xyz_mi = {"x": x, "y": y, "z": z, "carries": carries, "places": places, "y_indices_cleaned": y_indices_cleaned, "mi": mi}
        xyz_mi_list.append(xyz_mi)
    return xyz_mi_list

def calc_model_dataset_mi_v2(model, meta, lines, xyz_mi_list, reverse=False, batch_size=128, padding_token=0):
    """
    calc_model_dataset_mi estimate mutual information I(X; Y) for various pairs of digits X and Y from both data and model prediction probs, and conditional mutual information I(X; Y| Z)
    Arguments:
        lines: a list of lines, where each line is a string with format f"{str(operand1[j])}+{str(operand2[j])}={str(result[j])}$" (not reverse)
        reverse: bool indicating if result is reversed or not
        digit_places_list: a list of tuples, each tuple indicates the digit places of X, Y, Z taken from the operand and result of each expression. We always use the first operand for X, and the result for Y and Z
    Returns:
        mi_list: a list of arrays, where each array contains the mutual information metrics for specified digit places from digit_places_list
    """
    N = len(lines)
    mi_list = []
    for k, xyz_mi in enumerate(xyz_mi_list):
        probs = get_token_probabilities_at_indices(model, meta, lines, xyz_mi["y_indices_cleaned"], batch_size=batch_size, padding_token=padding_token)
        o1 = calc_mi_x_p(xyz_mi['x'], probs)
        o3 = calc_mi_x_p_z(xyz_mi['x'], probs, xyz_mi['z'])
        o5 = calc_mi_x_p_z(xyz_mi['x'], probs, xyz_mi['carries'])
        mi = [[o['mutual_info'], o['normalized_mutual_info']] for o in [o1, o3, o5]]
        mi_list.append(mi)
    return mi_list
        

    
# older version    

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
