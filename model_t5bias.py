# model_t5bias.py
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import inspect
from dataclasses import dataclass

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def _relative_position_bucket(distance, num_buckets=32, max_distance=128):
    """
    Translate absolute distances (non-negative int tensor) into relative position buckets.
    This follows the spirit of T5's bucketed relative positions:
      - small distances get their own buckets (linear)
      - larger distances are placed in log-spaced buckets up to max_distance
    distance: tensor of ints >= 0
    returns: tensor of same shape with bucket ids in [0, num_buckets-1]
    """
    if num_buckets <= 1:
        return torch.zeros_like(distance, dtype=torch.long)

    # number of exact (linear) buckets before switching to log scale
    max_exact = num_buckets // 2
    is_small = distance < max_exact

    # for large distances, compute a log-space bucket index
    # scale distances > max_exact into the remaining buckets
    if max_distance <= max_exact:
        # degenerate: just clip
        large_bucket = torch.full_like(distance, num_buckets - 1)
    else:
        # compute logarithmic buckets for large distances
        # convert to float for log
        distance_f = distance.float()
        # avoid log(0) by clamping distances at 1
        safe_dist = torch.clamp(distance_f, min=1.0)
        # scale factor for the log mapping
        # the formula maps [max_exact, max_distance] to [0, num_buckets-max_exact-1]
        max_exact_f = float(max_exact)
        max_distance_f = float(max_distance)
        # prevent division by zero
        divisor = math.log(max_distance_f / max_exact_f)
        # when divisor is 0 fallback to clip
        if divisor <= 0:
            large_bucket = torch.clamp(distance, max=num_buckets - 1)
        else:
            # normalized log rank
            val = (torch.log(safe_dist / max_exact_f) / divisor) * (num_buckets - max_exact - 1)
            val = val.floor().long()
            large_bucket = torch.clamp(val + max_exact, max=num_buckets - 1)

    return torch.where(is_small, distance, large_bucket)


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_flash = config.use_flash
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.use_flash

        # T5-style relative bias parameters
        self.num_relative_buckets = getattr(config, "num_relative_buckets", 32)
        self.max_relative_distance = getattr(config, "max_relative_distance", 128)
        # shape: (num_buckets, n_head)
        self.relative_attention_bias = nn.Parameter(torch.zeros(self.num_relative_buckets, self.n_head))

        # cache buffers for causal mask (used when flash unavailable)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))
        else:
            print("Using Flash attention")

        # initialize the bias
        nn.init.normal_(self.relative_attention_bias, mean=0.0, std=0.02)

    def _compute_relative_bias(self, seq_len, device, dtype):
        """
        Build a relative bias tensor of shape (1, n_head, seq_len, seq_len)
        Suitable for adding to attention logits (broadcasts over batch).
        """
        # query positions i (0..seq_len-1) and key positions j (0..seq_len-1)
        # for causal decoder, we use distance = i - j (>= 0 for valid attention entries)
        i = torch.arange(seq_len, device=device)
        j = torch.arange(seq_len, device=device)
        # distance shape (seq_len, seq_len), positive when query>=key
        # compute i - j and clamp at 0 (we don't need negative distances for causal)
        distance = (i[:, None] - j[None, :]).clamp(min=0)

        # bucketize
        buckets = _relative_position_bucket(distance, num_buckets=self.num_relative_buckets,
                                            max_distance=self.max_relative_distance).long()  # (seq_len, seq_len)

        # lookup bias: shape (seq_len, seq_len, n_head)
        bias = self.relative_attention_bias[buckets.view(-1)].view(seq_len, seq_len, self.n_head)
        # reshape to (1, n_head, seq_len, seq_len)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        # cast to dtype and device
        return bias.to(dtype=dtype, device=device)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # compute relative bias for this seq length: shape (1, nh, T, T)
        rel_bias = self._compute_relative_bias(T, device=x.device, dtype=x.dtype)  # (1, nh, T, T)

        if self.flash:
            # Build a causal additive mask (large negative values for j>i).
            # Use a large negative float like -1e9 instead of -inf to be friendly to tracing/compilation.
            causal_mask = torch.triu(torch.ones((T, T), device=x.device, dtype=rel_bias.dtype), diagonal=1) * -1e9
            # causal_mask shape (T, T) -> (1, 1, T, T) by unsqueeze; will broadcast with rel_bias (1, nh, T, T)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            attn_mask = rel_bias + causal_mask                     # (1, nh, T, T), broadcast OK

            # call scaled_dot_product_attention with is_causal=False because the causal constraint
            # is included inside attn_mask now.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = False
            )
        else:
            # manual implementation: compute logits, add causal mask and relative bias, then softmax
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))   # causal mask
            att = att + rel_bias                                                  # add T5 relative bias
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_flash: bool = True
    # T5-bias specific:
    num_relative_buckets: int = 32
    max_relative_distance: int = 128


class GPT(nn.Module):
    def __init__(self, config, pad_id=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.pad_id = pad_id

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights
        self.transformer.wte.weight = self.lm_head.weight

        # init
        self.apply(self._init_weights)

        # optionally zero pad row
        if self.pad_id is not None:
            with torch.no_grad():
                self.transformer.wte.weight[self.pad_id].zero_()

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None, get_hidden_embedding=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emd = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            ignore_idx = self.pad_id if self.pad_id is not None else -100
            loss = F.cross_entropy(logits, targets, ignore_index=ignore_idx)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        decay.remove('lm_head.weight')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused:{use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if (temperature is not None and float(temperature) == 0.0) or (top_k is not None and int(top_k) == 1):
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                temp = max(float(temperature), 1e-8)
                logits = logits / temp
                if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                    topk_vals, _ = torch.topk(logits, k=int(top_k), dim=-1)
                    min_topk = topk_vals[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_topk, torch.full_like(logits, -1e9), logits)
                probs = F.softmax(logits, dim=-1)
                if (probs < 0).any():
                    probs = torch.clamp(probs, min=0.0)
                row_sums = probs.sum(dim=-1, keepdim=True)
                bad_rows = (row_sums <= 0) | torch.isnan(row_sums)
                if bad_rows.any():
                    greedy_idx = torch.argmax(logits, dim=-1)
                    one_hot = torch.zeros_like(probs)
                    one_hot.scatter_(1, greedy_idx.unsqueeze(-1), 1.0)
                    probs = torch.where(bad_rows.unsqueeze(-1), one_hot, probs)
                    row_sums = probs.sum(dim=-1, keepdim=True)
                probs = probs / row_sums.clamp(min=1e-12)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
