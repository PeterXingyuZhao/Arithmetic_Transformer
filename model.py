import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import inspect
from dataclasses import dataclass

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CasualSelfAttention(nn.Module):
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    #output_projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    #regularization
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout
    self.use_flash = config.use_flash
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.use_flash
    if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    else:
      print("Using Flash attention")
  def forward(self,x):
    B, T, C = x.size() # batch_size, sequence length, embedding dimension
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

    if self.flash:
      y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = self.dropout if self.training else 0, is_causal = True)
    else:
      #manual implementation of attention
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))

    return y
  
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_flash: bool = True # use Flash Attention CUDA kernels for fast attention


class GPT(nn.Module):
  def __init__(self,config, pad_id):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config = config
    self.pad_id = pad_id

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        drop = nn.Dropout(config.dropout),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = LayerNorm(config.n_embd, bias=config.bias),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
    self.transformer.wte.weight = self.lm_head.weight # Tie weights

    # init all weights
    self.apply(self._init_weights)

    for pn, p in self.named_parameters():
      if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,) )

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
    pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

    tok_emd = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
    pos_emd = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
    x = self.transformer.drop(tok_emd + pos_emd)
    for block in self.transformer.h:
      x = block(x)

    # for i, block in enumerate(self.transformer.h):
    #   x = block(x)
    #   if get_hidden_embedding is not None and i == get_hidden_embedding:
    #      h_l = x
    x = self.transformer.ln_f(x)

    if targets is not None:
      logits = self.lm_head(x)
      logits = logits.view(-1, logits.size(-1))   # (B*L, V)
      targets = targets.view(-1)                  # (B*L,)
      loss = F.cross_entropy(logits, targets, ignore_index=self.pad_id)
    else:
      logits = self.lm_head(x[:,[-1],:])
      loss = None
    return logits, loss



  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
      for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
        # random note

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
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
      # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)

    print(f"using fused:{use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <=self.config.block_size else idx[:, -self.config.block_size:]
      logits, _ = self(idx_cond)

      logits = logits[:,-1,:] / temperature

      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:,[-1]]] = -float('Inf')
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx