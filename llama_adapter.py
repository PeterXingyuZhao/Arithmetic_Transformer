"""
LLaMA Model Adapter

Wraps HuggingFace's LlamaForCausalLM to match the interface of the custom GPT model
used in this codebase, enabling seamless integration with the existing training loop.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import inspect


class LlamaModelAdapter(nn.Module):
    """
    Adapter class that wraps HuggingFace's LlamaForCausalLM to match
    the interface of the custom GPT class used in this codebase.

    This allows the LLaMA model to be used as a drop-in replacement for
    the custom GPT model without modifying the training loop.
    """

    def __init__(self, model_name_or_path="EleutherAI/pythia-1b",
                 pad_id=0, block_size=1024, dropout=0.0, tokenizer=None):
        """
        Initialize the HuggingFace model adapter.

        Args:
            model_name_or_path: HuggingFace model identifier (e.g., "EleutherAI/pythia-1b", "meta-llama/Meta-Llama-3.1-8B")
            pad_id: Padding token ID (from tokenizer)
            block_size: Maximum sequence length
            dropout: Dropout rate (note: model has its own dropout, this is for compatibility)
            tokenizer: Optional tokenizer wrapper (to resize embeddings if tokens were added)
        """
        super().__init__()

        print(f"Loading pre-trained model from: {model_name_or_path}")
        print("This may take a few minutes...")

        # Try to load with Flash Attention 2 if available, fallback to standard attention
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency and stability
                attn_implementation="flash_attention_2",  # Try Flash Attention 2
                use_cache=True,  # Enable KV cache for faster generation
            )
            print("✓ Loaded with Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention 2 not available: {e}")
            print("Loading with standard attention...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                use_cache=True,
            )
            print("✓ Loaded with standard attention")

        self.pad_id = pad_id
        self.config = self.model.config
        self.config.block_size = block_size

        # Store vocab size for compatibility
        self.vocab_size = self.model.config.vocab_size

        # Enable gradient checkpointing for memory efficiency with large models
        self.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency")

        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {num_params / 1e9:.2f}B parameters")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Pad token ID: {self.pad_id}")

    def forward(self, idx, targets=None):
        """
        Forward pass matching GPT interface.

        Args:
            idx: Input token IDs of shape (batch_size, seq_len)
            targets: Target token IDs of shape (batch_size, seq_len), or None

        Returns:
            tuple: (logits, loss)
                - logits: shape (batch_size, seq_len, vocab_size) during training,
                         or (batch_size, 1, vocab_size) during inference
                - loss: scalar loss value if targets provided, else None
        """
        device = idx.device
        b, t = idx.size()

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (idx != self.pad_id).long()

        if targets is not None:
            # Training mode: compute loss
            outputs = self.model(
                input_ids=idx,
                attention_mask=attention_mask,
                use_cache=False  # Disable cache during training
            )

            # HuggingFace returns CausalLMOutput object
            logits = outputs.logits

            # Flatten for cross_entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            # verify shapes match
            if logits_flat.shape[0] != targets_flat.shape[0]:
                 # This can happen if padding/truncation was inconsistent
                 # but based on train.py, they should match.
                 # Just in case, we'll let it error out naturally if mismatch 
                 pass

            loss = nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.pad_id
            )

            return logits, loss
        else:
            # Inference mode: no loss computation
            outputs = self.model(
                input_ids=idx,
                attention_mask=attention_mask,
                use_cache=True  # Enable cache for faster inference
            )

            # Return only the last token logits (for generation)
            logits = outputs.logits[:, [-1], :]
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
            """
            Args:
                eos_token_id: The integer ID of the token that should stop generation (e.g., tokenizer.eos_token_id)
                            or a list of IDs (e.g. [eos_id, dollar_sign_id])
            """
            # Ensure eos_token_id is a list or set for easy checking
            if isinstance(eos_token_id, int):
                stop_tokens = {eos_token_id}
            elif isinstance(eos_token_id, (list, tuple)):
                stop_tokens = set(eos_token_id)
            else:
                stop_tokens = set()
            
            batch_size = idx.shape[0]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)

            for _ in range(max_new_tokens):
                # Crop context if it exceeds block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = torch.nn.functional.softmax(logits, dim=-1)

                if top_k == 1:
                    idx_next = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)

                # --- STOPPING LOGIC ---
                # Check if the generated token is one of our stop tokens
                # .item() pulls the scalar value out of the tensor (only works for batch_size=1)
                # --- BATCH STOPPING LOGIC ---
                if len(stop_tokens) > 0:
                # Check which rows generated a stop token in this step
                # We iterate because torch.isin can be tricky with device matching on older pytorch versions
                    for stop_id in stop_tokens:
                    # Update the finished mask where the new token matches a stop ID
                        finished |= (idx_next.squeeze(-1) == stop_id)
                
                    # If ALL sequences in the batch are finished, we can stop early
                    if finished.all():
                        idx = torch.cat((idx, idx_next), dim=1)
                        break
            # -----------------------------

                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)

            return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer matching GPT interface.
        Separate parameters into decay/no_decay groups.

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters (beta1, beta2)
            device_type: 'cuda' or 'cpu'

        Returns:
            torch.optim.AdamW optimizer
        """
        # Separate parameters into decay and no_decay groups
        decay = set()
        no_decay = set()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Biases, layer norms, and embeddings don't get weight decay
            # This follows standard practice for transformer fine-tuning
            if any(nd in name.lower() for nd in ['bias', 'norm', 'embed']):
                no_decay.add(name)
            else:
                decay.add(name)

        # Create parameter dictionaries
        param_dict = {name: param for name, param in self.named_parameters()}

        # Verify all parameters are categorized
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"

        # Log parameter counts
        decay_params = [param_dict[name] for name in sorted(list(decay))]
        no_decay_params = [param_dict[name] for name in sorted(list(no_decay))]

        print(f"Optimizer: {len(decay_params)} tensors with weight decay, "
              f"{len(no_decay_params)} tensors without weight decay")

        # Create optimizer parameter groups
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if available (faster on CUDA)
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)

        if use_fused:
            print("Using fused AdamW optimizer")
            extra_args = dict(fused=True)
        else:
            extra_args = dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def get_num_params(self, non_embedding=True):
        """
        Get number of parameters (for logging).

        Args:
            non_embedding: If True, exclude embedding parameters from count

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.model.model.embed_tokens.weight.numel()
        return n_params


if __name__ == "__main__":
    # Simple test
    print("Testing LlamaModelAdapter...")
    print("-" * 50)

    try:
        # Initialize model
        print("\n1. Initializing model...")
        model = LlamaModelAdapter(
            model_name_or_path="meta-llama/Meta-Llama-3.1-8B",
            pad_id=0,
            block_size=128,
            dropout=0.0
        )
        model.eval()

        # Test forward pass with targets (training mode)
        print("\n2. Testing forward pass (training mode)...")
        batch_size, seq_len = 2, 10
        x = torch.randint(0, 1000, (batch_size, seq_len))
        y = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss = model(x, y)
        print(f"   Input shape: {x.shape}")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
        assert logits.shape == (batch_size, seq_len, model.vocab_size)
        assert loss is not None

        # Test forward pass without targets (inference mode)
        print("\n3. Testing forward pass (inference mode)...")
        logits, loss = model(x, None)
        print(f"   Logits shape: {logits.shape}")
        print(f"   Loss: {loss}")
        assert logits.shape == (batch_size, 1, model.vocab_size)
        assert loss is None

        # Test generation
        print("\n4. Testing generation...")
        prompt = torch.randint(0, 1000, (1, 5))
        output = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=1)
        print(f"   Prompt shape: {prompt.shape}")
        print(f"   Output shape: {output.shape}")
        assert output.shape == (1, 10)

        # Test optimizer configuration
        print("\n5. Testing optimizer configuration...")
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-4,
            betas=(0.9, 0.999),
            device_type='cpu'
        )

        # Print parameter count
        print("\n6. Model parameters:")
        total_params = model.get_num_params(non_embedding=False)
        non_embed_params = model.get_num_params(non_embedding=True)
        print(f"   Total: {total_params / 1e9:.2f}B parameters")
        print(f"   Non-embedding: {non_embed_params / 1e9:.2f}B parameters")

        print("\n All tests passed!")

    except Exception as e:
        print(f"\n Error: {e}")
        print("  1. HuggingFace access to LLaMA models (run: huggingface-cli login)")
        print("  2. Sufficient GPU memory (~16GB+ for 8B model)")
        print("  3. Flash Attention installed (pip install flash-attn)")
        import traceback
        traceback.print_exc()
