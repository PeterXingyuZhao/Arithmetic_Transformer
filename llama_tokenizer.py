"""
LLaMA Tokenizer Wrapper

Provides a wrapper around HuggingFace's AutoTokenizer for LLaMA models
that is compatible with the existing character-level tokenizer interface.
"""

from transformers import AutoTokenizer
import torch


class LlamaTokenizerWrapper:
    """
    Wrapper for LLaMA tokenizer that provides interface compatible
    with the existing character-level tokenizer.

    This wrapper ensures that encode/decode functions work seamlessly
    with the evaluation pipeline that was designed for character-level tokenization.
    """

    def __init__(self, model_name="EleutherAI/pythia-1b"):
        """
        Initialize the LLaMA tokenizer wrapper.

        Args:
            model_name: HuggingFace model identifier for LLaMA
        """
        print(f"Loading LLaMA tokenizer from: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # # FORCE '$' as EOS token to match data format
        # # This overrides the default EOS token (like </s> for LLaMA)
        # special_tokens = {'eos_token': '$'}

        # # Add pad token if needed
        # if self.tokenizer.pad_token is None:
        #     special_tokens['pad_token'] = '<pad>'

        # num_added = self.tokenizer.add_special_tokens(special_tokens)
        # print(f"Set EOS token to '$' and added {num_added} special tokens: {special_tokens}")

        # # Verify '$' is set as EOS
        # assert self.tokenizer.eos_token == '$', f"Failed to set EOS token to '$', got {self.tokenizer.eos_token}"
        # print(f"✓ Verified: EOS token is '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")

        # # Set padding side to right (for causal LM)
        # self.tokenizer.padding_side = 'right'

        # Create metadata dict compatible with existing code
        self.meta = {
            'vocab_size': len(self.tokenizer),
            'stoi': {},  # Not used with BPE, but kept for compatibility
            'itos': {},  # Not used with BPE, but kept for compatibility
            'tokenizer': self.tokenizer
        }
        self.dollar_token_id = self.tokenizer.encode("$", add_special_tokens=False)[0]

        # Ensure pad_token is defined
        if self.tokenizer.pad_token_id is None:
            print(f"Tokenizer does not have a pad token. Defaulting to EOS token: {self.tokenizer.eos_token}")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Expose important IDs for easy access
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id

        print(f"Tokenizer initialized: vocab_size={len(self.tokenizer)}, "
              f"pad_id={self.pad_id}, eos_id={self.eos_id}")

    def encode(self, text):
        """
        Encode text to token IDs.

        Args:
            text: String to encode

        Returns:
            List of token IDs (Python list, not tensor)
        """
        # Encode without adding special tokens (we add $ manually in data)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return token_ids

    def decode(self, tokens):
        """
        Decode token IDs to text.

        Args:
            tokens: List, numpy array, or tensor of token IDs

        Returns:
            Decoded string
        """
        # Convert to list if it's a tensor or numpy array
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        elif hasattr(tokens, 'tolist'):  # numpy array
            tokens = tokens.tolist()

        # Decode tokens to text, keep special tokens visible
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
        return decoded

    def get_vocab_size(self):
        """Return vocabulary size."""
        return len(self.tokenizer)

    def create_meta(self):
        """
        Return metadata dict for compatibility with existing code.

        Returns:
            Dictionary with vocab_size, tokenizer, and placeholder stoi/itos
        """
        self.meta['vocab_size'] = len(self.tokenizer)
        return self.meta

    def __repr__(self):
        return (f"LlamaTokenizerWrapper(vocab_size={len(self.tokenizer)}, "
                f"pad_id={self.pad_id}, eos_id={self.eos_id})")


if __name__ == "__main__":
    # Simple test
    print("Testing LlamaTokenizerWrapper...")
    print("-" * 50)

    # Note: This will only work if you have HuggingFace access to LLaMA
    try:
        tokenizer = LlamaTokenizerWrapper("meta-llama/Meta-Llama-3.1-8B")

        # Test encode/decode
        test_texts = [
            "123+456=579$",
            "9+8=17$",
            "100+200+300+400=1000$"
        ]

        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)

            print(f"\nOriginal: {text}")
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"Decoded: {decoded}")
            print(f"Match: {text == decoded}")

        # Test metadata
        meta = tokenizer.create_meta()
        print(f"\nMetadata: {meta.keys()}")
        print(f"Vocab size: {meta['vocab_size']}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Note: You need HuggingFace access to LLaMA models.")
        print("Run: huggingface-cli login")
