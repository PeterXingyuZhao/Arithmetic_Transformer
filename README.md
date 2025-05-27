#Addition with Small Transformers



---

## Project Overview
The project explores training compact Transformer models from scratch to under addition problems under varying operands length.

---


## Model Architecture

- Layers: 6 (default) / 2 (small)  
- Heads: 6 (default) / 2 (small)  
- Embedding dimension: 384  
- Positional encoding: learned  
- Vocabulary: digits `0‑9`, symbols `+`, `=`, `\n`, `pad`

---

## Training Setup
- **Objective**  Causal‑LM (next‑token prediction)  
- **Optimizer**  AdamW (`β₁ = 0.9`, `β₂ = 0.99`, weight‑decay scheduling)  
- **Drop‑out**  0.20  
- **Batch size**  256  
---

## How to run
- python train.py 4_operands_addition.txt

---
