# Pretrained-model sources, attribution, and terms

This document records the model identifiers present in the released code and
their providers' licenses or terms. The paper's ten-model comparison, the
current notebook selection, and unused registry entries are distinguished below.
License information was checked against the linked official sources on
**2026-11-03**.

The public repository tree at `7bd0fb7af1023afa08c5ddca407ad98b3c06a3b3`
contains evaluation code and results, but no pretrained checkpoint files.
The notebooks and wrappers obtain weights and tokenizers from their providers
through Hugging Face. The MIT notices for nanoGPT, teaching_arithmetic, and
GSM8K do **not** license these pretrained models or supersede their terms.

## Models in the paper's ten-model Extended-GSM comparison

| Model repository ID | Original provider | License / official terms |
| --- | --- | --- |
| [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Mistral AI | [Apache License 2.0 (official model card)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Meta | [Meta Llama 3 Community License Agreement](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/LICENSE), including its acceptable-use policy |
| [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | Google | [Gemma Terms of Use](https://ai.google.dev/gemma/terms), including the Gemma Prohibited Use Policy |
| [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Alibaba Cloud / Qwen Team | [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE) |
| [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Alibaba Cloud / Qwen Team | [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE); not Apache-2.0 |
| [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Alibaba Cloud / Qwen Team | [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE) |
| [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Meta | [Llama 3.1 Community License Agreement](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE), including its acceptable-use policy |
| [`Qwen/Qwen2.5-Math-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) | Alibaba Cloud / Qwen Team | [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct/blob/main/LICENSE) |
| [`mistralai/Mathstral-7B-v0.1`](https://huggingface.co/mistralai/Mathstral-7B-v0.1) | Mistral AI | [Apache License 2.0 (official model card)](https://huggingface.co/mistralai/Mathstral-7B-v0.1) |
| [`deepseek-ai/deepseek-math-7b-instruct`](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct) | DeepSeek | [DeepSeek License Agreement, version 1.0 (October 23, 2023)](https://github.com/deepseek-ai/DeepSeek-Math/blob/main/LICENSE-MODEL); distinct from the repository's MIT code license |

The notebook uses the earlier identifier
`meta-llama/Meta-Llama-3.1-8B-Instruct`, which redirects to the current
`meta-llama/Llama-3.1-8B-Instruct` source linked above. This documentation records
the rename without changing the experimental code or claiming a checkpoint
change.

## Separate Pythia fine-tuning experiment

[`EleutherAI/pythia-1b`](https://huggingface.co/EleutherAI/pythia-1b), by
**EleutherAI**, is licensed under **Apache License 2.0**, as stated in its official
model card. See also Biderman et al., *Pythia: A Suite for Analyzing Large Language
Models Across Training and Scaling*, 2023,
<https://arxiv.org/abs/2304.01373>, and the
[Pythia source repository](https://github.com/EleutherAI/pythia).

This model is selected by `configuration_files/4_operands_addition_plain_pythia.txt`
and is the default identifier in `llama_adapter.py` and `llama_tokenizer.py`.
Those filenames do not identify it as a Meta Llama model.

## Additional models selected by the current notebook

`gsm_test/myGSM_test.ipynb` currently selects two further models that are absent
from the paper's ten-model comparison figure:

| Model repository ID | Original provider | License / official terms |
| --- | --- | --- |
| [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | Meta | [Llama 2 Community License Agreement](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/LICENSE.txt), including its acceptable-use policy |
| [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | Meta | [Meta Llama 3 Community License Agreement](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE), including its acceptable-use policy |

## Other registry entries

The notebook's registry also defines the following entries, which are not
selected by its current `model_dict`. Their presence is not evidence that they
were downloaded, evaluated, or included in the reported comparison.

| Model repository ID | Original provider | License / official terms |
| --- | --- | --- |
| [`EleutherAI/pythia-6.9b-deduped`](https://huggingface.co/EleutherAI/pythia-6.9b-deduped) | EleutherAI | Apache License 2.0; registry specifies `revision="step3000"` |
| [`EleutherAI/gpt-j-6B`](https://huggingface.co/EleutherAI/gpt-j-6b) | EleutherAI | Apache License 2.0, as stated in its official model card |
| `gpt2-medium`, now [`openai-community/gpt2-medium`](https://huggingface.co/openai-community/gpt2-medium) | OpenAI | MIT, as stated in its official model card |
| [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) | Google | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| [`google/gemma-2-9b`](https://huggingface.co/google/gemma-2-9b) | Google | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

## Model use, outputs, and redistribution

The linked agreements are authoritative. Obtain models from their official
sources, satisfy any access/acceptance requirements, and follow their use
restrictions. Model accessibility on Hugging Face does not make every model
Apache-2.0 or unrestricted for every purpose.

Specific distinctions relevant to this repository and its response CSVs include:

- **Qwen2.5-3B-Instruct:** its research license permits noncommercial research and
  evaluation; commercial use requires a separate license. It also includes
  attribution requirements for distributed models improved using its outputs.
- **Llama 2 and Llama 3:** the agreements restrict using their outputs or results
  to improve other large language model families. Their presence in a benchmark
  CSV does not remove that restriction.
- **Llama 3.1:** the agreement includes naming requirements when its outputs are
  used to create, train, or improve an AI model that is distributed or made
  available. It is a different agreement from Llama 3.
- **Gemma:** the terms distinguish outputs from model derivatives and state
  that Google claims no rights in the outputs; users remain responsible for
  subsequent uses.
- **DeepSeekMath:** the model agreement states that the licensor generally
  claims no rights in generated outputs, while their use must not contravene
  the agreement. The code repository's MIT license is not the model license.

These are selected distinctions, not substitutes for the full agreements.
We do not assign a blanket project MIT license to generated model responses.
See [gsm_test/README.md](gsm_test/README.md) for the separate provenance of
benchmark questions, reference answers, responses, and scores.

If model weights or derivatives are distributed in a later release, include
their own required licenses, notices, attribution, and applicable restrictions
with that release. The code-and-results release documented here does not itself
grant permission to redistribute every evaluated checkpoint.

## Version evidence

The identifiers above record model family, size, and variant as specified in
the source code. Except for the unused Pythia registry entry noted above, the
code does not establish exact Hugging Face commit revisions for the historical
evaluations. No current model-card revision is asserted to be the revision used
in those runs. Recover exact revisions from saved run metadata or cached
snapshot records when available.
