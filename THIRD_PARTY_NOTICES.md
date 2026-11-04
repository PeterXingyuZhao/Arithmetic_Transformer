# Third-party attribution and license notices

This document accompanies the code and data for *Shattered Compositionality:
Counterintuitive Learning Dynamics of Transformers for Arithmetic*. It identifies
reused materials and preserves the original upstream notices. It does not replace
their terms or assign a blanket license to every file in this repository.

## nanoGPT

- Creator: **Andrej Karpathy**.
- Source: <https://github.com/karpathy/nanoGPT>.
- License: **MIT**; complete original text in
  [licenses/nanoGPT-MIT.txt](licenses/nanoGPT-MIT.txt).
- Original copyright: **Copyright (c) 2022 Andrej Karpathy**.

Our arithmetic implementation builds on nanoGPT and teaching_arithmetic.
These notices accompany the related implementation files, including `model.py`,
`model_rope.py`, `model_t5bias.py`, `train.py`, and `configurator.py`.
The model variants retain this shared implementation while
adding or changing positional encodings and arithmetic experiment behavior.
These files have been modified for this project; they are not unmodified
upstream releases.

## Teaching Arithmetic to Small Transformers

- Creators: **Nayoung Lee, Kartik Sreenivasan, Jason D. Lee, Kangwook Lee,
  and Dimitris Papailiopoulos**.
- Paper: *Teaching Arithmetic to Small Transformers*, ICLR 2024.
- Source: <https://github.com/lee-ny/teaching_arithmetic>.
- License: **MIT**; complete original text in
  [licenses/teaching_arithmetic-MIT.txt](licenses/teaching_arithmetic-MIT.txt).
- Original copyright: **Copyright (c) 2023 lee-ny**.

This repository is a fork of `lee-ny/teaching_arithmetic`, which itself builds
on nanoGPT. The inherited arithmetic training and evaluation code and its
subsequent adaptations retain the upstream attribution. Our project extends
the code for addition, multiplication, comparison, sorting, subskill analysis,
and the additional experimental configurations described in the paper.
The teaching_arithmetic notice accompanies, rather than replaces, the nanoGPT
notice for code derived from both projects.

## GSM8K / grade-school-math

- Creators: **Karl Cobbe and collaborators at OpenAI**.
- Paper: Karl Cobbe et al., *Training Verifiers to Solve Math Word Problems*,
  2021, <https://arxiv.org/abs/2110.14168>.
- Source: <https://github.com/openai/grade-school-math>.
- License: **MIT**; complete original text in
  [licenses/GSM8K-MIT.txt](licenses/GSM8K-MIT.txt).
- Original copyright: **Copyright (c) 2021 OpenAI**.

We adapted selected GSM8K word problems into parameterized templates, added
clauses involving additional quantities or relationships, and recomputed
reference answers using the template-generating functions. The current
Extended-GSM evaluation uses 50 templates and six clause-count settings
(`k = 1, ..., 6`). This is a modified benchmark, not the unmodified GSM8K test set.

The notice applies to the GSM8K-derived question content in
`gsm_test/myGSM_test.ipynb`, the earlier generators in `gsm_test/gsm-test.ipynb`,
and the generated questions and associated reference answers in
`gsm_test/k_1/*.csv` through `gsm_test/k_6/*.csv`. Any GSM8K-derived content in
the accompanying `gsm_test/question_repo.numbers` workbook has the same upstream
provenance. See [gsm_test/README.md](gsm_test/README.md) for the distinction
between benchmark-derived content and model-generated responses.

This attribution does not label the independent synthetic arithmetic datasets
elsewhere in the repository as GSM8K, or apply GSM8K's license to model weights
or generated responses.

## Pretrained models and software dependencies

[MODEL_LICENSES.md](MODEL_LICENSES.md) identifies the pretrained model providers,
exact repository IDs used in the code, and their official licenses and use
terms. Model weights are obtained from their providers through Hugging Face;
the public repository tree reviewed for this notice does not include pretrained
checkpoint files. The `llama_adapter.py` and `llama_tokenizer.py` modules wrap
Hugging Face APIs and default to Pythia-1B despite their filenames.

External packages imported by the code, including PyTorch, Hugging Face
Transformers, NumPy, pandas, and others, remain under their respective licenses.
These imports do not make those packages part of the project's MIT-attributed
source portions. This notice does not claim to reproduce every installed
dependency's license or to grant rights to the dependency packages.

## License-source provenance and reuse

[licenses/SOURCES.json](licenses/SOURCES.json) records the pinned upstream URLs
and SHA-256 hashes of the complete license texts bundled here. Those revisions
identify the sources of the notices; they are not a claim that all experiments
used those exact upstream revisions. Historical experiment revisions and
checkpoint hashes should be taken from the corresponding saved run records.

When redistributing copies or substantial portions of the MIT-licensed upstream
materials, retain the applicable original copyright and permission notices.
The bundled texts preserve those notices and the original warranty disclaimers.
If a subset of this repository is distributed separately, include its applicable
license files and attribution with that subset.

These notices preserve third-party rights. They do not add a separate license
grant for otherwise unlicensed original contributions or override any
pretrained-model license, acceptable-use policy, or other applicable term.
