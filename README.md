# Arithmetic Transformer — README

A small repo for generating datasets and training transformer-style models on arithmetic tasks (addition, subtraction, sorting, scratchpad formats, etc.).
This README documents repository layout, a concise quickstart (generate data → move data into data/ → update a config → train/evaluate).

---

## 1 — At a glance / repo layout

Important files & folders (high level):

- data_generation_script/
  - data_generation.ipynb — interactive notebook that generates datasets. H1 headings correspond to task categories (e.g. Addition 4 Operand).
  - sample.py — sample a subset of train data for memorization checks (creates train_eval.txt).
  - reverse_results.py — produce “reverse” format outputs (aligns transformer next-token prediction with grade-school algorithm).

- data/ — store datasets here. Each task should get its own subdirectory (e.g. 4_operands_0_to_999_uniform).

- configuration_files/ — prototype config files. Edit one for your task (e.g. 4_operands_addition_reversed.txt).

- train.py — (entry point) training script.

- evaluation.py, result_analysis.ipynb, statistical_measurements.py — evaluation & analysis utilities.

- results/ — recommended place for trainer outputs (model checkpoints, logs).

- startHere.ipynb — quick-start notebook (check for runtime hints).

- other utilities: model.py, main_utilities.py, configurator.py.

---

## 2 — Quickstart example (4-operand 0–999 addition, reverse format)

Follow these steps to go from data generation to training.

### 2.1 Generate data
1. Open data_generation_script/data_generation.ipynb.
2. Find the H1 heading "Addition 4 Operand". Under that H1 there are subheadings for variations. To generate 4-operand 0–999 addition without padding locate the subheading "Generate 4 Operand 0-999 without Paddin"g and run the corresponding cell(s).
3. The notebook will write three files to the current working directory (where the notebook runs):

```
   train.txt
   val.txt
   test.txt
```

(Watch the notebook output messages — they show the exact write locations.)

### 2.2 (Optional) Sample a subset of training data for memorization checks
Under the H1 heading Sample Train Eval Data & Reverse Result Script, run the sample command (or run sample.py directly). Example:

```python
# produce a small train_eval set (3000 examples)
python data_generation_script/sample.py \
  --input ./train.txt \
  --output ./train_eval.txt \
  --sample-size 3000
```


### 2.3 (Optional) Create reverse-format datasets
To create the reverse-format datasets (so the model predicts the result in the reversed alignment), run the reverse script (from the same H1 section). That will output files like:

```
train_reverse.txt
train_eval_reverse.txt
val_reverse.txt
test_reverse.txt
```

### 2.4 Move data into data/
Create a directory under data/ named for the task (example):

```python
mkdir -p data/4_operands_0_to_999_uniform
```

Then move the generated files into that directory:

```python
mv train.txt val.txt test.txt train_reverse.txt val_reverse.txt test_reverse.txt data/4_operands_0_to_999_uniform/
# and train_eval.txt, train_eval_reverse.txt if you created it
```
---

## 3 — Configure the model

1. Open an existing proto-config in configuration_files/, e.g. 4_operands_addition_reversed.txt.
2. Edit the fields below (common ones you will likely change):

- eval_interval — do an evaluation every {eval_interval} iterations. Recommended: 1000.
- wandb / logging settings — set your experiment name, project, or disable if not using wandb.
- data_format — one of: plain, reverse, scratchpad, max, sorting. For reverse-format tasks set: reverse.
- max_new_tokens — maximum output tokens. For 4-operand addition set at least 5.
- max_iters — number of training iterations. Recommend at least 200000.
- out_dir — output directory for this run (e.g. results/4_operands_0_to_999_uniform/).
- train_data_path — path to training .txt file (e.g. data/4_operands_0_to_999_uniform/train_reverse.txt).
- train_data_test_path — path to sampled train-eval file (optional).
- val_data_path — path to validation .txt.
- test_file_path — path to test .txt (e.g. data/.../test_reverse.txt).
- mode — "compute_gold" or "read_gold_as_str". If your test files already include the gold answers, use "read_gold_as_str".

Example snippet can be found at configuration_files/4_operands_addition_reversed.txt.

---

## 4 — Run training & evaluation

Start training (example command can be found at startHere.ipynb):

```python
python train.py 4_operands_addition_reversed.txt
```

To run evaluation/analysis after training:

result_analysis.ipynb contains some useful result analysis functions, such as one counting and drawing the digit-wise error vs iterations.

---

