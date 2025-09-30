# Arithmetic Transformer — README

A small repo for generating datasets and training transformer-style models on arithmetic tasks (addition, subtraction, sorting, scratchpad formats, etc.).
This README documents repository layout, a concise quickstart (generate data → (optional) update a config → train/evaluate).

---

## 1 — At a glance / repo layout

Important files & folders (high level):

- data_generate.py — script for generating training/val/testing data. Highly automated.

- data/ — store datasets here. Each task should get its own subdirectory (e.g. 4_operands_0_to_999_uniform).

- configuration_files/ — prototype config files. Edit one for your task (e.g. 4_operands_addition_plain.txt).

- train.py — (entry point) training script.

- evaluation.py, result_analysis.ipynb, statistical_measurements.py — evaluation & analysis utilities.

- results/ — recommended place for trainer outputs (model checkpoints, logs).

- startHere.ipynb — quick-start notebook.

- other utilities: model.py, main_utilities.py, configurator.py.

---

## 2 — Quickstart example (4-operand 0–999 addition, reverse format)

Follow two simple steps to get the model start training.

### 2.1 Generate data

Run the data_generate.py file with desired arguments.

Usage:
```bash
python data_generate.py --task <task> --num_operands <n> --experiment_name <name> \
[--output_path <path>] [--train_size N] [--test_size N] [--val_size N] \
[--train_eval] [--sample-size N] [--generate_reverse]
```

Example:
```bash
python data_generate.py --task addition --num_operands 4 --experiment_name 4_operands_0_to_999_uniform \
--train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```

#### Arguments (explanation)

- `--task` **(required)**
  Which generation task to run. Supported: `addition`, `multiplication`, `sorting`. The code will select the generator script under `data_generation_script/individual_task_scripts/`.

- `--num_operands` *(int, default: `4`)*
  Number of operands to generate (e.g. `2`, `3`, `4`). **Note:** this flag is forwarded **only** to generators that accept it (e.g. the addition generator). If a generator does not accept `--num_operands`, it will be ignored.

- `--experiment_name` **(required)**
  Logical name of the experiment. The script will write results to the location:
  <project_root>/data/<experiment_name>/

- `--train_size` *(int, default: `1_000_000`)*
  Number of training samples to generate.

- `--test_size` *(int, default: `10_000`)*
  Number of test samples to generate.

- `--val_size` *(int, default: `10_000`)*
  Number of validation samples to generate.

- `--train_eval` *(boolean-like: True / False, default: False)*
  When True, after generation the script will run sample.py to create a train_eval.txt file sampled from train.txt. Accepted true values (case-insensitive): "true", "1", "yes". Any other value is treated as False.

- `--sample-size` *(int, default: 10000)*
  Number of lines to sample from train.txt when --train_eval is True. Passed to sample.py.

- `--generate_reverse` *(boolean-like: True / False, default: False)*
  When True the script will run reverse_results.py at the end to produce reverse-format files. Note: not every task may support the reverse step — the dispatcher checks whether the chosen task enables generate_reverse. Accepted true values (case-insensitive): "true", "1", "yes".

#### Output files (what to expect)

- The generator writes files into the directory: data/<experiment_name>:
  * train.txt
  * test.txt
  * val.txt
- If --train_eval True, a sampled train_eval.txt will also be created in the same directory.
- If --generate_reverse True and the task supports it, reverse_results.py is run on the generated files and additional reverse-format files are produced in the same directory.

### 2.2 (Optional) Configure the model

1. Open an existing proto-config in configuration_files/, e.g. 4_operands_addition_reversed.txt.
2. Edit the fields below (common ones you will likely change):

- eval_interval — do an evaluation every {eval_interval} iterations. Recommended: 1000.
- wandb / logging settings — set your experiment name, project, or disable if not using wandb.
- data_format — one of: plain, reverse, scratchpad, max, sorting. For reverse-format tasks set: reverse.
- max_new_tokens — maximum output tokens. For 4-operand addition set at least 5.
- max_iters — number of training iterations. Recommend at least 200000.
- out_dir — output directory for this run (e.g. 'results/4_operands_0_to_999_uniform/plain_out').
- data_dir — data directory, where all the data files to be used live under this directory. (e.g. 'data/4_operands_0_to_999_uniform/').
- train_data_name — name of the training data (e.g. 'train.txt').
- train_data_test_name — name of the sampled train_eval file (optional, e.g. "train_eval.txt").
- val_data_name — name of the validation data (e.g. 'val.txt').
- test_file_name — name of the test data (e.g. 'test.txt').
- mode — "compute_gold" or "read_gold_as_str". If your test files already include the gold answers, use "read_gold_as_str".

Example snippet can be found at configuration_files/4_operands_addition_plain.txt.

---

### 2.3 Run the model

#### For a quick start: use this command
#### only have to specify --task and --experiment_name (match the ones specified in the data generation step)

```bash
python train.py --task addition --experiment_name 4_operands_0_to_999_uniform
```

#### For a finer control: use this command
#### The .txt file is the configuration file, configured in Section 2.2

```bash
!python train.py 4_operands_addition_plain.txt
```

To run evaluation/analysis after training:

result_analysis.ipynb contains some useful result analysis functions, such as one counting and drawing the digit-wise error vs iterations.

---

