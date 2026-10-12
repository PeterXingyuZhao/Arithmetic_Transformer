# Reproducing the experiments

[Back to the paper overview](../README.md)

This guide covers synthetic data generation, model training, and analysis for *Shattered Compositionality: Counterintuitive Learning Dynamics of Transformers for Arithmetic*. Start with the addition example, then choose the experiments you want to reproduce.

**Run every command from the repository root**, where `train.py` and `data_generate.py` are located, rather than from `docs/`. Commands below use terminal syntax. In a notebook code cell, prefix each `python` command with `!`; [startHere.ipynb](../startHere.ipynb) organizes the workflow into data generation, training, and result analysis. Replace example CSV paths with the files from your own runs; some notebook cells contain paths from earlier experiments.

- [Repository layout and setup](#1--repository-layout-and-setup)
- [Quickstart: four-operand addition](#2--quickstart-example-4-operand-0999-addition-plain-format)
- [Addition and its ablation](#31-addition) · [Multiplication](#32-simple-multiplication) · [Comparison](#33-comparison) · [Sorting](#34-sorting)
- [Model scaling](#35-nanogpt-scaling) · [Extended-GSM evaluation](#36-extended-gsm-50-questions-evaluation) · [Scratchpad training](#37-training-addition-with-scratchpad) · [Pythia fine-tuning](#38-fine-tuning-pythia-on-addition)

---

## 1 — Repository layout and setup

Important files & folders (high level):

- `data_generate.py` — script for generating training/val/test data. Highly automated.

- `data/` — generated datasets, created when data generation runs and excluded from version control. Each experiment gets its own subdirectory (e.g. `4_operands_0_to_999_uniform`).

- `configuration_files/` — directory for storing prototype config files. Edit one for your task (e.g. 4_operands_addition_plain.txt).

- `train.py` — (entry point) training script.

- `evaluation.py`  — evaluation utilities.

- `statistical_measurements.py` — mutual-information and conditional-mutual-information diagnostics.

- `result_analysis_script/` — directory for storing training result analysis scripts

- `results/` — generated training outputs, including predictions, training metrics, and model checkpoints; excluded from version control.

- `startHere.ipynb` — quick-start notebook, actual commands you'll run.

- `model.py`, `model_rope.py`, `model_t5bias.py` — Transformer implementations with absolute, rotary, and T5-style relative positional encodings.
- `main_utilities.py`, `evaluation.py`, `result_analysis.py` — data/tokenization, evaluation, and training-output analysis utilities.

### Setup and configuration

Install a PyTorch build appropriate for your hardware. The training defaults use CUDA and `torch.compile`, so the supplied training configurations are intended for a CUDA GPU with PyTorch 2.x. The addition example is a full experiment: its default configuration uses a batch size of 512 and up to 800,000 iterations. Review the selected configuration before starting a run.

The repository does not include a pinned environment or dependency lockfile. The following packages are identified from the training and analysis imports; this is a dependency starting point, not a tested version matrix:

```bash
python -m pip install numpy pandas matplotlib tqdm requests pyyaml tiktoken wandb
```

`wandb` and `tiktoken` are imported by the training code even when logging is disabled or character-level tokenization is used. Install `transformers` for Pythia fine-tuning; `scipy` enables the optional chi-squared p-value in the normal-fit analysis. Install Jupyter if you want to run the notebooks locally.

The `.txt` files in `configuration_files/` contain Python assignments. Pass the filename, such as `4_operands_addition_plain.txt`, to `train.py` without the `configuration_files/` prefix. Edit that file to configure the device, batch size, training duration, data paths, and output paths. To disable model compilation, add `compile = False` to the selected configuration.

The supplied addition configuration enables Weights & Biases logging. Either authenticate with your own W&B account or set `wandb_log = False` in the configuration file. General `--key=value` overrides, including `--wandb_log=False`, are currently not applied by `train.py`; change those settings in the file. The explicitly parsed `--experiment_name` option does work and updates the experiment's data and output directories.

Training and data generation can create large files. Generated `data/` and `results/` directories are not included in a fresh checkout. The analysis commands below consume predictions saved during evaluation; they do not download pretrained experimental results.

---

## 2 — Quickstart example (4-operand 0–999 addition, plain format)

Go inside `startHere.ipynb`.
Follow two simple steps to start model training.

### 2.1 Generate data (training, val & test)
Go to `I. Generate Data` section of the notebook.
Choose one synthetic task (e.g. addition), and run the corresponding data generation command.

For 4 operand addition (including both plain and reverse output format), find this cell and run it:


```bash
python data_generate.py --task addition --num_operands 4 --experiment_name 4_operands_0_to_999_uniform --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```
---

### 2.2 Start training
Go to `II. Let's Start Training!` section of the notebook. Find this cell and run it. Training will start.

```bash
python train.py 4_operands_addition_plain.txt
```
The training results, including training dynamics and saved checkpoints will be stored under `results/4_operands_0_to_999_uniform/plain_out` directory.

Note: Depending on the number of runs you've initiated, there may be further subdirectory under it. So you may have to look inside  `results/4_operands_0_to_999_uniform/.../.../`. The `test_results.csv` is the saved training dynamics, which will be **crucial** in our later result analysis.

---

### 2.3 Result Analysis
The training code you've run in `2.2 Start training` is configured to save a digit-wise error vs training steps figure automatically once training finishes. That figure would appear under the same directory as the one `test_results.csv` is located.

If you do not see the digit-wise error figure, let's run a result analysis script directly.
Go to `III. Result Analysis` section of the notebook.

Find the cell under "Digitwise Error Rates" and run it:
```bash
python result_analysis_script/digitwise_error.py path/to/results.csv
```
The saved figure should appear under the same directory as where the result csv file is located.

---

## 3 — Detailed experiment guide

This section gives the generation, training, and analysis commands for the main experiments. Exact transition steps can vary across runs, so use your own learning curves to choose analysis windows.

### 3.1 Addition
#### 3.1.1 Generate data
Go to `I. Generate Data` section of the notebook. Find this cell under `Addition`, and run it:
```bash
python data_generate.py --task addition --num_operands 4 --experiment_name 4_operands_0_to_999_uniform --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```

If you want the thousands-place of the output randomized data (our ablation study), run this cell instead:

```bash
python data_generate.py --task addition --randomize thousands --num_operands 4 --experiment_name 4_operands_0_to_999_output_randomize_thousands --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```

#### 3.1.2 Start training
Go to `II. Let's Start Training!` section of the notebook.
Find this cell under `4 Operands Addition`, and run the corresponding output format you'd like.

For plain output format, run:
```bash
python train.py 4_operands_addition_plain.txt
```

For reverse output format, run:
```bash
python train.py 4_operands_addition_reversed.txt
```

For the thousands-place randomization ablation generated above, point the same configuration at the ablation experiment:

```bash
python train.py 4_operands_addition_plain.txt --experiment_name 4_operands_0_to_999_output_randomize_thousands
```

Use `4_operands_addition_reversed.txt` in that command for reversed output. Without `--experiment_name`, these configurations continue to read the ordinary uniform-addition data.
#### 3.1.3 Result analysis
For most result analysis scripts, one of the required input argument is the path to the test result csv file you'd like to be analyzed. That test result csv file is updated as training progresses and is stored under `results/experiment_name/` directory, where `experiment_name` is the one you provided to `data_generate.py` when generating data. If you had used the default data generation script we provided in `startHere.ipynb`, they have the following names:

| Task    | Default Experiment Name |
| -------- | ------- |
| Addition  | 4_operands_0_to_999_uniform    |
| Addition (ablation) | 4_operands_0_to_999_output_randomize_thousands     |
| Multiplication    | 40_digit_times_1_digit    |
| Comparison    | comparison_bal    |
| Sorting    | 4_operands_sorting_doubly_balanced    |

Once you've determined the path to the stored test result csv file, go to `III. Result Analysis` section of the notebook.

For digit-wise error rate vs training steps analysis, run:
```bash
python result_analysis_script/digitwise_error.py path/to/results.csv
```

To fit normal distributions to prediction errors, run the following commands.

Training step 1,000 to 1,800:
```bash
python result_analysis_script/fit_normal.py \
  --input path/to/results.csv \
  --iter-start 1000 --iter-end 1800 --iter-step 200 \
  --diff-min -800 --diff-max 800
```

Training step 8,000 to 12,000:
```bash
python result_analysis_script/fit_normal.py \
  --input path/to/results.csv \
  --iter-start 8000 --iter-end 12000 --iter-step 2000 \
  --diff-min -100 --diff-max 100
```

Training step 60,000 to 64,000:
```bash
python result_analysis_script/fit_normal.py \
  --input path/to/results.csv \
  --iter-start 60000 --iter-end 64000 --iter-step 2000 \
  --diff-min -20 --diff-max 20
```

Since the phase change might happen at different training steps for each training run, you may have to adjust the corresponding `--iter-start` and `--iter-end` based on the actual timing. (You may use "digit-wise error rate" figure to help decide the actual timing.)

---

### 3.2 Simple Multiplication
#### 3.2.1 Generate data
Go to `I. Generate Data` section of the notebook. Find the cell under `Multiplication` and run it:
```bash
python data_generate.py --task multiplication --experiment_name 40_digit_times_1_digit --train_size 1000000 --test_size 10000 --val_size 10000 \
--a_max_digits 40 --b_max_digits 1 --train_eval True --sample-size 10000 --generate_reverse True
```

This generates multiplication data with a first operand of up to 40 digits and a single-digit second operand. With `--a_max_digits 40`, the generator samples lengths from 1 to 40 with probability proportional to the length; it does not restrict every first operand to exactly 40 digits.

#### 3.2.2 Start training
Go to `II. Let's Start Training!` section of the notebook.
Find this cell under `Simpel Multiplication`, and run the corresponding output format you'd like.

For plain output format, run:
```bash
python train.py 40_1_digits_mul_plain.txt
```

For reverse output format, run:
```bash
python train.py 40_1_digits_mul_reversed.txt
```

#### 3.2.3 Result analysis
As before we need the test result csv file from `results/experiment_name/` directory (there may be further subdirectories) to run result analysis script on.

Once you've located the csv file, go to `III. Result Analysis` section of the notebook. Find the cell under `Simple Multiplication Task`, and run:
```bash
python result_analysis_script/mul_digitwise_error_colormap.py path/to/results.csv
```

---

### 3.3 Comparison
#### 3.3.1 Generate data
Go to `I. Generate Data` section of the notebook. Find the cell `Comparison (Balanced data)` and run it:
```bash
python data_generate.py --task comparison --experiment_name comparison_bal --train_eval True --sample-size 5000
```
This will generate the balanced training/val/test comparison data (which is composed of NCID Group 0-4). The script will also generate several additional test files under `data/comparison_bal/test` directory.

| Test filename    | Constraint |
| -------- | ------- |
| `thousands_diff_only.txt`  | $a_1 \ne b_1;\ a_{j\ne 1}=b_j$    |
| `hundreds_diff_only.txt` | $a_2 \ne b_2;\ a_{j\ne 2}=b_j$     |
| `tens_diff_only.txt`    | $a_3 \ne b_3;\ a_{j\ne 3}=b_j$    |
| `units_diff_only.txt`    | $a_4 \ne b_4;\ a_{j\ne 4}=b_j$    |
| `thousands.txt`    | No guaranteed match    |
| `hundreds.txt`    | $a_1 = b_1$    |
| `tens.txt`    | $a_{1:2} = b_{1:2}$    |
| `units.txt`    | $a_{1:3} = b_{1:3}$    |
| `equal.txt`    | $a_{1:4} = b_{1:4}$    |


#### 3.3.2 Start training
Go to `II. Let's Start Training!` section of the notebook. Find the cell under `Comparison`, and run it:
```bash
python train.py comparison_bal.txt
```

#### 3.3.3 Result analysis
Since we have a few extra test files, we have to use all of their corresponding test result csv files for result analysis. For example, to show how Contrast Pairs error rates change with training steps, we have to use the following under the `results/experiment_name/` folder (there may be further subfolders):

| Test Result CSV Filename |
| ------- |
| thousands_diff_only_results.csv |
| hundreds_diff_only_results.csv |
| tens_diff_only_results.csv |
| units_diff_only_results.csv |

Once we have located them, go to `III. Result Analysis` section of the notebook. Find the cell under `Comparison Task` and run it with the paths to the above csv files:
```bash
python result_analysis_script/comparison_error_rate.py \
  path/to/thousands_diff_only_results.csv \
  path/to/hundreds_diff_only_results.csv \
  path/to/tens_diff_only_results.csv \
  path/to/units_diff_only_results.csv \
  --output_file_name contrast_pair_error_rate
```

---


### 3.4 Sorting
#### 3.4.1 Generate data
Go to `I. Generate Data` section of the notebook. Find the cell under `Sorting (Doubly balanced data)` and run:
```bash
python data_generate.py --task sorting --experiment_name 4_operands_sorting_doubly_balanced --train_eval True --sample-size 5000
```
This will generate the doubly balanced training/eval/test data, which is both length balanced and is composed of NCID Group 0 to 2. In addition to the regular `test.txt`, the script will also generate a few extra test files under `data/4_operands_sorting_doubly_balanced/test` for later subskill learning order and mixing error analysis.

For all sorting data, we have 4 numbers to be sorted. Denote them as $a,b,c,d$. Except for the regular `test.txt` where each input number has equal probability of being 3-digit or 4-digit. The numbers in the rest additional test files are all 4-digit, i.e. $l(a, b, c, d) = (4, 4, 4, 4)$. The following table summarizes each additional test file. (We omitted the length constraint below)

| Test filename    | Constraint |
| -------- | ------- |
| `digitwise_random.txt`  | No guaranteed match    |
| `digitwise_thousand.txt` | $a_1=b_1=c_1=d_1$     |
| `digitwise_hundred.txt`    | $a_{1:2}=b_{1:2}=c_{1:2}=d_{1:2}$    |
| `digitwise_ten.txt`    | $a_{1:3}=b_{1:3}=c_{1:3}=d_{1:3}$    |
| `1_3_same_2_4_agreeing.txt`    | $a=1000,b_1=c_1, sgn(b_2 − c_2) · sgn(b_4 − c_4) = 1, b_3=c_3,d=9999$    |
| `1_3_same_2_4_conflicting.txt`    | $a=1000,b_1=c_1, sgn(b_2 − c_2) · sgn(b_4 − c_4) = -1, b_3=c_3,d=9999$    |
| `b1_eq_b3diff.txt`    | $a=1000,b_1=c_1, sgn(b_2 − c_2) · sgn(b_4 − c_4) = -1, b_3 \neq c_3,d=9999$    |
| `b3_eq_b1diff.txt`    | $a=1000,b_1 \neq c_1, sgn(b_2 − c_2) · sgn(b_4 − c_4) = -1, b_3 =c_3, d=9999$    |
| `b1c1_b3c3_bothdiff.txt`    | $a=1000,b_1 \neq c_1, sgn(b_2 − c_2) · sgn(b_4 − c_4) = -1, b_3 \neq c_3, d=9999$    |


#### 3.4.2 Start training
Go to `II. Let's Start Training!` section of the notebook. Find the cell under `Sorting` and run it:
```bash
python train.py 4_operands_sorting_doubly_bal.txt
```

#### 3.4.3 Result analysis
As before we need different test result csv files under `results/experiment_name/` for different result analysis.

Specifically, for determining subskill learning order, we will use:

| Test Result CSV Filename | Subskill assessed |
| ------- | ------- |
| test_results.csv | Length comparison|
| digitwise_random_results.csv | First digit comparison |
| digitwise_thousand_results.csv | Second digit comparison |
| digitwise_hundred_results.csv | Third digit comparison |
| digitwise_ten_results.csv | Fourth digit comparison |

For the conflicting vs agreeing pair mixing error analysis, we will use:

| Test Result CSV Filename | Condition |
| ------- | ------- |
| 1_3_same_2_4_agreeing_results.csv | Agreeing in $(b_2,c_2)$ and $(b_4, c_4)$ |
| 1_3_same_2_4_conflicting_results.csv | Conflicting in $(b_2,c_2)$ and $(b_4, c_4)$ |

Once we've located these csv files, go to `III. Result Analysis` section of the notebook. Find the cell under `Sorting Subskill from 10% to 90% Range`, and run it:
```bash
python result_analysis_script/sorting_acc_10_90_range.py \
  --csv \
    path/to/test_results.csv \
    path/to/digitwise_random_results.csv \
    path/to/digitwise_thousand_results.csv \
    path/to/digitwise_hundred_results.csv \
    path/to/digitwise_ten_results.csv \
  --positions 1,2,3,4 \
  --mode length first second third fourth
```

This should give the subskill learning progress bar plot.

For mixing error analysis, find the cell under `Sorting Mixing Error` and run it:
```bash
python result_analysis_script/mixing_error.py path/to/1_3_same_2_4_agreeing_results.csv
```
```bash
python result_analysis_script/mixing_error.py path/to/1_3_same_2_4_conflicting_results.csv
```
The script will output the swap, repeat & mixing error rate for each training step we evaluated, and the mean and standard deviation for the mixing error rate among the last 10 evaluated training steps.

---

### 3.5 NanoGPT Scaling
For pure model scaling, we use the same data as in the regular addition task, which is generated in `3.1.1 Generate data`.

To start training, go to `II. Let's Start Training!` section of the notebook. Find the cell under `NanoGPT Scaling (20M)`, and run it:
```bash
python train.py 20M_4_operands_addition_plain.txt
```

This will start training a 20M-parameter NanoGPT model.

Similarly, for scaling to 100M-parameter, find the cell under `NanoGPT Scaling (100M)`, and run it:
```bash
python train.py 100M_4_operands_addition_plain.txt
```

For result analysis, see `3.8.3 Result analysis`.

---

### 3.6 Extended-GSM (50 questions) evaluation
To reproduce our evaluation of LLMs on the extended GSM test, open [gsm_test/myGSM_test.ipynb](../gsm_test/myGSM_test.ipynb).

The notebook contains a few values that are specific to *your* environment:

- **Google Drive paths**
  - `wd` (working directory where outputs are written)
  - `model_path` (where Hugging Face model weights are cached/downloaded)

- **Hugging Face access token**
  - The notebook reads a token from a text file (e.g., `.../HF_token.txt`) and runs `huggingface-cli login`.
  - Create that file with your own HF token.

- **Model access / licenses**
  - Some models (e.g., Llama-family) require accepting Hugging Face model terms on your account before download.


Once you've configured these, just run the cells **top-to-bottom** (one by one). The last cell, which is titled `Evaluate the LLMs on our extended GSM question set`, should output each model's accuracy on the 50 Extended-GSM questions for each `k ∈ {1,2,3,4,5,6}`

---

### 3.7 Training Addition with Scratchpad
#### 3.7.1 Generate data
To train models with scratchpad, first go to `I. Generate Data` section of the notebook -- `startHere.ipynb`.

For D scratchpad, find the cell under `Addition with scratchpad (form 1)`, and run it:
```bash
python data_generate.py --task addition --reasoning_mode 1 --num_operands 4 --experiment_name 4_operands_0_to_999_uniform_scratchpad1 --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```

This should generate the scratchpad-added train/val/test data under `data/4_operands_0_to_999_uniform_scratchpad1/`, which are named `train_scratchpad1.txt`, `val_scratchpad1.txt`, `test_scratchpad1.txt` respectively.

For D+A scratchpad, find the cell under `Addition with scratchpad (form 2)`, and run it:
```bash
python data_generate.py --task addition --reasoning_mode 2 --num_operands 4 --experiment_name 4_operands_0_to_999_uniform_scratchpad2 --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```

This should generate the scratchpad-added train/val/test data under `data/4_operands_0_to_999_uniform_scratchpad2/`, which are named `train_scratchpad2.txt`, `val_scratchpad2.txt`, `test_scratchpad2.txt` respectively.

#### 3.7.2 Start training
Go to `II. Let's Start Training!` section of the notebook.

To train with D scratchpad, find the cell under `Scratchpad Form 1`, and run it:
```bash
python train.py 4_operands_addition_plain_scratchpad1.txt
```

To train with D+A scratchpad, find the cell under `Scratchpad Form 2`, and run it:
```bash
python train.py 4_operands_addition_plain_scratchpad2.txt
```

For result analysis, see `3.8.3 Result analysis`.

#### 3.7.3 Result analysis

Use the scratchpad CSVs in the [combined scaling, scratchpad, and fine-tuning analysis](#383-result-analysis) below. The `True` flag in each scratchpad `--test` entry tells the analysis script to handle the reasoning output.

### 3.8 Fine-tuning Pythia on addition
#### 3.8.1 Generate data

For Pythia fine-tuning on addition, use the same training, validation, and test data as the regular addition task in plain output format, generated in section 3.1.1.

#### 3.8.2 Start training

To start training, go to `II. Let's Start Training!` section of the notebook. Find the cell under `Pythia Finetuning ` and run it:
```bash
python train.py 4_operands_addition_plain_pythia.txt
```
#### 3.8.3 Result analysis
In this section, we'll plot a bar graph that shows digit-wise error rates at a fixed training step, for NanoGPT scaling, Pythia finetuning and training with two forms of scratchpad.

As before, the key is to locate all the test result csv files we need.

| Experiment | Test result csv filename | Location |
| ------- | ------- | ------- |
| NanoGPT Scaling 20M | test_results.csv | `results/4_operands_0_to_999_uniform/20M_plain_out` |
| NanoGPT Scaling 100M | test_results.csv | `results/4_operands_0_to_999_uniform/100M_reverse_out` |
| Pythia fine-tuning | test_results.csv | `results/4_operands_0_to_999_uniform/plain_out_pythia` |
| Scratchpad D | test_scratchpad1_results.csv | `results/4_operands_0_to_999_uniform_scratchpad1/plain_out` |
| Scratchpad D+A | test_scratchpad2_results.csv | `results/4_operands_0_to_999_uniform_scratchpad2/plain_out` |

These are parent output directories; look inside the run-specific subdirectory for each CSV. The current `100M_4_operands_addition_plain.txt` configuration uses the directory name `100M_reverse_out` even though its data format is plain. The table reflects that setting; inspect `out_dir` if you rename the output directory locally.

Once we find their paths, go to `III. Result Analysis` section of the notebook. Find the cell under `Scaling, Scratchpad, and Pythia finetuning`, and run it:
```bash
python result_analysis_script/scaling_scratchpad_finetuning.py \
  --test "20M NanoGPT" path/to/20M/test_results.csv 20000 False \
  --test "100M NanoGPT" path/to/100M/test_results.csv 50000 False \
  --test "Pythia 1B" path/to/pythia/test_results.csv 30000 False \
  --test "Scratchpad D" path/to/test_scratchpad1_results.csv 4500 True \
  --test "Scratchpad D+A" path/to/test_scratchpad2_results.csv 500 True
```

This will produce a bar plot that shows the digit-wise error rates for the 20M-parameter NanoGPT, 100M-parameter NanoGPT, finetuning a pretrained Pythia-1B model, training with D scratchpad, and training with D+A scratchpad, at training step 20000, 50000, 30000, 4500, and 500 respectively.
