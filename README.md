# Arithmetic Transformer — README

This repo is intended for generating synthetic datasets and training NanoGPTs on arithmetic tasks (including addition, simple multiplication, comparison, sorting).
This README documents repository layout, a concise quickstart (generate data → (optional) update a config → train → evaluate), and more detailed guide on how to reproduce the results in our paper .

---

## 1 — At a glance / repo layout

Important files & folders (high level):

- `data_generate.py` — script for generating training/val/test data. Highly automated.

- `data/` — directory for storing datasets. Each task should get its own subdirectory (e.g. 4_operands_0_to_999_uniform).

- `configuration_files/` — directory for storing prototype config files. Edit one for your task (e.g. 4_operands_addition_plain.txt).

- `train.py` — (entry point) training script.

- `evaluation.py`  — evaluation utilities.

- `statistical_measurements.py` - Mutual Information metric

- `result_analysis_script/` — directory for storing training result analysis scripts

- `results/` — directory for storing training outputs (e.g.training dynamics, model checkpoints).

- `startHere.ipynb` — quick-start notebook, actual commands you'll run.

- other utilities: model.py, main_utilities.py, configurator.py.

---

## 2 — Quickstart example (4-operand 0–999 addition, plain format)

Go inside `startHere.ipynb`.
Follow two simple steps to get the model start training.

### 2.1 Generate data (training, val & test)
Go to `I. Generate Data` secition of the notebook.
Choose one synthetic task (e.g. addition), and run the corresponding data generation command.

For 4 operand addition (including both plain and reverse output format), find this cell and run it:


```bash
!python data_generate.py --task addition --num_operands 4 --experiment_name 4_operands_0_to_999_uniform --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```
---

### 2.2 Start training
Go to `II. Let's Start Training!` section of the notebook. Find this cell and run it. Trainig will start.

```bash
!python train.py 4_operands_addition_plain.txt
```
The training results, including training dynamics and saved checkpoints will be stored under `results/4_operands_0_to_999_uniform/plain_out` directory. 

Note: Depending on the number of runs you've initiated, there may be further subdirectory under it. So you may have to look inside  `results/4_operands_0_to_999_uniform/.../.../`. The `test_results.csv` is the saved training dynamics, which will be **crucial** in our later result analysis.

---

### 2.3 Result Analysis
The training code you've run in `2.2 Start training` is supposed to print a digit-wise error vs training steps figure automatically once training finishes. That figure would appear under the same directory as the one `test_results.csv` is located.

If you do not see the digit-wise error figure, let's run a result analysis script directly. 
Go to `III. Result Analysis` section of the notebook.

Find the cell under "Digitwise Error Rates" and run it:
```bash
!python result_analysis_script/digitwise_error.py path/to/results.csv
```
The printed figure should appear under the same directory as where the result csv file is located.

---

## 3 — Detailed guide on reproducing

This section provides detailed instruction on how to reproduce all the main results in our paper.

### 3.1 Addition
#### 3.1.1 Generate data
Go to `I. Generate Data` secition of the notebook. Find this cell under `Addition`, and run it:
```bash
!python data_generate.py --task addition --num_operands 4 --experiment_name 4_operands_0_to_999_uniform --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```

If you want the thousands-place of the output randomized data (our ablation study), run this cell instead:

```bash
!python data_generate.py --task addition --randomize thousands --num_operands 4 --experiment_name 4_operands_0_to_999_output_randomize_thousands --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True
```

#### 3.1.2 Start trainng
Go to `II. Let's Start Training!` section of the notebook. 
Find this cell under `4 Operands Addition`, and run the corresponding output format you'd like.

For plain output format, run:
```bash
!python train.py 4_operands_addition_plain.txt
```

For reverse output format, run:
```bash
!python train.py 4_operands_addition_reversed.txt
```
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
!python result_analysis_script/digitwise_error.py path/to/results.csv
```

For normal fitting of prediction error distribution, run the following three.

Training step 1,000 to 1,800:
```bash
!python result_analysis_script/fit_normal.py \
  --input path/to/results.csv \
  --iter-start 1000 --iter-end 1800 --iter-step 200 \
  --diff-min -800 --diff-max 800
```

Training step 8,000 to 12,000:
```bash
!python result_analysis_script/fit_normal.py \
  --input path/to/results.csv \
  --iter-start 8000 --iter-end 12000 --iter-step 2000 \
  --diff-min -100 --diff-max 100
```

Training step 60,000 to 64,000:
```bash
!python result_analysis_script/fit_normal.py \
  --input input path/to/results.csv \
  --iter-start 60000 --iter-end 64000 --iter-step 2000 \
  --diff-min -20 --diff-max 20
```

Since the phase change might happen at different training steps for each training run, you may have to adjust the corresponding `--iter-start` and `--iter-end` based on the actual timing. (You may use "digit-wise error rate" figure to help decide the actual timing.)

---

### 3.2 Simple Multiplication
#### 3.2.1 Generate data
Go to `I. Generate Data` section of the notebook. Find the cell under `Multiplication` and run it:
```bash
!python data_generate.py --task multiplication --experiment_name 40_digit_times_1_digit --train_size 1000000 --test_size 10000 --val_size 10000 \
--a_max_digits 40 --b_max_digits 1 --train_eval True --sample-size 10000 --generate_reverse True
```

This will generate the 40-digit times 1-digit multiplication data.

#### 3.2.2 Start trainng
Go to `II. Let's Start Training!` section of the notebook. 
Find this cell under `Simpel Multiplication`, and run the corresponding output format you'd like.

For plain output format, run:
```bash
!python train.py 40_1_digits_mul_plain.txt
```

For reverse output format, run:
```bash
!python train.py 40_1_digits_mul_reversed.txt
```

#### 3.2.3 Result analysis
As before we need the test result csv file from `results/experiment_name/` directory (there may be further subdirectories) to run result analysis script on.

Once you've located the csv file, go to `III. Result Analysis` section of the notebook. Find the cell under `Simple Multiplication Task`, and run:
```bash
!python result_analysis_script/mul_digitwise_error_colormap.py path/to/results.csv
```

---

### 3.3 Comparison
#### 3.3.1 Generate data
Go to `I. Generate Data` section of the notebook. Find the cell `Comparison (Balanced data)` and run it:
```bash
!python data_generate.py --task comparison --experiment_name comparison_bal --train_eval True --sample-size 5000
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


#### 3.3.2 Start trainng
Go to `II. Let's Start Training!` section of the notebook. Find the cell under `Comparison`, and run it:
```bash
!python train.py comparison_bal.txt
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
!python result_analysis_script/comparison_error_rate.py \
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
!python data_generate.py --task sorting --experiment_name 4_operands_sorting_doubly_balanced --train_eval True --sample-size 5000
```
This will generate the doubly balanced training/eval/test data, which is both length balanced and is composed of NCID Group 0 to 2. In addition to the regular `test.txt`, the script will also generate a few extra test files under `data/4_operands_sorting_doubly_balanced/test` for later subskill learning order and mixing error analysis.

For all sorting data, we have 4 numbers to be sorted. Denote them as $a,b,c,d$. Except for the regular `test.txt` where each input number has equal probability be 3-digit & 4-digit. The numbers in the rest additional test files are all 4-digit, i.e. $l(a, b, c, d) = (4, 4, 4, 4)$. The following table summarizes each additinal test file. (We omitted the length contraint below)

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


#### 3.4.2 Start trainng
Go to `II. Let's Start Training!` section of the notebook. Find the cell under `Sorting` and run it: 
```bash
!python train.py 4_operands_sorting_doubly_bal.txt
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
| 1_3_same_2_4_agreeing_results.csv | Conflicting in $(b_2,c_2)$ and $(b_4, c_4)$ |
| 1_3_same_2_4_conflicting_results.csv | Agreeing in $(b_2,c_2)$ and $(b_4, c_4)$ |

Once we've located these csv files, go to `III. Result Analysis` section of the notebook. Find the cell under `Sorting Subskill from 10% to 90% Range`, and run it:
```bash
!python result_analysis_script/sorting_acc_10_90_range.py \
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
!python path/to/1_3_same_2_4_agreeing_results.csv
```
```bash
!python path/to/1_3_same_2_4_conflicting_results.csv
```
The script will output the swap, repeat & mixing error rate for each training step we evaluated, and the mean and standard deviation for the mixing error rate among the last 10 evaluated training steps.

---

### 3.5 NanoGPT Scaling
For pure model scaling, we use the same data as in the regular addition task, which is generated in `3.1.1 Generate data`.

Just find the cell under `IV. NanoGPT Scaling`, and run it:
```bash
!python train.py 20M_4_operands_addition_plain.txt
```

This will start training a 20M-parameter NanoGPT model. 

Similarly, for scaling to 100M-parameter, run:
```bash
!python train.py 100M_4_operands_addition_plain.txt
```

---

### 3.6 the Extended-GSM (50 questions) Evaluation
To reproduce our evaluation of LLMs on the extended GSM test, open the `myGSM_test.ipynb` under the `gsm_test\` directory.

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

