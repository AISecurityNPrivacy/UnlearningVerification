# LLM Unlearning Experiment

This module implements the unlearning verification experiment for LLMs, focusing on verifying whether an unlearned model can effectively forget specific data while retaining knowledge of others.

We conduct experiments using the **OPT-2.7B** model on two datasets:
- ✅ **TruthfulQA** (Retained dataset, Dr)
- 🛑 **SafeRLHF** (Unlearning dataset, Du)

---

## 📌 Overview

We fine-tune a pre-trained OPT-2.7B model sequentially on SafeRLHF and then TruthfulQA data. Then we treat SafeRLHF as the *Du* and apply unlearning techniques to remove its influence while preserving performance on *Dr*.
After that, we directly fine-tune the pre-trained OPT-2.7B model with the subsets of *Dr* to get the verification models *Mv*
Finally, we evaluate the effectiveness of unlearning using the proposed verification.

---

## 🚀 Running the Experiment

### 1. Fine-tune on SafeRLHF

```bash
python data/finetune_SafeRLHF.py
```

This script loads the OPT-2.7B base model and fine-tunes it on the SafeRLHF dataset.

### 2. Fine-tune on TruthfulQA

```bash
python data/finetune_TruthfulQA.py
```

Fine-tunes the model on *Dr* (TruthfulQA), resulting in **M**, or generates **Mv** when subset selection is enabled.

#### 🔧 `finetune_TruthfulQA.py` Arguments

| Argument         | Type    | Default                        | Description |
|------------------|---------|--------------------------------|-------------|
| `-model_path`    | `str`   | `"facebook/opt-2.7b"`          | Path to the pre-trained or checkpoint model. |
| `-split`         | `bool`  | `False`                        | Whether to split the dataset into two subsets. |
| `-subset`        | `str`   | `"A"`                          | Select `"A"` or `"B"` when `split=True`. |
| `-ratio`         | `float` | `0.5`                          | Ratio of data in subset A (if splitting is used). |
| `-epoch`         | `int`   | `2`                            | Number of fine-tuning epochs. |
| `-save_path`     | `str`   | `"opt2.7b_trufulqa_0.5_A"`     | Output directory for the trained model. |

#### 💡 Examples

**Use full dataset to get pre-unlearned model:**

```bash
python data/finetune_TruthfulQA.py -model_path opt2.7b_saferlhf -split False -ratio 1.0 -save_path opt2.7b_saferlhf_truthfulqa 
```

**Train on 50% subset A with dataset split to get a verification model:**

```bash
python data/finetune_TruthfulQA.py -model_path facebook/opt-2.7b -split True -subset A -ratio 0.5 -save_path opt2.7b_trufulqa_0.5_A
```


---

### 3. Unlearning SafeRLHF

```bash
python data/unlearn.py
```
Unlearns Du from model M and produces Mu.

#### 🔧 `unlearn.py` Arguments

| Argument           | Type    | Default                            | Description |
|--------------------|---------|------------------------------------|-------------|
| `-model_path`      | `str`   | `"opt2.7b_saferlhf_truthfulqa"`    | Path to the model to be unlearned (M). |
| `-use_GA`          | `bool`  | `False`                            | Enable Gradient Ascent unlearning. |
| `-use_mismatch`    | `bool`  | `False`                            | Enable Label Mismatch unlearning. |
| `-use_normal`      | `bool`  | `False`                            | Enable TruthfulQA-based Normal Loss. |
| `-epoch`           | `int`   | `1`                                | Number of unlearning epochs. |
| `-save_path`       | `str`   | `"opt2.7b_unlearn_GA_MIS_normal"`  | Path to save the unlearned model (Mu). |

#### 💡 Examples

**Unlearn with Gradient Ascent and Label Mismatch:**

```bash
python data/unlearn.py -use_GA True -use_mismatch True -save_path unlearned_model.pt
```


### 4. Evaluate Unlearning Effectiveness

```bash
python data/evaluate.py
```
Compare an unlearned model Mu with a verification model Mv on SafeRLHF data and TruthfulQA data.
#### 🔧 `evaluate.py` Arguments

| Argument      | Type   | Default                        | Description |
|---------------|--------|---------------------------------|-------------|
| `-Mv`         | `str`  | `"./opt2.7b_truthfulqa_0.5_A"`  | Path to the verification model trained only on Dr. |
| `-Mu`         | `str`  | `"./opt2.7b_unlearn_GA"`    | Path to the model after unlearning Du. |
| `-data_num`   | `int`  | `100`                           | Number of prompts from Dr used for evaluation. |

#### 💡 Example

```bash
python data/evaluate.py -Mu ./unlearned_model.pt -Mv ./verification_model.pt -data_num 100
```




