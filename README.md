# UnlearningVerification

A toolkit for verifying machine unlearning effectiveness, including certified unlearning verification, backdoor attack robustness, membership inference evaluation, and t-SNE visualizations.

---

## Installation

```bash
git lfs install
git clone https://github.com/AISecurityNPrivacy/UnlearningVerification.git
cd UnlearningVerification
pip install -r requirements.txt
```

---

## Usage Overview

### Unlearning Verification

```bash
python verify.py
```

**Arguments:**

| Argument | Default  | Description                                                                                                                           |
|----------|----------|---------------------------------------------------------------------------------------------------------------------------------------|
| `-du_r`  | `0.2`    | The proportion of Du (data to be unlearned) in the full dataset. Choices: `0.1`, `0.2`, `0.3`. Fixed to `0.2` in non-basic scenarios. |
| `-data`  | `CIFAR10`| Dataset name. Choices: `CIFAR10`, `SVHN`, `SkinCancer`.                                                                               |
| `-dev`   | `cuda`   | Device for evaluation. Choices: `cuda`, `cpu`.                                                                                        |
| `-rp`    | `result` | Directory path to save results.                                                                                                       |
| `-mv_r`  | `adapt`  | Voting strategy. Choices: `adapt`, `all`.                                                                                             |
| `-scene` | `basic`  | Verification scenario. Choices: `basic`, `SimpleCNN`, `ResNet18`, `unbalance`.                                                        |


---

### Backdoor Verification

```bash
python backdoor_verify.py
```

**Arguments:**

| Argument | Default  | Description |
|----------|----------|-------------|
| `-rp`    | `result` | Directory path to save results. |
| `-data`  | `CIFAR10`| Dataset name. Choices: `CIFAR10`, `SVHN`, `SkinCancer`.   |
| `-dev`   | `cuda`   | Device for evaluation. Choices: `cuda`, `cpu`. |

---

### Membership Inference Attack (MIA)

```bash
python MIA_verify.py
```

**Arguments:**

| Argument | Default  | Description |
|----------|----------|-------------|
| `-rp`    | `result` | Directory path to save results. |
| `-data`  | `CIFAR10`| Dataset name. Choices: `CIFAR10`, `SVHN`, `SkinCancer`.   |
| `-dev`   | `cuda`   | Device for evaluation. Choices: `cuda`, `cpu`. |

---

###  t-SNE Visualization

The `t-SNE.py` script supports generating t-SNE visualizations of learned features.

```bash
python t-SNE.py
```

**Arguments (relevant to t-SNE):**

| Argument | Default  | Description |
|----------|----------|-------------|
| `-du_r`  | `0.2`    | Unlearned dataset ratio. |
| `-scene` | `basic`  | Scenario type. |
| `-dev`   | `cuda`   | Evaluation device. |
| `-rp`    | `result` | Path to save visualization results. |

