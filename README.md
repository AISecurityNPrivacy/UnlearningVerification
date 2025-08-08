# UnlearningVerification

This repository contains source code of our new work on verifying the effectiveness of machine unlearning techniques. All experiments mentioned in the paper are covered to ensure reproduction. 

It supports three task types:

- 🖼️ **Image Classification**
- 📝 **Text Classification**
- 🤖 **LLM-based QA**

It includes experiments for:

- Multiple unlearned models
- Our proposed verification
- Multiple typical verifications for comparison (e.g., backdoor, MIA)
- Feature Visualization (e.g., t-SNE)

---

## 🚀 Installation

Please ensure [Git LFS](https://git-lfs.com/) is installed:

```bash
git lfs install
```

Then clone the repository and install dependencies:

```bash
git clone https://github.com/AISecurityNPrivacy/UnlearningVerification.git
cd UnlearningVerification
pip install -r requirements.txt
```

---

## 📁 Project Structure & Usage

This repository is organized into three submodules based on task type. Each folder contains its own `README.md` with specific instructions.

### 1. 🖼️ Image Classification  
**Directory**: `./Image/`  
Includes scripts for:
- Fine-tuning & unlearning image models
- Certified verification protocols
- Backdoor attack testing
- Membership inference evaluation
- Feature visualization (e.g., t-SNE)

👉 See [Image/README.md](./Image/README.md) for full details.

---

### 2. 📝 Text Classification  
**Directory**: `./Text/`  
Includes scripts for:
- Text model unlearning
- Membership inference assessment
- Verification metrics

👉 See [Text/README.md](./Text/README.md) for usage.

---

### 3. 🤖 LLM QA (Large Language Model - Question Answering)  
**Directory**: `./LLM/`  
Includes:
- Fine-tuning and unlearning LLMs
- Verification model construction (Mv)
- Behavioral comparison between Unlearned Model (Mu) and Verification Model (Mv)

👉 See [LLM/README.md](./LLM/README.md) for step-by-step guidance.

---
