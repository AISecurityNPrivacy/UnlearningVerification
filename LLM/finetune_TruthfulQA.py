import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset, Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler, DataCollatorForLanguageModeling,
)
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split, Subset
from bert_score import score as bertscore_score
from peft import AdaLoraConfig, TaskType, get_peft_model

from utils import set_seed

set_seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_truthfulqa_subsets(tokenizer, ratio=0.1, batch_size=4, seed=42):
    """
    从 TruthfulQA 数据集中采样两个不重叠的子集，各占原数据集的 ratio 比例，最多为 0.5。
    支持 ratio=1.0（返回全部数据作为 subset A）。

    Returns:
        dataloader1, dataloader2：两个不重叠的 DataLoader。
    """
    assert 0.0 <= ratio <= 1.0, "Ratio must be in [0, 1.0]."
    if ratio == 0.0:
        raise ValueError("Ratio = 0.0 results in empty dataset.")

    df = pd.read_csv("data/TruthfulQA.csv")
    questions, answers = df["Question"].values, df["Best Answer"].values

    texts = [f"### Question: {q}\n ### Answer: {a}" for q, a in zip(questions, answers)]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        tok = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    dataset = dataset.map(tokenize, remove_columns=["text"])
    dataset.set_format(type="torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    total_len = len(dataset)

    if ratio == 1.0:
        # 返回完整数据集为 A，空集为 B
        dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        dataloader2 = DataLoader(Dataset.from_dict({}), batch_size=batch_size, collate_fn=collator)
        return dataloader1, dataloader2

    # 正常 0 < ratio <= 0.5 的情况
    subset_len = int(total_len * ratio)
    np.random.seed(seed)
    all_indices = np.random.permutation(total_len)
    idx1 = all_indices[:subset_len]
    idx2 = all_indices[subset_len:2 * subset_len]

    subset1 = Subset(dataset, idx1)
    subset2 = Subset(dataset, idx2)

    dataloader1 = DataLoader(subset1, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dataloader2 = DataLoader(subset2, batch_size=batch_size, shuffle=True, collate_fn=collator)

    return dataloader1, dataloader2

def finetune_on_truthfulqa(model_path, split=True, subset='A', ratio=0.1, num_epochs=9, save_path=None, device="cuda"):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-2.7b",
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
        device_map="auto"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    train_loader_A, train_loader_B = create_truthfulqa_subsets(tokenizer, ratio, 4, 42)
    if split:
        if subset == 'A':
            train_loader = train_loader_A
        else:
            train_loader = train_loader_B
    else:
        train_loader = train_loader_A
    # Wrap with LoRA
    lora_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
    base_model = get_peft_model(base_model, lora_config)

    max_steps = len(train_loader) * num_epochs

    optimizer = AdamW(base_model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps
    )

    base_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        base_model, optimizer, train_loader, lr_scheduler
    )

    base_model.train()
    step = 0

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = base_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step += 1
            if step % 100 == 0:
                accelerator.print(f"Step {step}: loss = {loss.item():.4f}")
            if step % 200 == 0:
                save_path = f"opt2.7b_truthfulqa_lora_step{step}"
                base_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                accelerator.print(f"✅ 模型已保存至 {save_path}")
        print(epoch)

    accelerator.wait_for_everyone()
    base_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    accelerator.print("✅ TruthfulQA 微调完成，模型已保存。")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default="facebook/opt-2.7b",
                        help='The path to the model to be unlearned. (default: "opt2.7b_saferlhf_truthfulqa")')
    parser.add_argument('-split', type=bool, store=True,
                        help='Whether to split the dataset into two subsets')
    parser.add_argument('-subset', type=str, choices=['A', 'B'], default='A',
                        help='The label of two subsets. (default: "A")')
    parser.add_argument('-ratio', type=float, default=0.5,
                        help='Data ratio for the two subsets. Subset A will contain whole dataset if split=False or ratio=1.0.')
    parser.add_argument('-epoch', type=int, default=2,
                        help='Total fine-tuning epochs')
    parser.add_argument('-save_path', type=str, default="opt2.7b_trufulqa_0.5_A",
                        help='The path to save the finetuned model')
    args = parser.parse_args()

    finetune_on_truthfulqa(args.model_path, args.split, args.subset, args.ratio, args.epoch, args.save_path)


