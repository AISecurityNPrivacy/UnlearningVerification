import os
import json
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
from torch.optim import AdamW

# === 禁用 tokenizer 并行警告 ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === 参数 ===
model_name = "facebook/opt-2.7b"
output_dir = "opt2.7b_saferlhf_lora_200"
json_save_path = "saferlhf_200.json"
num_samples = 200
max_length = 512
batch_size = 8
learning_rate = 1e-4
num_epochs = 2


def extract_and_save_unsafe_200():
    """从 PKU-SafeRLHF 提取 200 条 unsafe 样本，并保存为 JSON"""
    raw = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    selected = []
    for ex in raw:
        prompt = ex["prompt"]
        if not ex["is_response_0_safe"]:
            selected.append({"text": f"### Question: {prompt}\n### Answer: {ex['response_0']}"})
        if len(selected) >= num_samples:
            break
        if not ex["is_response_1_safe"]:
            selected.append({"text": f"### Question: {prompt}\n### Answer: {ex['response_1']}"})
        if len(selected) >= num_samples:
            break

    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存 {len(selected)} 条 unsafe 样本至 {json_save_path}")
    return selected


def load_finetune_dataloader(tokenizer, examples):
    """构造 dataloader"""
    dataset = Dataset.from_list(examples)

    def tokenize(examples):
        output = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        output["labels"] = output["input_ids"].copy()
        return output

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )


def main():
    accelerator = Accelerator()

    # === 模型与 tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_config)

    # === 数据准备 ===
    examples = extract_and_save_unsafe_200()
    dataloader = load_finetune_dataloader(tokenizer, examples)
    max_steps = len(dataloader) * num_epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # === 微调过程 ===
    model.train()
    step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            if step >= max_steps:
                break

            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                accelerator.print(f"Step {step}/{max_steps}: loss = {loss.item():.4f}")

    # === 模型保存 ===
    accelerator.wait_for_everyone()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    accelerator.print(f"✅ 模型已保存至 {output_dir}")


if __name__ == "__main__":
    main()
