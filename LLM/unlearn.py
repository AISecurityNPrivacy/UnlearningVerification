import argparse
import os
import random
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset, Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator
from torch.utils.data import DataLoader
from bert_score import score as bertscore_score
from peft import AdaLoraConfig, TaskType, get_peft_model

from utils import (
    get_answer_loss,
    get_rand_ans_loss,
    compute_kl,
    get_truthfulQA_answers_plaintext,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_saferlhf_unlearning_dataloader(tokenizer, dataset, batch_size=4, max_length=1000, fraction=1.0):
    def preprocess(examples):
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}
        for i in range(len(examples["prompt"])):
            if random.random() > fraction:
                continue
            prompt = examples["prompt"][i]
            for resp, safe in zip(
                [examples["response_0"][i], examples["response_1"][i]],
                [examples["is_response_0_safe"][i], examples["is_response_1_safe"][i]]
            ):
                if not safe:
                    full_text = f"### Question: {prompt}\n### Answer: {resp}"
                    prefix_text = f"### Question: {prompt}\n### Answer: "
                    tokenized = tokenizer(full_text, padding="max_length", truncation=True, max_length=max_length)
                    prefix_tok = tokenizer(prefix_text, padding="max_length", truncation=True, max_length=max_length)
                    results["input_ids"].append(tokenized["input_ids"])
                    results["attention_mask"].append(tokenized["attention_mask"])
                    results["start_locs"].append(len(prefix_tok["input_ids"]) - 1)
        return results

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_locs"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

def create_truthfulqa_normal_dataloader(tokenizer, path="data/TruthfulQA.csv", batch_size=4, max_length=512):
    df = pd.read_csv(path)
    questions = df["Question"].values
    answers = df["Best Answer"].values
    texts = [f"### Question: {q}\n### Answer: {a}" for q, a in zip(questions, answers)]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        tok = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    dataset = dataset.map(tokenize, remove_columns=["text"])
    dataset.set_format(type="torch")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def unlearn(model_path, use_GA=True, use_mismatch=True, use_normal=True, epoch=1, save_path=None):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
        device_map="auto"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
    base_model = get_peft_model(base_model, lora_config)

    saferlhf_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_bad_loader = create_saferlhf_unlearning_dataloader(tokenizer, saferlhf_dataset, batch_size=2)
    train_loader = create_truthfulqa_normal_dataloader(tokenizer, batch_size=4)

    optimizer = AdamW(base_model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_bad_loader)
    )

    base_model, optimizer, train_bad_loader, train_loader, lr_scheduler = accelerator.prepare(
        base_model, optimizer, train_bad_loader, train_loader, lr_scheduler
    )

    base_model.train()
    step = 0
    normal_ans = get_truthfulQA_answers_plaintext()
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
        device_map="auto"
    )
    pretrained_model.eval()

    for epoch in range(epoch):  # 遍历一个epoch
        for bad_batch, normal_batch in zip(train_bad_loader, train_loader):
            try:
                bad_loss, random_loss, normal_loss = 0.0, 0.0, 0.0
                if use_GA:
                    bad_loss = get_answer_loss("ga", bad_batch, base_model, device=accelerator.device)
                if use_mismatch:
                    random_loss = get_rand_ans_loss(
                    bad_batch,
                    tokenizer,
                    normal_ans,
                    base_model,
                    K=5,
                    device=accelerator.device,
                    )
                if use_normal:
                    normal_loss = compute_kl(pretrained_model, base_model, normal_batch, accelerator.device)
                loss = 0.5 * bad_loss + 1.0 * random_loss + 1.0 * normal_loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                step += 1
                accelerator.print(f"[Step {step}] Loss: {loss.item():.4f} | Bad: {-bad_loss:.2f} | KL: {normal_loss:.2f}")

                if step % 200 == 0:
                    save_path = f"opt2.7b_truthfulqa_unlearned_step{step}"
                    base_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    accelerator.print(f"✅ 模型已保存至 {save_path}")

                del bad_loss, normal_loss, random_loss, loss
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    accelerator.print("⚠️ CUDA内存不足，尝试释放缓存继续执行...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    accelerator.wait_for_everyone()
    base_model.save_pretrained(save_path)
    tokenizer.save_pretrained("opt2.7b_unlearn_GA")
    accelerator.print("✅ Unlearning 完成，模型已保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default="opt2.7b_saferlhf_truthfulqa",
                        help='The path to the model to be unlearned. (default: "opt2.7b_saferlhf_truthfulqa")')
    parser.add_argument('-use_GA', type=bool, store=True,
                        help='Whether to enable the Gradient Ascent')
    parser.add_argument('-use_mismatch', type=bool, store=True,
                        help='Whether to enable the Label Mismatch')
    parser.add_argument('-use_normal', type=bool, store=True,
                        help='Whether to enable the TruthfulQA-based Normal Loss')
    parser.add_argument('-epoch', type=int, default=1,
                        help='Total unlearning epochs')
    parser.add_argument('-save_path', type=str, default="opt2.7b_unlearn_GA_MIS_normal",
                        help='The path to save the unlearned model')
    args = parser.parse_args()

    unlearn(args.model_path, args.use_GA, args.use_mismatch, args.use_normal, args.epoch, args.save_path)
