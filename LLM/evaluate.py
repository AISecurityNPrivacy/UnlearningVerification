import argparse
import os
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    get_answer_loss,
    get_rand_ans_loss,
    compute_kl,
    get_truthfulQA_answers_plaintext, set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate_model_difference(model1, model2, tokenizer, dataset, batch_size=4, max_length=512, num_prompts=None):
    model1.eval()
    model2.eval()
    model1.to("cuda")
    model2.to("cuda")

    if isinstance(dataset, str) and dataset == "truthfulqa":
        df = pd.read_csv("data/TruthfulQA.csv")
        prompts = df["Question"].tolist()
    else:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
        prompts = []
        for ex in ds:
            if not ex["is_response_0_safe"] or not ex["is_response_1_safe"]:
                prompts.append(ex["prompt"])

    if num_prompts is not None and num_prompts < len(prompts):
        prompts = random.sample(prompts, num_prompts)

    texts = [f"### Question: {p}\n### Answer:" for p in prompts]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        return tokenizer(
            example["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    gen_texts_1 = []
    gen_texts_2 = []
    kl_values = []

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")

        with torch.no_grad():
            outputs1 = model1.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    do_sample=False)
            outputs2 = model2.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id, do_sample=False )

            logits1 = model1(input_ids=input_ids, attention_mask=attention_mask).logits
            logits2 = model2(input_ids=input_ids, attention_mask=attention_mask).logits

            for i in range(input_ids.size(0)):
                prompt_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
                generated1 = outputs1[i][prompt_len:]
                generated2 = outputs2[i][prompt_len:]

                decoded1 = tokenizer.decode(generated1, skip_special_tokens=True)
                decoded2 = tokenizer.decode(generated2, skip_special_tokens=True)

                gen_texts_1.append(decoded1)
                gen_texts_2.append(decoded2)

            # KL divergence (last token only)
            logits1_last = logits1[:, -1, :]
            logits2_last = logits2[:, -1, :]
            p = F.softmax(logits1_last, dim=-1)
            q = F.softmax(logits2_last, dim=-1)
            kl = F.kl_div(p.log(), q, reduction="batchmean")
            kl_values.append(kl.item())

    P, R, F1 = bertscore_score(gen_texts_1, gen_texts_2, lang="en", rescale_with_baseline=True)
    avg_kl = sum(kl_values) / len(kl_values)

    # 打印3个示例
    print("\n📌 Example comparisons between models:")
    for i in range(min(3, len(prompts))):
        print(f"\n[Prompt] {texts[i]}")
        print(f"[Model 1] {gen_texts_1[i]}")
        print(f"[Model 2] {gen_texts_2[i]}")

    print(f"avg_bertscore_f1 {F1.mean().item()}")
    print(f"avg_kl_divergence {avg_kl}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Mv', type=str, default="./opt2.7b_truthfulqa_0.5_A",
                        help='The path to the verification model')
    parser.add_argument('-Mu', type=str, default="./opt2.7b_unlearn_GA_only",
                        help='The path to the unlearned model to be verified')
    parser.add_argument('-data_num', type=int, default=100,
                        help='The number of prompts used for evaluation')
    args = parser.parse_args()

    model1 = AutoModelForCausalLM.from_pretrained(args.Mv).to("cuda")
    model2 = AutoModelForCausalLM.from_pretrained(args.Mu).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print(f'evaluate model difference:{args.Mv}, {args.M}')
    evaluate_model_difference(model1, model2, tokenizer, dataset="saferlhf", num_prompts=args.data_num)

    evaluate_model_difference(model1, model2, tokenizer, dataset="truthfulqa", num_prompts=args.data_num)






