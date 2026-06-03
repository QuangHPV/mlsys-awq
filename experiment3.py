import argparse
import json
import os
import re
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

import awq_impl as awq

LETTERS = ["A", "B", "C", "D"]
RESULT_DIR = "results"
GSM8K_SHOTS = 5
MMLU_SHOTS = 5


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_model(args, device):
    if args.model:
        model_id, checkpoint = args.model, None
    else:
        checkpoint = torch.load(args.weight_path, map_location=device)
        model_id = checkpoint["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # so the final token aligns across a batch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map=device)
    if checkpoint is not None:
        model = awq.replace_with_awq_linear(model, checkpoint["quant_params"], group_size=checkpoint["group_size"])
    model.eval()
    return model, tokenizer


def evaluate_gsm8k(model, tokenizer, batch_size, rank, world_size):
    device = next(model.parameters()).device
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    test = test.select(range(rank, len(test), world_size))  # stride shard

    prefix = ""
    for ex in train.select(range(GSM8K_SHOTS)):
        reasoning, _, final = ex["answer"].partition("####")
        prefix += f"Question: {ex['question']}\nAnswer: {reasoning.strip()}\nThe answer is {final.strip()}.\n\n"

    def extract(text):
        m = re.findall(r"The answer is\s*([\-\d.,]+)", text)
        nums = m if m else re.findall(r"-?[\d.,]+", text)
        return nums[-1].rstrip(".").replace(",", "") if nums else None

    prompts = [prefix + f"Question: {ex['question']}\nAnswer:" for ex in test]
    golds = [ex["answer"].split("####")[-1].strip().replace(",", "") for ex in test]

    correct = 0
    idx = 0
    for batch in tqdm(list(chunked(prompts, batch_size)), desc="GSM8k", disable=rank != 0):
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for g in gen:
            if extract(g.split("Question:")[0]) == golds[idx]:
                correct += 1
            idx += 1
    return correct, len(prompts)


def evaluate_mmlu(model, tokenizer, batch_size, rank, world_size):
    device = next(model.parameters()).device
    dev = load_dataset("cais/mmlu", "all", split="dev")
    test = load_dataset("cais/mmlu", "all", split="test")
    test = test.select(range(rank, len(test), world_size))  # stride shard

    shots = defaultdict(list)
    for ex in dev:
        shots[ex["subject"]].append(ex)

    def format_q(ex, with_answer):
        s = ex["question"] + "\n"
        for letter, choice in zip(LETTERS, ex["choices"]):
            s += f"{letter}. {choice}\n"
        s += "Answer:"
        if with_answer:
            s += f" {LETTERS[ex['answer']]}\n\n"
        return s

    letter_ids = [tokenizer(f" {l}", add_special_tokens=False).input_ids[-1] for l in LETTERS]

    prompts, answers = [], []
    for ex in test:
        header = f"The following are multiple choice questions (with answers) about {ex['subject'].replace('_', ' ')}.\n\n"
        prefix = "".join(format_q(s, True) for s in shots[ex["subject"]][:MMLU_SHOTS])
        prompts.append(header + prefix + format_q(ex, False))
        answers.append(ex["answer"])

    correct = 0
    idx = 0
    for batch in tqdm(list(chunked(prompts, batch_size)), desc="MMLU", disable=rank != 0):
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]  # left-padded => -1 is the real last token
        choice_logits = logits[:, letter_ids]  # [B, 4]
        preds = choice_logits.argmax(dim=-1).tolist()
        for p in preds:
            if p == answers[idx]:
                correct += 1
            idx += 1
    return correct, len(prompts)


def worker(rank, world_size, args, q):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    model, tokenizer = load_model(args, device)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    torch.cuda.reset_peak_memory_stats(device)
    result = {}
    if "gsm8k" in tasks:
        result["gsm8k"] = evaluate_gsm8k(model, tokenizer, args.batch_size, rank, world_size)
    if "mmlu" in tasks:
        result["mmlu"] = evaluate_mmlu(model, tokenizer, args.batch_size, rank, world_size)
    result["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    q.put(result)


def append_result(result):
    path = os.path.join(RESULT_DIR, "exp3.json")
    existing = json.load(open(path)) if os.path.exists(path) else []
    existing.append(result)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="HF model ID for FP16/quantized baseline.")
    src.add_argument("--weight_path", help="Path to saved quant checkpoint (.pt).")
    parser.add_argument("--tasks", default="gsm8k,mmlu", help="Comma-separated: gsm8k,mmlu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=0, help="0 = all visible GPUs.")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    world_size = args.num_gpus or torch.cuda.device_count()

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(rank, world_size, args, q)) for rank in range(world_size)]
    for p in procs:
        p.start()
    shards = [q.get() for _ in range(world_size)]  # collect before join to avoid queue deadlock
    for p in procs:
        p.join()

    entry = {"model": args.model} if args.model else {"weight_path": args.weight_path}
    for task in ("gsm8k", "mmlu"):
        if all(task in s for s in shards):
            correct = sum(s[task][0] for s in shards)
            total = sum(s[task][1] for s in shards)
            entry[task] = correct / total if total else None
    entry["peak_vram_gb"] = max(s["peak_vram_gb"] for s in shards)

    print(json.dumps(entry, indent=2))
    append_result(entry)


if __name__ == "__main__":
    main()
