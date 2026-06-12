import argparse
import json
import os
import re
import time
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

import kernel

LETTERS = ["A", "B", "C", "D"]
RESULT_DIR = "results"
GSM8K_SHOTS = 5
MMLU_SHOTS = 5


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_model(args, rank):
    if args.model:
        model_id, checkpoint = args.model, None
    else:
        checkpoint = torch.load(args.weight_path, map_location="cpu", weights_only=True)
        model_id = checkpoint["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map={"": rank}
        )
        model.eval()
    else:
        model = kernel.load_quantized_model(checkpoint, kernel="marlin", device=f"cuda:{rank}")
    return model, tokenizer


def evaluate_gsm8k(model, tokenizer, batch_size, rank=0, n_gpus=1):
    device = next(model.parameters()).device
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

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

    prompts = prompts[rank::n_gpus]
    golds = golds[rank::n_gpus]

    correct = 0
    idx = 0
    gpu_time, n_batches = 0.0, 0
    for batch in tqdm(list(chunked(prompts, batch_size)), desc=f"GSM8k[{rank}]", position=rank):
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(device)
        gpu_time += time.perf_counter() - t0
        n_batches += 1
        gen = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for g in gen:
            if extract(g.split("Question:")[0]) == golds[idx]:
                correct += 1
            idx += 1
    return correct, len(prompts), gpu_time / n_batches


def evaluate_mmlu(model, tokenizer, batch_size, max_seq_len=2048, rank=0, n_gpus=1):
    device = next(model.parameters()).device
    dev = load_dataset("cais/mmlu", "all", split="dev")
    test = load_dataset("cais/mmlu", "all", split="test")

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

    encoded = tokenizer(prompts, add_special_tokens=True, truncation=True, max_length=max_seq_len)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    sorted_prompts = [prompts[i] for i in order]
    sorted_answers = [answers[i] for i in order]

    sorted_prompts = sorted_prompts[rank::n_gpus]
    sorted_answers = sorted_answers[rank::n_gpus]

    correct = 0
    idx = 0
    gpu_time, n_batches = 0.0, 0
    for batch in tqdm(list(chunked(sorted_prompts, batch_size)), desc=f"MMLU[{rank}]", position=rank):
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
                logits = out.logits[:, -1, :]
            except TypeError:
                out = model.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state[:, -1:, :]
                logits = model.lm_head(hidden).squeeze(1)
        torch.cuda.synchronize(device)
        gpu_time += time.perf_counter() - t0
        n_batches += 1
        preds = logits[:, letter_ids].argmax(dim=-1).tolist()
        for p in preds:
            if p == sorted_answers[idx]:
                correct += 1
            idx += 1
    return correct, len(sorted_prompts), gpu_time / n_batches


def _save_result(entry):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "exp3.json")
    try:
        with open(path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    existing.append(entry)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(json.dumps(entry, indent=2))


def run_worker(rank, n_gpus, args, tasks, queue):
    model, tokenizer = load_model(args, rank)
    torch.cuda.reset_peak_memory_stats()

    results = {}
    for task in tasks:
        try:
            if task == "gsm8k":
                correct, total, latency = evaluate_gsm8k(model, tokenizer, args.batch_size, rank, n_gpus)
            else:
                correct, total, latency = evaluate_mmlu(model, tokenizer, args.batch_size, rank=rank, n_gpus=n_gpus)
            results[task] = (correct, total, latency, None)
        except Exception as e:
            results[task] = (0, 0, 0.0, str(e))

    queue.put((rank, results, torch.cuda.max_memory_allocated() / 1e9))


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model")
    src.add_argument("--weight_path")
    parser.add_argument("--tasks", default="gsm8k,mmlu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    base = {"model": args.model} if args.model else {"weight_path": args.weight_path}

    n_gpus = torch.cuda.device_count()

    if n_gpus <= 1:
        model, tokenizer = load_model(args, 0)
        torch.cuda.reset_peak_memory_stats()
        for task in tasks:
            print(f"\n=== {task.upper()} ===")
            try:
                if task == "gsm8k":
                    correct, total, latency = evaluate_gsm8k(model, tokenizer, args.batch_size)
                else:
                    correct, total, latency = evaluate_mmlu(model, tokenizer, args.batch_size)
            except Exception as e:
                print(f"{task} failed: {e}")
                _save_result({**base, "task": task, "error": str(e)})
                continue
            _save_result({
                **base,
                task: correct / total if total else None,
                f"{task}_correct": correct,
                f"{task}_total": total,
                f"{task}_latency_s": latency,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            })
        return

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=run_worker, args=(rank, n_gpus, args, tasks, queue))
             for rank in range(n_gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    all_results = {}
    peak_vrams = []
    while not queue.empty():
        _, results, peak_vram = queue.get()
        peak_vrams.append(peak_vram)
        for task, val in results.items():
            all_results.setdefault(task, []).append(val)

    for task in tasks:
        print(f"\n=== {task.upper()} ===")
        worker_results = all_results.get(task, [])
        if any(err for _, _, _, err in worker_results):
            errs = [err for _, _, _, err in worker_results if err]
            _save_result({**base, "task": task, "error": errs[0]})
            continue
        total_correct = sum(c for c, _, _, _ in worker_results)
        total_total = sum(t for _, t, _, _ in worker_results)
        avg_latency = sum(l for _, _, l, _ in worker_results) / len(worker_results)
        _save_result({
            **base,
            task: total_correct / total_total if total_total else None,
            f"{task}_correct": total_correct,
            f"{task}_total": total_total,
            f"{task}_latency_s": avg_latency,
            "peak_vram_gb": max(peak_vrams),
        })


if __name__ == "__main__":
    main()
