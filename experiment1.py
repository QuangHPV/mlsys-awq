import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import trange

import awq_impl as awq

parser = argparse.ArgumentParser()
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--model", help="HF model ID for FP16 baseline.")
src.add_argument("--weight_path", help="Path to saved quant checkpoint (.pt).")
parser.add_argument("--dataset", default="wikitext-2-raw-v1")
parser.add_argument("--result_dir", default="results")
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)

SEQ_LEN = 512
N_SAMPLES = 40


def compute_perplexity(model, tokenizer):
    dataset = load_dataset("wikitext", args.dataset, split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(next(model.parameters()).device)

    nlls = []
    for i in trange(0, N_SAMPLES * SEQ_LEN, SEQ_LEN, desc="Perplexity on Wikitext-2"):
        chunk = input_ids[:, i:i + SEQ_LEN]
        if chunk.shape[1] < SEQ_LEN:
            break
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        nlls.append(loss.item())

    return math.exp(sum(nlls) / len(nlls))


def append_result(result):
    path = os.path.join(args.result_dir, "exp1.json")
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(result)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


if args.model:
    model_id, checkpoint = args.model, None
else:
    checkpoint = torch.load(args.weight_path, map_location="cuda")
    model_id = checkpoint["model_id"]

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
if checkpoint is not None:
    model = awq.replace_with_awq_linear(model, checkpoint["quant_params"], group_size=checkpoint["group_size"])
model.eval()

torch.cuda.reset_peak_memory_stats()
ppl = compute_perplexity(model, tokenizer)
peak_vram = torch.cuda.max_memory_allocated() / 1e9

if checkpoint is None:
    entry = {"model": model_id, "perplexity": ppl, "peak_vram_gb": peak_vram}
    print(f"[{model_id}] perplexity={ppl:.2f}, peak_vram={peak_vram:.2f}GB")
else:
    entry = {"weight_path": args.weight_path, "perplexity": ppl, "peak_vram_gb": peak_vram}
    print(f"[{args.weight_path}] perplexity={ppl:.2f}, peak_vram={peak_vram:.2f}GB")
append_result(entry)
