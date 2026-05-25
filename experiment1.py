import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import awq_linear

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
parser.add_argument("--dataset", default="wikitext-2-raw-v1")
parser.add_argument("--result_dir", default="results")
parser.add_argument("--weight_path", required=True, help="Path to saved AWQ quant params (.pt).")
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)

SEQ_LEN = 512
N_SAMPLES = 40


def compute_perplexity(model, tokenizer):
    dataset = load_dataset("wikitext", args.dataset, split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(next(model.parameters()).device)

    nlls = []
    for i in range(0, N_SAMPLES * SEQ_LEN, SEQ_LEN):
        chunk = input_ids[:, i:i + SEQ_LEN]
        if chunk.shape[1] < SEQ_LEN:
            break
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        nlls.append(loss.item())

    return math.exp(sum(nlls) / len(nlls))


def append_result(result):
    path = os.path.join(args.result_dir, "experiments.json")
    existing = json.load(open(path)) if os.path.exists(path) else []
    existing.append(result)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


tokenizer = AutoTokenizer.from_pretrained(args.model)

# FP16 baseline
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
model.eval()

torch.cuda.reset_peak_memory_stats()
ppl_fp16 = compute_perplexity(model, tokenizer)
peak_vram_fp16 = torch.cuda.max_memory_allocated() / 1e9

append_result({"model": args.model, "mode": "fp16", "perplexity": ppl_fp16, "peak_vram_gb": peak_vram_fp16})
print(f"[fp16] perplexity={ppl_fp16:.2f}, peak_vram={peak_vram_fp16:.2f}GB")

del model
torch.cuda.empty_cache()

# AWQ INT4
checkpoint = torch.load(args.weight_path, map_location="cuda")
quant_params = checkpoint["quant_params"]
group_size = checkpoint["group_size"]

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
model = awq_linear.replace_with_awq_linear(model, quant_params, group_size=group_size)
model.eval()

torch.cuda.reset_peak_memory_stats()
ppl_awq = compute_perplexity(model, tokenizer)
peak_vram_awq = torch.cuda.max_memory_allocated() / 1e9

append_result({"model": args.model, "mode": "awq_int4", "perplexity": ppl_awq, "peak_vram_gb": peak_vram_awq})
print(f"[awq_int4] perplexity={ppl_awq:.2f}, peak_vram={peak_vram_awq:.2f}GB")
