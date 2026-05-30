import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import trange

import awq

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
parser.add_argument("--dataset", default="wikitext-2-raw-v1")
parser.add_argument("--result_dir", default="results")
parser.add_argument("--weight_path", default=None, help="Path to saved quant params (.pt).")
parser.add_argument("--baseline", action="store_true", help="Run FP16 baseline evaluation.")
args = parser.parse_args()

if not args.baseline and args.weight_path is None:
    parser.error("Provide --weight_path, --baseline, or both.")

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
    path = os.path.join(args.result_dir, "experiments.json")
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(result)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


tokenizer = AutoTokenizer.from_pretrained(args.model)

if args.baseline:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    ppl_fp16 = compute_perplexity(model, tokenizer)
    peak_vram_fp16 = torch.cuda.max_memory_allocated() / 1e9

    append_result({"model": args.model, "mode": "fp16", "perplexity": ppl_fp16, "peak_vram_gb": peak_vram_fp16})
    print(f"[fp16] perplexity={ppl_fp16:.2f}, peak_vram={peak_vram_fp16:.2f}GB")

    del model
    torch.cuda.empty_cache()

if args.weight_path is not None:
    checkpoint = torch.load(args.weight_path, map_location="cuda")
    quant_params = checkpoint["quant_params"]
    group_size = checkpoint["group_size"]
    method = checkpoint.get("method", "awq")  # backward-compat with old checkpoints
    ckpt_model_id = checkpoint.get("model_id", args.model)
    if ckpt_model_id != args.model:
        print(f"checkpoint is for {ckpt_model_id}, overriding --model={args.model}")

    model = AutoModelForCausalLM.from_pretrained(ckpt_model_id, torch_dtype=torch.float16, device_map="cuda")
    model = awq.replace_with_awq_linear(model, quant_params, group_size=group_size)
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    ppl_int4 = compute_perplexity(model, tokenizer)
    peak_vram_int4 = torch.cuda.max_memory_allocated() / 1e9

    mode_label = f"{method}_int4"
    append_result({"model": args.model, "mode": mode_label, "perplexity": ppl_int4, "peak_vram_gb": peak_vram_int4})
    print(f"[{mode_label}] perplexity={ppl_int4:.2f}, peak_vram={peak_vram_int4:.2f}GB")
