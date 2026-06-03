"""Standalone GPTQ via the `gptqmodel` library, to sanity-check our hand-rolled
gptq.py against a reference implementation.

Stage 1: quantize a model with gptqmodel and save it (skip with --weight_path).
Stage 2: load the quantized model and measure wikitext-2 perplexity (mirrors
experiment1.py). Results append to results/gptq.json.

The quantized output lands in ../checkpoint (parent dir, outside the repo).

  uv run test_gptq.py                                             # quantize Llama-3.1-8B-Instruct, then eval
  uv run test_gptq.py --weight_path ../checkpoint/gptqmodel_...   # eval only
"""
import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import trange

from gptqmodel import GPTQModel, QuantizeConfig

SEQ_LEN = 512
N_SAMPLES = 40


def get_calib_data(tokenizer, n_samples=128, seq_len=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids[0]
    samples = []
    for i in range(n_samples):
        chunk = ids[i * seq_len:(i + 1) * seq_len]
        if len(chunk) < seq_len:
            break
        samples.append(tokenizer.decode(chunk))
    return samples


def quantize(model_id, save_path, group_size):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # desc_act=False to match gptq.py, which does plain GPTQ with no activation reordering.
    quant_config = QuantizeConfig(bits=4, group_size=group_size, desc_act=False)
    model = GPTQModel.load(model_id, quant_config)
    model.quantize(get_calib_data(tokenizer), batch_size=2)
    model.save(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")
    del model
    torch.cuda.empty_cache()


def compute_perplexity(model, tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
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
    path = os.path.join("results", "gptq.json")
    existing = json.load(open(path)) if os.path.exists(path) else []
    existing.append(result)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model for stage 1.")
    parser.add_argument("--weight_path", help="Skip stage 1 and eval this saved gptqmodel dir.")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--save_dir", default="../checkpoint")
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    if args.weight_path:
        weight_path = args.weight_path
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        weight_path = os.path.join(args.save_dir, f"gptqmodel_{args.model.replace('/', '_')}")
        quantize(args.model, weight_path, args.group_size)

    tokenizer = AutoTokenizer.from_pretrained(weight_path)
    model = AutoModelForCausalLM.from_pretrained(weight_path, device_map="cuda")
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    ppl = compute_perplexity(model, tokenizer)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    entry = {"weight_path": weight_path, "perplexity": ppl, "peak_vram_gb": peak_vram}
    print(f"[{weight_path}] perplexity={ppl:.2f}, peak_vram={peak_vram:.2f}GB")
    append_result(entry)
