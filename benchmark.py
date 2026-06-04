import argparse
import gc
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import trange
from vllm import LLM, SamplingParams

import awq_impl as awq

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--weight_path", required=True)
parser.add_argument("--vllm_model", required=True)
parser.add_argument("--result_dir", default="results")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--n_samples", type=int, default=40)
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)


def get_wikitext(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.n_samples * args.seq_len,
    ).input_ids


def compute_perplexity(model, input_ids):
    nlls = []
    for i in trange(0, args.n_samples * args.seq_len, args.seq_len, desc="PPL"):
        chunk = input_ids[:, i:i + args.seq_len].to(next(model.parameters()).device)
        if chunk.shape[1] < args.seq_len:
            break
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        nlls.append(loss.item())
    return math.exp(sum(nlls) / len(nlls))


def compute_throughput(model, tokenizer, prompt="The quick brown fox"):
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    ids, mask = enc.input_ids, enc.attention_mask
    with torch.no_grad():
        model.generate(ids, attention_mask=mask, max_new_tokens=16, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=mask, max_new_tokens=128, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (out.shape[1] - ids.shape[1]) / elapsed


def compute_perplexity_vllm(llm, input_ids):
    params = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=1)
    nlls = []
    for i in trange(0, args.n_samples * args.seq_len, args.seq_len, desc="PPL (vllm)"):
        chunk = input_ids[0, i:i + args.seq_len].tolist()
        if len(chunk) < args.seq_len:
            break
        outputs = llm.generate(prompt_token_ids=[chunk], sampling_params=params)
        logprobs = outputs[0].prompt_logprobs
        nll = -sum(
            list(lp.values())[0].logprob
            for lp in logprobs[1:]
            if lp is not None
        )
        nlls.append(nll / (args.seq_len - 1))
    return math.exp(sum(nlls) / len(nlls))


def compute_throughput_vllm(llm, prompt="The quick brown fox"):
    params = SamplingParams(temperature=0, max_tokens=128)
    llm.generate([prompt], params)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_tokens = sum(len(o.token_ids) for o in outputs[0].outputs)
    return n_tokens / elapsed


def benchmark(model, tokenizer, input_ids):
    torch.cuda.reset_peak_memory_stats()
    ppl = compute_perplexity(model, input_ids)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    tput = compute_throughput(model, tokenizer)
    return {"perplexity": ppl, "peak_vram_gb": peak_vram, "tokens_per_sec": tput}


def benchmark_vllm(llm, input_ids):
    compute_throughput_vllm(llm)
    torch.cuda.synchronize()
    peak_vram = torch.cuda.memory_allocated() / 1e9
    ppl = compute_perplexity_vllm(llm, input_ids)
    tput = compute_throughput_vllm(llm)
    return {"perplexity": ppl, "peak_vram_gb": peak_vram, "tokens_per_sec": tput}


results = {}
tokenizer = AutoTokenizer.from_pretrained(args.model)
input_ids = get_wikitext(tokenizer)

for kernel in ("awq", "marlin"):
    checkpoint = torch.load(args.weight_path, map_location="cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    if kernel == "marlin":
        marlin_params = awq.convert_quant_params_to_marlin(checkpoint["quant_params"])
        model = awq.replace_with_marlin_linear(model, marlin_params, group_size=checkpoint["group_size"])
    else:
        model = awq.replace_with_awq_linear(model, checkpoint["quant_params"], group_size=checkpoint["group_size"])
    model = model.to("cuda")
    model.eval()

    results[f"custom_{kernel}"] = benchmark(model, tokenizer, input_ids)
    r = results[f"custom_{kernel}"]
    print(f"[custom/{kernel}] ppl={r['perplexity']:.3f} vram={r['peak_vram_gb']:.2f}GB tput={r['tokens_per_sec']:.1f} tok/s")
    del model, checkpoint
    torch.cuda.empty_cache()
    gc.collect()

llm = LLM(
    model=args.vllm_model,
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.9,
)

results["vllm_awq"] = benchmark_vllm(llm, input_ids)
r = results["vllm_awq"]
print(f"[vllm/awq] ppl={r['perplexity']:.3f} vram={r['peak_vram_gb']:.2f}GB tput={r['tokens_per_sec']:.1f} tok/s")
del llm
torch.cuda.empty_cache()
gc.collect()

out_path = os.path.join(args.result_dir, "benchmark.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")