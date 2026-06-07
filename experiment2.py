import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

import kernel

parser = argparse.ArgumentParser()
parser.add_argument("--weight_path", required=True, help="Our INT4 quant checkpoint (.pt).")
parser.add_argument("--result_dir", default="results")
parser.add_argument("--prompt_len", type=int, default=512)
parser.add_argument("--decode_tokens", type=int, default=64)
parser.add_argument("--batch_sweep", default="1,4,16,32")
parser.add_argument("--skip_baseline", action="store_true", help="Skip the FP16 model (saves ~16GB).")
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)
DEVICE = "cuda"


def make_batch(tokenizer, batch_size, prompt_len):
    # tokens are arbitrary; only shape matters for latency
    tok = tokenizer("The quick brown fox jumps over the lazy dog. " * 200,
                    return_tensors="pt").input_ids[0, :prompt_len]
    return tok.unsqueeze(0).repeat(batch_size, 1).to(DEVICE)


def measure_prefill_decode(model, tokenizer, batch_size, prompt_len, decode_tokens):
    ids = make_batch(tokenizer, batch_size, prompt_len)
    gen_kwargs = dict(do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)

    with torch.no_grad():
        model(ids)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(ids)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # generate(1+D) minus generate(1) cancels the shared prefill, isolating D decode steps
    def gen(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            model.generate(ids, max_new_tokens=n, **gen_kwargs)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    gen(1)  # warmup
    t1 = gen(1)
    t2 = gen(1 + decode_tokens)
    per_step_s = (t2 - t1) / decode_tokens

    return {
        "batch_size": batch_size,
        "prefill_s": prefill_s,
        "prefill_tok_per_s": batch_size * prompt_len / prefill_s,
        "decode_s_per_step": per_step_s,
        "decode_tok_per_s": batch_size / per_step_s,
    }


def _top_ops(prof, k=12):
    rows = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
    return [{"name": e.key, "cuda_us": e.self_device_time_total, "count": e.count} for e in rows[:k]]


def profile_step(model, tokenizer, prompt_len):
    from torch.profiler import ProfilerActivity, profile

    ids = make_batch(tokenizer, 1, prompt_len)
    gen_kwargs = dict(do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    out = {}
    with torch.no_grad():
        model(ids)  # warmup / autotune
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(ids)
    out["prefill_top_ops"] = _top_ops(prof)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, **gen_kwargs)  # warmup
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model.generate(ids, max_new_tokens=8, **gen_kwargs)
    out["decode_top_ops"] = _top_ops(prof)
    return out


def main():
    results = {"config": vars(args), "device": torch.cuda.get_device_properties(0).name}

    print("\n== end-to-end prefill/decode (HF static batching) ==\n")
    batch_sweep = [int(b) for b in args.batch_sweep.split(",")]
    model_id = torch.load(args.weight_path, map_location="cpu", weights_only=True)["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configs = [("triton", "triton"), ("vanilla", "vanilla")]
    if not args.skip_baseline:
        configs.append(("fp16_baseline", None))

    e2e = {}
    for label, kname in configs:
        print(f"\tLoading {label} model...")
        if kname is None:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=DEVICE)
            model.eval()
        else:
            checkpoint = torch.load(args.weight_path, map_location="cpu", weights_only=True)
            model = kernel.load_quantized_model(checkpoint, kernel=kname, device=DEVICE)
        torch.cuda.reset_peak_memory_stats()
        e2e[label] = {"by_batch": [measure_prefill_decode(model, tokenizer, b, args.prompt_len, args.decode_tokens)
                                   for b in tqdm(batch_sweep, desc=f"Prefill/Decode ({label})")]}
        e2e[label]["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if kname == "triton":  # profile the kernel we care about, once, to its own file
            profile_path = os.path.join(args.result_dir, "exp2_profile.json")
            with open(profile_path, "w") as f:
                json.dump(profile_step(model, tokenizer, args.prompt_len), f, indent=2)
            print(f"  wrote profile to {profile_path}")
        for row in e2e[label]["by_batch"]:
            print(f"  [{label}] bs={row['batch_size']:>3} "
                  f"prefill={row['prefill_tok_per_s']:.0f} tok/s  decode={row['decode_tok_per_s']:.1f} tok/s")
        del model
        torch.cuda.empty_cache()
        gc.collect()
    results["e2e"] = e2e

    out_path = os.path.join(args.result_dir, "exp2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
