import argparse
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

# Reference ceiling (NOT our kernel): vLLM's AWQ-Marlin kernel + full serving stack.

parser = argparse.ArgumentParser()
parser.add_argument("--vllm_model", required=True, help="HF AWQ model id.")
parser.add_argument("--result_dir", default="results")
parser.add_argument("--decode_tokens", type=int, default=64)
parser.add_argument("--batch_sweep", default="1,4,16,32")
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)


def main():
    llm = LLM(model=args.vllm_model, quantization="awq_marlin", dtype="float16", gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=0, max_tokens=args.decode_tokens)

    out = {"vllm_model": args.vllm_model}
    for bs in [int(b) for b in args.batch_sweep.split(",")]:
        prompts = ["The quick brown fox jumps over the lazy dog."] * bs
        llm.generate(prompts, params)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        out[f"batch_{bs}"] = {"decode_tok_per_s": n_tokens / elapsed, "total_tokens": n_tokens}
        print(f"  [vllm/batch_{bs}] decode={n_tokens / elapsed:.1f} tok/s")

    out_path = os.path.join(args.result_dir, "vllm_reference.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
