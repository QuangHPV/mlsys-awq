import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import awq_impl as awq
import gptq


def run_quantization(model_id, method, save_path, group_size):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")

    if method == "awq":
        calib_data = awq.get_calib_data(tokenizer)
        act_scales, act_inputs = awq.collect_activation_stats(model, calib_data)
        quant_params = awq.quantize_model(model, act_scales, act_inputs, group_size=group_size)
    elif method == "gptq":
        calib_data = awq.get_calib_data(tokenizer)
        quant_params = gptq.quantize_model_gptq(model, calib_data, group_size=group_size)
    elif method == "rtn":
        quant_params = awq.quantize_model_rtn(model, group_size=group_size)
    else:
        raise ValueError(f"unknown method: {method}")

    model = awq.replace_with_awq_linear(model, quant_params, group_size=group_size)
    print(f"Peak VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    cpu_params = {
        name: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in p.items()}
        for name, p in quant_params.items()
    }
    torch.save({"quant_params": cpu_params, "model_id": model_id, "group_size": group_size, "method": method}, save_path)
    print(f"Saved to {save_path}")

    return model, tokenizer


# HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py --model "Qwen/Qwen2.5-7B" --method gptq
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ])
    parser.add_argument("--method", required=True, choices=["awq", "gptq", "rtn"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--save_dir", default="./checkpoint")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Running {args.method.upper()} quantization on {args.model}")
    model_slug = args.model.replace("/", "_")
    run_quantization(
        args.model,
        method=args.method,
        save_path=os.path.join(args.save_dir, f"{args.method}_{model_slug}.pt"),
        group_size=args.group_size,
    )
