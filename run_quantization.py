import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import awq


def run_awq(model_id, save_path, group_size):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")

    calib_data = awq.get_calib_data(tokenizer)
    act_scales = awq.collect_activation_scales(model, calib_data)
    quant_params = awq.quantize_model(model, act_scales, group_size=group_size)
    model = awq.replace_with_awq_linear(model, quant_params, group_size=group_size)

    print(f"Peak VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    cpu_params = {
        name: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in p.items()}
        for name, p in quant_params.items()
    }
    torch.save({"quant_params": cpu_params, "model_id": model_id, "group_size": group_size}, save_path)
    print(f"Saved to {save_path}")

    return model, tokenizer

# HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py --model "meta-llama/Meta-Llama-3.1-8B"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "Qwen/Qwen2.5-7B",
        "meta-llama/Meta-Llama-3.1-8B"
    ]) 
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--save_dir", default="checkpoint")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Running AWQ quantization on {args.model}")
    model_slug = args.model.replace("/", "_")
    run_awq(
        args.model,
        save_path=os.path.join(args.save_dir, f"awq_{model_slug}.pt"),
        group_size=args.group_size,
    )