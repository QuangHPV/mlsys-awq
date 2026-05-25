import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import get_calib_data, collect_activation_scales, quantize_model
from awq_linear import replace_with_awq_linear


def run_awq(model_id, group_size=128, save_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")

    calib_data = get_calib_data(tokenizer)
    act_scales = collect_activation_scales(model, calib_data)
    quant_params = quantize_model(model, act_scales, group_size=group_size)
    model = replace_with_awq_linear(model, quant_params, group_size=group_size)

    print(f"peak VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if save_path:
        torch.save({"quant_params": quant_params, "model_id": model_id, "group_size": group_size}, save_path)
        print(f"saved to {save_path}")

    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    run_awq(args.model, group_size=args.group_size, save_path=args.save)