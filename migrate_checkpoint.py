import argparse
import torch
from transformers import AutoModelForCausalLM

import kernel

# Port an old quant_params-only checkpoint to the self-contained format via one CPU model load.

parser = argparse.ArgumentParser()
parser.add_argument("--old", required=True, help="Path to an old quant_params-only .pt")
parser.add_argument("--out", help="Output path (default: <old>.migrated.pt)")
args = parser.parse_args()

old = torch.load(args.old, map_location="cpu", weights_only=True)
quant_params = old["quant_params"]
model_id = old["model_id"]
group_size = old["group_size"]
method = old.get("method", "unknown")

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True,
)
model = kernel.replace_with_quant_linear(model, quant_params, group_size=group_size)

checkpoint = {
    "model_id": model_id,
    "group_size": group_size,
    "method": method,
    "quant_layers": list(quant_params.keys()),
    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
}

out = args.out or (args.old[:-3] if args.old.endswith(".pt") else args.old) + ".migrated.pt"
torch.save(checkpoint, out)
print(f"Migrated {args.old} -> {out}")
