import torch
import torch.nn as nn
from datasets import load_dataset


def get_calib_data(tokenizer, n_samples=128, seq_len=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    samples = []
    for item in dataset:
        enc = tokenizer(item["text"], return_tensors="pt", max_length=seq_len, truncation=True)
        if enc.input_ids.shape[1] == seq_len:
            samples.append(enc.input_ids)
        if len(samples) == n_samples:
            break
    return torch.cat(samples, dim=0)


def collect_activation_scales(model, calib_data, batch_size=4):
    # max over the calibration set: one large activation is enough to cause
    # clipping, so mean would underestimate the true channel magnitude
    act_scales = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        def make_hook(n):
            def hook(_, inp, __):
                x = inp[0].detach().float()
                channel_max = x.abs().view(-1, x.shape[-1]).max(0).values
                act_scales[n] = torch.maximum(act_scales[n], channel_max) if n in act_scales else channel_max
            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(calib_data), batch_size):
            model(calib_data[i:i + batch_size].to(next(model.parameters()).device))

    for h in hooks:
        h.remove()

    return act_scales


def pseudo_quantize(w, n_bits=4, group_size=128):
    # quantize then immediately dequantize to measure error in fp32 space
    C_out, C_in = w.shape
    w = w.reshape(-1, group_size)
    w_min = w.min(dim=1, keepdim=True).values
    w_max = w.max(dim=1, keepdim=True).values
    scale = (w_max - w_min) / (2 ** n_bits - 1)
    scale = scale.clamp(min=1e-8)
    zero = (-w_min / scale).round()
    w_int = ((w / scale) + zero).round().clamp(0, 2 ** n_bits - 1)
    return ((w_int - zero) * scale).reshape(C_out, C_in)


def search_best_scale(weight, act_scale, group_size=128, n_grid=20):
    # grid search over s = act_scale^alpha, alpha in [0, 1]
    w = weight.float()
    s = act_scale.float().to(w.device).clamp(min=1e-4)
    best_scale = torch.ones(w.shape[1], device=w.device)
    best_error = float("inf")

    for alpha in torch.linspace(0, 1, n_grid):
        candidate = s.pow(alpha.item())
        w_scaled = w * candidate.unsqueeze(0)
        w_q = pseudo_quantize(w_scaled, group_size=group_size) / candidate.unsqueeze(0)
        err = (w_q - w).pow(2).mean().item()
        if err < best_error:
            best_error = err
            best_scale = candidate.clone()

    return best_scale


def quantize_and_pack(w, n_bits=4, group_size=128):
    # asymmetric per-group: pack two INT4 values into one uint8 (low/high nibble)
    C_out, C_in = w.shape
    w = w.float().reshape(-1, group_size)
    w_min = w.min(dim=1, keepdim=True).values
    w_max = w.max(dim=1, keepdim=True).values
    scales = (w_max - w_min) / (2 ** n_bits - 1)
    scales = scales.clamp(min=1e-8)
    zeros = (-w_min / scales).round()
    w_int = ((w / scales) + zeros).round().clamp(0, 2 ** n_bits - 1).to(torch.uint8)
    w_packed = (w_int[:, 0::2] & 0xF) | ((w_int[:, 1::2] & 0xF) << 4)
    n_groups = C_in // group_size
    return (
        w_packed.reshape(C_out, C_in // 2),
        scales.reshape(C_out, n_groups).half(),
        zeros.reshape(C_out, n_groups).half(),
    )


def quantize_model(model, act_scales, group_size=128):
    quant_params = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in act_scales:
            continue

        w = module.weight.data
        s = act_scales[name].to(w.device)
        best_scale = search_best_scale(w, s, group_size)
        w_int4, scales, zeros = quantize_and_pack(w.float() * best_scale.unsqueeze(0), group_size=group_size)

        quant_params[name] = {
            "weight_int4": w_int4,
            "scales": scales,
            "zeros": zeros,
            "awq_scale": best_scale.half(),
            "bias": module.bias,
        }
        print(f"quantized {name}")

    return quant_params