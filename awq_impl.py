import torch
import torch.nn as nn
from tqdm.auto import trange, tqdm


def collect_activation_stats(model, calib_data, batch_size=2, n_samples_for_loss=512):
    # returns per-channel mean |x| (s_X in AWQ) and cached inputs for the alpha search
    sums, counts, inputs = {}, {}, {}
    hooks = []

    def make_hook(n):
        def hook(_, inp, __):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])  # (T, C_in), fp16
            abs_sum = x.float().abs().sum(0)
            if n in sums:
                sums[n] += abs_sum
                counts[n] += x.shape[0]
            else:
                sums[n] = abs_sum
                counts[n] = x.shape[0]
            have = inputs[n].shape[0] if n in inputs else 0
            if have < n_samples_for_loss:
                take = x[: n_samples_for_loss - have].cpu()
                inputs[n] = torch.cat([inputs[n], take], dim=0) if n in inputs else take
        return hook

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i in trange(0, len(calib_data), batch_size, desc="Collecting activations"):
            model(calib_data[i:i + batch_size].to(next(model.parameters()).device))

    for h in hooks:
        h.remove()

    act_scales = {n: sums[n] / counts[n] for n in sums}
    return act_scales, inputs


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


def search_best_scale(weight, act_scale, act_input, group_size=128, n_grid=20):
    # AWQ Eq.4: minimise || (X/s) @ Q(W*s)^T - X @ W^T ||^2 over s = s_X^alpha
    w = weight.float()
    s_x = act_scale.float().to(w.device).clamp(min=1e-4)
    X = act_input.float().to(w.device)
    Y_ref = X @ w.t()

    best_scale = torch.ones(w.shape[1], device=w.device)
    best_error = float("inf")
    for alpha in tqdm(torch.linspace(0, 1, n_grid), desc="Grid search"):
        s = s_x.pow(alpha.item())
        w_eff = pseudo_quantize(w * s.unsqueeze(0), group_size=group_size) / s.unsqueeze(0)
        Y_q = X @ w_eff.t()
        err = (Y_q - Y_ref).pow(2).mean().item()
        if err < best_error:
            best_error = err
            best_scale = s.clone()
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


def quantize_model(model, act_scales, act_inputs, group_size=128):
    quant_params = {}
    for name, module in tqdm(model.named_modules(), total=len(list(model.named_modules())), desc="AWQ quantization"):
        if not isinstance(module, nn.Linear) or name not in act_scales:
            continue
        if name in ("lm_head", "model.embed_tokens"):
            continue

        w = module.weight.data
        s = act_scales[name].to(w.device)
        X = act_inputs[name]
        best_scale = search_best_scale(w, s, X, group_size=group_size)
        print(name, "scale stats:", best_scale.min().item(), best_scale.max().item(), best_scale.mean().item())

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


def quantize_model_rtn(model, group_size=128):
    quant_params = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in ("lm_head", "model.embed_tokens"):
            continue
        w = module.weight.data
        w_int4, scales, zeros = quantize_and_pack(w.float(), group_size=group_size)
        quant_params[name] = {
            "weight_int4": w_int4,
            "scales": scales,
            "zeros": zeros,
            "awq_scale": torch.ones(w.shape[1], dtype=torch.float16, device=w.device),
            "bias": module.bias,
        }
        print(f"quantized {name}")
    return quant_params
