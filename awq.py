import torch
import torch.nn as nn
import triton
import triton.language as tl
from datasets import load_dataset


# ── Triton dequant-GEMM kernel ────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 256}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _dequant_gemm_kernel(
    X_ptr, W_ptr, S_ptr, Z_ptr, O_ptr,
    M, N, K,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # which output tile this program owns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row indices into X (shape M×K) and W (shape N×K//2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # accumulate in fp32 to avoid precision loss from many fp16 additions
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + rk

        # load input tile X[rm, k], shape (BLOCK_M, BLOCK_K)
        x_mask = (rm[:, None] < M) & (k[None, :] < K)
        x = tl.load(X_ptr + rm[:, None] * K + k[None, :], mask=x_mask, other=0.0)

        # W is stored as uint8 with 2 INT4 values per byte, so column index is k//2
        w_col = k // 2
        w_mask = (rn[:, None] < N) & (w_col[None, :] < K // 2)
        w_packed = tl.load(W_ptr + rn[:, None] * (K // 2) + w_col[None, :], mask=w_mask, other=0)

        # unpack nibbles: even k → low 4 bits, odd k → high 4 bits
        w_int4 = tl.where(k[None, :] % 2 == 0, w_packed & 0xF, (w_packed >> 4) & 0xF).to(tl.float16)

        # each group of GROUP_SIZE columns shares one scale and zero point
        g = k // GROUP_SIZE
        sg_mask = (rn[:, None] < N) & (g[None, :] < K // GROUP_SIZE)
        scale = tl.load(S_ptr + rn[:, None] * (K // GROUP_SIZE) + g[None, :], mask=sg_mask, other=1.0)
        zero  = tl.load(Z_ptr + rn[:, None] * (K // GROUP_SIZE) + g[None, :], mask=sg_mask, other=0.0)

        w_fp16 = (w_int4 - zero) * scale

        # tl.dot computes (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N)
        # allow_tf32=False: TF32 drops mantissa bits 23→10, which compounds INT4 noise
        acc += tl.dot(x, tl.trans(w_fp16), allow_tf32=False)

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(O_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=out_mask)


def dequant_gemm(x, w_int4, scales, zeros, group_size=128):
    """
    x:       (M, K)        fp16
    w_int4:  (N, K//2)     uint8, two INT4 per byte
    scales:  (N, K//group) fp16
    zeros:   (N, K//group) fp16
    returns: (M, N)        fp16
    """
    assert x.dtype == torch.float16
    assert x.is_contiguous() and w_int4.is_contiguous()

    M, K = x.shape
    N = w_int4.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)

    # grid defines how many programs launch: one per output tile
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _dequant_gemm_kernel[grid](x, w_int4, scales, zeros, out, M, N, K, GROUP_SIZE=group_size)
    return out


# ── Calibration & quantization ────────────────────────────────────────────────

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

    def make_hook(n):
        def hook(_, inp, __):
            x = inp[0].detach().float()
            channel_max = x.abs().view(-1, x.shape[-1]).max(0).values
            act_scales[n] = torch.maximum(act_scales[n], channel_max) if n in act_scales else channel_max
        return hook

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
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
        if name in ("lm_head", "model.embed_tokens"):
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


# ── AWQ linear layer ──────────────────────────────────────────────────────────

class AWQLinear(nn.Module):
    def __init__(self, w_int4, scales, zeros, awq_scale, bias, group_size=128):
        super().__init__()
        self.group_size = group_size
        self.register_buffer("w_int4", w_int4)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        # dividing x by awq_scale inverts the per-channel scaling baked into weights
        self.register_buffer("awq_scale", awq_scale)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1]).half()
        x = x / self.awq_scale.unsqueeze(0)
        out = dequant_gemm(x, self.w_int4, self.scales, self.zeros, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], -1)


def replace_with_awq_linear(model, quant_params, group_size=128):
    for name, params in quant_params.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], AWQLinear(
            w_int4=params["weight_int4"],
            scales=params["scales"],
            zeros=params["zeros"],
            awq_scale=params["awq_scale"],
            bias=params["bias"],
            group_size=group_size,
        ))
    return model
