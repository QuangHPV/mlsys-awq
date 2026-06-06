import torch
import torch.nn as nn
import triton
import triton.language as tl


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

    N, K_half = w_int4.shape
    K = K_half * 2

    # unpack two INT4 per byte: low nibble → even k, high nibble → odd k
    lo = (w_int4 & 0xF).to(torch.int16)
    hi = ((w_int4 >> 4) & 0xF).to(torch.int16)
    w_int = torch.stack([lo, hi], dim=-1).reshape(N, K).float()

    # per-group dequant: (w_int - zero) * scale, broadcast across each group
    n_groups = K // group_size
    w_int = w_int.reshape(N, n_groups, group_size)
    w_fp = (w_int - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)
    w_fp = w_fp.reshape(N, K)

    # fp32 accumulate to match what the Triton kernel was doing
    return (x.float() @ w_fp.t()).to(torch.float16)


def pack_marlin_weights(w_int4_awq, scales, zeros, group_size=128):
    # tl.dot handles MMA layout, so we skip the row permutation; only scales/zeros transpose
    N, K_half = w_int4_awq.shape
    K = K_half * 2

    lo = (w_int4_awq & 0xF).to(torch.uint8)
    hi = ((w_int4_awq >> 4) & 0xF).to(torch.uint8)
    w_int = torch.stack([lo, hi], dim=-1).reshape(N, K)

    w_packed = ((w_int[:, 0::2] & 0xF) | ((w_int[:, 1::2] & 0xF) << 4)).contiguous()

    scales_t = scales.t().contiguous()
    zeros_t = zeros.t().contiguous()

    return w_packed, scales_t, zeros_t


# Design notes and divergences from canonical Marlin: see README "Marlin-style kernel".

@triton.autotune(
    configs=[
        # small BLOCK_M: decode / small-batch regime (paired with split-K)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        # large BLOCK_M: prefill / large-M regime (big M-tiles maximise weight reuse)
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K", "GROUP_SIZE"],
)
@triton.jit
def _marlin_gemm_kernel(
    X_ptr, W_ptr, S_ptr, Z_ptr, O_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sg, stride_sn,
    stride_ok, stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)  # which K-slice this program reduces (0..SPLIT_K-1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # program pid_k handles K-blocks pid_k, pid_k+SPLIT_K, ... (SPLIT_K=1 => all)
    num_k = tl.cdiv(K, BLOCK_K)
    for ki in range(pid_k, num_k, SPLIT_K):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)

        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        w_col = offs_k // 2
        w_packed = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + w_col[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (w_col[None, :] < K // 2),
            other=0,
        )
        w_int = tl.where(offs_k[None, :] % 2 == 0, w_packed & 0xF, (w_packed >> 4) & 0xF).to(tl.float16)

        g = offs_k // GROUP_SIZE
        sg_mask = (offs_n[:, None] < N) & (g[None, :] < (K // GROUP_SIZE))

        scale = tl.load(
            S_ptr + g[None, :] * stride_sg + offs_n[:, None] * stride_sn,
            mask=sg_mask, other=1.0,
        ).to(tl.float16)
        zero = tl.load(
            Z_ptr + g[None, :] * stride_sg + offs_n[:, None] * stride_sn,
            mask=sg_mask, other=0.0,
        ).to(tl.float16)

        w_fp16 = (w_int - zero) * scale

        acc = tl.dot(x, tl.trans(w_fp16), acc=acc, allow_tf32=False)

    # each pid_k writes its own FP32 partial plane; marlin_gemm sums them
    tl.store(
        O_ptr + pid_k * stride_ok + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _choose_split_k(M, N, device, block_n=128, max_split=8):
    # split-K only helps at small M where the grid can't fill the SMs; cap 8 per arXiv:2402.00025
    if M > 64:
        return 1
    n_sms = torch.cuda.get_device_properties(device).multi_processor_count
    col_tiles = max(1, N // block_n)
    target = (2 * n_sms) // col_tiles  # aim for ~2 SM-waves of column tiles
    for p in (max_split, 4, 2):
        if target >= p:
            return p
    return 1


def marlin_gemm(x, w_int4, scales, zeros, group_size=128):
    assert x.dtype == torch.float16
    assert x.is_contiguous() and w_int4.is_contiguous()
    M, K = x.shape
    N = w_int4.shape[0]

    split_k = _choose_split_k(M, N, x.device)
    # one FP32 (M, N) partial plane per K-slice; each element written once, then summed
    partials = torch.empty((split_k, M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), split_k)

    _marlin_gemm_kernel[grid](
        x, w_int4, scales, zeros, partials,
        M, N, K,
        x.stride(0), x.stride(1),
        w_int4.stride(0), w_int4.stride(1),
        scales.stride(0), scales.stride(1),
        partials.stride(0), partials.stride(1), partials.stride(2),
        GROUP_SIZE=group_size, SPLIT_K=split_k,
    )
    return partials.sum(0).to(torch.float16)


def _get_parent(model, name):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


_GEMM = {"vanilla": dequant_gemm, "marlin": marlin_gemm}


class QuantLinear(nn.Module):
    """INT4 linear, dispatched to one of the GEMM kernels by `kernel`:

    - "vanilla": dequantize the weight then a plain fp16 matmul (dequant_gemm).
    - "marlin":  fused Marlin-style Triton GEMM (marlin_gemm).

    The only layout difference is scales/zeros: marlin stores them transposed
    ([n_groups, out]) so the kernel reads them with coalesced strides; vanilla
    keeps them [out, n_groups].
    """

    def __init__(self, w_int4, scales, zeros, awq_scale, bias, group_size=128, kernel="vanilla"):
        super().__init__()
        self.group_size = group_size
        self.kernel = kernel
        self.register_buffer("w_int4", w_int4)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        # dividing x by awq_scale inverts the per-channel scaling baked into weights
        self.register_buffer("awq_scale", awq_scale)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def empty(cls, in_features, out_features, bias, group_size=128, kernel="vanilla", device="meta"):
        # skeleton with correctly-shaped buffers, to be filled by load_state_dict(assign=True)
        n_groups = in_features // group_size
        sz_shape = (n_groups, out_features) if kernel == "marlin" else (out_features, n_groups)
        return cls(
            w_int4=torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            scales=torch.empty(*sz_shape, dtype=torch.float16, device=device),
            zeros=torch.empty(*sz_shape, dtype=torch.float16, device=device),
            awq_scale=torch.empty(in_features, dtype=torch.float16, device=device),
            bias=torch.empty(out_features, dtype=torch.float16, device=device) if bias else None,
            group_size=group_size,
            kernel=kernel,
        )

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1]).half()
        x = x / self.awq_scale.unsqueeze(0)
        out = _GEMM[self.kernel](x, self.w_int4, self.scales, self.zeros, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], -1)


def replace_with_quant_linear(model, quant_params, group_size=128):
    # builds the quantized model in vanilla layout; load_quantized_model repacks to marlin
    for name, params in quant_params.items():
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, QuantLinear(
            w_int4=params["weight_int4"],
            scales=params["scales"],
            zeros=params["zeros"],
            awq_scale=params["awq_scale"],
            bias=params["bias"],
            group_size=group_size,
        ))
    return model


def load_quantized_model(checkpoint, kernel="vanilla", device="cuda"):
    """Build the model from a self-contained checkpoint without ever materializing
    the original FP16 weights.

    The architecture skeleton is created on the meta device (zero memory), the
    quantized linears are swapped in as empty INT4 modules, then load_state_dict
    with assign=True materializes everything straight from the checkpoint —
    only INT4 weights + the small FP16 remainder (embeddings, norms, lm_head)
    ever touch memory.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    model_id = checkpoint["model_id"]
    group_size = checkpoint["group_size"]
    quant_layers = checkpoint["quant_layers"]
    state_dict = checkpoint["state_dict"]

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    if kernel == "marlin":
        # checkpoints store the vanilla INT4 layout; repack into the Marlin layout
        state_dict = dict(state_dict)
        for name in quant_layers:
            w, s, z = pack_marlin_weights(
                state_dict[f"{name}.w_int4"],
                state_dict[f"{name}.scales"],
                state_dict[f"{name}.zeros"],
                group_size,
            )
            state_dict[f"{name}.w_int4"] = w
            state_dict[f"{name}.scales"] = s
            state_dict[f"{name}.zeros"] = z

    for name in quant_layers:
        parent, attr = _get_parent(model, name)
        lin = getattr(parent, attr)
        setattr(parent, attr, QuantLinear.empty(
            lin.in_features, lin.out_features, lin.bias is not None, group_size, kernel=kernel,
        ))

    model.load_state_dict(state_dict, assign=True, strict=False)
    model.tie_weights()  # re-link lm_head to embeddings if tied (assign breaks the alias)
    model.eval()
    return model.to(device)
