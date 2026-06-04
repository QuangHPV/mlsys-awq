import torch
import torch.nn as nn
import triton
import triton.language as tl


def pack_marlin_weights(w_int4_awq, scales, zeros, group_size=128):
    N, K_half = w_int4_awq.shape
    K = K_half * 2
    n_groups = K // group_size

    BLOCK_N = 64

    lo = (w_int4_awq & 0xF).to(torch.uint8)
    hi = ((w_int4_awq >> 4) & 0xF).to(torch.uint8)
    w_int = torch.stack([lo, hi], dim=-1).reshape(N, K)

    perm = torch.tensor(
        [(r % 8) * (BLOCK_N // 8) + (r // 8) for r in range(BLOCK_N)],
        dtype=torch.long,
    )

    out_tiles = []
    for n_start in range(0, N, BLOCK_N):
        tile = w_int[n_start:n_start + BLOCK_N, :]
        out_tiles.append(tile[perm[:tile.shape[0]], :])
    w_reordered = torch.cat(out_tiles, dim=0)

    w_packed = ((w_reordered[:, 0::2] & 0xF) | ((w_reordered[:, 1::2] & 0xF) << 4)).contiguous()

    scales_t = scales.t().contiguous()
    zeros_t = zeros.t().contiguous()

    return w_packed, scales_t, zeros_t


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 256}, num_warps=8, num_stages=3),
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
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

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

    tl.store(
        O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


_WARMED_UP = set()


def marlin_gemm(x, w_int4, scales, zeros, group_size=128):
    assert x.dtype == torch.float16
    assert x.is_contiguous() and w_int4.is_contiguous()
    M, K = x.shape
    N = w_int4.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    def launch():
        _marlin_gemm_kernel[grid](
            x, w_int4, scales, zeros, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w_int4.stride(0), w_int4.stride(1),
            scales.stride(0), scales.stride(1),
            out.stride(0), out.stride(1),
            GROUP_SIZE=group_size,
        )

    key = (M, N, K)
    if key not in _WARMED_UP:
        launch()
        torch.cuda.synchronize()
        _WARMED_UP.add(key)

    launch()
    return out


class MarlinLinear(nn.Module):
    def __init__(self, w, s, z, awq_scale, perm, bias, group_size=128):
        super().__init__()
        self.register_buffer("w_int4", w)
        self.register_buffer("scales", s)
        self.register_buffer("zeros", z)
        self.register_buffer("awq_scale", awq_scale)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.group_size = group_size

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1]).half()
        x = x / self.awq_scale.unsqueeze(0)
        out = marlin_gemm(x, self.w_int4, self.scales, self.zeros, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], -1)