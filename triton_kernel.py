import torch
import triton
import triton.language as tl

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

        X_ptr += BLOCK_K
        W_ptr += BLOCK_K // 2

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