"""Correctness + speed checks for the INT4 GEMM kernels in kernel.py.

Needs CUDA + Triton, so run it on the GPU box:

    uv run --with pytest pytest test_kernel.py -v

The correctness checks also settle whether pack_marlin_weights' row permutation
is actually inverted by the kernel: if it isn't, marlin output rows are
scrambled and the asserts fail (cosine ≈ 0); if they pass, the layout is fine.
"""
import time

import torch

import awq_impl
import kernel

GROUP = 128
# (M, N, K) = (tokens, out, in): decode (M=1), small prefill, and real Llama-3.1-8B shapes
SHAPES = [
    (1, 4096, 4096),
    (16, 4096, 4096),
    (64, 4096, 4096),
    (16, 14336, 4096),
    (16, 4096, 14336),
]


def make_case(M, N, K, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.1
    w_int4, scales, zeros = awq_impl.quantize_and_pack(w, group_size=GROUP)
    return x, w_int4, scales, zeros


def dequant_reference(x, w_int4, scales, zeros):
    # independent fp32 reference; validates the marlin packing, not just the arithmetic
    N, K_half = w_int4.shape
    K = K_half * 2
    lo = (w_int4 & 0xF).to(torch.int16)
    hi = ((w_int4 >> 4) & 0xF).to(torch.int16)
    w_int = torch.stack([lo, hi], dim=-1).reshape(N, K).float()
    n_groups = K // GROUP
    w_int = w_int.reshape(N, n_groups, GROUP)
    w_dq = (w_int - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)
    return (x.float() @ w_dq.reshape(N, K).t()).half()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def test_marlin_matches_reference():
    for M, N, K in SHAPES:
        x, w_int4, scales, zeros = make_case(M, N, K)
        ref = dequant_reference(x, w_int4, scales, zeros)

        w_m, s_m, z_m = kernel.pack_marlin_weights(w_int4, scales, zeros, GROUP)
        out = kernel.marlin_gemm(x, w_m, s_m, z_m, GROUP)

        cos = cosine(out, ref)
        assert cos > 0.999, f"{(M, N, K)}: marlin diverges from reference (cos={cos:.4f})"


def test_marlin_matches_vanilla():
    for M, N, K in SHAPES:
        x, w_int4, scales, zeros = make_case(M, N, K)
        vanilla = kernel.dequant_gemm(x, w_int4, scales, zeros, GROUP)

        w_m, s_m, z_m = kernel.pack_marlin_weights(w_int4, scales, zeros, GROUP)
        marlin = kernel.marlin_gemm(x, w_m, s_m, z_m, GROUP)

        cos = cosine(marlin, vanilla)
        assert cos > 0.999, f"{(M, N, K)}: marlin disagrees with vanilla (cos={cos:.4f})"


def bench(fn, iters=50):
    for _ in range(5):  # warmup: autotune + first-launch sync
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def test_marlin_faster_than_vanilla():
    for M, N, K in [(16, 4096, 4096), (64, 14336, 4096)]:
        x, w_int4, scales, zeros = make_case(M, N, K)
        w_m, s_m, z_m = kernel.pack_marlin_weights(w_int4, scales, zeros, GROUP)

        t_vanilla = bench(lambda: kernel.dequant_gemm(x, w_int4, scales, zeros, GROUP))
        t_marlin = bench(lambda: kernel.marlin_gemm(x, w_m, s_m, z_m, GROUP))

        assert t_marlin < t_vanilla, \
            f"{(M, N, K)}: marlin {t_marlin * 1e3:.3f}ms not faster than vanilla {t_vanilla * 1e3:.3f}ms"
