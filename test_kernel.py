"""Correctness + speed checks for the INT4 GEMM kernels in kernel.py.

Needs CUDA + Triton, so run it on the GPU box (-s to see the timing table):

    uv run --with pytest --with vllm pytest test_kernel.py -v -s

The correctness checks also settle whether pack_marlin_weights' row permutation
is actually inverted by the kernel: if it isn't, marlin output rows are
scrambled and the asserts fail (cosine ≈ 0); if they pass, the layout is fine.

Speed target (perf, higher = better):  dequant < FP16 < marlin < vLLM.
marlin > FP16 is xfail for now (not yet beating dense cuBLAS); the vLLM ceiling
check skips if vLLM's Marlin GEMM op isn't importable on this box.
"""
import functools
import time

import pytest
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
    return x, w, w_int4, scales, zeros


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
        x, _, w_int4, scales, zeros = make_case(M, N, K)
        ref = dequant_reference(x, w_int4, scales, zeros)

        w_m, s_m, z_m = kernel.pack_marlin_weights(w_int4, scales, zeros, GROUP)
        out = kernel.marlin_gemm(x, w_m, s_m, z_m, GROUP)

        cos = cosine(out, ref)
        assert cos > 0.999, f"{(M, N, K)}: marlin diverges from reference (cos={cos:.4f})"


def test_marlin_matches_vanilla():
    for M, N, K in SHAPES:
        x, _, w_int4, scales, zeros = make_case(M, N, K)
        vanilla = kernel.dequant_gemm(x, w_int4, scales, zeros, GROUP)

        w_m, s_m, z_m = kernel.pack_marlin_weights(w_int4, scales, zeros, GROUP)
        marlin = kernel.marlin_gemm(x, w_m, s_m, z_m, GROUP)

        cos = cosine(marlin, vanilla)
        assert cos > 0.999, f"{(M, N, K)}: marlin disagrees with vanilla (cos={cos:.4f})"


def bench(fn, iters=100):
    for _ in range(10):  # warmup: Triton autotune sweep + first-launch sync
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _vllm_marlin_callable(x, w):
    """No-arg callable running vLLM's Marlin W4A16 GEMM on (x, w), or None if vLLM
    (or a compatible kernel signature) isn't importable -> the ceiling test skips.

    Uses vLLM's own quantizer so the packed layout is guaranteed valid for its
    kernel; we compare kernel speed, not numerics. gptq_marlin_gemm's signature is
    version-sensitive -- if your installed vLLM differs, adjust the call below.
    """
    try:
        from vllm import _custom_ops as ops
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            MarlinWorkspace, marlin_quantize,
        )
    except Exception:
        return None

    M, K = x.shape
    N = w.shape[0]
    quant_type = scalar_types.uint4b8
    try:
        # vLLM's marlin_quantize wants the weight as (K, N)
        _, q_w, s, _, sort_indices, _ = marlin_quantize(w.t().contiguous(), quant_type, GROUP, act_order=False)
        workspace = MarlinWorkspace(N, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL)
        g_idx = torch.empty(0, dtype=torch.int32, device=x.device)
        b_zeros = torch.empty(0, dtype=torch.int32, device=x.device)

        def run():
            return ops.gptq_marlin_gemm(
                x, q_w, s, b_zeros, g_idx, sort_indices, workspace.scratch,
                quant_type, M, N, K, True, False, True, False,
            )

        run()  # trial call now so a signature mismatch -> skip, not a benched crash
        torch.cuda.synchronize()
        return run
    except Exception:
        return None


_HEADER_PRINTED = False
_COLS = ["dequant", "fp16", "marlin", "vllm"]


def _print_row(M, N, K, t):
    global _HEADER_PRINTED
    if not _HEADER_PRINTED:
        print("\n" + f"{'(M,N,K)':>18} " + " ".join(f"{c:>10}" for c in _COLS) + "   (ms)")
        _HEADER_PRINTED = True
    row = " ".join(f"{t[c] * 1e3:>10.3f}" if c in t else f"{'-':>10}" for c in _COLS)
    print(f"{str((M, N, K)):>18} {row}")


@functools.lru_cache(maxsize=None)
def _timings(M, N, K):
    # cached so the four kernels are benched once per shape and shared across tests
    x, w, w_int4, scales, zeros = make_case(M, N, K)
    w_m, s_m, z_m = kernel.pack_marlin_weights(w_int4, scales, zeros, GROUP)

    t = {
        "dequant": bench(lambda: kernel.dequant_gemm(x, w_int4, scales, zeros, GROUP)),
        "fp16": bench(lambda: torch.matmul(x, w.t())),
        "marlin": bench(lambda: kernel.marlin_gemm(x, w_m, s_m, z_m, GROUP)),
    }
    vfn = _vllm_marlin_callable(x, w)
    if vfn is not None:
        t["vllm"] = bench(vfn)

    _print_row(M, N, K, t)
    return t


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_marlin_faster_than_dequant(M, N, K):
    t = _timings(M, N, K)
    assert t["marlin"] < t["dequant"], \
        f"{(M, N, K)}: marlin {t['marlin'] * 1e3:.3f}ms not faster than dequant {t['dequant'] * 1e3:.3f}ms"


@pytest.mark.xfail(reason="marlin not yet faster than dense FP16 cuBLAS", strict=False)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_marlin_faster_than_fp16(M, N, K):
    t = _timings(M, N, K)
    assert t["marlin"] < t["fp16"], \
        f"{(M, N, K)}: marlin {t['marlin'] * 1e3:.3f}ms not faster than FP16 {t['fp16'] * 1e3:.3f}ms"


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_marlin_slower_than_vllm(M, N, K):
    t = _timings(M, N, K)
    if "vllm" not in t:
        pytest.skip("vLLM Marlin GEMM op unavailable on this box")
    assert t["marlin"] >= t["vllm"], \
        f"{(M, N, K)}: marlin {t['marlin'] * 1e3:.3f}ms unexpectedly beat vLLM {t['vllm'] * 1e3:.3f}ms"
