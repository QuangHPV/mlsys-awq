"""Correctness + speed checks for the INT4 GEMM kernels in kernel.py.

Needs CUDA + Triton, so run it on the GPU box (-s to see the timing table):

    uv run --with pytest --with vllm pytest test_kernel.py -v -s

The correctness checks also settle whether pack_triton_weights' row permutation
is actually inverted by the kernel: if it isn't, triton output rows are
scrambled and the asserts fail (cosine ≈ 0); if they pass, the layout is fine.

Speed target (perf, higher = better):  dequant < FP16 < triton < marlin <= vLLM.
triton > FP16 is xfail (our split-K kernel doesn't beat dense cuBLAS); the vendored
canonical-Marlin CUDA kernel is expected to beat both FP16 and triton at low batch.
Marlin's correctness vs our AWQ reference is xfail until the zero-point port (it's
symmetric only today); the marlin/vLLM checks skip if those kernels aren't built.
"""
import functools
import time

import pytest
import torch
import torch.nn as nn

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
    # independent fp32 reference; validates the triton packing, not just the arithmetic
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


def test_triton_matches_reference():
    for M, N, K in SHAPES:
        x, _, w_int4, scales, zeros = make_case(M, N, K)
        ref = dequant_reference(x, w_int4, scales, zeros)

        w_m, s_m, z_m = kernel.pack_triton_weights(w_int4, scales, zeros, GROUP)
        out = kernel.triton_gemm(x, w_m, s_m, z_m, GROUP)

        cos = cosine(out, ref)
        assert cos > 0.999, f"{(M, N, K)}: triton diverges from reference (cos={cos:.4f})"


def test_triton_matches_vanilla():
    for M, N, K in SHAPES:
        x, _, w_int4, scales, zeros = make_case(M, N, K)
        vanilla = kernel.dequant_gemm(x, w_int4, scales, zeros, GROUP)

        w_m, s_m, z_m = kernel.pack_triton_weights(w_int4, scales, zeros, GROUP)
        triton = kernel.triton_gemm(x, w_m, s_m, z_m, GROUP)

        cos = cosine(triton, vanilla)
        assert cos > 0.999, f"{(M, N, K)}: triton disagrees with vanilla (cos={cos:.4f})"


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


def _marlin_callable(x, w):
    """No-arg callable running the vendored canonical Marlin GEMM on (x, w), or None
    if the `marlin_cuda` extension isn't built / the shape is unsupported.

    Packs w with a *symmetric* 4-bit grid (Marlin's only mode today), so numerics
    differ from our asymmetric AWQ pipeline — the correctness test is xfail. We pack
    the actual w (not random) so the speed comparison runs on a realistic layout.
    Needs `marlin`/`marlin_cuda` installed (built by `uv sync`).
    """
    try:
        import marlin
    except Exception:
        return None
    N, K = w.shape
    if N % 256 or K % 128:  # marlin.Layer requires outfeatures%256, infeatures%128
        return None

    maxq, gs = 15, GROUP
    wt = w.t().contiguous()  # (K, N)
    wg = wt.reshape((-1, gs, N)).permute(1, 0, 2).reshape((gs, -1))
    s = torch.max(torch.abs(wg), dim=0, keepdim=True)[0] * (2.0 / maxq)
    q = torch.clamp(torch.round(wg / s).int() + (maxq + 1) // 2, 0, maxq)
    ref = (q - (maxq + 1) // 2).half() * s
    ref = ref.reshape((gs, -1, N)).permute(1, 0, 2).reshape((K, N)).contiguous()
    s = s.reshape((-1, N)).contiguous()

    try:
        linear = nn.Linear(K, N, bias=False, dtype=torch.half, device=x.device)
        linear.weight.data = ref.t().contiguous()
        layer = marlin.Layer(K, N, gs).to(x.device)
        layer.pack(linear, s.t())
        run = lambda: kernel.marlin_gemm(x, layer.B, layer.s, layer.workspace)
        run()  # trial now so a build/shape error -> skip, not a benched crash
        torch.cuda.synchronize()
        return run
    except Exception:
        return None


_HEADER_PRINTED = False
_COLS = ["dequant", "fp16", "triton", "marlin", "vllm"]


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
    w_m, s_m, z_m = kernel.pack_triton_weights(w_int4, scales, zeros, GROUP)

    t = {
        "dequant": bench(lambda: kernel.dequant_gemm(x, w_int4, scales, zeros, GROUP)),
        "fp16": bench(lambda: torch.matmul(x, w.t())),
        "triton": bench(lambda: kernel.triton_gemm(x, w_m, s_m, z_m, GROUP)),
    }
    mfn = _marlin_callable(x, w)
    if mfn is not None:
        t["marlin"] = bench(mfn)
    vfn = _vllm_marlin_callable(x, w)
    if vfn is not None:
        t["vllm"] = bench(vfn)

    _print_row(M, N, K, t)
    return t


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_triton_faster_than_dequant(M, N, K):
    t = _timings(M, N, K)
    assert t["triton"] < t["dequant"], \
        f"{(M, N, K)}: triton {t['triton'] * 1e3:.3f}ms not faster than dequant {t['dequant'] * 1e3:.3f}ms"


@pytest.mark.xfail(reason="triton not yet faster than dense FP16 cuBLAS", strict=False)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_triton_faster_than_fp16(M, N, K):
    t = _timings(M, N, K)
    assert t["triton"] < t["fp16"], \
        f"{(M, N, K)}: triton {t['triton'] * 1e3:.3f}ms not faster than FP16 {t['fp16'] * 1e3:.3f}ms"


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_triton_slower_than_vllm(M, N, K):
    t = _timings(M, N, K)
    if "vllm" not in t:
        pytest.skip("vLLM Marlin GEMM op unavailable on this box")
    assert t["triton"] >= t["vllm"], \
        f"{(M, N, K)}: triton {t['triton'] * 1e3:.3f}ms unexpectedly beat vLLM {t['vllm'] * 1e3:.3f}ms"


# --- vendored canonical Marlin (CUDA): symmetric today, AWQ zero-point port pending ---

@pytest.mark.xfail(reason="marlin is symmetric; mismatches the asymmetric AWQ reference "
                          "until the zero-point port", strict=False)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_marlin_matches_reference(M, N, K):
    x, w, w_int4, scales, zeros = make_case(M, N, K)
    mfn = _marlin_callable(x, w)
    if mfn is None:
        pytest.skip("marlin_cuda extension not built")
    ref = dequant_reference(x, w_int4, scales, zeros)
    cos = cosine(mfn(), ref)
    assert cos > 0.999, f"{(M, N, K)}: marlin diverges from AWQ reference (cos={cos:.4f})"


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_marlin_faster_than_triton(M, N, K):
    t = _timings(M, N, K)
    if "marlin" not in t:
        pytest.skip("marlin_cuda extension not built")
    assert t["marlin"] < t["triton"], \
        f"{(M, N, K)}: marlin {t['marlin'] * 1e3:.3f}ms not faster than triton {t['triton'] * 1e3:.3f}ms"


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_marlin_faster_than_fp16(M, N, K):
    t = _timings(M, N, K)
    if "marlin" not in t:
        pytest.skip("marlin_cuda extension not built")
    assert t["marlin"] < t["fp16"], \
        f"{(M, N, K)}: marlin {t['marlin'] * 1e3:.3f}ms not faster than FP16 {t['fp16'] * 1e3:.3f}ms"
