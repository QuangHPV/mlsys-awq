import argparse
import json
import os
import time

import torch
from transformers import AutoConfig
from tqdm.auto import tqdm

import awq_impl
import kernel

parser = argparse.ArgumentParser()
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--model", help="HF model id (its config sets the layer shapes).")
src.add_argument("--weight_path", help="Quant checkpoint (.pt); its model_id sets the shapes.")
parser.add_argument("--result_dir", default="results")
parser.add_argument("--group_size", type=int, default=128)
parser.add_argument("--m_sweep", default="1,2,4,8,16,32,64,128,256,512,1024")
parser.add_argument("--plot", action="store_true", help="Emit roofline.png if matplotlib is available.")
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)
DEVICE = "cuda"
GROUP = args.group_size


def layer_shapes(config):
    # (N=out, K=in) of the four distinct linear shapes; varied N/K spreads intensity at fixed M
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", None) or hidden // n_heads
    inter = config.intermediate_size
    return {
        "attn_qo": (n_heads * head_dim, hidden),
        "attn_kv": (n_kv * head_dim, hidden),
        "mlp_gate_up": (inter, hidden),
        "mlp_down": (hidden, inter),
    }


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):  # also triggers Triton autotune on the first call
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_peak_compute():
    n = 8192  # big square FP16 matmul saturates the tensor cores
    a = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
    b = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
    t = bench(lambda: torch.matmul(a, b), iters=30)
    return (2 * n ** 3) / t / 1e12  # TFLOP/s


def measure_peak_bandwidth():
    x = torch.empty(256 * 1024 * 1024, device=DEVICE, dtype=torch.float16)  # 512 MB
    y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x), iters=30)
    return 2 * x.numel() * 2 / t / 1e9  # read + write = 2x bytes -> GB/s


def gemm_bytes(M, N, K, dtype):
    """Ideal minimum DRAM traffic for C(M,N) = A(M,K) @ W(N,K)^T."""
    act = M * K * 2
    out = M * N * 2
    if dtype == "fp16":
        w = N * K * 2
    else:  # INT4 weight + fp16 scales & zeros (one each per group)
        w = N * K * 0.5 + (N * K / GROUP) * 2 * 2
    return act + out + w


def gemm_roofline(shapes, m_values):
    results = []
    for shape_name, (N, K) in shapes.items():
        torch.manual_seed(0)
        w = torch.randn(N, K, device=DEVICE, dtype=torch.float16) * 0.1
        w_int4, scales, zeros = awq_impl.quantize_and_pack(w, group_size=GROUP)
        w_m, s_m, z_m = kernel.pack_triton_weights(w_int4, scales, zeros, GROUP)
        marlin_packed = kernel.pack_marlin_weights(w_int4, scales, zeros, GROUP)

        for M in tqdm(m_values, desc=f"GEMM roofline ({shape_name})", leave=False):
            x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
            kernels = {
                "fp16": (lambda: torch.matmul(x, w.t()), "fp16"),
                "vanilla": (lambda: kernel.dequant_gemm(x, w_int4, scales, zeros, GROUP), "int4"),
                "triton": (lambda: kernel.triton_gemm(x, w_m, s_m, z_m, GROUP), "int4"),
            }
            if marlin_packed is not None:
                B, s, z, ws = marlin_packed
                kernels["marlin"] = (lambda: kernel.marlin_gemm(x, B, s, z, ws), "int4")
            for kname, (fn, wdtype) in kernels.items():
                t = bench(fn)
                flops = 2 * M * N * K
                byts = gemm_bytes(M, N, K, wdtype)
                results.append({
                    "shape": shape_name, "N": N, "K": K, "M": M, "kernel": kname,
                    "latency_ms": t * 1e3,
                    "tflops": flops / t / 1e12,
                    "gbps": byts / t / 1e9,
                    "intensity": flops / byts,  # FLOP/byte -> roofline x-axis
                })
            del x
        del w, w_int4, scales, zeros, w_m, s_m, z_m
        torch.cuda.empty_cache()
    return results


def ridge_crossing(roofline, peak_tflops, peak_bw):
    # smallest M per (kernel, shape) that crosses the ridge into compute-bound
    ridge_intensity = peak_tflops * 1e12 / (peak_bw * 1e9)
    crossings = {}
    for r in roofline:
        key = f"{r['kernel']}/{r['shape']}"
        if r["intensity"] >= ridge_intensity and key not in crossings:
            crossings[key] = r["M"]
    return {"ridge_intensity_flop_per_byte": ridge_intensity, "first_compute_bound_M": crossings}


def plot_roofline(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak_tflops = results["hardware"]["peak_tflops_fp16"]
    peak_bw = results["hardware"]["peak_bw_gbps"]
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [2 ** i for i in range(-2, 12)]
    ax.plot(xs, [min(peak_tflops, peak_bw * x / 1e3) for x in xs], "k--", label="roofline")
    markers = {"fp16": "o", "vanilla": "s", "triton": "^"}
    for kname in markers:
        pts = [(r["intensity"], r["tflops"]) for r in results["gemm_roofline"]
               if r["kernel"] == kname and r["shape"] == "mlp_gate_up"]
        if pts:
            ax.scatter(*zip(*pts), marker=markers[kname], label=kname)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_title("W4A16 roofline (mlp_gate_up, M sweep)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved roofline plot to {path}")


def main():
    model_id = args.model or torch.load(args.weight_path, map_location="cpu", weights_only=True)["model_id"]
    config = AutoConfig.from_pretrained(model_id)
    shapes = layer_shapes(config)
    results = {"config": vars(args), "model_id": model_id, "layer_shapes": shapes}

    print("== hardware peaks ==")
    peak_tflops = measure_peak_compute()
    peak_bw = measure_peak_bandwidth()
    dev = torch.cuda.get_device_properties(0)
    results["hardware"] = {
        "device": dev.name, "sm_count": dev.multi_processor_count,
        "peak_tflops_fp16": peak_tflops, "peak_bw_gbps": peak_bw,
    }
    print(f"  {dev.name}: ~{peak_tflops:.0f} TFLOP/s fp16, ~{peak_bw:.0f} GB/s")

    print("\n== GEMM roofline sweep ==\n")
    m_values = [int(m) for m in args.m_sweep.split(",")]
    results["gemm_roofline"] = gemm_roofline(shapes, m_values)
    results["roofline_ridge"] = ridge_crossing(results["gemm_roofline"], peak_tflops, peak_bw)
    print(f"  ridge at {results['roofline_ridge']['ridge_intensity_flop_per_byte']:.1f} FLOP/byte; "
          f"compute-bound onset (M): {results['roofline_ridge']['first_compute_bound_M']}")

    out_path = os.path.join(args.result_dir, "exp4.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    if args.plot:
        try:
            plot_roofline(results, os.path.join(args.result_dir, "roofline.png"))
        except ImportError:
            print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
