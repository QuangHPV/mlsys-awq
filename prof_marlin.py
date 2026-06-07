import torch
import torch.cuda.nvtx as nvtx

import awq_impl
import kernel

# mlp_gate_up at M=1: the decode shape that dominates exp4 (split_k=1, pure weight-stream)
N, K, G, M = 18944, 3584, 128, 1

torch.manual_seed(0)
w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.1
w_int4, scales, zeros = awq_impl.quantize_and_pack(w, group_size=G)
w_m, s_m, z_m = kernel.pack_marlin_weights(w_int4, scales, zeros, G)
x = torch.randn(M, K, device="cuda", dtype=torch.float16)

for _ in range(50):  # complete Triton autotune so the profiled launch is steady-state
    kernel.marlin_gemm(x, w_m, s_m, z_m, G)
torch.cuda.synchronize()

nvtx.range_push("profile")  # ncu --nvtx-include "profile/" profiles only what's in here
kernel.marlin_gemm(x, w_m, s_m, z_m, G)
torch.cuda.synchronize()
nvtx.range_pop()
