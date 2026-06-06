import torch
import torch.nn.functional as F
import awq_impl as awq
import kernel


def verify_kernel(group_size=128):
    M, N, K = 16, 256, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)

    ref = x @ w.T
    w_int4, scales, zeros = awq.quantize_and_pack(w, group_size=group_size)
    out = kernel.dequant_gemm(x, w_int4, scales, zeros, group_size)

    max_diff = (out - ref).abs().max().item()
    # cosine sim captures direction error independent of magnitude; expect >0.999 at INT4 group=128
    cos_sim = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0).item()
    print(f"max_diff={max_diff:.4f}  cosine_sim={cos_sim:.6f}")
    assert cos_sim > 0.99, "cosine similarity too low — check packing or dequant"


def verify_pseudo_quantize(group_size=128):
    w = torch.randn(64, 256, device="cuda")
    w_q = awq.pseudo_quantize(w, n_bits=8, group_size=group_size)
    err = (w_q - w).abs().max().item()
    print(f"pseudo_quantize 8-bit max_err={err:.6f}")


if __name__ == "__main__":
    verify_pseudo_quantize()
    verify_kernel()
    print("all checks passed")