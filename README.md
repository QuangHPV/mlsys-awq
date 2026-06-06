# MLSys Project: Activation-aware Weight Quantization

## AWQ Implementation

## Experiment 1 - Quantization perplexity
**Why ~2-2.5x VRAM reduction instead of the theoretical 4x**

INT4 compresses only the weights, but VRAM holds more than weights:

- Embedding table (embed_tokens) and lm_head — skipped from quantization, stay FP16
- Scales + zeros for each group (FP16 overhead on top of INT4 weights)
- Activations during the forward pass — always FP16 regardless
- The actual memory breakdown: quantized weight bytes = N_params × 0.5 bytes instead of × 2 bytes, but the unquantized components form a fixed FP16 floor.

**Why Llama got more reduction than Qwen**

The key difference is embedding weight tying:

- Llama 3.1 uses tied embeddings — lm_head.weight and embed_tokens.weight are the same tensor in memory. One copy, ~1.05 GB unquantized.
- Qwen 2.5 does NOT tie embeddings — they're separate tensors. Both stay FP16 and we skip both from quantization. That's 2 × 151,936 × 3,584 × 2 bytes ≈ 2.18 GB unquantized.
- So Qwen carries ~1.1 GB more uncompressed FP16 weight that INT4 never touches, which raises its floor and explains the smaller reduction ratio.

**Is the perplexity increase normal?**

Yes. The degradation:

- Llama: 7.92 → 8.77 (+10.7%)
- Qwen: 7.77 → 8.38 (+7.9%)

This is expected for INT4. The AWQ paper reports ~5-8% degradation with their full pipeline — your numbers are slightly higher mainly because the grid search is coarse (n_grid=20) and calibration set is small (128 samples). The direction is correct and the models are usable.

## RTN implementation

RTN is the same packing/storage path as AWQ (uint8 INT4, per-group scales + zeros, group_size=128) with `awq_scale = 1`. So peak VRAM is identical — the only difference is whether weights get scaled before quantization.

**First AWQ run came out tied with RTN.** Llama 8.78 (RTN) vs 8.77 (AWQ), basically a tie. The bug was in the alpha search: the loss was weight reconstruction MSE, with no calibration activations involved. Without activations weighting the loss, the search has no reason to prefer scaling salient channels, and `alpha ≈ 0` (i.e. `scale ≈ 1`) usually wins. Effectively RTN.

**Fix matched the AWQ paper.** Three changes: switch `act_scales` from per-channel max to mean `|x|` (paper's `s_X`), cache a sample of per-layer inputs `X` during the calibration pass, and change the loss to `||(X/s) @ Q(W·diag(s))^T − X @ W^T||²`. After this, Llama AWQ → 8.34.

**Geometric-mean normalization didn't help.** The reference `llm-awq` divides scales by `sqrt(max·min)` so they straddle 1. Tried it; perplexity didn't move. Reverted.

**Result.**

| Method | Llama-3.1-8B ppl | Peak VRAM |
| --- | --- | --- |
| FP16 | 7.92 | 16.67 GB |
| RTN INT4 | 8.78 | 6.42 GB |
| AWQ INT4 | 8.34 | 6.42 GB |

AWQ closes ~0.4 of the 0.85 gap to FP16. Not the full paper number, but the rest of the gap needs invasive changes (module-grouped scales fused into the preceding LayerNorm, per-group weight clipping). At INT4 there's a floor — this is in range.

## Instruct vs base

On base Qwen 2.5 7B, our AWQ run blew up to ~5000 PPL. RTN on the same base model did the same. Spent a lot of time on it. Switched to the Instruct variants and the same code ran cleanly: our AWQ on Qwen 2.5 7B Instruct landed essentially on top of HF's official AWQ-quantized checkpoint for that model. Llama 3.1 8B Instruct also behaved as expected.

Hypothesis (unconfirmed): instruction fine-tuning regularizes the weight distribution — fewer extreme outlier channels, fewer near-constant groups — and that's what INT4 group quantization is sensitive to. Base models, especially Qwen 2.5 which uses heavy pretraining-only mixtures, may have weight groups our `quantize_and_pack` handles badly (e.g. groups where `scale.clamp(min=1e-8)` triggers and the zero-point goes to fp16 inf). Worth checking later; not on the critical path.

**Decision: use Instruct variants for all subsequent quantization experiments.** The base-model failure is a real bug in our path, but the production AWQ packages handle it (HF's checkpoint exists for base too); chasing it would burn the rest of the timeline.

*Another observation*: though we don't know the calibration dataset of AWQ Qwen 2.5 on HF, it still did worse than RTN without any calibration, so it validates our implementation, but also makes the improvement from AWQ feel marginal

## Marlin-style kernel

The fused INT4 GEMM in `kernel.py` (`_marlin_gemm_kernel` + `marlin_gemm`) borrows ideas from two references and deliberately diverges from both where Triton changes the tradeoffs.

**References**
- Marlin (CUDA): Frantar, Castro, Chen, Hoefler, Alistarh, *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs*, [arXiv:2408.11743](https://arxiv.org/abs/2408.11743); code [github.com/IST-DASLab/marlin](https://github.com/IST-DASLab/marlin). Near-ideal 3.87× up to batch 16–32.
- W4A16 SplitK (Triton): Hoque, Wright et al., *Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK work decomposition*, [arXiv:2402.00025](https://arxiv.org/abs/2402.00025).
- Triton split-K mechanics: [split-K tutorial](https://medium.com/@michael.diggin/implementing-a-split-k-matrix-multiplication-kernel-in-triton-7ad93fe4a54c), [`tl.atomic_add` docs](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html).

**What we take**
- **Tiling + transposed scale layout.** Scales/zeros are stored `(n_groups, N)` (`pack_marlin_weights`) so neighbouring programs read neighbouring scale values (coalesced), as Marlin reshuffles scales offline for ideal access.
- **cp.async + software pipelining — delegated to the compiler, not hand-written.** Marlin issues `cp.async` global→shared copies with double-buffering explicitly. We instead rely on Triton's loop pipeliner: the `num_stages=` field in each `triton.Config` (2–4) tells Triton to prefetch the next K-iterations' tiles via async copies and multi-buffer them while `tl.dot` runs. The async pipeline is exactly the in-loop `tl.load`s + `num_stages`; there is no explicit `cp.async` in our source.
- **Split-K for the decode regime.** At small M (decode) the data-parallel grid is `(M/BLOCK_M)×(N/BLOCK_N)` ≈ a handful of blocks — too few to fill an A100/H100, the "wave quantization" gap from arXiv:2402.00025 (they measure +61% waves/SM on A100). We add a 3rd grid dimension `SPLIT_K` that partitions the K reduction across more blocks. `_choose_split_k` turns it on only for small M and backs off to 1 for prefill (where the grid already saturates and split-K's reduction is pure overhead — the same large-M degradation the paper reports). The cap of 8 matches their empirical sweet spot (4 on A100, 8 on H100), and our "~2 SM-waves of column tiles" target reproduces those picks for the 4096-wide attention projections.
- **Large-`BLOCK_M` configs for prefill.** The earlier kernel capped `BLOCK_M` at 32, so for large-M prefill (MMLU) each weight tile was re-loaded and re-dequantized `M/32` times → worse than the pure-PyTorch path that dequantizes once. Adding `BLOCK_M` 64/128 configs lets autotune pick big M-tiles that amortize the fused dequant.

**Where we diverge (and why)**
- **No `ldmatrix` row permutation.** Marlin permutes weight rows offline into the exact `ldmatrix` fragment order the tensor cores want, then unpermutes on output. That only helps a hand-written MMA kernel; `tl.dot` chooses the fragment layout itself, so the permutation can't help in Triton — and the old code applied it without inverting it (and without permuting scales to match), which scrambled output rows (cosine ≈ 0.14). Removed.
- **Reduction via FP32 partial planes + a torch sum, not atomics.** Marlin reduces partials in the output buffer kept in L2; the W4A16 paper uses `tl.atomic_add`. We instead have each `pid_k` write its own FP32 `(M,N)` plane (pure idempotent `tl.store`), then `partials.sum(0)` — CUTLASS's "parallel split-K". It's deterministic, fully FP32-accurate, and autotune-safe (no `reset_to_zero`/pre-run-hook needed, since stores don't accumulate). The cost is one extra global round-trip for the partials, which is cheap at decode where `M×N` is small and L2-resident. The atomic variant would save that round-trip but adds nondeterminism and autotuning fragility.

## Experiment 2 (kernel latency: is quantization worth deploying?)

About the INT4 GEMM *kernels*, not quality (exp1 owns perplexity). The old
`benchmark.py` labelled runs `custom_awq` / `custom_marlin`, conflating the AWQ
*method* with the *kernel*, and called `awq_impl` Marlin helpers that no longer
exist. We now hold the quantized weights fixed and vary only the GEMM: `vanilla`
(dequant→fp16 matmul) vs `marlin` (fused Triton), via `load_quantized_model`
(which never materialises the FP16 model, so `peak_vram` is honest).

- **`experiment2.py` → `exp2.json`.** End-to-end prefill vs decode latency under
  plain HF static batching (what exp1/exp3 use), across batch sizes, vs the FP16
  baseline. Also writes a `torch.profiler` top-ops breakdown of one prefill + one
  decode step to `exp2_profile.json`.
- **`experiment2_vllm.py` → `vllm_reference.json`.** A labelled reference
  *ceiling*, **not our kernel**: vLLM's own AWQ-Marlin kernel + full serving stack
  (paged attention, CUDA graphs, continuous batching). Running our kernel inside
  vLLM would need a custom `QuantizationConfig`/`LinearMethodBase` plugin (the
  W4A16 interface exists in vLLM 0.7.3 but weight-loader/sharding/`awq_scale`
  matching is fragile, and it still wouldn't touch paged-attn/CUDA-graphs). Used
  Qwen 2.5 7B Instruct AWQ (Llama 3.1 8B has no official AWQ checkpoint on HF).

**Result (A10, Qwen 2.5 7B Instruct): marlin loses to FP16 by ~3.5–3.8×.**
End-to-end decode tok/s — marlin 8.3 / fp16 31.5 (bs=1) … marlin 223 / fp16 770
(bs=32). marlin beats vanilla ~3.8×, so it's a fine INT4 kernel; it's just not
competitive with cuBLAS. The only win is VRAM (11.8 vs 21.3 GB). Exp4 diagnoses why.

## Experiment 4 (GEMM roofline / kernel characterization)

`experiment4.py` → `roofline.json` (+`roofline.png`). A synthetic single-GEMM
microbenchmark (no model/data) that sweeps M (=batch·seq tokens) for
`{fp16-cuBLAS, vanilla, marlin}`, recording latency, achieved TFLOP/s, GB/s and
arithmetic intensity. **Layer shapes are derived from the model config** (not
hardcoded — the original Llama 4096/14336 shapes were wrong for Qwen's
3584/18944 + GQA). GPU peaks are *measured* (big matmul + big copy), so the ridge
point (=peak_compute/peak_bw) is hardware-agnostic.

**Why marlin loses: it's overhead-bound, not memory-bound.** At M=1 (decode,
attn_qo) FP16 runs at 494 GB/s = 89% of the 557 GB/s peak — bandwidth-saturated,
as it should be. marlin moves 4× fewer bytes yet runs 2.4× *slower*, at only
55 GB/s = 10% of peak. It sits at a flat ~0.16 ms floor for all M≤32 (FP16's
floor is ~0.065 ms). The exp2 profile shows the cause: each `QuantLinear.forward`
fires ~5 kernels vs FP16's single cuBLAS call — `x/awq_scale` (div),
`_marlin_gemm_kernel`, `partials.sum(0)` (reduce), `.to(fp16)` (copy), bias-add —
plus `marlin_gemm` queries `get_device_properties()` and `torch.empty`s the FP32
split-K partials every call. Even at large M marlin plateaus at ~18 TFLOP/s =
~24% of the 75 TFLOP/s peak. The roofline says a clean decode kernel could hit
~0.017 ms (4× under FP16), so ~10× of headroom is lost to per-call overhead.
Quantization *can* win at decode; this kernel doesn't yet.

Still TODO:
- Kernel: kill the per-call overhead (fold `awq_scale` into stored scales, drop
  split-K's separate FP32 partial+sum at small M, emit fp16 from the epilogue,
  cache device props).
- Comprehensive experiment 3 run (accuracy must stay on our kernel — vLLM would
  measure vLLM's quantization, not ours).
- Experiment 5: damage control.