# MLSys Project: Activation-aware Weight Quantization

## AWQ Implementation

## Experiment 1
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


## Experiment 2 (batched decoding)
