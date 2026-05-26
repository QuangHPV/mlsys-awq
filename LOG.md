# Update May 26
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

