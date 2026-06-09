#!/usr/bin/env bash
# Usage:
#   chmod +x run_exp3_parallel.sh
#   HF_ENDPOINT=https://hf-mirror.com ./run_exp3_parallel.sh
#
# Env knobs:
#   CKPT_DIR   path to save/load checkpoints (default: ./checkpoint)
#   BS_INT4    batch size for INT4 models     (default: 16, reduce to 8 on OOM)
#   BS_FP16    batch size for FP16 models     (default: 32, reduce to 16 on OOM)
#   SKIP_QUANT set to 1 to skip Phase 1
#   SKIP_FP16  set to 1 to skip FP16 eval

set -uo pipefail

CKPT_DIR="${CKPT_DIR:-./checkpoint}"
BS_INT4="${BS_INT4:-16}"
BS_FP16="${BS_FP16:-32}"
SKIP_QUANT="${SKIP_QUANT:-0}"
SKIP_FP16="${SKIP_FP16:-0}"
HF="${HF_ENDPOINT:-https://hf-mirror.com}"

QWEN="Qwen/Qwen2.5-7B-Instruct"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN_SLUG="Qwen_Qwen2.5-7B-Instruct"
LLAMA_SLUG="meta-llama_Llama-3.1-8B-Instruct"

mkdir -p "$CKPT_DIR" results

log() { echo "[$(date '+%H:%M:%S')] $*"; }

quantize_model() {
    local gpu=$1 model=$2 method=$3 slug=$4
    local out="$CKPT_DIR/${method}_${slug}.pt"
    if [[ -f "$out" ]]; then
        log "SKIP quantize (exists): $out"
        return 0
    fi
    log "GPU$gpu | quantize $method $model"
    CUDA_VISIBLE_DEVICES=$gpu HF_ENDPOINT=$HF \
        uv run run_quantization.py \
            --model "$model" --method "$method" \
            --save_dir "$CKPT_DIR"
    log "GPU$gpu | done: $out"
}

eval_int4() {
    local gpu=$1 ckpt=$2 label=$3
    if [[ ! -f "$ckpt" ]]; then
        log "SKIP eval [$label]: checkpoint not found: $ckpt"
        return 0
    fi
    log "GPU$gpu | eval INT4 [$label] batch=$BS_INT4"
    CUDA_VISIBLE_DEVICES=$gpu HF_ENDPOINT=$HF \
        uv run experiment3.py \
            --weight_path "$ckpt" \
            --batch_size "$BS_INT4"
    log "GPU$gpu | done: $label"
}

eval_hf() {
    local gpu=$1 model_id=$2 label=$3
    log "GPU$gpu | eval HF [$label] batch=$BS_INT4"
    CUDA_VISIBLE_DEVICES=$gpu HF_ENDPOINT=$HF \
        uv run experiment3.py \
            --model "$model_id" \
            --batch_size "$BS_INT4"
    log "GPU$gpu | done: $label"
}

eval_fp16() {
    local gpus=$1 model_id=$2 label=$3
    log "GPUs[$gpus] | eval FP16 [$label] batch=$BS_FP16"
    CUDA_VISIBLE_DEVICES=$gpus HF_ENDPOINT=$HF \
        uv run experiment3.py \
            --model "$model_id" \
            --batch_size "$BS_FP16"
    log "GPUs[$gpus] | done: $label"
}

if [[ "$SKIP_QUANT" != "1" ]]; then
    log "=== Phase 1: Quantize (Qwen on GPU0, Llama on GPU1, parallel) ==="

    (
        quantize_model 0 "$QWEN" awq  "$QWEN_SLUG"
        quantize_model 0 "$QWEN" rtn  "$QWEN_SLUG"
        quantize_model 0 "$QWEN" gptq "$QWEN_SLUG"
    ) &
    PID_QWEN=$!

    (
        quantize_model 1 "$LLAMA" awq  "$LLAMA_SLUG"
        quantize_model 1 "$LLAMA" rtn  "$LLAMA_SLUG"
        quantize_model 1 "$LLAMA" gptq "$LLAMA_SLUG"
    ) &
    PID_LLAMA=$!

    wait $PID_QWEN
    log "Qwen quantization complete"
    wait $PID_LLAMA
    log "Llama quantization complete"
    log "=== Phase 1 done ==="
else
    log "=== Phase 1 skipped ==="
fi

log "=== Phase 2A: INT4 eval (4 × 1 GPU) ==="

eval_int4 0 "$CKPT_DIR/awq_${QWEN_SLUG}.pt"   "qwen_awq"        &
eval_int4 1 "$CKPT_DIR/gptq_${QWEN_SLUG}.pt"  "qwen_gptq_ours"  &
eval_int4 2 "$CKPT_DIR/awq_${LLAMA_SLUG}.pt"  "llama_awq"       &
eval_int4 3 "$CKPT_DIR/gptq_${LLAMA_SLUG}.pt" "llama_gptq_ours" &
wait
log "=== Phase 2A done ==="

log "=== Phase 2B: RTN + HF GPTQ reference ==="

eval_int4 0 "$CKPT_DIR/rtn_${QWEN_SLUG}.pt"  "qwen_rtn"    &
eval_hf   1 "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" "qwen_gptq_hf" &
eval_int4 2 "$CKPT_DIR/rtn_${LLAMA_SLUG}.pt" "llama_rtn"   &
wait
log "=== Phase 2B done ==="

if [[ "$SKIP_FP16" != "1" ]]; then
    log "=== Phase 2C: FP16 baselines (all 4 GPUs) ==="
    eval_fp16 "0,1,2,3" "$QWEN"  "qwen_fp16"
    eval_fp16 "0,1,2,3" "$LLAMA" "llama_fp16"
    log "=== Phase 2C done ==="
else
    log "=== Phase 2C skipped ==="
fi

log "=== All done. Results in results/exp3.json ==="