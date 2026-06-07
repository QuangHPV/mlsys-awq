# Env setup
# gptqmodel & autoawq are sdist-only (no wheels exist on PyPI), so uv compiles
# their CUDA kernels from source. Restrict the build to THIS server's GPU arch
# and parallelize nvcc, or it compiles for ~6 archs and takes hours.
#   nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 8.0 -> A100, 8.9 -> L40/4090, 9.0 -> H100
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=$(nproc) uv sync

# Download model
HF_ENDPOINT=https://hf-mirror.com hf download "meta-llama/Llama-3.1-8B-Instruct"
HF_ENDPOINT=https://hf-mirror.com hf download "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
HF_ENDPOINT=https://hf-mirror.com hf download "Qwen/Qwen2.5-7B-Instruct"
HF_ENDPOINT=https://hf-mirror.com hf download "Qwen/Qwen2.5-7B-Instruct-AWQ"


# Load checkpoint
cp -r /cephfs/ylshare/huangyg/models--meta-llama--Meta-Llama-3.1-8B ~/.cache/huggingface/hub
cp -r /cephfs/ylshare/huangyg/models--Qwen--Qwen2.5-7B ~/.cache/huggingface/hub

cp -r checkpoint /cephfs/ylshare/huangyg 
# Run experiment 1
HF_ENDPOINT=https://hf-mirror.com uv run experiment1.py --model "meta-llama/Meta-Llama-3.1-8B" \
    --weight_path "checkpoint/awq_meta-llama_Meta-Llama-3.1-8B.pt"

HF_ENDPOINT=https://hf-mirror.com uv run experiment1.py --model "Qwen/Qwen2.5-7B" \
    --weight_path "checkpoint/awq_Qwen_Qwen2.5-7B.pt"


# Test quantization
HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py --model "meta-llama/Llama-3.1-8B-Instruct" --method awq
HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py --model "meta-llama/Llama-3.1-8B-Instruct" --method gptq
HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py --model "Qwen/Qwen2.5-7B"
uv run run_quantization.py --model Qwen/Qwen2.5-7B --method gptq
uv run run_quantization.py --model Qwen/Qwen2.5-7B --method rtn
uv run run_quantization.py --model meta-llama/Meta-Llama-3.1-8B --method rtn
uv run run_quantization.py --model meta-llama/Meta-Llama-3.1-8B --method awq
uv run run_quantization.py --model Qwen/Qwen2.5-7B-Instruct --method awq
uv run run_quantization.py --model Qwen/Qwen2.5-7B-Instruct --method rtn
uv run experiment1.py --weight_path checkpoint/gptq_Qwen_Qwen2.5-7B.pt
uv run experiment1.py --baseline --model Qwen/Qwen2.5-7B
uv run experiment1.py --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B.pt
uv run experiment1.py --weight_path ../checkpoint/awq_meta-llama_Llama-3.1-8B-Instruct.migrated.pt
uv run experiment1.py --weight_path ../checkpoint/rtn_meta-llama_Llama-3.1-8B-Instruct.pt
uv run experiment1.py --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt
uv run experiment1.py --weight_path ../checkpoint/rtn_Qwen_Qwen2.5-7B-Instruct.pt
uv run experiment1.py --model meta-llama/Llama-3.1-8B-Instruct
uv run experiment1.py --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
uv run experiment1.py --weight_path ./checkpoint/gptq_meta-llama_Llama-3.1-8B-Instruct.pt

# Debug baseline Qwen AWQ
uv run experiment1.py --baseline --model Qwen/Qwen2.5-7B-Instruct-AWQ
uv run experiment1.py --baseline --model Qwen/Qwen2.5-7B-Instruct

# Migrate old checkpoint
HF_ENDPOINT=https://hf-mirror.com uv run migrate_checkpoint.py --old ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt

# Experiment 3: GSM8k + MMLU accuracy. Spawns one worker per visible GPU
# (override with --num_gpus), stride-shards the test set across workers, batches
# generation. Restrict GPUs with CUDA_VISIBLE_DEVICES; tune per-worker throughput
# with --batch_size. Pass exactly one source: an HF model id (FP16 or a
# pre-quantized HF checkpoint) OR our own quant checkpoint (.pt).
#
# usage: [CUDA_VISIBLE_DEVICES=0,1,2,3] HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py \
#            (--model HF_ID | --weight_path CHECKPOINT.pt) \
#            [--tasks gsm8k,mmlu] [--batch_size 16] [--num_gpus 0]   # 0 = all visible GPUs

# FP16 baseline, both tasks, all visible GPUs
HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py \
    --model meta-llama/Llama-3.1-8B-Instruct

# our INT4 (triton kernel) checkpoint on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py \
    --weight_path ../checkpoint/awq_meta-llama_Llama-3.1-8B-Instruct.migrated.pt \
    --batch_size 8

# pre-quantized HF baselines (their own kernels), for reference
HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batch_size 32

# single task: mmlu only (one forward, no generation -> fast)
HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tasks mmlu

# single task on a single GPU: gsm8k only, force 1 worker
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py \
    --weight_path ../checkpoint/gptq_Qwen_Qwen2.5-7B-Instruct.migrated.pt \
    --tasks gsm8k --num_gpus 1

# Experiment 2: Prefil/decode latency
HF_ENDPOINT=https://hf-mirror.com uv run experiment2.py \
    --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.migrated.pt
HF_ENDPOINT=https://hf-mirror.com uv run experiment2_vllm.py --vllm_model Qwen/Qwen2.5-7B-Instruct-AWQ

# Experiment 4 - GEMM roofline
HF_ENDPOINT=https://hf-mirror.com uv run experiment4.py \
    --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.migrated.pt --plot

ncu --nvtx --nvtx-include "profile/" -k regex:triton \
    --section SpeedOfLight --section WarpStateStats --section MemoryWorkloadAnalysis \
    uv run prof_triton.py

ncu --nvtx --nvtx-include "profile/" -k regex:triton \
    --section SpeedOfLight --section WarpStateStats --section MemoryWorkloadAnalysis \
    --target-processes all \
    python prof_triton.py

ncu python -c "import torch; print(torch.randn(8, device='cuda').sum())"