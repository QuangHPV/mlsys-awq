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
uv run experiment1.py --weight_path ../checkpoint/awq_meta-llama_Llama-3.1-8B-Instruct.pt
uv run experiment1.py --weight_path ../checkpoint/rtn_meta-llama_Llama-3.1-8B-Instruct.pt
uv run experiment1.py --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt
uv run experiment1.py --weight_path ../checkpoint/rtn_Qwen_Qwen2.5-7B-Instruct.pt
uv run experiment1.py --model meta-llama/Llama-3.1-8B-Instruct
uv run experiment1.py --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4

# Debug baseline Qwen AWQ
uv run experiment1.py --baseline --model Qwen/Qwen2.5-7B-Instruct-AWQ
uv run experiment1.py --baseline --model Qwen/Qwen2.5-7B-Instruct

# Experiment 3: GSM8k + MMLU accuracy. Spawns one worker per visible GPU
# (override with --num_gpus), stride-shards the test set, batches generation.
# Restrict GPUs with CUDA_VISIBLE_DEVICES; tune throughput with --batch_size.
HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py --model meta-llama/Llama-3.1-8B-Instruct
HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py --weight_path ../checkpoint/awq_meta-llama_Llama-3.1-8B-Instruct.pt
HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py --model Qwen/Qwen2.5-7B-Instruct --batch_size 32
HF_ENDPOINT=https://hf-mirror.com uv run experiment3.py --model Qwen/Qwen2.5-7B-Instruct --tasks mmlu  # mmlu only