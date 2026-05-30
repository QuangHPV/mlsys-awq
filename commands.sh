# Download model
HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=<> hf download "meta-llama/Meta-Llama-3.1-8B"

# Test quantization
HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py --model "meta-llama/Meta-Llama-3.1-8B"
HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py --model "Qwen/Qwen2.5-7B"

# Load checkpoint
cp -r /cephfs/ylshare/huangyg/models--meta-llama--Meta-Llama-3.1-8B ~/.cache/huggingface/hub
cp -r /cephfs/ylshare/huangyg/models--Qwen--Qwen2.5-7B ~/.cache/huggingface/hub

cp -r checkpoint /cephfs/ylshare/huangyg 
# Run experiment 1
HF_ENDPOINT=https://hf-mirror.com uv run experiment1.py --model "meta-llama/Meta-Llama-3.1-8B" \
    --weight_path "checkpoint/awq_meta-llama_Meta-Llama-3.1-8B.pt"

HF_ENDPOINT=https://hf-mirror.com uv run experiment1.py --model "Qwen/Qwen2.5-7B" \
    --weight_path "checkpoint/awq_Qwen_Qwen2.5-7B.pt"


# GPTQ/RTN
uv run run_quantization.py --model Qwen/Qwen2.5-7B --method gptq
uv run run_quantization.py --model Qwen/Qwen2.5-7B --method rtn
uv run run_quantization.py --model meta-llama/Meta-Llama-3.1-8B --method rtn
uv run run_quantization.py --model meta-llama/Meta-Llama-3.1-8B --method awq
uv run experiment1.py --weight_path checkpoint/gptq_Qwen_Qwen2.5-7B.pt
uv run experiment1.py --baseline --model Qwen/Qwen2.5-7B
uv run experiment1.py --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B.pt
uv run experiment1.py --weight_path ../checkpoint/awq_meta-llama_Meta-Llama-3.1-8B.pt