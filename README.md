# mlsys-awq — INT4 weight quantization for LLMs, end to end

A from-scratch study of **4-bit weight-only quantization** for Llama 3.1 8B and
Qwen 2.5 7B: our own implementations of **AWQ**, **GPTQ**, and **RTN**; a fused
**Triton split-K W4A16 GEMM** to actually run the INT4 weights; and a modified vendored
copy of the canonical **Marlin** CUDA kernel for asymmetric quantization schemes like AWQ. Four
experiments measure the things that matter for deployment — perplexity, task
accuracy (GSM8k / MMLU), prefill/decode latency, and a GEMM roofline that
explains *why* the kernel performs the way it does.

This is a research/coursework repository, not a library. The narrative findings
(why AWQ beat RTN by ~0.4 ppl, why the Triton kernel loses to cuBLAS, the
instruct-vs-base blow-up, etc.) live in [LOGS.md](LOGS.md). 

## What's here

- **Quantization methods**, all writing the same INT4 checkpoint format:
  - *AWQ* — activation-aware per-channel scaling searched against cached
    calibration activations (`awq_impl.py`).
  - *GPTQ* — Hessian-based error-compensating quantization (`gptq.py`).
  - *RTN* — round-to-nearest baseline (same packing path as AWQ, `awq_scale=1`).
- **INT4 inference kernels**, selectable per layer (`kernel.py`):
  - *vanilla* — dequantize to FP16, then a plain matmul (correctness baseline).
  - *triton* — fused split-K dequant+GEMM, autotuned, tuned for the decode regime.
  - *marlin* — the vendored canonical CUDA kernel, used as a speed ceiling.
- **Experiments 1–4** + notebooks that visualize their JSON outputs.

## Architecture

```
mlsys-awq/
├── awq_impl.py            # AWQ + RTN: activation stats, scale search, pack to INT4
├── gptq.py                # GPTQ: per-layer Hessian, error compensation, INT4 pack
├── kernel.py              # the heart: Triton GEMM, QuantLinear/MarlinLinear,
│                          #   weight (re)packing, checkpoint load/replace helpers
├── run_quantization.py    # CLI: quantize a model (awq|gptq|rtn) -> self-contained .pt
├── migrate_checkpoint.py  # port old quant_params-only .pt to the new format
│
├── experiment1.py         # perplexity (wikitext-2) -> results/exp1.json
├── experiment2.py         # prefill/decode latency, HF static batching -> exp2.json
├── experiment2_vllm.py    # vLLM AWQ-Marlin reference ceiling -> vllm_reference.json
├── experiment3.py         # GSM8k + MMLU accuracy, multi-GPU sharded -> exp3.json
├── experiment4.py         # single-GEMM roofline sweep -> roofline.json/.png
├── run_exp3.sh            # orchestrates the full exp3 matrix across 4 GPUs
│
├── verify.py              # quick kernel/quant sanity checks
├── test_kernel.py         # pytest: triton vs reference/vanilla, micro-bench
├── test_gptq.py           # pytest: our GPTQ vs gptqmodel, perplexity
├── prof_triton.py         # NVTX-wrapped launch for ncu profiling of the kernel
│
├── analysis.ipynb         # visualizes exp1–4 JSON (the report figures)
├── diagnostics.ipynb      # PyPI wheel survey + build/latency scratch work
│
├── marlin/                # VENDORED canonical Marlin CUDA kernel (see below)
│   ├── marlin_cuda.cpp        # pybind entry
│   ├── marlin_cuda_kernel.cu  # the FP16xINT4 mma kernel (modified with zero points)
│   ├── __init__.py            # mul() wrapper + offline weight/scale perms
│   ├── setup.py               # CUDA_HOME autodetect + header diagnostics
│   └── pyproject.toml         # built from source by uv as the `marlin` dep
│
├── results/               # experiment JSON + generated figures (*.png)
├── commands.sh            # copy-paste log of every command used (env, runs)
├── LOGS.md                # the research narrative: findings, bugs, decisions
└── pyproject.toml         # uv project; pins torch 2.5.1 / triton 3.1 / cu121
```

### The checkpoint format

`run_quantization.py` saves a **self-contained** `.pt`: the INT4 `QuantLinear`
layers *plus* the untouched FP16 remainder (embeddings, norms, lm_head), so
loading never needs the original FP16 model in memory. Each checkpoint carries
`model_id`, `group_size`, `method`, the list of quantized layer names, and the
full `state_dict`. `kernel.load_quantized_model` rebuilds the empty model on the
`meta` device and repacks scales/zeros into the chosen kernel's layout.

## Building

The project is managed with [uv](https://docs.astral.sh/uv/) and pins
`torch==2.5.1`, `triton==3.1.0` against the CUDA 12.1 wheels. It is Linux/x86_64
+ NVIDIA only (the `tool.uv.environments` marker enforces this).

```bash
# Build for THIS machine's GPU arch only, and parallelize nvcc — otherwise
# the source deps compile for ~6 archs and take hours.
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 8.6 -> A10/3090, 9.0 -> H100
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=$(nproc) uv sync
```

### Kernel build nuances

Three dependencies compile CUDA from source: `autoawq`, `gptqmodel`, and our
vendored `marlin`. 

- **A full CUDA toolkit is required, not just `nvcc`.** `marlin/setup.py`
  autodetects `CUDA_HOME` (env → `nvcc` on PATH → `/usr/local/cuda` → the pip
  `nvidia-cuda-nvcc-cu12` wheel) and prints a diagnostic. The pip nvcc wheel
  ships the compiler but **not** the headers (`cuda.h`, `cuda_runtime.h`,
  `cuda_fp16.h`), so the build fails late with a missing-header error. If
  autodetect picks the wrong root, point it at a real toolkit explicitly:
  ```bash
  CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=$(nproc) uv sync
  ```
- **`TORCH_CUDA_ARCH_LIST` must match your GPU** or the kernels build for the
  wrong SM and fail to launch. Set it to your `compute_cap`.
- **Rebuild just Marlin** after editing the `.cu`/`.cpp` without re-syncing the
  whole world:
  ```bash
  CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=$(nproc) \
      uv sync --reinstall-package marlin
  # or, in-place during kernel hacking:
  cd marlin && CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST=8.6 \
      python setup.py build_ext --inplace
  ```
- **The Triton kernel needs no build step** — it JITs on first use and autotunes
  across a config grid (warm it up once before benchmarking/profiling).

### Marlin is vendored from upstream

[marlin/](marlin/) is a vendored copy of the canonical Marlin kernel from
**[IST-DASLab/marlin](https://github.com/IST-DASLab/marlin)** (Frantar, Castro,
Chen, Hoefler, Alistarh — *MARLIN: Mixed-Precision Auto-Regressive Parallel
Inference on LLMs*, [arXiv:2408.11743](https://arxiv.org/abs/2408.11743)),
Apache-2.0. We further modified this kernel to support asymmetric quantization schemes. We keep it as a local editable dependency so it builds against the
project's exact torch/CUDA and so the offline weight/scale permutations can be
called from `kernel.py`. Note our Triton kernel is **not** Marlin
— it shares none of Marlin's hand-written MMA machinery; the rationale for where
the two diverge is in [LOGS.md](LOGS.md).

### Models & calibration

```bash
HF_ENDPOINT=https://hf-mirror.com hf download "meta-llama/Llama-3.1-8B-Instruct"
HF_ENDPOINT=https://hf-mirror.com hf download "Qwen/Qwen2.5-7B-Instruct"
# pre-quantized HF baselines used as references:
HF_ENDPOINT=https://hf-mirror.com hf download "Qwen/Qwen2.5-7B-Instruct-AWQ"
HF_ENDPOINT=https://hf-mirror.com hf download "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
```

Calibration uses the `mit-han-lab/pile-val-backup` set (shared by AWQ and GPTQ);
perplexity uses wikitext-2. **Use the Instruct variants** — the base models blow
up under our INT4 path (documented in [LOGS.md](LOGS.md)).

## Quantizing a model

`run_quantization.py` produces a checkpoint. Pick a model and a method:

```bash
HF_ENDPOINT=https://hf-mirror.com uv run run_quantization.py \
    --model Qwen/Qwen2.5-7B-Instruct --method awq   # or gptq | rtn
# -> ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt
```

`--group-size` defaults to 128; `--save_dir` defaults to `../checkpoint`. An old
`quant_params`-only checkpoint can be ported to the self-contained format with
`uv run migrate_checkpoint.py --old <path>.pt`.

## Running the experiments

All experiments take **either** an HF model id (`--model`, for an FP16 or
pre-quantized-HF baseline) **or** one of our checkpoints (`--weight_path`).
The full command log lives in [commands.sh](commands.sh).

**Experiment 1 — perplexity** (wikitext-2):
```bash
uv run experiment1.py --model Qwen/Qwen2.5-7B-Instruct                    # FP16 baseline
uv run experiment1.py --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt \
    --kernel triton                                                      # our INT4
# -> results/exp1.json
```

**Experiment 2 — prefill/decode latency** (HF static batching, vs FP16, with a
torch.profiler op breakdown):
```bash
uv run experiment2.py \
    --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt \
    --vllm_model Qwen/Qwen2.5-7B-Instruct-AWQ        # optional vLLM ceiling
uv run experiment2_vllm.py --vllm_model Qwen/Qwen2.5-7B-Instruct-AWQ
# -> results/exp2.json, exp2_profile.json, vllm_reference.json
```

**Experiment 3 — GSM8k / MMLU accuracy** (one worker per visible GPU,
stride-sharded test set, batched generation):
```bash
# our INT4 checkpoint across 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com \
    uv run experiment3.py --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt \
    --tasks gsm8k,mmlu --batch_size 16
# FP16 baseline, single task
uv run experiment3.py --model Qwen/Qwen2.5-7B-Instruct --tasks mmlu
# -> results/exp3.json
```
Or run the whole matrix (quantize Qwen+Llama × awq/rtn/gptq, then eval ours, HF
references, and FP16) with `HF_ENDPOINT=https://hf-mirror.com ./run_exp3.sh`.

**Experiment 4 — GEMM roofline** (synthetic single-GEMM sweep over batch tokens;
layer shapes derived from the model config, GPU peaks measured):
```bash
uv run experiment4.py \
    --weight_path ../checkpoint/awq_Qwen_Qwen2.5-7B-Instruct.pt --plot
uv run experiment4.py --plot_only --plot --plot_mode line     # re-plot from JSON
# -> results/roofline.json, roofline.png
```

### Tests

```bash
uv run pytest test_kernel.py      # triton GEMM vs reference & vanilla (cosine, bench)
uv run pytest test_gptq.py        # our GPTQ vs gptqmodel + perplexity
uv run python verify.py           # quick kernel + pseudo-quantize sanity checks
```

Kernel profiling with Nsight Compute (the launch is NVTX-wrapped so only the
steady-state call after autotune is captured):
```bash
ncu --nvtx --nvtx-include "profile/" -k regex:triton \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    uv run prof_triton.py
```

## Visualization

[analysis.ipynb](analysis.ipynb) is the report notebook: it reads the
`results/*.json` from each experiment and renders the figures — perplexity and
peak-VRAM tables (exp1), prefill/decode throughput curves (exp2), GSM8k/MMLU
accuracy comparisons (exp3), and the GEMM roofline (exp4, also saved to
`results/roofline.png` / `linear.png`). Run experiments first, then execute the
notebook top to bottom.
