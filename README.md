# Quantization-Aware EAGLE-3 W4A16 Pilot

This workspace now contains a runnable research scaffold for the pilot study:

- offline hidden-state extraction for BF16 and AWQ teachers
- resumable draft-head training with checkpointing
- local speculative-evaluation utilities for MT-Bench style prompts
- cluster-friendly example configs and a 20-question mini benchmark

The code is intentionally narrow for v1:

- it targets **EAGLE-3-style multi-layer fused draft training**
- it assumes **W4A16 AWQ** as the first quantized setting
- it prioritizes **resumable offline runs** over production serving
- it works best when the main pilot is run on a multi-GPU cluster node such as Sol's `4x A100 80 GiB` nodes

## Layout

```text
peagle_q/
  cli.py
  data.py
  dataset.py
  eval.py
  extract.py
  layers.py
  manifest.py
  model.py
  teacher.py
  train.py
configs/
benchmarks/
tests/
```

## Install

Core package:

```bash
pip install -e .
```

For a local Windows CPU smoke test with Gemma:

```bash
pip install -e .[smoke]
```

For research runs:

```bash
pip install -e .[research,quant]
```

If you want to experiment with a future vLLM integration layer:

```bash
pip install -e .[serve]
```

## Local Gemma 1B Smoke Test

This is the safest first check on a laptop-class Windows machine before attempting the full quantization-aware pipeline.

What it validates:

- the local Python environment can load an instruction-tuned Gemma model
- chat-template rendering works
- hidden-state extraction works on the chosen layer taps
- plain autoregressive generation runs end to end

What it does **not** validate:

- vLLM integration
- AWQ verifier loading
- draft-head training throughput
- realistic paper-scale latency claims

### Quick Start

1. If Hugging Face asks for authentication, run:

```bash
hf auth login
```

You may also need to accept the Gemma model terms for the specific model on Hugging Face before the first download succeeds.

2. Recommended Windows path:

```powershell
.\scripts\run_gemma_smoke.ps1
```

This helper installs `sentencepiece` and `protobuf` into a workspace-local `.vendor-smoke` folder and runs the smoke test with `python -m peagle_q.cli`.

3. Manual fallback:

```powershell
pip install --target .vendor-smoke sentencepiece "protobuf<6"
$env:PYTHONPATH = ".vendor-smoke;."
python -m peagle_q.cli smoke --config configs\smoke_gemma_1b_cpu.json
```

Expected result:

- a JSON summary with `prompt_length`
- selected `layer_indices`
- `hidden_state_shape`
- a short `autoregressive.text` sample
- rough CPU latency metrics

If this passes, the next sensible step is a tiny extraction smoke test on a few prompts before considering any heavier training work.

## Bigger-System First Steps

For a stronger machine, the safest progression is:

1. Run a 3-prompt extraction first
2. Verify `manifest.jsonl` and `tensors/*.pt`
3. Only then scale to a larger prompt set
4. After that, run training

Ready-made assets:

- [extract_bigmodel_3prompt_template.json](/f:/Research%20paper/configs/extract_bigmodel_3prompt_template.json)
- [extract_bigmodel_pilot_template.json](/f:/Research%20paper/configs/extract_bigmodel_pilot_template.json)
- [train_bigmodel_smoke_template.json](/f:/Research%20paper/configs/train_bigmodel_smoke_template.json)
- [run_bigmodel_first_steps.ps1](/f:/Research%20paper/scripts/run_bigmodel_first_steps.ps1)
- [run_bigmodel_first_steps.sh](/f:/Research%20paper/scripts/run_bigmodel_first_steps.sh)

Example usage on Windows:

```powershell
.\scripts\run_bigmodel_first_steps.ps1 -ModelPath "google/gemma-3-4b-it" -RunName "gemma4b"
```

Example usage on Linux / Slurm node:

```bash
bash ./scripts/run_bigmodel_first_steps.sh "google/gemma-3-4b-it" gemma4b
```

That script will:

- generate a 3-prompt extract config for the model you pass
- run the 3-prompt extraction
- verify that `manifest.jsonl` exists
- verify that at least 3 tensor files exist
- print the exact next commands for scaled extraction and training

## Data Preparation

Training extraction expects JSONL with one record per line. Supported shapes:

1. `{"prompt_id": "...", "prompt": "plain text"}`
2. `{"id": "...", "messages": [{"role": "user", "content": "..."}, ...]}`
3. ShareGPT-style `{"id": "...", "conversations": [{"from": "human", "value": "..."}, ...]}`

## Hidden-State Extraction

BF16 teacher:

```bash
peagle-q extract ^
  --dataset path\to\sharegpt.jsonl ^
  --teacher-model-path meta-llama/Llama-3.1-8B-Instruct ^
  --teacher-quantization none ^
  --output-dir runs\extract_bf16 ^
  --max-examples 1000
```

AWQ teacher:

```bash
peagle-q extract ^
  --dataset path\to\sharegpt.jsonl ^
  --teacher-model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 ^
  --teacher-quantization awq ^
  --output-dir runs\extract_awq ^
  --max-examples 1000
```

Outputs:

- `manifest.jsonl`
- `run_config.json`
- `tensors/<dataset_index>_<prompt_id>_<hash>.pt`

Each manifest row stores prompt id, teacher model path, quantization, tokenizer metadata, tapped layers, and the tensor file path.

## Training

Smoke, pilot, and main phases map directly to the v1 plan. Example:

```bash
peagle-q train ^
  --manifest runs\extract_awq\manifest.jsonl ^
  --output-dir runs\train_awq_1k ^
  --projection-model-path meta-llama/Llama-3.1-8B-Instruct ^
  --epochs 1 ^
  --max-examples 1000 ^
  --checkpoint-every 200
```

Resume a stopped run:

```bash
peagle-q train ^
  --manifest runs\extract_awq\manifest.jsonl ^
  --output-dir runs\train_awq_32k ^
  --projection-model-path meta-llama/Llama-3.1-8B-Instruct ^
  --epochs 3 ^
  --max-examples 32000 ^
  --resume-from runs\train_awq_32k\checkpoints\last.pt
```

## Evaluation

The local evaluator is meant for research validation, not production latency claims. It:

- computes accepted lengths and position-wise acceptance
- compares speculative generation against autoregressive generation on the same verifier
- writes per-turn results plus a summary JSON

Mini benchmark:

```bash
peagle-q evaluate ^
  --benchmark benchmarks\mini_mt_bench.jsonl ^
  --verifier-model-path meta-llama/Llama-3.1-8B-Instruct ^
  --verifier-quantization none ^
  --draft-checkpoint runs\train_bf16_1k\checkpoints\best.pt ^
  --output-path runs\eval_bf16_on_bf16.json
```

For the three v1 study rows, run:

1. BF16-trained draft -> BF16 verifier
2. BF16-trained draft -> AWQ verifier
3. AWQ-trained draft -> AWQ verifier

## Compute Notes

- For a local laptop or desktop, use `Gemma 1B` only for smoke tests and basic environment validation.
- If you have access to ASU Sol, use the `A100 80 GiB` nodes for the first real `Qwen3-8B` BF16 and AWQ runs.
- Keep BF16 and AWQ extractions in separate run folders.
- Use the same normalized dataset file for both extractions.
- Store large hidden-state dumps, checkpoints, and benchmark logs on cluster scratch space rather than small home-directory quotas.
- Keep checkpointing enabled for any longer training run.

## Sol Runbook

This is the recommended branch-ready path if you have access to Sol.

If you prefer to work from a git branch on Sol and keep cluster-side steps minimal, also see [SOL_BRANCH_WORKFLOW.md](F:/Research paper/SOL_BRANCH_WORKFLOW.md).

### 1. Set the required environment variables

At minimum, export these before running the Sol scripts:

```bash
export PEAGLE_ROOT=/path/to/this/repo
export PEAGLE_BF16_MODEL=Qwen/Qwen3-8B
export PEAGLE_AWQ_MODEL=Qwen/Qwen3-8B-AWQ
export PEAGLE_DATASET=/path/to/sharegpt_or_other_train.jsonl
```

Recommended run locations:

```bash
export PEAGLE_RUN_ROOT=$SCRATCH/peagle-q/$USER
export PEAGLE_BENCHMARK=$PEAGLE_ROOT/benchmarks/mini_mt_bench.jsonl
export PEAGLE_CACHE_ROOT=$PEAGLE_RUN_ROOT/cache
```

### 2. Build the Sol environment once

Run this on a login node:

```bash
bash scripts/sol/setup_env.sh
```

This creates `.venv-sol`, installs the research dependencies, and defaults large outputs to `$SCRATCH` when available.

The quantized AWQ path also depends on `gptqmodel`. `setup_env.sh` installs it separately with `--no-build-isolation`, which is the install mode recommended by the package.

If your Sol environment already provides a suitable Python stack and you do not want the helper to run `pip install`, set:

```bash
export PEAGLE_SKIP_PIP_INSTALL=1
```

### 3. Sanity-check the first real model path

Submit a single-GPU Qwen smoke job:

```bash
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=01:00:00 \
  --export=ALL,PEAGLE_COMMAND=smoke,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_smoke.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm
```

### 4. Run the first paper-relevant 1k pilot

BF16 extraction:

```bash
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=02:00:00 \
  --export=ALL,PEAGLE_COMMAND=extract,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_extract_bf16_1k.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm
```

AWQ extraction:

```bash
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=02:00:00 \
  --export=ALL,PEAGLE_COMMAND=extract,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_extract_awq_1k.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm
```

If an existing `.venv-sol` was created before the AWQ dependency step was added, refresh it with:

```bash
source .venv-sol/bin/activate
python -m pip install -e ".[research,quant,dev]"
python -m pip install -v --no-build-isolation gptqmodel
```

BF16-control training:

```bash
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=04:00:00 \
  --export=ALL,PEAGLE_COMMAND=train,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_train_bf16_1k.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm
```

AWQ-aware training:

```bash
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=04:00:00 \
  --export=ALL,PEAGLE_COMMAND=train,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_train_awq_1k.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm
```

Core evaluation rows:

```bash
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=02:00:00 \
  --export=ALL,PEAGLE_COMMAND=evaluate,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_eval_bf16_on_bf16.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm

sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=02:00:00 \
  --export=ALL,PEAGLE_COMMAND=evaluate,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_eval_bf16_on_awq.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm

sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=02:00:00 \
  --export=ALL,PEAGLE_COMMAND=evaluate,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_eval_awq_on_awq.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm
```

### 5. Scale to the 8k pilot only after the 1k run works

The repo includes ready-made 8k configs:

- `configs/sol/qwen3_8b_extract_bf16_8k.json`
- `configs/sol/qwen3_8b_extract_awq_8k.json`
- `configs/sol/qwen3_8b_train_bf16_8k.json`
- `configs/sol/qwen3_8b_train_awq_8k.json`

### 6. Notes for Sol usage

- Add your cluster-specific `--account`, `--partition`, or QoS flags to each `sbatch` command.
- The generic Slurm wrapper lives at `scripts/sol/run_peagle_job.slurm`.
- A quick command reference is available in `scripts/sol/submit_examples.sh`.
- For branch-based Sol usage, use `scripts/sol/bootstrap_branch.sh`, `scripts/sol/run_from_branch.sh`, and `scripts/sol/submit_from_branch.sh`.
- To refresh a branch and submit the full first 1k pipeline in one step, use `scripts/sol/submit_1k_pilot_from_branch.sh`.
- Keep the repo itself on a location you can version, and keep large generated outputs on scratch.
- The Sol helpers default Hugging Face, Transformers, datasets, and pip caches into `PEAGLE_CACHE_ROOT`, which should usually live under scratch.

## vLLM Status

This scaffold is ready for a vLLM-backed extractor or serving adapter, but the included implementation uses the robust local `transformers` path today. That keeps the study executable even when `vllm` or `autoawq` are not available in the current environment.
