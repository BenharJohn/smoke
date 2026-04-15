# Sol First Run

## 0. Set paths

```bash
export PEAGLE_ROOT=/path/to/this/repo
export PEAGLE_BF16_MODEL=Qwen/Qwen3-8B
export PEAGLE_AWQ_MODEL=Qwen/Qwen3-8B-AWQ
export PEAGLE_DATASET=/path/to/train.jsonl
export PEAGLE_RUN_ROOT=$SCRATCH/peagle-q/$USER
export PEAGLE_BENCHMARK=$PEAGLE_ROOT/benchmarks/mini_mt_bench.jsonl
export PEAGLE_CACHE_ROOT=$PEAGLE_RUN_ROOT/cache
```

## 1. Build the environment

```bash
bash $PEAGLE_ROOT/scripts/sol/setup_env.sh
```

If AWQ later errors with `Loading an AWQ quantized model requires gptqmodel`, refresh the environment once with:

```bash
source $PEAGLE_ROOT/.venv-sol/bin/activate
python -m pip install -e "$PEAGLE_ROOT[research,quant,dev]"
```

## 2. Run smoke tests (BF16 and AWQ)

```bash
# BF16 smoke
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=01:00:00 \
  --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,PEAGLE_COMMAND=smoke,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_smoke.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm

# AWQ smoke — verifies the quantized model loads and generates correctly
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=01:00:00 \
  --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,PEAGLE_COMMAND=smoke,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_smoke_awq.json \
  $PEAGLE_ROOT/scripts/sol/run_peagle_job.slurm
```

## 3. Submit the full 1k pilot chain

```bash
bash $PEAGLE_ROOT/scripts/sol/submit_1k_pilot.sh --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

## 4. Check outputs

```bash
ls $PEAGLE_RUN_ROOT/qwen3_8b
find $PEAGLE_RUN_ROOT/qwen3_8b -maxdepth 2 -type f | sort
```

## 5. Key result files

```bash
cat $PEAGLE_RUN_ROOT/qwen3_8b/eval_bf16_on_bf16.json
cat $PEAGLE_RUN_ROOT/qwen3_8b/eval_bf16_on_awq.json
cat $PEAGLE_RUN_ROOT/qwen3_8b/eval_awq_on_awq.json
```

## 6. If you want the 8k step next

```bash
ls $PEAGLE_ROOT/configs/sol/*8k.json
```
