# Sol First Run

## 0. Set paths

```bash
export PEAGLE_ROOT=/path/to/this/repo
export PEAGLE_BF16_MODEL=Qwen/Qwen3-8B
export PEAGLE_AWQ_MODEL=/path/or/hf/id/of/your/qwen3-awq-checkpoint
export PEAGLE_DATASET=/path/to/train.jsonl
export PEAGLE_RUN_ROOT=$SCRATCH/peagle-q/$USER
export PEAGLE_BENCHMARK=$PEAGLE_ROOT/benchmarks/mini_mt_bench.jsonl
export PEAGLE_CACHE_ROOT=$PEAGLE_RUN_ROOT/cache
```

## 1. Build the environment

```bash
bash $PEAGLE_ROOT/scripts/sol/setup_env.sh
```

## 2. Run one smoke job

```bash
sbatch --chdir=$PEAGLE_ROOT --gpus=1 --time=01:00:00 \
  --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,PEAGLE_COMMAND=smoke,PEAGLE_CONFIG=$PEAGLE_ROOT/configs/sol/qwen3_8b_smoke.json \
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
