# Sol Branch Workflow

## 1. Move the Sol checkout onto your branch

```bash
bash scripts/sol/bootstrap_branch.sh YOUR_BRANCH
```

This does:

- `git fetch --prune`
- `git checkout` the branch if it exists locally
- or creates a tracking branch from `origin/YOUR_BRANCH`
- `git pull --ff-only`
- `bash scripts/sol/setup_env.sh`

## 2. Run one command directly on the branch

```bash
bash scripts/sol/run_from_branch.sh YOUR_BRANCH smoke configs/sol/qwen3_8b_smoke.json
```

Another example:

```bash
bash scripts/sol/run_from_branch.sh YOUR_BRANCH train configs/sol/qwen3_8b_train_bf16_1k.json
```

## 3. Submit one Slurm job from the branch

```bash
bash scripts/sol/submit_from_branch.sh \
  YOUR_BRANCH \
  smoke \
  configs/sol/qwen3_8b_smoke.json \
  origin \
  --account YOUR_ACCOUNT \
  --partition YOUR_PARTITION \
  --gpus=1 \
  --time=01:00:00
```

Train example:

```bash
bash scripts/sol/submit_from_branch.sh \
  YOUR_BRANCH \
  train \
  configs/sol/qwen3_8b_train_bf16_1k.json \
  origin \
  --account YOUR_ACCOUNT \
  --partition YOUR_PARTITION \
  --gpus=1 \
  --time=04:00:00
```

## 4. Submit the whole 1k pilot chain from the branch

```bash
bash scripts/sol/submit_1k_pilot_from_branch.sh \
  YOUR_BRANCH \
  origin \
  --account YOUR_ACCOUNT \
  --partition YOUR_PARTITION
```

## 5. Suggested habit

Use this pattern each time:

1. Push changes from your main machine to the branch.
2. On Sol, run `bootstrap_branch.sh YOUR_BRANCH`.
3. Run or submit jobs with `run_from_branch.sh`, `submit_from_branch.sh`, or `submit_1k_pilot_from_branch.sh`.

That way the Sol side stays dumb: pull latest branch, run config, collect outputs.
