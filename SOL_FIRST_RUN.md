# Sol First Run

**Git repo**: `https://github.com/BenharJohn/smoke.git`
Clone path on Sol: `/home/$USER/smoke`
Push from your local machine, then pull on Sol with `bash scripts/sol/bootstrap_branch.sh YOUR_BRANCH`.

## 0. Set paths and persist to ~/.bashrc

Set all required env vars **and** append them to `~/.bashrc` so they survive across login
sessions and are forwarded to batch jobs via `--export=ALL`.

```bash
export PEAGLE_ROOT=/home/$USER/smoke          # adjust to your repo clone path
export PEAGLE_BF16_MODEL=Qwen/Qwen3-8B
export PEAGLE_AWQ_MODEL=Qwen/Qwen3-8B-AWQ
export PEAGLE_RUN_ROOT=/scratch/bjohn10/peagle-q/bjohn10   # MUST be hardcoded — $SCRATCH is unset when .bashrc is sourced by sbatch
export PEAGLE_BENCHMARK=$PEAGLE_ROOT/benchmarks/mini_mt_bench.jsonl
export PEAGLE_CACHE_ROOT=$PEAGLE_RUN_ROOT/cache

# Persist — jobs inherit these via --export=ALL
cat >> ~/.bashrc <<'BASHRC'
export PEAGLE_ROOT=/home/bjohn10/smoke
export PEAGLE_BF16_MODEL=Qwen/Qwen3-8B
export PEAGLE_AWQ_MODEL=Qwen/Qwen3-8B-AWQ
export PEAGLE_RUN_ROOT=/scratch/bjohn10/peagle-q/bjohn10
export PEAGLE_BENCHMARK=/home/bjohn10/smoke/benchmarks/mini_mt_bench.jsonl
export PEAGLE_CACHE_ROOT=/scratch/bjohn10/peagle-q/bjohn10/cache
BASHRC
```

> **Why this matters**: if `PEAGLE_ROOT` is empty when `sbatch` runs, the config path
> expands to `/configs/sol/...` and the job immediately fails with `FileNotFoundError`.
> Same for `PEAGLE_AWQ_MODEL` — the config loader rejects unresolved `${}` placeholders.

## 1. Build the environment

```bash
bash $PEAGLE_ROOT/scripts/sol/setup_env.sh
```

`setup_env.sh` automatically uninstalls `gptqmodel` after the pip install step.
`gptqmodel` 6.x is a transitive dependency of `autoawq` but crashes on Sol at import time:

```
RuntimeError: Tensor.item() cannot be called on meta tensors
```

This is caused by `exllamav3_torch.py` running scalar ops during module-level init,
which conflicts with Sol's `/etc/python/sitecustomize.py` wrapper. It is not needed
for AWQ inference (we use `AutoAWQForCausalLM.from_quantized()` directly).

## 2. Fix Python headers for Triton (one-time, login node)

`autoawq` 0.2.9 uses the Triton GEMM backend unconditionally. Triton JIT-compiles
CUDA utilities at first use and requires `Python.h`. Sol's system Python 3.11 is
installed **without** `python3.11-devel`, so compilation fails:

```
fatal error: Python.h: No such file or directory
```

### 2a. Download Python 3.11 headers and generate pyconfig.h

Run this **inside the venv** so `python` resolves to the venv's 3.11 interpreter and
the downloaded source version matches exactly.

```bash
source ~/smoke/.venv-sol/bin/activate
python_ver=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Fetching Python-${python_ver}"   # must print 3.11.x

mkdir -p ~/pyheaders/python3.11 /tmp/py_src_${python_ver}
wget -q "https://www.python.org/ftp/python/${python_ver}/Python-${python_ver}.tgz" \
  -O /tmp/py${python_ver}.tgz
tar xzf /tmp/py${python_ver}.tgz -C /tmp/py_src_${python_ver} --strip-components=1

# Generate pyconfig.h (must use the 3.11 source — 3.6 pyconfig.h causes
# "undefined symbol: PyObject_CallMethodNoArgs" at runtime)
cd /tmp/py_src_${python_ver}
./configure --quiet 2>/dev/null

# Copy the FULL Include tree (top-level headers + cpython/ subdir)
# Omitting cpython/ causes: fatal error: cpython/pymem.h: No such file or directory
cp -r Include/. ~/pyheaders/python3.11/
cp pyconfig.h ~/pyheaders/python3.11/
```

### 2c. Set CPATH and persist

```bash
export CPATH=$HOME/pyheaders/python3.11:${CPATH:-}

# Persist so batch jobs inherit it
echo "export CPATH=\$HOME/pyheaders/python3.11:\${CPATH:-}" >> ~/.bashrc
```

## 3. Pre-compile Triton CUDA utilities (login node)

Triton caches compiled `.so` files in `~/.triton/` on the shared filesystem.
Pre-compile once on the login node so compute nodes reuse the cache without
re-compiling (which would also fail if `CPATH` is not set in the job environment).

```bash
source $PEAGLE_VENV/bin/activate
python -c "from triton.backends.nvidia.driver import CudaUtils; CudaUtils(); print('done')"
```

Expected output: `done` (may take 30–60 s the first time).

## 4. Prepare training dataset

The extraction step requires a `.jsonl` file pointed to by `PEAGLE_DATASET`.
A 1 000-example ShareGPT slice is sufficient for the 1 k pilot:

```bash
pip install datasets  # already in the venv
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                  data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                  split="train")
import json, pathlib
out = pathlib.Path("$PEAGLE_RUN_ROOT/train_1k.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for row in ds.select(range(1000)):
        f.write(json.dumps(row) + "\n")
print(f"Wrote {out}")
PY

export PEAGLE_DATASET=$PEAGLE_RUN_ROOT/train_1k.jsonl
echo "export PEAGLE_DATASET=\$PEAGLE_RUN_ROOT/train_1k.jsonl" >> ~/.bashrc
```

> `PEAGLE_DATASET` must point to a **file**, not a directory. If it points to a
> directory the loader silently iterates nothing and extraction produces 0 rows.

## 5. Run smoke tests (BF16 and AWQ)

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

Check the job output for `"status": "ok"` to confirm success.

## 6. Submit the full 1k pilot chain

```bash
bash $PEAGLE_ROOT/scripts/sol/submit_1k_pilot.sh --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

## 7. Check outputs

```bash
ls $PEAGLE_RUN_ROOT/qwen3_8b
find $PEAGLE_RUN_ROOT/qwen3_8b -maxdepth 2 -type f | sort
```

## 8. Key result files

```bash
cat $PEAGLE_RUN_ROOT/qwen3_8b/eval_bf16_on_bf16.json
cat $PEAGLE_RUN_ROOT/qwen3_8b/eval_bf16_on_awq.json
cat $PEAGLE_RUN_ROOT/qwen3_8b/eval_awq_on_awq.json
```

### 1k Pilot Results (April 2026, 20-question MT-Bench mini, seed 42)

| Row | Drafter trained on | Verifier | τ | α[0] | α[1] | α[4] | theoretical speedup (τ+1) |
|-----|-------------------|----------|-------|------|------|------|--------------------------|
| (a) bf16_on_bf16 | BF16 teacher | BF16 | 1.648 | 0.663 | 0.395 | 0.113 | 2.648 |
| (b) bf16_on_awq  | BF16 teacher | AWQ W4A16 | 1.645 | 0.667 | 0.400 | 0.116 | 2.645 |
| (c) awq_on_awq   | AWQ teacher  | AWQ W4A16 | 1.627 | 0.666 | 0.392 | 0.114 | 2.627 |

**Key finding**: W4A16 (AWQ) is a null cell — the BF16-trained drafter loses only ~1.3% on τ against the AWQ verifier. QAT draft training does not improve on the BF16 control for this regime.

**Note**: The `speedup_vs_ar` values in the 1 k pilot result JSONs (0.26–0.33) come from the old evaluator that ran `draft_length` full verifier passes per speculative step. The eval loop now has three paths:

1. `speculation_mode: "sequential"` (default, and what the 1 k pilot configs use) — KV-cached, 2 verifier passes per speculative step. This is the matched-control path for comparing τ/α across regimes.
2. `speculation_mode: "parallel"` — draft proposes K tokens autoregressively via `Eagle3StyleDraftHead.propose_sequence`, verified in a single pass plus one tip-extension pass. This path produces realistic `tokens_per_second` and `speedup_vs_ar`.
3. `speculation_mode: "both"` — run each prompt through both paths in one job and emit a `by_mode` section in the summary. Recommended for the 8 k matrix.

For the 1 k pilot numbers shown above, use `theoretical_speedup_ideal` (= τ + 1) as the drafter-quality metric; the new 8 k eval configs (`qwen3_8b_eval_*_8k.json`) default to `speculation_mode: "both"` and will produce real speedup figures.

## 9. If you want the 8k step next

```bash
ls $PEAGLE_ROOT/configs/sol/*8k.json
```

---

## Known Sol-specific issues (quick reference)

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `FileNotFoundError: /configs/sol/...` | `PEAGLE_ROOT` not set | Export and add to `~/.bashrc` |
| `config contains unresolved environment variables` | `PEAGLE_AWQ_MODEL` not set | Export and add to `~/.bashrc` |
| `RuntimeError: Tensor.item() cannot be called on meta tensors` | `gptqmodel` 6.x import crash | `setup_env.sh` now uninstalls it automatically |
| `fatal error: Python.h: No such file or directory` | `python3.11-devel` absent | Download Python 3.11 source + `./configure` (§2) |
| `undefined symbol: PyObject_CallMethodNoArgs` | Wrong `pyconfig.h` (3.6 instead of 3.11) | Redo §2 with `python` from inside the venv; `rm -rf ~/.triton/cache/` then recompile |
| `fatal error: cpython/pymem.h: No such file or directory` | Only top-level `*.h` copied, missing `cpython/` subdir | Use `cp -r Include/. ~/pyheaders/python3.11/` (not `Include/*.h`) |
| Triton re-compiles on every job | `~/.triton/` cache missing | Pre-compile on login node (§3) |
| Extraction produces 0 rows | `PEAGLE_DATASET` points to a directory | Set it to the `.jsonl` file path |
