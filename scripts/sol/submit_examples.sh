#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cat <<EOF
Example submissions. Supply your cluster-specific flags such as --account and --partition.

1. Environment setup on a login node:
   bash ${ROOT}/scripts/sol/setup_env.sh

2. Qwen3 smoke test:
   sbatch --chdir=${ROOT} --gpus=1 --time=01:00:00 \\
     --export=ALL,PEAGLE_COMMAND=smoke,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_smoke.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm

3. BF16 extraction:
   sbatch --chdir=${ROOT} --gpus=1 --time=02:00:00 \\
     --export=ALL,PEAGLE_COMMAND=extract,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_extract_bf16_1k.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm

4. AWQ extraction:
   sbatch --chdir=${ROOT} --gpus=1 --time=02:00:00 \\
     --export=ALL,PEAGLE_COMMAND=extract,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_extract_awq_1k.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm

5. BF16-control training:
   sbatch --chdir=${ROOT} --gpus=1 --time=04:00:00 \\
     --export=ALL,PEAGLE_COMMAND=train,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_train_bf16_1k.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm

6. AWQ-aware training:
   sbatch --chdir=${ROOT} --gpus=1 --time=04:00:00 \\
     --export=ALL,PEAGLE_COMMAND=train,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_train_awq_1k.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm

7. Evaluations:
   sbatch --chdir=${ROOT} --gpus=1 --time=02:00:00 \\
     --export=ALL,PEAGLE_COMMAND=evaluate,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_eval_bf16_on_bf16.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm

   sbatch --chdir=${ROOT} --gpus=1 --time=02:00:00 \\
     --export=ALL,PEAGLE_COMMAND=evaluate,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_eval_bf16_on_awq.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm

   sbatch --chdir=${ROOT} --gpus=1 --time=02:00:00 \\
     --export=ALL,PEAGLE_COMMAND=evaluate,PEAGLE_CONFIG=${ROOT}/configs/sol/qwen3_8b_eval_awq_on_awq.json \\
     ${ROOT}/scripts/sol/run_peagle_job.slurm
EOF
