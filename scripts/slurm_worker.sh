#!/usr/bin/env bash
# One Slurm array task: run every n-th run of a .jsonl run list with PROCS persistent processes on this task's GPU.
#   sbatch ... --array=0-(n-1) scripts/slurm_worker.sh results/jobs/<exp>/stage1.jsonl 8
# Submitted by scripts/launch.py; the work is done by scripts/worker.py. Runs resume from finished frames.
set -uo pipefail
CMDS="$1"
PROCS="${2:-6}"
source /etc/profile.d/modules.sh 2>/dev/null && module load cuda/12.8 2>/dev/null   # runtime used to build pydiffvg
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

i="${SLURM_ARRAY_TASK_ID:-0}"
n="${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[worker] $(date '+%F %T') host $(hostname) task $i/$n gpu ${CUDA_VISIBLE_DEVICES:-?} procs $PROCS file $CMDS"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

REPO="${FLIPBOOK_REPO:?set by launch.py}"
ENV_BIN="${FLIPBOOK_ENV_BIN:-$REPO/../env/bin}"   # Python environment; default: env/ next to the repository
export PATH="$ENV_BIN:$PATH"
python "$REPO/scripts/worker.py" "$CMDS" --procs "$PROCS" --shard "$i" --nshards "$n"
rc=$?
echo "[worker] $(date '+%F %T') finished, exit $rc"
exit $rc
