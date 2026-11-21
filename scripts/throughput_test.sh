#!/usr/bin/env bash
# Throughput of CLIPasso on one GPU with 1, 4 and 8 concurrent processes (200 iterations on dog frame 1).
# The 1-process run is the isolated per-iteration time used for cost; the others set --procs for launch.py.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source /etc/profile.d/modules.sh 2>/dev/null && module load cuda/12.8 2>/dev/null
export PATH="${FLIPBOOK_ENV_BIN:-$REPO/../env/bin}:$PATH" OMP_NUM_THREADS=2
OUT="$REPO/results/timing"; rm -rf "$OUT"/p*; mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
for P in 1 4 8; do
  t0=$(date +%s.%N)
  for k in $(seq 1 $P); do
    python "$REPO/src/clipasso_runner.py" --target "$REPO/data/clips/dog/frames/01.png" --out_dir "$OUT/p${P}_$k" \
      --num_iter 200 --no_early_stop --seed $k > "$OUT/p${P}_$k.log" 2>&1 &
  done
  wait
  t1=$(date +%s.%N)
  python - "$OUT" "$P" "$t0" "$t1" <<'PY'
import json, sys, glob
out, P, t0, t1 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
s = [json.load(open(f)) for f in glob.glob(f"{out}/p{P}_*/summary.json")]
spi = sum(x["sec_per_iter_mean"] for x in s) / len(s)
init = sum(x["init_gpu_seconds"] for x in s) / len(s)
print(f"procs {P}: done {len(s)}/{P}, s/iter per process {spi:.4f}, init {init:.1f}s, wall {t1-t0:.0f}s, "
      f"throughput {P*200/(t1-t0):.1f} iter/s total")
json.dump({"procs": P, "sec_per_iter": spi, "init_s": init, "wall_s": t1 - t0, "n_ok": len(s)},
          open(f"{out}/throughput_p{P}.json", "w"))
PY
done
