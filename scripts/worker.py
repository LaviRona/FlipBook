"""Run a shard of launch.py's run list with persistent worker processes on one GPU.

  python scripts/worker.py results/jobs/<exp>/stage1.jsonl --procs 8 [--shard i --nshards n]

Why: starting a fresh Python process per run costs minutes on the cluster (importing torch and loading the CLIP
models from the network home directory while other processes do the same). Here each of the --procs processes
imports once, keeps the CLIP loss models cached (src/clipasso_runner.py), and runs its runs one after another.
Each run's output goes to its own log file. Exit code 1 if any run failed; rerunning resumes from finished frames.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _init():
    time.sleep(random.uniform(0, 20))  # stagger the heavy imports a little
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    import run_sequence  # noqa: F401  (imports torch once per process)


def _run(task):
    import run_sequence
    log = Path(task["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    saved = os.dup(1), os.dup(2)
    with open(log, "a") as f:
        sys.stdout.flush(), sys.stderr.flush()
        os.dup2(f.fileno(), 1), os.dup2(f.fileno(), 2)
        ok = True
        try:
            print(f"=== start {time.strftime('%F %T')} pid {os.getpid()}", flush=True)
            run_sequence.main(task["args"])
        except BaseException:  # noqa: BLE001  (log everything, keep the worker alive)
            ok = False
            traceback.print_exc()
        finally:
            print(f"=== end {time.strftime('%F %T')} ok={ok} wall {time.time() - t0:.0f}s", flush=True)
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(saved[0], 1), os.dup2(saved[1], 2)
            os.close(saved[0]), os.close(saved[1])
    return task["tag"], ok, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--runs_per_process", type=int, default=3)
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    a = p.parse_args()
    tasks = [json.loads(x) for x in Path(a.tasks).read_text().splitlines() if x.strip()]
    mine = [t for i, t in enumerate(tasks) if i % a.nshards == a.shard]
    print(f"[worker.py] {len(mine)} of {len(tasks)} runs, {a.procs} processes", flush=True)
    if not mine:
        return
    ctx = mp.get_context("spawn")
    failed = []
    # maxtasksperchild: recycle each process after a few runs as a safety net against GPU memory growth
    with ctx.Pool(min(a.procs, len(mine)), initializer=_init, maxtasksperchild=a.runs_per_process) as pool:
        for tag, ok, sec in pool.imap_unordered(_run, mine):
            print(f"[worker.py] {time.strftime('%T')} {'ok  ' if ok else 'FAIL'} {tag} ({sec:.0f}s)", flush=True)
            if not ok:
                failed.append(tag)
    print(f"[worker.py] done, {len(failed)} failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
