"""Expand an experiment config into run_sequence.py commands and (optionally) submit them to Slurm.

  python scripts/launch.py configs/dev.yaml --dry_run
  python scripts/launch.py configs/dev.yaml --submit --gpus 2 --procs 6

Config (YAML): name, clips, seeds, n_frames, a_chunk (frames per process for method A, default 4), runs, and
optionally sweeps_seed0 (runs for the first seed only). Each run is a dict of run_sequence.py options; list values
expand as a grid, e.g. {method: C, motion: [zero, flow], lam: [0, 4.0e-4], K: 6} gives four runs.

Stage 0 computes method A's frame 0 for every (clip, seed), which every other method starts from; stage 1 runs
everything else and starts only after stage 0 succeeded. Runs are written to results/jobs/<name>/stage{0,1}.jsonl
(and as shell commands to stage{0,1}.txt); each Slurm array task takes every n-th run and executes them with
--procs persistent worker processes on its GPU (scripts/worker.py). Every command resumes
from its last finished frame, so resubmitting the same config is safe and skips finished work.
"""
import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from run_sequence import parse as rs_parse, run_name  # noqa: E402

PY = sys.executable


def expand(spec):
    keys = list(spec)
    vals = [v if isinstance(v, list) else [v] for v in spec.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def to_args(d):
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def commands(cfg, log_dir):
    st0, st1, names = [], [], set()
    for clip in cfg["clips"]:
        for seed in cfg["seeds"]:
            base = {"clip": clip, "seed": seed}
            if cfg.get("n_frames"):
                base["n_frames"] = cfg["n_frames"]
            args = to_args(base | {"method": "A", "frame0_only": True})
            st0.append((f"{clip}_A_frame0_s{seed}", args))
            specs = [(1, x) for x in cfg["runs"]]
            specs += [(2, x) for x in cfg.get("sweeps_seed0", [])] if seed == cfg["seeds"][0] else []
            for prio, spec in specs:
                for d in expand(spec):
                    args = to_args(base | d)
                    name = run_name(rs_parse(args))
                    names.add(name)
                    if d.get("method") == "A":  # frames are independent: split them over processes
                        T, ch = cfg.get("n_frames", 24), cfg.get("a_chunk", 4)
                        for lo in range(1, T, ch):
                            hi = min(lo + ch - 1, T - 1)
                            st1.append((f"{clip}_{name}_s{seed}_f{lo:02d}-{hi:02d}", args + ["--frames", f"{lo}-{hi}"], 0))
                    elif all(t != f"{clip}_{name}_s{seed}" for t, _, _ in st1):  # skip duplicates
                        st1.append((f"{clip}_{name}_s{seed}", args, prio))
    st1 = [x[:2] for x in sorted(st1, key=lambda x: x[2])]  # A first, then main runs, then sweeps
    line = lambda tag, args: (f"cd {shlex.quote(str(REPO))} && {shlex.quote(PY)} scripts/run_sequence.py "
                              + " ".join(map(shlex.quote, args)) + f" >> {shlex.quote(str(log_dir / tag))}.log 2>&1")
    task = lambda tag, args: json.dumps({"tag": tag, "args": args, "log": str(log_dir / f"{tag}.log")})
    return ([line(*x) for x in st0], [line(*x) for x in st1], sorted(names),
            [task(*x) for x in st0], [task(*x) for x in st1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--gpus", type=int, default=2, help="array tasks (one GPU each) for stage 1")
    p.add_argument("--procs", type=int, default=6, help="concurrent runs per GPU")
    p.add_argument("--partition", default="rtx6k-shocher")
    p.add_argument("--account", default="shocher_prj")
    p.add_argument("--qos", default="24h_4g")
    p.add_argument("--time", default="24:00:00")
    p.add_argument("--cpus", type=int, default=15)
    p.add_argument("--mem", default="96G")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    jdir = REPO / "results/jobs" / cfg["name"]
    log_dir = jdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    st0, st1, names, t0, t1 = commands(cfg, log_dir)
    (jdir / "stage0.txt").write_text("\n".join(st0) + "\n")          # human-readable / manual use
    (jdir / "stage1.txt").write_text("\n".join(st1) + ("\n" if st1 else ""))
    (jdir / "stage0.jsonl").write_text("\n".join(t0) + "\n")         # what the Slurm worker runs
    (jdir / "stage1.jsonl").write_text("\n".join(t1) + ("\n" if t1 else ""))
    print(f"[launch] {cfg['name']}: {len(st0)} stage-0 and {len(st1)} stage-1 commands, runs: {', '.join(names)}")
    print(f"[launch] command files in {jdir}")
    if a.dry_run or not a.submit:
        for c in st0 + st1:
            print("  " + c.split(" && ", 1)[1].split(" >> ")[0])
        return

    worker = REPO / "scripts/slurm_worker.sh"
    common = ["sbatch", "--parsable", "-N", "1", "-p", a.partition, "-A", a.account, "--qos", a.qos,
              "--gres=gpu:1", "-c", str(a.cpus), "--mem", a.mem, "-t", a.time,
              "-o", str(jdir / "slurm_%A_%a.out"), f"--export=ALL,FLIPBOOK_REPO={REPO}"]
    n0 = min(a.gpus, len(st0))
    j0 = subprocess.run(common + ["-J", f"fb0_{cfg['name']}", f"--array=0-{n0 - 1}", str(worker),
                                  str(jdir / "stage0.jsonl"), str(a.procs)],
                        check=True, capture_output=True, text=True).stdout.strip()
    if not st1:
        (jdir / "slurm_jobs.txt").write_text(f"stage0 {j0}\n")
        print(f"[launch] submitted stage 0 = {j0} ({n0} tasks); no stage-1 commands")
        return
    n1 = min(a.gpus, len(st1))
    j1 = subprocess.run(common + ["-J", f"fb1_{cfg['name']}", f"--array=0-{n1 - 1}", f"--dependency=afterok:{j0}",
                                  str(worker), str(jdir / "stage1.jsonl"), str(a.procs)],
                        check=True, capture_output=True, text=True).stdout.strip()
    (jdir / "slurm_jobs.txt").write_text(f"stage0 {j0}\nstage1 {j1}\n")
    print(f"[launch] submitted stage 0 = {j0} ({n0} tasks), stage 1 = {j1} ({n1} tasks, after stage 0)")


if __name__ == "__main__":
    main()
