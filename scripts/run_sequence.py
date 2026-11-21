"""Sketch one prepared clip frame by frame with one of the compared methods. Resumable per frame.

  python scripts/run_sequence.py --clip dog --method A --seed 0                                    # per-frame
  python scripts/run_sequence.py --clip dog --method B --seed 0                                    # warm start
  python scripts/run_sequence.py --clip dog --method C --motion zero --lam 4e-4 --K 6 --seed 0   # without flow
  python scripts/run_sequence.py --clip dog --method C --motion flow --lam 4e-4 --K 6 --seed 0   # flow propagation

Run names (folders) and the names used in the report:
A    per-frame: CLIPasso from scratch on every frame (saliency initialization, --first_iters iterations, snapshots
     for the cost curve).
B    warm start: frame t+1 starts from the final sketch of frame t as is and is refined for N iterations.
C_*  flow propagation: frame t+1 starts from frame t's control points moved by RAFT forward flow (--motion flow).
     Strokes with at least --reseed_min of 4 control points in occluded or off-canvas pixels are re-initialized
     from frame t+1's saliency map. N iterations with the regularizer lam * mean((theta - theta0)^2); every K frames
     a keyframe refinement of --anchor_iters iterations with --anchor_lam.
BL_* propagation without flow: the same code path with --motion zero (strokes not moved; the RAFT occlusion mask is
     still used for re-initialization), so comparing C_* with BL_* isolates the effect of the flow.
BF   warm start + flow (exploratory): C with lam 0, no re-initialization and no keyframes.

Layout: results/runs/<clip>/<run name>/seed_<s>/frame_XX/. Frame 0 of every method is A's frame 0 of the same seed
(a symlink), so differences come from frames 1..T-1. --frame0_only computes just that frame (launcher stage 0).
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from src.flow import warp_points  # noqa: E402

A_SNAPSHOTS = "150,300,500,750,1000,1500"


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--method", choices=["A", "B", "C"], required=True)
    p.add_argument("--motion", choices=["flow", "zero"], default="flow", help="C only: flow = flow propagation, zero = propagation without flow")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="run name; default derived from the settings (see run_name)")
    p.add_argument("--data_root", default=str(REPO / "data/clips"))
    p.add_argument("--runs_root", default=str(REPO / "results/runs"))
    p.add_argument("--n_frames", type=int, default=0, help="use only the first n frames (0 = all)")
    p.add_argument("--first_iters", type=int, default=2001)
    p.add_argument("--snapshots", default=A_SNAPSHOTS, help="A only: iterations at which to save a snapshot")
    p.add_argument("--N", type=int, default=150)
    p.add_argument("--K", type=int, default=0, help="re-anchor every K frames; 0 = never")
    p.add_argument("--anchor_iters", type=int, default=500)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--anchor_lam", type=float, default=0.0,
                   help="C: lambda on re-anchor frames; 0 lets the long refinement remove drift")
    p.add_argument("--reseed_min", type=int, default=2,
                   help="points warp: re-seed a stroke if at least this many of its 4 control points are occluded")
    p.add_argument("--warp", choices=["points", "curve"], default="points",
                   help="C: move the 4 control points by the flow (points), or move 8 points sampled on the curve "
                        "and refit the control points by least squares (curve; occlusion checked on the samples, "
                        "re-seed if at least half are occluded)")
    p.add_argument("--occluded_mode", choices=["replace", "keep"], default="replace",
                   help="C: flagged strokes are re-seeded from saliency (replace) or kept and left untied (keep)")
    p.add_argument("--reg_reseeded", action="store_true",
                   help="C: also apply the regularizer to re-initialized strokes (off by default)")
    p.add_argument("--frame0_only", action="store_true", help="only compute A's frame 0 for this seed")
    p.add_argument("--frames", default="", help="A only: frame range 'a-b' for this process (frames are independent)")
    p.add_argument("--subprocess", action="store_true",
                   help="one process per frame; by default all frames run in this process so the CLIP models load "
                        "once")
    return p.parse_args(argv)


def run_name(a):
    if a.name:
        return a.name
    if a.method == "A":
        return "A" if a.first_iters == 2001 else f"A_it{a.first_iters}"
    if a.method == "B":
        return "B" if a.N == 150 else f"B_N{a.N}"
    s = ("C" if a.motion == "flow" else "BL") + f"_lam{a.lam:g}_K{a.K or 'inf'}"
    if a.N != 150:
        s += f"_N{a.N}"
    if a.anchor_iters != 500 and a.K:
        s += f"_AI{a.anchor_iters}"
    if a.occluded_mode == "keep":
        s += "_keep"
    if a.reg_reseeded:
        s += "_regreseed"
    if a.warp == "curve":
        s += "_curve"
    return s


SUBPROCESS = False


def run_frame(target, out, iters, extra=()):
    if (out / "DONE").exists():
        return
    argv = ["--target", str(target), "--out_dir", str(out), "--num_iter", str(iters), "--ckpt_every", "100",
            "--no_early_stop"] + list(extra)
    print("[seq]", " ".join(argv), flush=True)
    if SUBPROCESS:
        subprocess.run([sys.executable, str(REPO / "src/clipasso_runner.py")] + argv, check=True)
    else:
        import clipasso_runner
        clipasso_runner.main(argv)


def atomic_write(path, text):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


_U = torch.linspace(0, 1, 8)[:, None]
BERN = torch.cat([(1 - _U) ** 3, 3 * (1 - _U) ** 2 * _U, 3 * (1 - _U) * _U ** 2, _U ** 3], 1)  # (8, 4)
BERN_PINV = torch.linalg.pinv(BERN)                                                             # (4, 8)


def flagged_and_moved(p, flow_t, valid_t, a):
    """Decide whether stroke p (4,2) is occluded in frame t, and where it moves in frame t+1."""
    S = flow_t.shape[-1]
    q = BERN @ p if a.warp == "curve" else p   # points that are checked (and moved)
    xi, yi = q[:, 0].round().long().clamp(0, S - 1), q[:, 1].round().long().clamp(0, S - 1)
    inside = (q[:, 0] >= 0) & (q[:, 0] <= S - 1) & (q[:, 1] >= 0) & (q[:, 1] <= S - 1)
    bad = int((~(valid_t[yi, xi] & inside)).sum())
    flagged = bad >= (len(q) / 2 if a.warp == "curve" else a.reseed_min)
    if a.motion == "zero":
        return flagged, p.clone()
    if a.warp == "curve":
        return flagged, BERN_PINV @ warp_points(q, flow_t)
    return flagged, warp_points(p, flow_t)


def is_anchor(t, K):
    return K > 0 and t % K == 0


def main(argv=None):
    global SUBPROCESS
    a = parse(argv)
    SUBPROCESS = a.subprocess
    data = Path(a.data_root) / a.clip
    meta = json.loads((data / "meta.json").read_text())
    T = len(meta["source_frames"]) if a.n_frames == 0 else min(a.n_frames, len(meta["source_frames"]))
    root = Path(a.runs_root) / a.clip
    name = run_name(a)
    out_m = root / name / f"seed_{a.seed}"
    frame = lambda t: data / f"frames/{t:02d}.png"
    # frame 0 comes from the reference A run; an A variant (e.g. A_it500) draws its own frame 0
    a_name = name if a.method == "A" else "A"
    a_dir = lambda t: root / a_name / f"seed_{a.seed}" / f"frame_{t:02d}"
    seed_args = ["--seed", str(a.seed)]
    snap_args = ["--snapshots", a.snapshots] if a.snapshots else []

    # frame 0: always method A's sketch for this seed
    run_frame(frame(0), a_dir(0), a.first_iters, seed_args + snap_args)
    if a.frame0_only:
        return
    out_m.mkdir(parents=True, exist_ok=True)
    if a.method != "A":
        link = out_m / "frame_00"
        if not link.is_symlink() and not link.exists():
            link.symlink_to(Path("..") / ".." / a_name / f"seed_{a.seed}" / "frame_00")

    flow_fwd = valid_fwd = None
    if a.method == "C":
        flow_fwd = torch.from_numpy(np.load(data / "flow_fwd.npy"))
        valid_fwd = torch.from_numpy(np.load(data / "valid_fwd.npy"))
    # Propagation without flow moves nothing but still uses RAFT's occlusion mask for re-initialization.
    raft_sec = meta["raft_gpu_seconds_per_pair_both_dirs"]

    reseed_log = {}
    if a.method == "A" and a.frames:
        lo, hi = (int(x) for x in a.frames.split("-"))
        todo = range(max(lo, 1), min(hi, T - 1) + 1)
    else:
        todo = range(1, T)
    for t in todo:
        out = out_m / f"frame_{t:02d}"
        prev = out_m / f"frame_{t - 1:02d}" / "final_points.pt"
        if a.method == "A":
            run_frame(frame(t), out, a.first_iters, seed_args + snap_args)
            continue
        if a.method == "B":
            run_frame(frame(t), out, a.N, seed_args + ["--init_pt", str(prev)])
            continue
        # flow propagation (motion = flow) or propagation without flow (motion = zero)
        anchor = is_anchor(t, a.K)
        iters = a.anchor_iters if anchor else a.N
        out.mkdir(parents=True, exist_ok=True)
        init_path = out / "warped_init.pt"
        pts_prev = torch.load(prev, map_location="cpu", weights_only=False)
        moved, reseed = [], []
        for i, p in enumerate(pts_prev):
            flagged, m = flagged_and_moved(p.float(), flow_fwd[t - 1], valid_fwd[t - 1], a)
            if flagged:
                reseed.append(i)
            moved.append(m)
        if not (out / "DONE").exists():
            torch.save(moved, init_path)
        reseed_log[t] = reseed
        flagged = ",".join(map(str, reseed))
        extra = seed_args + ["--init_pt", str(init_path),
                             "--reseed" if a.occluded_mode == "replace" else "--no_reg_strokes", flagged,
                             "--lam", str(a.anchor_lam if anchor else a.lam),
                             "--extra_gpu_seconds", f"{raft_sec:.6f}"]
        if not a.reg_reseeded:
            extra.append("--no_reg_reseeded")
        run_frame(frame(t), out, iters, extra)

    # per-frame table (frame 0 is shared with A); for A split over processes, only once every frame is done
    if not all((out_m / f"frame_{t:02d}" / "DONE").exists() for t in range(T)):
        print(f"[seq] {a.clip}/{name}/seed_{a.seed}: frames {a.frames or 'all'} done, others still running")
        return
    rows = []
    for t in range(T):
        s = json.loads((out_m / f"frame_{t:02d}" / "summary.json").read_text())
        rows.append({"frame": t, "iters": s["iters_run"], "opt_gpu_s": round(s["gpu_seconds_total"], 3),
                     "init_gpu_s": round(s["init_gpu_seconds"], 3), "extra_gpu_s": round(s["extra_gpu_seconds"], 3),
                     "frame_gpu_s": round(s["frame_gpu_seconds"], 3), "final_eval_loss": s["final_eval_loss"],
                     "anchor": a.method == "C" and is_anchor(t, a.K),
                     "reseeded": len(reseed_log.get(t, [])) if a.method == "C" else "",
                     "shared_with_A": t == 0 and a.method != "A"})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    atomic_write(out_m / "timing.csv", buf.getvalue())
    cfg = {k: v for k, v in vars(a).items() if k not in ("frames", "subprocess")}
    cfg |= {"run_name": name, "frames": T, "reseed_per_frame": reseed_log}
    atomic_write(out_m / "config.json", json.dumps(cfg, indent=2))
    atomic_write(out_m / "DONE", "ok\n")
    later = [r["frame_gpu_s"] for r in rows[1:]]
    print(f"[seq] {a.clip}/{name}/seed_{a.seed} done: mean frame GPU s (frames 1..{T - 1}) = {np.mean(later):.2f}")


if __name__ == "__main__":
    main()
