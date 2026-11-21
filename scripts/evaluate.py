"""Evaluate every run of one clip: flicker, fidelity, cost, per frame and per frame pair.

  python scripts/evaluate.py --clip dog                       # all runs and seeds found under results/runs/dog
  python scripts/evaluate.py --clip dog --runs A B C_lam0.0004_K6 --seeds 0

Flicker (per pair t -> t+1, lower is smoother):
  ewarp_eval   mean L1 between sketch t+1 and sketch t warped by the GMFlow flow, on GMFlow-valid pixels (MAIN)
  ewarp_raft   the same with RAFT, the flow the methods use (biased toward C; reported to show the bias)
  chamfer_eval symmetric mean distance in px between the ink of warped sketch t and the ink of sketch t+1,
               measured from ink pixels that are GMFlow-valid; robust to 1 px shifts of thin lines
Fidelity (per frame, frames 1..T-1; frame 0 is shared by all methods):
  clip_q  cosine similarity of CLIP ViT-L/14 embeddings, sketch vs masked frame (not the loss model), higher better
  lpips   LPIPS (AlexNet), sketch vs masked frame, lower better; only meaningful relative to A
Cost: optimization iterations per frame and measured GPU seconds (noisy when jobs share a GPU; see aggregate.py).
A's snapshots become extra runs named "A@<iterations>" (per-frame CLIPasso stopped early), for the cost curve.

Outputs in results/eval/<clip>/: per_frame.csv, per_pair.csv, runs.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.flow import backward_warp  # noqa: E402

SHORT_PX = 15.0  # a stroke shorter than this (arc length, px) counts as a fragment
INK = 0.5        # a pixel is ink if darker than this


def load_gray(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0)


def warp_seq(sk, flow_bwd):
    """sk (T,H,W) -> W(S_t) predicted at t+1 for t = 0..T-2, shape (T-1,H,W)."""
    return backward_warp(sk[:-1, None], flow_bwd[: len(sk) - 1])[:, 0]


def ewarp_pairs(sk, warped, valid):
    m = valid[: len(sk) - 1].float()
    return (((sk[1:] - warped).abs() * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)).tolist()


def chamfer(a_img, b_img, valid):
    """Symmetric mean distance (px) between ink of a and ink of b, from ink pixels inside `valid`."""
    a, b = a_img < INK, b_img < INK
    a_from, b_from = a & valid, b & valid
    if not a_from.any() or not b_from.any() or not a.any() or not b.any():
        return float("nan")
    da = distance_transform_edt(~a)  # distance to nearest ink of a
    db = distance_transform_edt(~b)
    return float(0.5 * (db[a_from].mean() + da[b_from].mean()))


def chamfer_pairs(sk, warped, valid):
    s, w, v = sk.cpu().numpy(), warped.cpu().numpy(), valid.cpu().numpy()
    return [chamfer(w[t], s[t + 1], v[t]) for t in range(len(sk) - 1)]


def bezier_lengths(points_list, n=50):
    s = torch.linspace(0, 1, n)[:, None]
    out = []
    for p in points_list:
        p = p.float().cpu()
        c = ((1 - s) ** 3) * p[0] + 3 * ((1 - s) ** 2) * s * p[1] + 3 * (1 - s) * s ** 2 * p[2] + s ** 3 * p[3]
        out.append((c[1:] - c[:-1]).norm(dim=1).sum().item())
    return out


def steps_since_anchor(cfg, T):
    if cfg is None:  # A: every frame from scratch
        return [0] * T
    K = cfg.get("K", 0) if cfg.get("method") == "C" else 0
    out, last = [], 0
    for t in range(T):
        if t == 0 or (K and t % K == 0):
            last = t
        out.append(t - last)
    return out


def selftest(flow_bwd, valid_bwd, sk):
    """Blank pages give 0; a sequence built by warping with the same flow gives 0 E_warp and ~0 chamfer."""
    blank = torch.ones_like(sk)
    assert max(ewarp_pairs(blank, warp_seq(blank, flow_bwd), valid_bwd)) < 1e-6
    seq = [sk[0]]
    for t in range(len(sk) - 1):
        seq.append(backward_warp(seq[-1][None, None], flow_bwd[t:t + 1])[0, 0])
    seq = torch.stack(seq)
    w = warp_seq(seq, flow_bwd)
    e = max(ewarp_pairs(seq, w, valid_bwd))
    c = np.nanmax(chamfer_pairs(seq[:3], w[:2], valid_bwd[:2]))
    assert e < 1e-6 and c < 1e-6, (e, c)
    return True


def discover(clip_root):
    runs = {}
    for d in sorted(clip_root.iterdir()):
        if d.is_dir():
            seeds = sorted(int(s.name.split("_")[1]) for s in d.glob("seed_*") if (s / "DONE").exists())
            if seeds:
                runs[d.name] = seeds
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--data_root", default=str(REPO / "data/clips"))
    p.add_argument("--runs_root", default=str(REPO / "results/runs"))
    p.add_argument("--out_root", default=str(REPO / "results/eval"))
    p.add_argument("--no_snapshots", action="store_true")
    a = p.parse_args()

    data = Path(a.data_root) / a.clip
    clip_root = Path(a.runs_root) / a.clip
    out = Path(a.out_root) / a.clip
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((data / "meta.json").read_text())
    flows = {k: torch.from_numpy(np.load(data / f"flow{s}_bwd.npy")).to(dev) for k, s in (("raft", ""), ("eval", "_eval"))}
    valids = {k: torch.from_numpy(np.load(data / f"valid{s}_bwd.npy")).to(dev) for k, s in (("raft", ""), ("eval", "_eval"))}
    motion = {r["t"]: r for r in meta["per_pair_motion"]}

    found = discover(clip_root)
    runs = {r: found[r] for r in (a.runs or found) if r in found}
    if a.seeds is not None:
        runs = {r: [s for s in v if s in a.seeds] for r, v in runs.items()}
    if "A" not in runs:
        raise SystemExit("method A is the reference and must be evaluated too")

    import clip
    import lpips
    cm, pre = clip.load("ViT-L/14", device=dev)
    cm.eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    @torch.no_grad()
    def embed(paths):
        x = torch.stack([pre(Image.open(q).convert("RGB")) for q in paths]).to(dev)
        e = cm.encode_image(x).float()
        return e / e.norm(dim=-1, keepdim=True)

    def rgb(q):
        x = torch.from_numpy(np.asarray(Image.open(q).convert("RGB"), dtype=np.float32) / 255.0)
        return (x.permute(2, 0, 1)[None] * 2 - 1).to(dev)

    # Units: (run name, seed, list of PNG paths, list of point files, iterations, gpu seconds, cfg)
    units = []
    for r, seeds in runs.items():
        for s in seeds:
            sd = clip_root / r / f"seed_{s}"
            cfg = json.loads((sd / "config.json").read_text())
            T = cfg["frames"]
            fr = [sd / f"frame_{t:02d}" for t in range(T)]
            summ = [json.loads((f / "summary.json").read_text()) for f in fr]
            units.append((r, s, [f / "final.png" for f in fr], [f / "final_points.pt" for f in fr],
                          [x["iters_run"] for x in summ], [x["frame_gpu_seconds"] for x in summ],
                          None if r == "A" else cfg))
            if r == "A" and not a.no_snapshots:
                for it in sorted(int(k) for k in summ[0].get("snapshots", {})):
                    ok = all(str(it) in x.get("snapshots", {}) for x in summ)
                    if ok:
                        units.append((f"A@{it}", s, [f / f"snapshots/it{it:04d}.png" for f in fr],
                                      [f / f"snapshots/it{it:04d}_points.pt" for f in fr], [it] * T,
                                      [x["snapshots"][str(it)]["opt_gpu_seconds"] + x["init_gpu_seconds"] for x in summ],
                                      None))

    T_all = min(len(u[2]) for u in units)
    targets = [data / f"frames/{t:02d}.png" for t in range(T_all)]
    e_tgt = embed(targets)
    tgt_rgb = [rgb(q) for q in targets]
    sk_ref = torch.stack([load_gray(q) for q in units[0][2][:T_all]]).to(dev)
    selftest(flows["eval"], valids["eval"], sk_ref)
    print("[eval] self-test passed (blank pages = 0; flow-built sequence = 0)")

    per_frame, per_pair, summary = [], [], {}
    for r, s, pngs, pts, iters, gpu_s, cfg in units:
        pngs, pts, iters, gpu_s = pngs[:T_all], pts[:T_all], iters[:T_all], gpu_s[:T_all]
        T = len(pngs)
        sk = torch.stack([load_gray(q) for q in pngs]).to(dev)
        fl = {}
        for k in ("eval", "raft"):
            w = warp_seq(sk, flows[k])
            fl[f"ewarp_{k}"] = ewarp_pairs(sk, w, valids[k])
            if k == "eval":
                fl["chamfer_eval"] = chamfer_pairs(sk, w, valids[k])
        q = (embed(pngs) * e_tgt[:T]).sum(-1).cpu().numpy()
        with torch.no_grad():
            lpv = [lp(rgb(pngs[t]), tgt_rgb[t]).item() for t in range(T)]
        lens = [bezier_lengths(torch.load(f, weights_only=False)) for f in pts]
        short = [sum(1 for x in L if x < SHORT_PX) for L in lens]
        ink = (sk < INK).float().mean((1, 2)).cpu().numpy()
        ssa = steps_since_anchor(cfg, T)
        for t in range(T):
            per_frame.append({"clip": a.clip, "run": r, "seed": s, "frame": t, "steps_since_anchor": ssa[t],
                              "clip_q": float(q[t]), "lpips": lpv[t], "ink": float(ink[t]), "short_strokes": short[t],
                              "iters": iters[t], "gpu_s": gpu_s[t]})
        for t in range(T - 1):
            row = {"clip": a.clip, "run": r, "seed": s, "t": t}
            row.update({k: v[t] for k, v in fl.items()})
            row.update({k: v for k, v in motion[t].items() if k != "t"})
            per_pair.append(row)
        later = slice(1, T)
        summary.setdefault(r, {})[s] = {
            "frames": T,
            "ewarp_eval": float(np.mean(fl["ewarp_eval"])), "ewarp_raft": float(np.mean(fl["ewarp_raft"])),
            "chamfer_eval": float(np.nanmean(fl["chamfer_eval"])),
            "clip_q": float(np.mean(q[later])), "lpips": float(np.mean(lpv[later])),
            "ink": float(np.mean(ink)), "short_strokes": float(np.mean(short[later])),
            "iters_per_frame": float(np.mean(iters[later])), "gpu_s_per_frame": float(np.mean(gpu_s[later])),
        }
        print(f"[eval] {a.clip} {r} seed {s}: " + ", ".join(f"{k} {v:.4g}" for k, v in summary[r][s].items()))

    # A run-to-run noise: the same frame sketched by two seeds (a floor for how much A's pages differ anyway)
    a_seeds, seed_noise = runs.get("A", []), None
    if len(a_seeds) >= 2:
        s0, s1 = a_seeds[:2]
        g = lambda s, t: load_gray(clip_root / "A" / f"seed_{s}" / f"frame_{t:02d}" / "final.png").numpy()
        full = np.ones((g(s0, 0).shape), bool)
        seed_noise = {"seeds": [s0, s1], "chamfer_px": float(np.nanmean(
            [chamfer(g(s0, t), g(s1, t), full) for t in range(1, T_all)]))}

    for name, rows in (("per_frame", per_frame), ("per_pair", per_pair)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "runs.json").write_text(json.dumps({"clip": a.clip, "frames": T_all, "eval_flow": meta.get("eval_flow"),
                                               "A_seed_to_seed": seed_noise, "runs": summary}, indent=2))
    print(f"[eval] wrote {out}/per_frame.csv, per_pair.csv, runs.json")


if __name__ == "__main__":
    main()
