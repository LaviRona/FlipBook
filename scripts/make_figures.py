"""Figures from results/eval/<clip>/ and results/summary/<exp>/summary.json.

  python scripts/make_figures.py --clips dog --summary results/summary/dev --out results/figures/dev \
      --show C_lam0.0004_K6 BL_lam0.0004_K6 --qual_clip dog --qual_seed 0

All quantities are relative to per-frame CLIPasso (run A) on the same clip and seed.
overview.png  flicker and LPIPS bars for the compared methods at their chosen settings (--chosen)
tradeoff.png  flicker vs LPIPS: lambda sweeps of flow propagation (C_*) and propagation without flow (BL_*) as
              curves, per-frame stopped early (A@iterations) as a cost curve, warm start (B, BF) as points
drift.png     LPIPS gap to per-frame, per frame and per frames since the last keyframe, for the --show runs
motion.png    flicker per frame pair vs motion size, for the --show runs
bias.png      E_warp measured with RAFT vs with GMFlow for every run: points below the diagonal are favoured by RAFT
strip.png, onion.png, side_by_side.gif   qualitative, one clip and seed
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.flow import backward_warp  # noqa: E402
from src.names import display  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"C": "#1f6feb", "BL": "#d9822b", "A": "#6e7781", "B": "#24292f", "BF": "#1f6feb"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def family(r):
    return "A" if r.startswith("A") else r.split("_")[0]


def lam_of(r):
    m = re.search(r"_lam([0-9.e-]+)_", r + "_")
    return float(m.group(1)) if m else None


def K_of(r):
    m = re.search(r"_K(\w+?)(_|$)", r)
    return m.group(1) if m else None


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def tradeoff(S, out, chosen=()):
    """chosen: runs to ring and label (the settings chosen on the development clip)."""
    R = S["runs"]
    fk = S.get("flicker_metric", "ewarp_eval")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    pt = lambda r: (R[r]["rel_A"]["lpips"]["mean"], R[r]["rel_A"][fk]["mean"])  # noqa: E731
    for fam in ("BL", "C"):
        name = "Flow propagation" if fam == "C" else "Propagation w/o flow"
        rs = sorted([r for r in R if family(r) == fam and K_of(r) == "6"], key=lam_of)
        if rs:  # the lambda sweep at K = 6 as a curve, labelled at its two ends
            x, y = zip(*map(pt, rs))
            ax.plot(x, y, "-o", color=COL[fam], ms=4, lw=1.2, label=f"{name}, λ sweep")
            for r in rs:
                ci = R[r]["rel_A"][fk]["ci"]
                if np.isfinite(ci[0]):
                    ax.plot([pt(r)[0]] * 2, ci, color=COL[fam], lw=0.6, alpha=0.35)
            for r, dy in ((rs[0], -9), (rs[-1], 4)):
                if r in chosen:
                    continue
                ax.annotate(f"λ={lam_of(r):g}", pt(r), fontsize=6, xytext=(4, dy), textcoords="offset points",
                            color=COL[fam])
        for r in sorted(r for r in R if fam == "C" and family(r) == fam and K_of(r) not in ("6", None)):  # K 12, never
            ax.plot(*pt(r), "o", color=COL[fam], ms=4, mfc="white")
            ax.annotate("K=∞" if K_of(r) == "inf" else f"K={K_of(r)}", pt(r), fontsize=6, xytext=(4, -3),
                        textcoords="offset points", color=COL[fam])
    for r in chosen:
        if r in R:
            ax.plot(*pt(r), "o", ms=11, mfc="none", mec=COL[family(r)], mew=1.2)
            ci = R[r]["rel_A"][fk]["ci"]
            ax.plot([pt(r)[0]] * 2, ci, color=COL[family(r)], lw=1.2)
            c = family(r) == "C"
            ax.annotate("proposed setting" if c else "w/o flow (λ=0)", pt(r), fontsize=7,
                        xytext=(0, 9) if c else (40, 22), textcoords="offset points", ha="center",
                        color=COL[family(r)], fontweight="bold",
                        arrowprops=None if c else dict(arrowstyle="-", color=COL[family(r)], lw=0.6))
    snaps = sorted([r for r in R if r.startswith("A@")], key=lambda r: int(r[2:]))
    if snaps:
        x = [R[r]["rel_A"]["lpips"]["mean"] for r in snaps] + [1.0]
        y = [R[r]["rel_A"][fk]["mean"] for r in snaps] + [1.0]
        ax.plot(x, y, "--s", color=COL["A"], ms=3, lw=1, label="Per-frame, stopped early (iterations)")
        for r, xi, yi in zip(snaps, x, y):
            ax.annotate(r[2:], (xi, yi), fontsize=6, xytext=(3, -7), textcoords="offset points", color=COL["A"])
    ax.plot([1], [1], "s", color=COL["A"], ms=6, label="Per-frame, 2000 iterations")
    if "B" in R:
        ax.plot([pt("B")[0]] * 2, R["B"]["rel_A"][fk]["ci"], color=COL["B"], lw=1.2)
        ax.plot(*pt("B"), "D", color=COL["B"], ms=6, label="Warm start")
    if "BF" in R:  # exploratory: warm start + flow only
        ax.plot([pt("BF")[0]] * 2, R["BF"]["rel_A"][fk]["ci"], color=COL["C"], lw=0.8, ls="--")
        ax.plot(*pt("BF"), "^", color=COL["C"], ms=7, mfc="white", label="Warm start + flow (exploratory)")
    ax.axvline(S["selection_limits"]["lpips"], color="#999", lw=0.6, ls=":")
    ax.set_xlabel("LPIPS relative to per-frame (lower = closer to the frame)")
    ax.set_ylabel({"chamfer_eval": "Flicker relative to per-frame (lower = smoother)",
                   "ewarp_eval": "Flicker (E_warp) relative to per-frame"}.get(fk, fk))
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.45 * (hi - lo))  # free space on the right for the legend
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=200)
    plt.close(fig)


def overview(S, out, chosen=()):
    """overview.png: flicker and LPIPS of the compared methods at their chosen settings, as horizontal bars with 95%
    intervals, relative to per-frame drawing (= 1)."""
    R = S["runs"]
    fk = S.get("flicker_metric", "chamfer_eval")
    C = next((r for r in chosen if family(r) == "C"), None)
    BL = next((r for r in chosen if family(r) == "BL"), None)
    rows = [("A", "Per-frame", "#8c959f"), ("B", "Warm start", "#24292f"), (BL, "Propagation w/o flow", "#d9822b"),
            (C, "Flow propagation (ours)", "#1f6feb"), ("BF", "Warm start + flow*", "#9ec5fe")]
    rows = [r for r in rows if r[0] in R]
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, key, title in ((axs[0], fk, "Flicker ↓"), (axs[1], "lpips", "LPIPS ↓")):
        for yi, (r, lab, col) in zip(y, rows):
            v = R[r]["rel_A"][key]
            lo, hi = v["ci"]
            err = None if not np.isfinite(lo) else [[v["mean"] - lo], [hi - v["mean"]]]
            ax.barh(yi, v["mean"], color=col, height=0.62, xerr=err, error_kw=dict(lw=0.8, capsize=2))
            ax.text(v["mean"] if err is None else hi, yi, f" {v['mean']:.2f}", va="center", fontsize=6.5)
        ax.axvline(1, color="#57606a", lw=0.7, ls="--")
        ax.set_title(title, fontsize=7.5, fontweight="bold")
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(0, 1.95 if key == "lpips" else 1.45)
        ax.set_xticks([0, 0.5, 1, 1.5])
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([lab for _, lab, _ in rows], fontsize=7)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "overview.png", dpi=250)
    plt.close(fig)


def drift(clips, eval_root, show, out):
    gap = defaultdict(lambda: defaultdict(list))
    by_ssa = defaultdict(lambda: defaultdict(list))
    for clip in clips:
        pf = read_csv(Path(eval_root) / clip / "per_frame.csv")
        a = defaultdict(list)
        for row in pf:
            if row["run"] == "A":
                a[int(row["frame"])].append(float(row["lpips"]))
        for row in pf:
            t = int(row["frame"])
            if row["run"] in show and t > 0:
                g = float(row["lpips"]) - np.mean(a[t])
                gap[row["run"]][t].append(g)
                by_ssa[row["run"]][int(row["steps_since_anchor"])].append(g)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for r in show:
        if r not in gap:
            continue
        c = COL.get(family(r), "k")
        ls = {"6": "-", "12": "--", "inf": ":"}.get(K_of(r) or "", "-")
        t = sorted(gap[r])
        axs[0].plot(t, [np.mean(gap[r][k]) for k in t], ls, color=c, lw=1.2, label=display(r))
        s = sorted(by_ssa[r])
        axs[1].plot(s, [np.mean(by_ssa[r][k]) for k in s], ls + "o", color=c, ms=3, lw=1.2, label=display(r))
    for ax, xl in zip(axs, ("frame", "frames since the last keyframe")):
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_xlabel(xl)
    axs[0].set_ylabel("LPIPS minus per-frame LPIPS")
    axs[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "drift.png", dpi=200)
    plt.close(fig)


def motion(clips, eval_root, show, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for r in show:
        xs, ys = [], []
        for clip in clips:
            pp = read_csv(Path(eval_root) / clip / "per_pair.csv")
            a = defaultdict(list)
            for row in pp:
                if row["run"] == "A":
                    a[int(row["t"])].append(float(row["ewarp_eval"]))
            for row in pp:
                if row["run"] == r:
                    xs.append(float(row["flow_mag_mean_px_fg"]))
                    ys.append(float(row["ewarp_eval"]) / np.mean(a[int(row["t"])]))
        if xs:
            c = COL.get(family(r), "k")
            ax.scatter(xs, ys, s=8, color=c, alpha=0.6, label=display(r))
            if len(xs) > 2:
                k, b = np.polyfit(xs, ys, 1)
                xx = np.linspace(min(xs), max(xs), 10)
                ax.plot(xx, k * xx + b, color=c, lw=1)
    ax.axhline(1, color="#999", lw=0.6)
    ax.set_xlabel("mean motion on the object (px per frame)")
    ax.set_ylabel("flicker relative to per-frame (per pair)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "motion.png", dpi=200)
    plt.close(fig)


def bias(S, out):
    R = S["runs"]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for r, v in R.items():
        ax.scatter(v["rel_A"]["ewarp_eval"]["mean"], v["rel_A"]["ewarp_raft"]["mean"], s=12,
                   color=COL.get(family(r), "k"))
    lo = min(min(v["rel_A"]["ewarp_eval"]["mean"], v["rel_A"]["ewarp_raft"]["mean"]) for v in R.values()) * 0.95
    ax.plot([lo, 1.05], [lo, 1.05], color="#999", lw=0.6)
    ax.set_xlabel("E_warp with GMFlow (rel. per-frame)")
    ax.set_ylabel("E_warp with RAFT (rel. per-frame)")
    fig.tight_layout()
    fig.savefig(out / "bias.png", dpi=200)
    plt.close(fig)


def load_gray(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0)


def qualitative(clip, seed, runs, runs_root, data_root, out, strip_frames, onion_pairs, size=150):
    data = Path(data_root) / clip
    LW = 160  # width of the label column
    root = Path(runs_root) / clip
    path = lambda r, t: root / r / f"seed_{seed}" / f"frame_{t:02d}" / "final.png"
    T = len(list((data / "frames").glob("*.png")))
    T = min([T] + [len(list((root / r / f"seed_{seed}").glob("frame_*"))) for r in runs])
    # strip
    rows = [("input", [Image.open(data / f"frames/{t:02d}.png") for t in strip_frames])]
    rows += [(r, [Image.open(path(r, t)) for t in strip_frames]) for r in runs]
    W = Image.new("RGB", (LW + len(strip_frames) * (size + 4), len(rows) * (size + 4) + 20), "white")
    d = ImageDraw.Draw(W)
    for j, t in enumerate(strip_frames):
        d.text((LW + j * (size + 4) + 4, 4), f"frame {t}", fill="black")
    for i, (lab, ims) in enumerate(rows):
        d.text((4, 20 + i * (size + 4) + size // 2), display(lab), fill="black")
        for j, im in enumerate(ims):
            W.paste(im.convert("RGB").resize((size, size)), (LW + j * (size + 4), 20 + i * (size + 4)))
    W.save(out / f"strip_{clip}_s{seed}.png")
    # onion skin with the evaluation flow: red = sketch t warped to t+1, black = sketch t+1
    flow = torch.from_numpy(np.load(data / "flow_eval_bwd.npy"))
    W = Image.new("RGB", (LW + len(onion_pairs) * (size + 4), len(runs) * (size + 4) + 22), "white")
    d = ImageDraw.Draw(W)
    for j, t in enumerate(onion_pairs):
        d.text((LW + j * (size + 4) + 4, 4), f"{t} warped (red) vs {t + 1}", fill="black")
    for i, r in enumerate(runs):
        d.text((4, 22 + i * (size + 4) + size // 2), display(r), fill="black")
        for j, t in enumerate(onion_pairs):
            w = backward_warp(load_gray(path(r, t))[None, None], flow[t:t + 1])[0, 0].numpy()
            s1 = load_gray(path(r, t + 1)).numpy()
            im = np.ones((*s1.shape, 3), np.float32)
            im[..., 1] -= (1 - w) * 0.85
            im[..., 2] -= (1 - w) * 0.85
            im *= s1[..., None]
            W.paste(Image.fromarray((im.clip(0, 1) * 255).astype(np.uint8)).resize((size, size)),
                    (LW + j * (size + 4), 22 + i * (size + 4)))
    W.save(out / f"onion_{clip}_s{seed}.png")
    # GIF, 12 fps
    frames = []
    for t in range(T):
        ims = [Image.open(data / f"frames/{t:02d}.png")] + [Image.open(path(r, t)) for r in runs]
        G = Image.new("RGB", (len(ims) * 228 + 4, 252), "white")
        dd = ImageDraw.Draw(G)
        for k, (im, lab) in enumerate(zip(ims, ["input"] + [display(r) for r in runs])):
            G.paste(im.convert("RGB").resize((224, 224)), (4 + k * 228, 24))
            dd.text((8 + k * 228, 6), lab, fill="black")
        frames.append(G)
    frames[0].save(out / f"side_by_side_{clip}_s{seed}.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / 12), loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--summary", required=True, help="folder with summary.json from aggregate.py")
    p.add_argument("--out", required=True)
    p.add_argument("--show", nargs="*", default=[], help="runs for drift, motion and qualitative figures")
    p.add_argument("--chosen", nargs="*", default=[], help="runs with the chosen settings (marked in tradeoff.png)")
    p.add_argument("--eval_root", default=str(REPO / "results/eval"))
    p.add_argument("--runs_root", default=str(REPO / "results/runs"))
    p.add_argument("--data_root", default=str(REPO / "data/clips"))
    p.add_argument("--qual_clip", default=None)
    p.add_argument("--qual_seed", type=int, default=0)
    p.add_argument("--strip_frames", nargs="+", type=int, default=[0, 4, 8, 12, 16, 20])
    p.add_argument("--onion_pairs", nargs="+", type=int, default=[2, 8, 14, 20])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = json.loads((Path(a.summary) / "summary.json").read_text())
    tradeoff(S, out, a.chosen)
    if a.chosen:
        overview(S, out, a.chosen)
    bias(S, out)
    if a.show:
        drift(a.clips, a.eval_root, a.show, out)
        motion(a.clips, a.eval_root, a.show, out)
    if a.qual_clip:
        qualitative(a.qual_clip, a.qual_seed, ["A"] + [r for r in a.show if r != "A"], a.runs_root, a.data_root,
                    out, a.strip_frames, a.onion_pairs)
    print(f"[figures] wrote {sorted(x.name for x in out.iterdir())}")


if __name__ == "__main__":
    main()
