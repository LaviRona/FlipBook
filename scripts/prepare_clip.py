"""Prepare one DAVIS clip: frames, masks, RAFT flow (method), GMFlow flow (evaluation), occlusion masks, stats.

  python scripts/prepare_clip.py --zip data/davis/DAVIS-2017-trainval-480p.zip --clip blackswan

Outputs in data/clips/<clip>/ (gitignored):
  frames/NN.png        masked on white, SIZE x SIZE (the sketch targets)
  rgb/NN.png           same crop without mask (what RAFT sees)
  masks/NN.png         binary object mask
  flow_fwd.npy         (T-1, 2, S, S) float32, frame t -> t+1, pixels at SIZE
  flow_bwd.npy         (T-1, 2, S, S) float32, frame t+1 -> t
  valid_fwd.npy        (T-1, S, S) bool, non-occluded pixels of frame t   (for warping points of t)
  valid_bwd.npy        (T-1, S, S) bool, non-occluded pixels of frame t+1
  flow_eval_fwd.npy, flow_eval_bwd.npy, valid_eval_fwd.npy, valid_eval_bwd.npy
                       the same from GMFlow (src/eval_flow.py), used only by the evaluation (flicker metric)
  meta.json            source frames, crop box, stats, per-pair motion and occlusion, timings
and figures in results/clips/<clip>/: flow_viz.png, warp_check.png
Idempotent: skips if meta.json exists (use --force to recompute).
"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.flow import load_raft, raft_flow, backward_warp, fb_valid  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=str(REPO / "data/davis/DAVIS-2017-trainval-480p.zip"))
    p.add_argument("--clip", required=True)
    p.add_argument("--n_frames", type=int, default=24)
    p.add_argument("--stride", type=int, default=2, help="DAVIS is 24 fps; stride 2 gives 12 fps")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--flow_res", type=int, default=448)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--out_root", default=str(REPO / "data/clips"))
    p.add_argument("--fig_root", default=str(REPO / "results/clips"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_clip(zpath, clip):
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    jp = sorted(n for n in names if f"/JPEGImages/480p/{clip}/" in n and n.endswith(".jpg"))
    an = sorted(n for n in names if f"/Annotations/480p/{clip}/" in n and n.endswith(".png"))
    if not jp:
        clips = sorted({n.split("/")[3] for n in names if "/JPEGImages/480p/" in n and n.count("/") >= 4})
        raise SystemExit(f"clip '{clip}' not found. Available: {clips}")
    assert len(jp) == len(an), (len(jp), len(an))
    imgs = [np.array(Image.open(io.BytesIO(z.read(n))).convert("RGB")) for n in jp]
    anns = [np.array(Image.open(io.BytesIO(z.read(n)))) for n in an]  # palette indices
    return imgs, anns, [Path(n).name for n in jp]


def square_box(masks, margin, H, W):
    ys, xs = np.where(np.any(np.stack(masks), 0))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    side = min(side, H, W)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = int(np.clip(round(cx - side / 2), 0, W - side))
    by = int(np.clip(round(cy - side / 2), 0, H - side))
    return bx, by, side


def to_t(a, dev):
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255)[None].to(dev)


def resize(t, s, mode="bilinear"):
    return F.interpolate(t, size=(s, s), mode=mode, align_corners=False, antialias=(mode == "bilinear"))


def main():
    a = parse()
    out = Path(a.out_root) / a.clip
    fig_dir = Path(a.fig_root) / a.clip
    if (out / "meta.json").exists() and not a.force:
        print(f"[prepare] {out}/meta.json exists, skipping (use --force)")
        return
    for d in ("frames", "rgb", "masks"):
        (out / d).mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t_start = time.time()

    imgs, anns, names = read_clip(a.zip, a.clip)
    idx = [a.start + i * a.stride for i in range(a.n_frames)]
    assert idx[-1] < len(imgs), f"clip has {len(imgs)} frames, need index {idx[-1]}"
    obj_ids = sorted(set(np.unique(np.stack([anns[i] for i in idx]))) - {0, 255})
    masks = [(anns[i] > 0) & (anns[i] != 255) for i in idx]
    H, W = masks[0].shape
    bx, by, side = square_box(masks, a.margin, H, W)

    S, R = a.size, a.flow_res
    rgb_R, masked_S, rgb_S, mask_S = [], [], [], []
    for k, i in enumerate(idx):
        crop = imgs[i][by:by + side, bx:bx + side]
        m = masks[k][by:by + side, bx:bx + side].astype(np.float32)
        ct = to_t(crop, dev)
        mt = torch.from_numpy(m)[None, None].to(dev)
        rgb_R.append(resize(ct, R))
        cs, ms = resize(ct, S), (resize(mt, S) > 0.5).float()
        masked = cs * ms + (1 - ms)  # object on white
        rgb_S.append(cs), mask_S.append(ms), masked_S.append(masked)
        Image.fromarray((masked[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(out / f"frames/{k:02d}.png")
        Image.fromarray((cs[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(out / f"rgb/{k:02d}.png")
        Image.fromarray((ms[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(out / f"masks/{k:02d}.png")

    # RAFT at flow_res on the unmasked crop, then resize to SIZE and rescale vectors.
    model, tf = load_raft(dev)
    torch.cuda.synchronize(); t0 = time.time()
    fwd, bwd = [], []
    for t in range(a.n_frames - 1):
        f = raft_flow(model, tf, rgb_R[t], rgb_R[t + 1])
        b = raft_flow(model, tf, rgb_R[t + 1], rgb_R[t])
        fwd.append(resize(f, S) * (S / R)), bwd.append(resize(b, S) * (S / R))
    torch.cuda.synchronize(); flow_sec = time.time() - t0
    fwd, bwd = torch.cat(fwd), torch.cat(bwd)
    valid_fwd = fb_valid(fwd, bwd)   # frame t coordinates
    valid_bwd = fb_valid(bwd, fwd)   # frame t+1 coordinates
    np.save(out / "flow_fwd.npy", fwd.cpu().numpy().astype(np.float32))
    np.save(out / "flow_bwd.npy", bwd.cpu().numpy().astype(np.float32))
    np.save(out / "valid_fwd.npy", valid_fwd.cpu().numpy())
    np.save(out / "valid_bwd.npy", valid_bwd.cpu().numpy())
    del model

    # Independent flow for the evaluation only (GMFlow), same resolution and resizing as RAFT.
    from src.eval_flow import load_gmflow, gmflow_flow, NAME as EVAL_FLOW_NAME
    gm = load_gmflow(dev)
    torch.cuda.synchronize(); t0 = time.time()
    efwd, ebwd = [], []
    for t in range(a.n_frames - 1):
        f = gmflow_flow(gm, rgb_R[t], rgb_R[t + 1])
        b = gmflow_flow(gm, rgb_R[t + 1], rgb_R[t])
        efwd.append(resize(f, S) * (S / R)), ebwd.append(resize(b, S) * (S / R))
    torch.cuda.synchronize(); eval_flow_sec = time.time() - t0
    efwd, ebwd = torch.cat(efwd), torch.cat(ebwd)
    evalid_fwd, evalid_bwd = fb_valid(efwd, ebwd), fb_valid(ebwd, efwd)
    np.save(out / "flow_eval_fwd.npy", efwd.cpu().numpy().astype(np.float32))
    np.save(out / "flow_eval_bwd.npy", ebwd.cpu().numpy().astype(np.float32))
    np.save(out / "valid_eval_fwd.npy", evalid_fwd.cpu().numpy())
    np.save(out / "valid_eval_bwd.npy", evalid_bwd.cpu().numpy())
    del gm

    # Sanity check: predict frame t+1 from frame t, compare with no warp.
    rows = []
    for t in range(a.n_frames - 1):
        for name, seq in (("masked", masked_S), ("rgb", rgb_S)):
            pred = backward_warp(seq[t], bwd[t:t + 1])
            v = valid_bwd[t].float()
            l1_warp = ((pred - seq[t + 1]).abs().mean(1)[0] * v).sum() / v.sum()
            l1_none = ((seq[t] - seq[t + 1]).abs().mean(1)[0] * v).sum() / v.sum()
            rows.append({"t": t, "img": name, "l1_no_warp": l1_none.item(), "l1_warp": l1_warp.item()})
    fg = torch.cat(mask_S)[:-1, 0] > 0.5
    mag = fwd.norm(dim=1)
    # per pair: motion size, occlusion, and how much the two flow models agree (end-point error on the object)
    pair_stats = []
    for t in range(a.n_frames - 1):
        m = fg[t]
        pair_stats.append({
            "t": t,
            "flow_mag_mean_px_fg": mag[t][m].mean().item(),
            "flow_mag_p95_px_fg": mag[t][m].quantile(0.95).item(),
            "occluded_frac_fg": 1 - valid_fwd[t][m].float().mean().item(),
            "eval_flow_mag_mean_px_fg": efwd[t].norm(dim=0)[m].mean().item(),
            "eval_occluded_frac_fg": 1 - evalid_fwd[t][m].float().mean().item(),
            "raft_vs_eval_epe_px_fg": (fwd[t] - efwd[t]).norm(dim=0)[m].mean().item(),
        })
    stats = {
        "flow_mag_mean_px_fg": mag[fg].mean().item(),
        "flow_mag_p95_px_fg": mag[fg].quantile(0.95).item(),
        "flow_mag_mean_px_all": mag.mean().item(),
        "occluded_frac_fg_frame_t": 1 - valid_fwd[fg].float().mean().item(),
        "occluded_frac_all_frame_t1": 1 - valid_bwd.float().mean().item(),
        "fg_frac": torch.cat(mask_S).mean().item(),
    }
    for name in ("masked", "rgb"):
        r = [x for x in rows if x["img"] == name]
        stats[f"l1_{name}_no_warp_mean"] = float(np.mean([x["l1_no_warp"] for x in r]))
        stats[f"l1_{name}_warp_mean"] = float(np.mean([x["l1_warp"] for x in r]))

    make_figures(fig_dir, masked_S, fwd, bwd, valid_bwd, a)
    meta = {
        "clip": a.clip, "source_frames": [names[i] for i in idx], "source_total_frames": len(imgs),
        "stride": a.stride, "assumed_source_fps": 24, "fps": 24 / a.stride, "object_ids": [int(o) for o in obj_ids],
        "crop_box_xy_side_480p": [bx, by, side], "size": S, "flow_res": R,
        "raft": "torchvision raft_large, Raft_Large_Weights.DEFAULT",
        "occlusion_check": "Sundaram 2010: |f+b|^2 <= 0.01(|f|^2+|b|^2)+0.5, plus out-of-frame",
        "eval_flow": EVAL_FLOW_NAME, "eval_flow_gpu_seconds_total": eval_flow_sec,
        "stats": stats, "per_pair": rows, "per_pair_motion": pair_stats,
        "raft_gpu_seconds_total": flow_sec, "raft_gpu_seconds_per_pair_both_dirs": flow_sec / (a.n_frames - 1),
        "wall_seconds": time.time() - t_start,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (fig_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("clip", "object_ids", "crop_box_xy_side_480p", "raft_gpu_seconds_per_pair_both_dirs")}))
    print(json.dumps(stats, indent=1))


def make_figures(fig_dir, masked, fwd, bwd, valid_bwd, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import flow_to_image

    np_img = lambda t: t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    T = a.n_frames
    cols = list(range(0, T - 1, 2))  # every other pair
    fig, ax = plt.subplots(3, len(cols), figsize=(2.1 * len(cols), 6.6))
    fimg = flow_to_image(fwd).cpu()
    for j, t in enumerate(cols):
        ax[0, j].imshow(np_img(masked[t])); ax[0, j].set_title(f"frame {t}", fontsize=8)
        ax[1, j].imshow(fimg[t].permute(1, 2, 0).numpy()); ax[1, j].set_title(f"flow {t}->{t+1}", fontsize=8)
        ax[2, j].imshow(valid_bwd[t].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax[2, j].set_title(f"valid in {t+1} (white)", fontsize=8)
    for x in ax.ravel():
        x.axis("off")
    fig.suptitle(f"{a.clip}: RAFT forward flow and forward-backward valid mask", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "flow_viz.png", dpi=130); plt.close(fig)

    # warp check on the pair with the largest motion
    t = int(fwd.norm(dim=1).mean((1, 2)).argmax())
    pred = backward_warp(masked[t], bwd[t:t + 1])
    v = valid_bwd[t].cpu().numpy()
    d0 = (masked[t] - masked[t + 1]).abs().mean(1)[0].cpu().numpy()
    d1 = (pred - masked[t + 1]).abs().mean(1)[0].cpu().numpy() * v
    panels = [(np_img(masked[t]), f"frame {t}"), (np_img(masked[t + 1]), f"frame {t+1}"),
              (np_img(pred), f"frame {t} warped to {t+1}"),
              (d0, f"|{t+1} - {t}|  mean {d0[v].mean():.3f}"),
              (d1, f"|{t+1} - warped|, valid  mean {d1[v].mean():.3f}")]
    fig, ax = plt.subplots(1, 5, figsize=(12, 2.9))
    for x, (im, title) in zip(ax, panels):
        x.imshow(im, cmap="magma" if im.ndim == 2 else None, vmin=0, vmax=1); x.set_title(title, fontsize=8); x.axis("off")
    fig.suptitle(f"{a.clip}: warp sanity check (pair with largest motion)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "warp_check.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
