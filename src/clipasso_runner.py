"""Resumable CLIPasso runner.

Runs the same optimization loop as CLIPasso's painterly_rendering.main(), using
CLIPasso's own Loss, Painter and PainterOptimizer, and adds:
  * an atomic checkpoint every --ckpt_every iterations (points, opacities,
    Adam state, RNG states, loss history, timings), and automatic resume;
  * per-iteration GPU timing;
  * outputs: final.svg, best.svg, final.png, loss_curve.png, losses.csv,
    summary.json, and a DONE marker.

Usage:
  python src/clipasso_runner.py --target IMG --out_dir DIR [--num_strokes 16]
      [--num_iter 2001] [--seed 0] [--mask_object 0] [--ckpt_every 100]
"""
import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CLIPASSO = REPO / "third_party" / "CLIPasso"


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_strokes", type=int, default=16)
    p.add_argument("--num_iter", type=int, default=2001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask_object", type=int, default=0)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--no_early_stop", action="store_true",
                   help="disable CLIPasso's eval-loss plateau stop")
    p.add_argument("--init_pt", default=None,
                   help="torch file with a list of per-stroke (4,2) points to initialize from (methods B, C)")
    p.add_argument("--reseed", default="",
                   help="comma-separated stroke indices that keep the fresh saliency init instead of --init_pt")
    p.add_argument("--lam", type=float, default=0.0,
                   help="weight of lam * mean((theta - theta0)^2), theta0 = initialization (method C)")
    p.add_argument("--no_reg_strokes", default="",
                   help="comma-separated stroke indices excluded from the regularizer (kept in place, not re-seeded)")
    p.add_argument("--no_reg_reseeded", action="store_true",
                   help="exclude re-seeded strokes from the regularizer (they have no flow prediction)")
    p.add_argument("--extra_gpu_seconds", type=float, default=0.0,
                   help="added to the reported per-frame GPU time (e.g. RAFT flow for this frame)")
    p.add_argument("--snapshots", default="",
                   help="comma-separated iteration counts at which to save points and a PNG (cost/quality curve)")
    return p.parse_args(argv)


def clipasso_args(a, out_dir):
    """Build CLIPasso's own args object exactly as run_object_sketching.py does."""
    import config  # CLIPasso's config.py
    argv = [
        "painterly_rendering.py", str(Path(a.target).resolve()),
        "--num_paths", str(a.num_strokes),
        "--output_dir", str(out_dir.parent),
        "--wandb_name", out_dir.name,
        "--num_iter", str(a.num_iter),
        "--save_interval", "10",
        "--seed", str(a.seed),
        "--use_gpu", "1",
        "--fix_scale", "0",
        "--mask_object", str(a.mask_object),
        "--mask_object_attention", str(a.mask_object),
        "--display_logs", "0",
        "--display", "0",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        return config.parse_arguments()
    finally:
        sys.argv = old


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class _MemoClip:
    """Stands in for CLIP in CLIPasso's painter: the saliency model is loaded once per process instead of twice per
    frame. Each load leaked about 0.2 GB of GPU memory: the attention gradient hooks reference the model through the
    autograd graph, a cycle Python's garbage collector cannot see. With one model, each frame's attention tensors
    replace the previous frame's, which frees them. The saliency code calls model.zero_grad() itself."""

    def __init__(self, module):
        self._m, self._cache = module, {}

    def __getattr__(self, k):
        return getattr(self._m, k)

    def load(self, name, device="cuda", jit=False, **kw):
        key = (name, str(device), jit)
        if key not in self._cache:
            self._cache[key] = self._m.load(name, device=device, jit=jit, **kw)
        return self._cache[key]


# Models that do not depend on the frame (CLIP loss models), kept across frames when frames run in one process.
# Loss holds no per-frame state: its only counter feeds commented-out plotting code.
_CACHE = {}


def get_loss(Loss, args):
    key = repr((args.clip_model_name, args.clip_conv_layer_weights, args.clip_fc_loss_weight, args.clip_conv_loss,
           args.num_aug_clip, args.device))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = Loss(args)
    return _CACHE[key]


def main(argv=None):
    a = parse(argv)
    a.target = str(Path(a.target).resolve())  # resolve before chdir below
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        print(f"[runner] {out_dir} already DONE, nothing to do")
        return

    # CLIPasso uses paths relative to its own folder (U2Net weights).
    sys.path.insert(0, str(CLIPASSO))
    os.chdir(CLIPASSO)
    import sketch_utils as utils
    from models.loss import Loss
    from models.painter_params import Painter, PainterOptimizer
    import models.painter_params as painter_params
    if not isinstance(painter_params.clip, _MemoClip):
        painter_params.clip = _MemoClip(painter_params.clip)
    from painterly_rendering import get_target

    args = clipasso_args(a, out_dir)
    loss_func = get_loss(Loss, args)
    torch.cuda.synchronize()
    t_init = time.time()
    inputs, mask = get_target(args)
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments, imsize=args.image_scale,
                       device=args.device, target_im=inputs, mask=mask).to(args.device)
    optimizer = PainterOptimizer(args, renderer)
    renderer.set_random_noise(0)
    renderer.init_image(stage=0)
    optimizer.init_optimizers()

    # Methods B/C: replace the saliency init by given points, except re-seeded strokes.
    reseed = [int(i) for i in a.reseed.split(",") if i.strip() != ""]
    theta0_path = out_dir / "theta0.pt"
    if a.init_pt and not theta0_path.exists():
        init_pts = torch.load(a.init_pt, map_location="cpu", weights_only=False)
        assert len(init_pts) == len(renderer.shapes), (len(init_pts), len(renderer.shapes))
        with torch.no_grad():
            for i, path in enumerate(renderer.shapes):
                if i not in reseed:
                    path.points.copy_(init_pts[i])
    if theta0_path.exists():  # resume: use the saved initialization
        theta0 = torch.load(theta0_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for path, pts in zip(renderer.shapes, theta0):
                path.points.copy_(pts)
    else:
        theta0 = [p.points.detach().cpu().clone() for p in renderer.shapes]
        torch.save(theta0, theta0_path)
    theta0_cat = torch.cat([t.reshape(-1) for t in theta0]).to(args.device)
    # per-coordinate weight of the regularizer; normalization stays 1/n over all coordinates
    free = {int(i) for i in a.no_reg_strokes.split(",") if i.strip() != ""}
    reg_w = torch.cat([torch.full((t.numel(),), 0.0 if ((a.no_reg_reseeded and i in reseed) or i in free) else 1.0)
                       for i, t in enumerate(theta0)]).to(args.device)
    torch.cuda.synchronize()
    init_seconds = time.time() - t_init

    st = {"epoch": 0, "counter": 0, "best_loss": 100.0, "best_iter": 0,
          "terminate": False, "train_loss": [], "iter_time": [],
          "eval": {"iter": [], "loss_eval": []}, "resumed_from": [],
          "init_seconds": init_seconds, "reg_loss": [], "snapshots": {}}
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        # load to CPU: RNG states must stay CPU ByteTensors; copy_() and
        # Optimizer.load_state_dict() move tensors to the parameters' device.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for path, pts in zip(renderer.shapes, ck["points"]):
                path.points.copy_(pts)
            for grp, col in zip(renderer.shape_groups, ck["colors"]):
                grp.stroke_color.copy_(col)
        optimizer.points_optim.load_state_dict(ck["points_optim"])
        if optimizer.optim_color and ck.get("color_optim"):
            optimizer.color_optim.load_state_dict(ck["color_optim"])
        st = ck["state"]
        st.setdefault("snapshots", {})
        st["resumed_from"].append(st["epoch"])
        set_rng_state(ck["rng"])
        print(f"[runner] resumed from checkpoint at iteration {st['epoch']}")
    else:
        renderer.save_svg(str(out_dir), "init")

    def checkpoint():
        atomic_save({
            "points": [p.points.detach().clone() for p in renderer.shapes],
            "colors": [g.stroke_color.detach().clone() for g in renderer.shape_groups],
            "points_optim": optimizer.points_optim.state_dict(),
            "color_optim": optimizer.color_optim.state_dict() if optimizer.optim_color else None,
            "rng": rng_state(), "state": st,
        }, ckpt_path)

    snap_iters = {int(x) for x in a.snapshots.split(",") if x.strip()}

    def snapshot(it):
        """Points and a PNG after `it` optimization steps, plus the GPU time spent so far."""
        from PIL import Image
        d = out_dir / "snapshots"
        d.mkdir(exist_ok=True)
        torch.save([p.points.detach().cpu().clone() for p in renderer.shapes], d / f"it{it:04d}_points.pt")
        with torch.no_grad():
            img = renderer.get_image().detach().cpu()[0].permute(1, 2, 0).clamp(0, 1).numpy()
        Image.fromarray((img * 255).astype(np.uint8)).save(d / f"it{it:04d}.png")
        st["snapshots"][str(it)] = {
            "opt_gpu_seconds": float(np.sum(st["iter_time"])),
            "eval_loss": st["eval"]["loss_eval"][-1] if st["eval"]["loss_eval"] else None}

    min_delta = 1e-5
    early_stopped = False
    while st["epoch"] < args.num_iter:
        epoch = st["epoch"]
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(st["counter"])
        torch.cuda.synchronize()
        t0 = time.time()
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)
        losses = loss_func(sketches, inputs.detach(), renderer.get_color_parameters(),
                           renderer, st["counter"], optimizer)
        loss = sum(list(losses.values()))
        if a.lam > 0:
            theta = torch.cat([p.points.reshape(-1) for p in renderer.shapes])
            reg = a.lam * (reg_w * (theta - theta0_cat) ** 2).mean()
            st["reg_loss"].append(reg.item())
            loss = loss + reg
        loss.backward()
        optimizer.step_()
        torch.cuda.synchronize()
        st["iter_time"].append(time.time() - t0)
        st["train_loss"].append(loss.item())

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                ev = loss_func(sketches, inputs, renderer.get_color_parameters(),
                               renderer.get_points_parans(), st["counter"], optimizer, mode="eval")
                loss_eval = sum(list(ev.values())).item()
            st["eval"]["iter"].append(epoch)
            st["eval"]["loss_eval"].append(loss_eval)
            for k, v in ev.items():
                st["eval"].setdefault(k, []).append(v.item())
            delta = loss_eval - st["best_loss"]
            if abs(delta) > min_delta and delta < 0:
                st["best_loss"], st["best_iter"], st["terminate"] = loss_eval, epoch, False
                renderer.save_svg(str(out_dir), "best")
            if abs(delta) <= min_delta and not a.no_early_stop:
                if st["terminate"]:
                    early_stopped = True
                    st["epoch"] += 1
                    break
                st["terminate"] = True

        if epoch == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs,
                             renderer.get_inds(), 0, str(out_dir / "attention_map.jpg"),
                             args.saliency_model, 0)

        st["counter"] += 1
        st["epoch"] += 1
        if st["epoch"] in snap_iters:
            snapshot(st["epoch"])
        if st["epoch"] % a.ckpt_every == 0:
            checkpoint()
            print(f"[runner] iter {st['epoch']}/{args.num_iter} loss {loss.item():.4f} "
                  f"mean_it {np.mean(st['iter_time']):.4f}s (checkpoint saved)", flush=True)

    checkpoint()
    torch.save([p.points.detach().cpu().clone() for p in renderer.shapes], out_dir / "final_points.pt")
    finish(out_dir, renderer, inputs, st, args, early_stopped, a)
    # CLIPasso's saliency step loads a CLIP model per frame and its attention hooks create reference cycles, which
    # Python frees only on a garbage-collection pass; without this, long-lived worker processes run out of GPU memory.
    del renderer, optimizer, inputs, mask
    gc.collect()
    torch.cuda.empty_cache()


def finish(out_dir, renderer, inputs, st, args, early_stopped, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    renderer.save_svg(str(out_dir), "final")
    with torch.no_grad():
        img = renderer.get_image().detach().cpu()[0].permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(out_dir / "final.png")
    tgt = inputs.detach().cpu()[0].permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((tgt * 255).astype(np.uint8)).save(out_dir / "target.png")

    with open(out_dir / "losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = [k for k in st["eval"] if k != "iter"]
        w.writerow(["iter"] + keys)
        for i, it in enumerate(st["eval"]["iter"]):
            w.writerow([it] + [st["eval"][k][i] for k in keys])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(st["train_loss"], lw=0.6, alpha=0.5, label="train loss (with augmentations)")
    ax.plot(st["eval"]["iter"], st["eval"]["loss_eval"], lw=1.5, label="eval loss")
    ax.axvline(st["best_iter"], ls="--", lw=0.8, color="gray", label=f"best iter {st['best_iter']}")
    ax.set_xlabel("iteration"), ax.set_ylabel("loss"), ax.legend(frameon=False)
    ax.set_title("CLIPasso loss")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)

    t = np.array(st["iter_time"])
    summary = {
        "target": str(args.target), "num_strokes": args.num_paths, "seed": args.seed,
        "num_iter_requested": args.num_iter, "iters_run": int(st["epoch"]),
        "early_stopped": early_stopped, "best_iter": st["best_iter"],
        "best_eval_loss": st["best_loss"],
        "final_eval_loss": st["eval"]["loss_eval"][-1] if st["eval"]["loss_eval"] else None,
        "gpu_seconds_total": float(t.sum()),
        "sec_per_iter_mean": float(t.mean()),
        "sec_per_iter_median": float(np.median(t)),
        "resumed_from_iters": st["resumed_from"],
        "init_gpu_seconds": st["init_seconds"],
        "extra_gpu_seconds": a.extra_gpu_seconds,
        "frame_gpu_seconds": float(t.sum()) + st["init_seconds"] + a.extra_gpu_seconds,
        "init_pt": a.init_pt, "reseed": a.reseed, "lam": a.lam, "snapshots": st.get("snapshots", {}),
        "final_reg_loss": st["reg_loss"][-1] if st["reg_loss"] else None,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "DONE").write_text("ok\n")
    print("[runner] DONE", json.dumps(summary))


if __name__ == "__main__":
    main()
