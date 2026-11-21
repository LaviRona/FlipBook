"""Labelled comparison GIFs: the video next to the main methods, with this clip's numbers under each drawing.

  python scripts/make_gifs.py --clips blackswan camel horsejump-high dog --seed 0

Writes results/figures/gifs/<clip>_s<seed>.gif (12 fps, the flipbook speed) and a slowed copy *_slow.gif (6 fps).
Numbers under each drawing are for this clip and seed only, relative to per-frame CLIPasso (run A): flicker = mean
chamfer stroke jump, LPIPS = mean LPIPS to the frame. The tables in results/summary/ average over clips and seeds.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
RUNS = [("A", "Per-frame CLIPasso"), ("B", "Warm start"), ("C_lam0.0016_K6", "Flow propagation (ours)"),
        ("BF", "Warm start + flow")]


def means(path, key, seed):
    acc = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) == seed and row[key] not in ("", "nan"):
                acc[row["run"]].append(float(row[key]))
    return {r: sum(v) / len(v) for r, v in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO / "results/figures/gifs"))
    p.add_argument("--tile", type=int, default=224)
    p.add_argument("--runs", nargs="+", default=None, help="run names to show (default: A, B, C, BF)")
    p.add_argument("--labels", nargs="+", default=None, help="column titles, one per --runs entry")
    p.add_argument("--name", default="", help="suffix for the output file, e.g. 'vs' -> <clip>_s0_vs.gif")
    p.add_argument("--no_numbers", action="store_true", help="leave out the numbers under each drawing (for slides)")
    p.add_argument("--font_size", type=int, default=15, help="label size in pixels")
    p.add_argument("--video_label", default=None, help="title of the video column (default: '<clip> (input)')")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = RUNS
    if a.runs:
        default = dict(RUNS)
        sel = list(zip(a.runs, a.labels or [default.get(r, r) for r in a.runs]))
    stem = f"_s{a.seed}" + (f"_{a.name}" if a.name else "")
    font = ImageFont.truetype(fm.findfont("DejaVu Sans"), a.font_size)
    small = ImageFont.truetype(fm.findfont("DejaVu Sans"), 13)
    S, pad, top, bot = a.tile, 8, a.font_size + 13, 24
    for clip in a.clips:
        ev = REPO / "results/eval" / clip
        fl = means(ev / "per_pair.csv", "chamfer_eval", a.seed)
        lp = means(ev / "per_frame.csv", "lpips", a.seed)
        runs = [(r, lab) for r, lab in sel if (REPO / "results/runs" / clip / r / f"seed_{a.seed}" / "DONE").exists()]
        video = sorted((REPO / "data/clips" / clip / "frames").glob("*.png"))
        cols = [("video", a.video_label or f"{clip} (input)")] + runs
        W, H = len(cols) * (S + pad) + pad, top + S + bot
        frames = []
        for t in range(len(video)):
            G = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(G)
            for k, (r, lab) in enumerate(cols):
                x = pad + k * (S + pad)
                src = video[t] if r == "video" else REPO / "results/runs" / clip / r / f"seed_{a.seed}" / f"frame_{t:02d}/final.png"
                G.paste(Image.open(src).convert("RGB").resize((S, S), Image.LANCZOS), (x, top))
                d.text((x, 6), lab, fill="black", font=font)
                if a.no_numbers:
                    pass
                elif r == "video":
                    d.text((x, top + S + 4), f"frame {t:02d}", fill="#57606a", font=small)
                elif r != "A" and r in fl and "A" in fl:
                    d.text((x, top + S + 4), f"flicker {fl[r] / fl['A']:.2f}x   LPIPS {lp[r] / lp['A']:.2f}x",
                           fill="#57606a", font=small)
                elif r == "A":
                    d.text((x, top + S + 4), "reference (1.00x)", fill="#57606a", font=small)
            frames.append(G.quantize(colors=128, dither=Image.Dither.NONE))
        for name, fps in ((f"{clip}{stem}.gif", 12), (f"{clip}{stem}_slow.gif", 6)):
            frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                           optimize=True)
        msg = "   ".join(f"{r}: flicker {fl[r] / fl['A']:.2f}x LPIPS {lp[r] / lp['A']:.2f}x" for r, _ in runs
                         if r in fl and "A" in fl)
        print(f"[gifs] {clip} -> {out / (clip + stem + '.gif')}   {msg}")


if __name__ == "__main__":
    main()
