"""Print-sized qualitative figures for the report (results/figures/report/).

  python scripts/make_report_figures.py

fig_onion.png  Flicker made visible: drawing t warped to frame t+1 with GMFlow (red) over drawing t+1 (black), for
               per-frame drawing and warm start. Red and black coincide when strokes move only with the object.
fig_strip.png  horsejump-high, seed 0: the video and four methods at six frames, each column cropped to the subject.
fig_methods.png How each method draws the next frame (horsejump-high, frames 2 to 3): the starting strokes (red) over the faded
               next frame, and the drawing after optimization.
"""
import io
import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.flow import backward_warp  # noqa: E402

FONT = fm.findfont("DejaVu Sans")


def gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def page(clip, run, seed, t):
    return REPO / "results/runs" / clip / run / f"seed_{seed}" / f"frame_{t:02d}/final.png"


def bbox(ims, pad=10):
    """Union bounding box of non-white pixels over RGB uint8 arrays, padded and made square."""
    m = np.zeros(ims[0].shape[:2], bool)
    for a in ims:
        m |= a.min(-1) < 235
    ys, xs = np.nonzero(m)
    H, W = m.shape
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    s = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = int(np.clip(cy - s // 2, 0, max(H - s, 0))), int(np.clip(cx - s // 2, 0, max(W - s, 0)))
    return x0, y0, min(x0 + s, W), min(y0 + s, H)


def grid(cells, row_labels, col_labels, tile=224, lab_w=120, head=56, fs=40, col_fs=None, gap=6):
    """cells[i][j]: RGB uint8 arrays already cropped; pasted at tile x tile with labels ('\\n' breaks a label)."""
    f = ImageFont.truetype(FONT, fs)
    fc = ImageFont.truetype(FONT, col_fs or fs)
    R, C = len(cells), len(cells[0])
    G = Image.new("RGB", (lab_w + C * (tile + gap), head + R * (tile + gap)), "white")
    d = ImageDraw.Draw(G)
    for j, s in enumerate(col_labels):
        d.multiline_text((lab_w + j * (tile + gap) + tile // 2, head // 2), s, fill="black", font=fc, anchor="mm",
                         align="center", spacing=4)
    for i, s in enumerate(row_labels):
        d.multiline_text((lab_w // 2, head + i * (tile + gap) + tile // 2), s, fill="black", font=f, anchor="mm",
                         align="center", spacing=6)
        for j in range(C):
            G.paste(Image.fromarray(cells[i][j]).resize((tile, tile), Image.LANCZOS),
                    (lab_w + j * (tile + gap), head + i * (tile + gap)))
    return G


def add_legend(G, items, fs=32, h=64):
    """Put a one-line legend [(text, colour), ...] above image G."""
    f = ImageFont.truetype(FONT, fs)
    W = Image.new("RGB", (G.width, G.height + h), "white")
    W.paste(G, (0, h))
    d = ImageDraw.Draw(W)
    x = 20
    for text, col in items:
        d.line([(x, h // 2), (x + 50, h // 2)], fill=col, width=7)
        d.text((x + 62, h // 2), text, fill="black", font=f, anchor="lm")
        x += 62 + int(d.textlength(text, font=f)) + 50
    return W


def onion(clip, seed, pairs, runs, out):
    flow = torch.from_numpy(np.load(REPO / "data/clips" / clip / "flow_eval_bwd.npy"))
    cols = []
    for t in pairs:
        col = []
        for r in runs:
            w = backward_warp(torch.from_numpy(gray(page(clip, r, seed, t)))[None, None], flow[t:t + 1])[0, 0].numpy()
            s1 = gray(page(clip, r, seed, t + 1))
            im = np.ones((*s1.shape, 3), np.float32)
            im[..., 1] -= (1 - w) * 0.9
            im[..., 2] -= (1 - w) * 0.9
            im *= s1[..., None]
            col.append((im.clip(0, 1) * 255).astype(np.uint8))
        x0, y0, x1, y1 = bbox(col)
        cols.append([c[y0:y1, x0:x1] for c in col])
    cells = [[cols[j][i] for j in range(len(pairs))] for i in range(len(runs))]
    G = grid(cells, ["Per-\nframe", "Warm\nstart"], [f"pages {t}, {t + 1}" for t in pairs], lab_w=190, fs=34,
             col_fs=28, gap=24)
    add_legend(G, [("page t, moved with the swan", (230, 25, 25)), ("page t+1", (0, 0, 0))]).save(out / "fig_onion.png")


def render_svg(path, size=224):
    import cairosvg
    png = cairosvg.svg2png(url=str(path), output_width=size, output_height=size, background_color="white")
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float32) / 255.0


def methods_figure(clip, seed, t, runs, labels, out, name="fig_methods.png", fs=30, col_fs=25, legend_fs=30,
                   lab_w=190, head=120, tile=224, gap=24):
    """Row 1: starting strokes of frame t+1 (red) over the faded frame t+1. Row 2: the drawing after optimization."""
    frame = np.asarray(Image.open(REPO / "data/clips" / clip / "frames" / f"{t + 1:02d}.png").convert("RGB"),
                       dtype=np.float32) / 255.0
    faded = 1 - 0.5 * (1 - frame)
    starts, results = [], []
    for r in runs:
        d = REPO / "results/runs" / clip / r / f"seed_{seed}" / f"frame_{t + 1:02d}"
        ink = 1 - render_svg(d / "init.svg")
        im = faded.copy()
        im[..., 1] *= 1 - ink
        im[..., 2] *= 1 - ink
        starts.append((im.clip(0, 1) * 255).astype(np.uint8))
        fin = 1 - gray(d / "final.png")
        im = faded * (1 - fin[..., None])
        results.append((im.clip(0, 1) * 255).astype(np.uint8))
    x0, y0, x1, y1 = bbox(starts + results, pad=6)
    cells = [[c[y0:y1, x0:x1] for c in starts], [c[y0:y1, x0:x1] for c in results]]
    G = grid(cells, ["Starting\nstrokes", "Finished\npage"], labels, tile=tile, lab_w=lab_w, head=head, fs=fs,
             col_fs=col_fs, gap=gap)
    add_legend(G, [("strokes the optimization starts from", (230, 25, 25))], fs=legend_fs).save(out / name)


def strip(clip, seed, frames, runs, labels, out):
    cols = []
    for t in frames:
        col = [np.asarray(Image.open(REPO / "data/clips" / clip / "frames" / f"{t:02d}.png").convert("RGB"))]
        col += [np.asarray(Image.open(page(clip, r, seed, t)).convert("RGB")) for r in runs]
        x0, y0, x1, y1 = bbox(col)
        cols.append([c[y0:y1, x0:x1] for c in col])
    cells = [[cols[j][i] for j in range(len(frames))] for i in range(len(runs) + 1)]
    grid(cells, ["Video"] + labels, [f"t={t}" for t in frames], lab_w=250, fs=36).save(out / "fig_strip.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO / "results/figures/report"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    onion("blackswan", 0, [6, 12, 18], ["A", "B"], out)
    strip("horsejump-high", 0, [0, 4, 8, 12, 16, 20], ["A", "B", "C_lam0.0016_K6", "BF"], ["Per-\nframe", "Warm\nstart", "Flow\npropagation", "Warm start\n+ flow"], out)
    methods_figure("horsejump-high", 0, 2, ["A", "B", "C_lam0.0016_K6"], ["Per-frame\n(random start)", "Warm start\n(previous page)", "Flow propagation\n(previous page,\nmoved by flow)"], out)
    print(f"[report figures] -> {out}")


if __name__ == "__main__":
    main()
