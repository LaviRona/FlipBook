"""Printable flipbook: one sketch per page, 8 pages per A4 sheet (2 x 4), cut lines, binding margin, page numbers.

  python scripts/make_flipbook.py --clip dog --run C_lam0.0004_K6 --seed 0 --out results/flipbooks/dog_C.pdf
  python scripts/make_flipbook.py --clip dog --run A --seed 0 --line_scale 1.6 --repeat 2

Pages are rendered from each frame's final.svg with the stroke width multiplied by --line_scale (export only, no
re-optimization). Print at 100% scale on 160 to 200 g/m2 card, cut along the grey lines, stack page 1 on top and
clamp the left margin with a binder clip. --repeat shows every frame on several pages (slower flip, more pages).
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DPI = 300
MM = DPI / 25.4
A4 = (round(210 * MM), round(297 * MM))
COLS, ROWS = 2, 4


def render_svg(path, size_px, line_scale):
    import cairosvg
    svg = Path(path).read_text()
    svg = re.sub(r'stroke-width="([0-9.]+)"', lambda m: f'stroke-width="{float(m.group(1)) * line_scale:g}"', svg)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size_px, output_height=size_px,
                           background_color="white")
    import io
    return Image.open(io.BytesIO(png)).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs_root", default=str(REPO / "results/runs"))
    p.add_argument("--out", default=None)
    p.add_argument("--line_scale", type=float, default=1.6)
    p.add_argument("--binding_mm", type=float, default=15)
    p.add_argument("--repeat", type=int, default=1, help="pages per frame")
    p.add_argument("--title", default=None, help="text on the cover page (page 0); default: clip and run")
    a = p.parse_args()

    sd = Path(a.runs_root) / a.clip / a.run / f"seed_{a.seed}"
    svgs = sorted(sd.glob("frame_*/final.svg"))
    if not svgs:
        raise SystemExit(f"no frames in {sd}")
    out = Path(a.out or REPO / "results/flipbooks" / f"{a.clip}_{a.run}_s{a.seed}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    cw, ch = A4[0] // COLS, A4[1] // ROWS
    bind = round(a.binding_mm * MM)
    side = min(cw - bind - round(6 * MM), ch - round(8 * MM))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", round(2.5 * MM))
        big = ImageFont.truetype("DejaVuSans.ttf", round(5 * MM))
    except OSError:
        font = big = ImageFont.load_default()

    pages = []
    cover = Image.new("RGB", (cw, ch), "white")
    d = ImageDraw.Draw(cover)
    d.text((bind + round(4 * MM), ch // 2 - round(6 * MM)), a.title or f"{a.clip}", fill="black", font=big)
    d.text((bind + round(4 * MM), ch // 2 + round(2 * MM)), f"{a.run}, seed {a.seed}", fill="black", font=font)
    pages.append(cover)
    for t, s in enumerate(svgs):
        img = render_svg(s, side, a.line_scale)
        for _ in range(a.repeat):
            pg = Image.new("RGB", (cw, ch), "white")
            x0 = bind + (cw - bind - side) // 2
            pg.paste(img, (x0, (ch - side) // 2))
            ImageDraw.Draw(pg).text((cw - round(9 * MM), ch - round(5 * MM)), str(len(pages)), fill="#888", font=font)
            pages.append(pg)

    sheets = []
    for k in range(0, len(pages), COLS * ROWS):
        sheet = Image.new("RGB", A4, "white")
        d = ImageDraw.Draw(sheet)
        for j, pg in enumerate(pages[k:k + COLS * ROWS]):
            sheet.paste(pg, ((j % COLS) * cw, (j // COLS) * ch))
        for c in range(1, COLS):
            d.line([(c * cw, 0), (c * cw, A4[1])], fill="#bbb", width=2)
        for r in range(1, ROWS):
            d.line([(0, r * ch), (A4[0], r * ch)], fill="#bbb", width=2)
        for c in range(COLS):  # binding margin guide
            d.line([(c * cw + bind, 0), (c * cw + bind, A4[1])], fill="#eee", width=1)
        sheets.append(sheet)
    sheets[0].save(out, save_all=True, append_images=sheets[1:], resolution=DPI)
    sheets[0].resize((A4[0] // 4, A4[1] // 4)).save(out.with_suffix(".preview.png"))  # first sheet, for a quick look
    print(f"[flipbook] {len(pages) - 1} pages + cover on {len(sheets)} A4 sheets -> {out}")


if __name__ == "__main__":
    main()
