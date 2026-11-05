"""Build the README's progress GIF: the view up the bowl from the stage, stage by stage.

Each stage of the project adds one frame to ``docs/media/progress/stages.json``
(the picture, a short title, the PR). This script scales every frame to one
size, puts a caption bar under it ("3/6  Look pass: Lumen, sandstone  #41") and
writes ``docs/media/progress.gif``, holding the newest stage longest:

    .venv/bin/python tools/make_progress_gif.py

Add a stage: capture the same view (``unreal/playtest_explore.py``'s ``seating``
scenario shoots ``redrocks_explore_seating_stage``, or ``bowl`` shoots
``bowl_stage``), copy it to ``docs/media/progress/``, append it to
``stages.json`` and re-run. Needs only Pillow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "docs" / "media" / "progress" / "stages.json"
OUT = ROOT / "docs" / "media" / "progress.gif"

WIDTH = 800
ASPECT = 9 / 16
CAPTION_H = 44
HOLD_MS = 1800
LAST_HOLD_MS = 4000
BAR = (24, 18, 16)
TEXT = (245, 236, 224)
DIM = (176, 150, 132)
FONTS = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to cover w x h, then centre-crop: every frame the same size, none letterboxed."""
    img = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def frame(img: Image.Image, index: int, total: int, title: str, pr: str, width: int) -> Image.Image:
    h = round(width * ASPECT)
    out = Image.new("RGB", (width, h + CAPTION_H), BAR)
    out.paste(_fit(img, width, h), (0, 0))
    d = ImageDraw.Draw(out)
    big, small = _font(int(CAPTION_H * 0.45)), _font(int(CAPTION_H * 0.36))
    y = h + CAPTION_H // 2
    d.text((14, y), f"{index}/{total}", font=small, fill=DIM, anchor="lm")
    d.text((14 + int(CAPTION_H * 1.2), y), title, font=big, fill=TEXT, anchor="lm")
    d.text((width - 14, y), pr, font=small, fill=DIM, anchor="rm")
    # Progress ticks along the top edge of the bar.
    for k in range(total):
        x0 = 14 + k * (width - 28) / total
        x1 = x0 + (width - 28) / total - 4
        d.rectangle((x0, h + 3, x1, h + 5), fill=TEXT if k < index else (70, 58, 52))
    return out


def build(stages: list[dict], out: Path, width: int = WIDTH) -> dict:
    """Write the GIF from ``stages`` (each {frame, title, pr}, oldest first)."""
    if not stages:
        raise ValueError("no stages")
    frames = []
    for i, s in enumerate(stages, start=1):
        path = Path(s["frame"])
        path = path if path.is_absolute() else ROOT / path
        with Image.open(path) as img:
            frames.append(frame(img, i, len(stages), s["title"], s["pr"], width))
    # One shared adaptive palette keeps the colours steady from frame to frame.
    sheet = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for k, f in enumerate(frames):
        sheet.paste(f, (0, k * f.height))
    palette = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    durations = [HOLD_MS] * (len(frames) - 1) + [LAST_HOLD_MS]
    out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(out, save_all=True, append_images=quantized[1:], duration=durations, loop=0,
                      optimize=False, disposal=1)
    return {"frames": len(frames), "size": frames[0].size, "bytes": out.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", type=Path, default=STAGES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--width", type=int, default=WIDTH)
    args = ap.parse_args(argv)
    info = build(json.loads(args.stages.read_text())["stages"], args.out, args.width)
    print(f"{info['frames']} stages, {info['size'][0]}x{info['size'][1]}, {info['bytes'] / 1e6:.1f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
