"""Before | after composites of the amphitheatre beauty shots for the README.

Reads out/beauty/<before>_<view>.png and <after>_<view>.png (shot on the render
pod by unreal/shoot_board.py with REDROCKS_VIEWS=docs/media/beauty/views.json)
and writes, in docs/media/beauty/:

  <view>.jpg          the after shot, 1600 px wide
  compare_<view>.jpg  before | after side by side, captioned
  compare.jpg         every view's before | after, one row per view

    .venv/bin/python tools/make_beauty_compare.py --before before_ --after after4_
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _font(size):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(img, text, size=28):
    d = ImageDraw.Draw(img)
    f = _font(size)
    box = d.textbbox((0, 0), text, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    d.rectangle((0, 0, w + 24, h + 18), fill=(0, 0, 0))
    d.text((12, 6), text, fill=(255, 255, 255), font=f)
    return img


def pair(before: Path, after: Path, width: int) -> Image.Image:
    a = Image.open(before).convert("RGB")
    b = Image.open(after).convert("RGB")
    h = round(width * a.height / a.width)
    a, b = a.resize((width, h), Image.LANCZOS), b.resize((width, h), Image.LANCZOS)
    out = Image.new("RGB", (2 * width + 8, h), (255, 255, 255))
    out.paste(_label(a, "before"), (0, 0))
    out.paste(_label(b, "after"), (width + 8, 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shots", default=str(ROOT / "out" / "beauty"))
    ap.add_argument("--before", default="before_")
    ap.add_argument("--after", required=True)
    ap.add_argument("--dest", default=str(ROOT / "docs" / "media" / "beauty"))
    args = ap.parse_args(argv)
    shots, dest = Path(args.shots), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    views = [v["name"] for v in json.loads((dest / "views.json").read_text())["views"]]
    rows = []
    for name in views:
        before, after = shots / f"{args.before}{name}.png", shots / f"{args.after}{name}.png"
        if not after.exists():
            print(f"skip {name}: no {after.name}")
            continue
        img = Image.open(after).convert("RGB")
        img.resize((1600, round(1600 * img.height / img.width)), Image.LANCZOS).save(dest / f"{name}.jpg", quality=86)
        if before.exists():
            pair(before, after, 800).save(dest / f"compare_{name}.jpg", quality=84)
            rows.append(pair(before, after, 640))
        print(f"wrote {name}")
    if rows:
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 8 * (len(rows) - 1)), (255, 255, 255))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 8
        sheet.save(dest / "compare.jpg", quality=84)
        print("wrote compare.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
