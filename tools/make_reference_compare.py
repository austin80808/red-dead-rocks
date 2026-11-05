"""Before | after | photo composites for the vegetation, and their colour error.

For each view in docs/media/vegetation-real/views.json, reads
``<shots>/before_<name>.png`` and ``<shots>/after_<name>.png`` (screenshots from
unreal/shoot_vegetation.py), fetches the view's Wikimedia Commons photo into
``--cache`` (never committed), and writes ``<out>/<name>.jpg``: three panels
cropped to the view's aspect, captioned "Before", "After" and the photo's
title, author and licence. Each view's ``samples`` pair a pixel (``photo_xy``/``game_xy``)
or a region (``photo_box``/``game_box`` with ``kind`` "green" or "brown") in the
photo with the same feature in the after shot; their median colours and CIE76 ΔE go
to ``<out>/delta_e.json`` and a printed table.

    .venv/bin/python tools/make_reference_compare.py --shots out/shots \
        --out docs/media/vegetation-real --cache out/refs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from redrocks.vegetation_metrics import delta_e, median_hex

REPO = Path(__file__).resolve().parent.parent
VIEWS = REPO / "docs" / "media" / "vegetation-real" / "views.json"
USER_AGENT = "redrocks-ue-reference/0.1"
CAPTION_H = 34


def load_views(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["views"]


def fetch_photo(photo: dict, cache_dir: Path) -> Path:
    """The photo's 1280 px Commons thumb, downloaded once into ``cache_dir``."""
    cache_dir = Path(cache_dir)
    path = cache_dir / ("%s.jpg" % photo["id"])
    if path.exists():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(photo["file_url"], headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS)
    x0, y0 = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x0, y0, x0 + w, y0 + h))


def _font(size: int):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(before: Image.Image, after: Image.Image, photo: Image.Image, captions: tuple[str, str, str],
            aspect: tuple[int, int], panel_w: int = 640) -> Image.Image:
    """Three panels side by side, each cover-cropped to ``aspect``, over a caption strip."""
    panel_h = round(panel_w * aspect[1] / aspect[0])
    out = Image.new("RGB", (panel_w * 3, CAPTION_H + panel_h), (18, 18, 18))
    draw = ImageDraw.Draw(out)
    font = _font(15)
    for i, (img, text) in enumerate(zip((before, after, photo), captions)):
        out.paste(_cover(img, panel_w, panel_h), (i * panel_w, CAPTION_H))
        while draw.textlength(text, font=font) > panel_w - 12 and len(text) > 4:
            text = text[:-2].rstrip() + "…"
        draw.text((i * panel_w + 6, 9), text, fill=(235, 235, 235), font=font)
    return out


def region_hex(img, box, kind: str) -> str:
    """Median colour, as hex, of the pixels in ``box`` (x0, y0, x1, y1) that look like
    ``kind``: "green" foliage (not sky) or "brown" bark."""
    x0, y0, x1, y1 = (int(v) for v in box)
    r = np.asarray(img, float)[y0:y1, x0:x1, :3].reshape(-1, 3)
    red, green, blue = r[:, 0], r[:, 1], r[:, 2]
    if kind == "green":
        keep = (green >= red * 0.95) & (green > blue * 1.1) & (blue < 200)
    elif kind == "brown":
        keep = (red > green) & (green > blue) & (red - blue > 25) & (red < 200)
    else:
        raise ValueError(kind)
    sel = r[keep] if keep.any() else r
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in np.median(sel, 0))


def _sample(img, s: dict, which: str) -> str:
    if "%s_box" % which in s:
        return region_hex(img, s["%s_box" % which], s["kind"])
    return median_hex(img, s["%s_xy" % which])


def _photo_caption(photo: dict) -> str:
    return "%s — %s, %s" % (photo["title"], photo["author"], photo["licence"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--views", type=Path, default=VIEWS)
    ap.add_argument("--panel-width", type=int, default=640)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report, missing = {}, []
    for view in load_views(args.views):
        name = view["name"]
        shots = [args.shots / ("%s_%s.png" % (k, name)) for k in ("before", "after")]
        if not all(p.exists() for p in shots):
            missing.append(name)
            continue
        before, after = (Image.open(p) for p in shots)
        photo = Image.open(fetch_photo(view["photo"], args.cache))
        compose(before, after, photo, ("Before", "After", _photo_caption(view["photo"])), tuple(view["aspect"]),
                args.panel_width).save(args.out / ("%s.jpg" % name), quality=88)
        rows = {}
        for s in view.get("samples", []):
            p_hex = _sample(np.asarray(photo.convert("RGB")), s, "photo")
            g_hex = _sample(np.asarray(after.convert("RGB")), s, "game")
            rows[s["label"]] = {"photo": p_hex, "game": g_hex, "delta_e": round(delta_e(p_hex, g_hex), 1)}
            print("%-10s %-10s photo %s  game %s  ΔE %5.1f" % (name, s["label"], p_hex, g_hex, rows[s["label"]]["delta_e"]))
        report[name] = rows
    (args.out / "delta_e.json").write_text(json.dumps(report, indent=1) + "\n")
    if missing:
        print("missing shots for: %s" % ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
