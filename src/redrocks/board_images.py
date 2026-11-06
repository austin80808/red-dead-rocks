"""Images for the photo board and the vegetation side-by-sides.

Commons photos (fetched once into a cache that is never committed), captioned
side-by-side panels, the "is this shot black?" check, and the magenta
calibration markers that unreal/shoot_board.py draws at a view's landmarks.
Shared by tools/make_board_compare.py and tools/make_reference_compare.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

USER_AGENT = "redrocks-ue-reference/0.1"
CAPTION_H = 34
CAPTION_FONT_SIZE = 15
BACKGROUND = (18, 18, 18)


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


def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """``img`` scaled to cover ``w`` x ``h``, centre-cropped to exactly that."""
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


@lru_cache(maxsize=None)
def _caption_font():
    return _font(CAPTION_FONT_SIZE)


def compose_panels(images, captions, aspect: tuple[int, int], panel_w: int = 640) -> Image.Image:
    """Panels side by side, each cover-cropped to ``aspect``, over a caption strip."""
    panel_h = round(panel_w * aspect[1] / aspect[0])
    out = Image.new("RGB", (panel_w * len(images), CAPTION_H + panel_h), BACKGROUND)
    draw = ImageDraw.Draw(out)
    font = _caption_font()
    for i, (img, text) in enumerate(zip(images, captions)):
        out.paste(cover(img, panel_w, panel_h), (i * panel_w, CAPTION_H))
        while draw.textlength(text, font=font) > panel_w - 12 and len(text) > 4:
            text = text[:-2].rstrip() + "…"
        draw.text((i * panel_w + 6, 9), text, fill=(235, 235, 235), font=font)
    return out


def compose(before: Image.Image, after: Image.Image, photo: Image.Image, captions: tuple[str, str, str],
            aspect: tuple[int, int], panel_w: int = 640) -> Image.Image:
    """Three panels side by side, each cover-cropped to ``aspect``, over a caption strip."""
    return compose_panels((before, after, photo), captions, aspect, panel_w)


def text_width(text: str) -> float:
    """Width in pixels of ``text`` in the caption font."""
    return ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=_caption_font())


def photo_caption(photo: dict, max_px: float | None = None) -> str:
    """"<title> — <author>, <licence>". With ``max_px``, the title is shortened (…)
    until the caption fits, so the credit is never the part cut off."""
    credit = " — %s, %s" % (photo["author"], photo["licence"])
    title = photo["title"]
    if max_px is None or text_width(title + credit) <= max_px:
        return title + credit
    while title and text_width(title + "…" + credit) > max_px:
        title = title[:-1].rstrip()
    return (title + "…" if title else "") + credit


def mean_luminance(img: Image.Image) -> float:
    """Mean Rec. 709 luma of ``img``, on 0-255 (a black frame is near 0)."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
    return float((rgb @ np.array([0.2126, 0.7152, 0.0722])).mean())


def _runs(mask: np.ndarray):
    """Horizontal runs of True: (rows, starts, ends), ``ends`` exclusive, row-major."""
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), np.int8)
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    rows, starts = np.nonzero(step == 1)
    _, ends = np.nonzero(step == -1)
    return rows, starts, ends


def marker_pixels(img: Image.Image) -> np.ndarray:
    """Centroids (x, y) of the magenta blobs (r > 200, g < 60, b > 200), shape (N, 2).

    Blobs are 8-connected. Pixel (i, j) covers [i, i + 1) x [j, j + 1), so a
    centroid is in the same continuous pixel frame as ``photo_camera.project``.
    Sorted left to right (then top to bottom).
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    mask = (rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] > 200)
    rows, starts, ends = _runs(mask)
    n = len(rows)
    if n == 0:
        return np.zeros((0, 2))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Runs on neighbouring rows join when they overlap or touch at a corner.
    first = np.searchsorted(rows, np.arange(rows.max() + 2))
    for r in range(rows.min(), rows.max()):
        a0, a1, b1 = first[r], first[r + 1], first[r + 2]
        j = a1
        for i in range(a0, a1):
            while j < b1 and ends[j] < starts[i]:
                j += 1
            k = j
            while k < b1 and starts[k] <= ends[i]:
                parent[find(k)] = find(i)
                k += 1
    count = (ends - starts).astype(float)
    sx = count * (starts + ends) / 2.0  # sum of the pixel centres (i + 0.5) along each run
    sy = count * (rows + 0.5)
    roots = np.array([find(i) for i in range(n)])
    blobs = []
    for root in np.unique(roots):
        sel = roots == root
        total = count[sel].sum()
        blobs.append((sx[sel].sum() / total, sy[sel].sum() / total))
    return np.array(sorted(blobs))
