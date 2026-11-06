"""Scores for the photo board: how close a game render is to a real photo.

Colour: CIE76 delta E of the median colour of a region (``region_hex``,
``vegetation_metrics.delta_e``). Shape: how far the skyline sits from the photo's,
as a fraction of the image height (``skyline_error``).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from redrocks.vegetation_metrics import delta_e

KINDS = ("sky", "rock_lit", "rock_shade", "tread", "riser", "plants", "plains")
MIN_SHARED_SKY = 0.20  # share of columns that must have sky in both images


def fit_to(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Cover-crop (centred) and resize ``img`` to ``size`` = (width, height)."""
    w, h = img.size
    tw, th = size
    if (w, h) == (tw, th):
        return img
    scale = max(tw / w, th / h)
    cw, ch = min(w, round(tw / scale)), min(h, round(th / scale))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((tw, th), Image.LANCZOS)


def _sky_like(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    blue = (b > r + 10) & (b > g)
    lo, hi = rgb[..., :3].min(-1).astype(int), rgb[..., :3].max(-1).astype(int)
    return blue | ((lo >= 170) & (hi - lo <= 30))


def _spread_row(like: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Sky-like pixels of one row in the same horizontal run as a seed pixel."""
    run = np.cumsum(~like)  # equal ids = same run of like pixels
    hit = np.unique(run[seed & like])
    return like & np.isin(run, hit)


def sky_mask(rgb: np.ndarray) -> np.ndarray:
    """Sky-like pixels (blue-dominant, or bright and grey) 4-connected to the top row."""
    rgb = np.asarray(rgb)[..., :3]
    like = _sky_like(rgb)
    sky = np.zeros_like(like)
    if not like.size:
        return sky
    sky[0] = _spread_row(like[0], like[0])
    h = like.shape[0]
    while True:
        before = sky.copy()
        for i in range(1, h):  # down
            sky[i] = _spread_row(like[i], sky[i - 1] | sky[i])
        for i in range(h - 2, -1, -1):  # back up, for sky reached from below
            sky[i] = _spread_row(like[i], sky[i + 1] | sky[i])
        if (sky == before).all():
            return sky


def skyline(mask: np.ndarray) -> np.ndarray:
    """Per column, the lowest sky row; -1 where the column has no sky."""
    mask = np.asarray(mask, bool)
    h = mask.shape[0]
    lowest = h - 1 - np.argmax(mask[::-1], axis=0)
    return np.where(mask.any(0), lowest, -1).astype(int)


def _in_boxes(xs: np.ndarray, rows: np.ndarray, boxes) -> np.ndarray:
    out = np.zeros(len(xs), bool)
    for x0, y0, x1, y1 in boxes:
        out |= (xs >= x0) & (xs <= x1) & (rows >= y0) & (rows <= y1)
    return out


def skyline_error(game: Image.Image, photo: Image.Image, ignore: list[list[int]]) -> float | None:
    """Mean |game row - photo row| / height over columns with sky in both.

    Measured at the photo's size after ``fit_to``. Columns without sky in either
    image are left out (never row 0), as are columns where either skyline point
    lies in an ``ignore`` box (photo pixels). ``None`` when fewer than 20 % of
    the columns qualify.
    """
    photo_rgb = np.asarray(photo.convert("RGB"))
    game_rgb = np.asarray(fit_to(game.convert("RGB"), photo.size))
    sp, sg = skyline(sky_mask(photo_rgb)), skyline(sky_mask(game_rgb))
    keep = (sp >= 0) & (sg >= 0)
    xs = np.arange(len(sp))
    if ignore:
        keep &= ~_in_boxes(xs, sp, ignore) & ~_in_boxes(xs, sg, ignore)
    if keep.sum() < MIN_SHARED_SKY * len(sp):
        return None
    return float(np.abs(sg[keep] - sp[keep]).mean() / photo_rgb.shape[0])


def _is_green(r: np.ndarray) -> np.ndarray:
    """The vegetation tool's "green" rule (tools/make_reference_compare.py)."""
    red, green, blue = r[:, 0], r[:, 1], r[:, 2]
    return (green >= red * 0.95) & (green > blue * 1.1) & (blue < 200)


def region_hex(rgb: np.ndarray, box: list[int], kind: str) -> str:
    """Median colour, as hex, of ``box`` = (x0, y0, x1, y1).

    ``sky`` keeps the box's sky_mask pixels (the mask is found on the whole
    image, so the box need not touch the top row), ``plants`` keeps green
    pixels, the other kinds take the whole box.
    """
    if kind not in KINDS:
        raise ValueError(kind)
    rgb = np.asarray(rgb)[..., :3]
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    if kind == "sky":
        keep = sky_mask(rgb)[y0:y1, x0:x1].reshape(-1)
    elif kind == "plants":
        keep = _is_green(patch)
    else:
        keep = np.ones(len(patch), bool)
    sel = patch[keep] if keep.any() else patch
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in np.median(sel, 0))


def score_view(view: dict, game: Image.Image, photo: Image.Image) -> dict:
    """Colour delta E per sample region and the skyline error for one board view.

    ``view["samples"]`` carry ``photo_box`` and ``game_box``, both in photo
    pixels; the game render is fitted to the photo's size first.
    """
    photo_rgb = np.asarray(photo.convert("RGB"))
    game_fit = fit_to(game.convert("RGB"), photo.size)
    game_rgb = np.asarray(game_fit)
    regions = {}
    for s in view.get("samples", []):
        p = region_hex(photo_rgb, s["photo_box"], s["kind"])
        g = region_hex(game_rgb, s["game_box"], s["kind"])
        regions[s["label"]] = {"kind": s["kind"], "photo": p, "game": g, "delta_e": delta_e(g, p)}
    return {"regions": regions,
            "skyline_error": skyline_error(game, photo, view.get("skyline_ignore", []))}


def worst_regions(scores: dict[str, dict], n: int = 3) -> list[tuple[str, str, float]]:
    """The ``n`` regions furthest from their photo, as (view, label, delta E), worst first."""
    rows = [(v, label, r["delta_e"]) for v, s in scores.items() for label, r in s["regions"].items()]
    return sorted(rows, key=lambda t: -t[2])[:n]
