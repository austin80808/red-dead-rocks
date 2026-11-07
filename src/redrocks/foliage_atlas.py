"""Leaf-card atlases for the Red Rocks vegetation, painted at real size.

Each species' atlas is 2x2 tiles; a tile is one leaf card, ``CARD_M`` metres
wide, twig end at the bottom. The painters work in metres (``px_per_m`` = tile
pixels / card width), so a Gambel oak leaf is 5-12 cm on the card whatever the
texture resolution, and record every size they draw: ``painted_sizes`` is what
the tests check against ``SIZE_M``. Colours centre on ``PHOTO_COLOUR``, median
pixels of sunlit summer foliage in Wikimedia Commons photos of Red Rocks Park
(docs/superpowers/specs/2026-11-04-realistic-vegetation-design.md).
"""

from __future__ import annotations

import math
import zlib

import numpy as np

ATLASES = ("Juniper", "Pine", "Oak", "Mahogany")

# Card width (m) per atlas: the leaf cards in vegetation.py use these.
CARD_M = {"Juniper": 0.32, "Pine": 0.48, "Oak": 0.38, "Mahogany": 0.28}
# What a card holds, length in m: juniper sprays, ponderosa needles, oak and mahogany leaves.
SIZE_M = {"Juniper": (0.10, 0.20), "Pine": (0.10, 0.18), "Oak": (0.05, 0.12), "Mahogany": (0.02, 0.035)}
JUNIPER_BRANCHLET_M = (0.001, 0.002)  # scale-leaf branchlet thickness
# Sunlit photo colour per atlas (mahogany has no clean sample; it is greyer and lighter than oak).
PHOTO_COLOUR = {"Juniper": "#505B46", "Pine": "#68744B", "Oak": "#5C662F"}

_PALETTE = {
    "Juniper": ["#4A5640", "#56614B", "#5E6A52", "#47523F", "#535E47"],
    "Pine": ["#627454", "#6C7E5A", "#74865E", "#667853", "#6A7B58"],
    "Oak": ["#56633A", "#5E6A40", "#647046", "#5A6640", "#617043"],
    "Mahogany": ["#6E7560", "#7A806A", "#737A64", "#838872", "#6A715C"],
}


def _hex(c, jitter, rng):
    h = c.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)
    rgb = rgb * rng.uniform(1 - jitter, 1 + jitter) + rng.normal(scale=jitter * 30, size=3)
    return tuple(int(v) for v in np.clip(rgb, 0, 255)) + (255,)


def _rec(record, key, value):
    if record is not None:
        record.setdefault(key, []).append(float(value))


def _paint_juniper(draw, rng, s, record):
    """Fluffy sprays of rope-like scale-leaf branchlets, 1-2 mm thick, from one twig."""
    k = s / CARD_M["Juniper"]
    pal = _PALETTE["Juniper"]
    twig_top = (s * 0.5, s - 0.05 * k)
    draw.line([(s * 0.5, s * 0.995), twig_top], fill=(88, 68, 52, 255), width=max(2, int(0.004 * k)))

    def cord(x, y, angle, length_m, width_m, depth):
        steps = max(3, int(length_m / 0.004))
        col = _hex(rng.choice(pal), 0.08, rng)
        _rec(record, "branchlet", width_m)
        for i in range(steps):
            a = angle + rng.normal(scale=0.06)
            nx, ny = x + math.cos(a) * length_m * k / steps, y - math.sin(a) * length_m * k / steps
            w = width_m * (1.0 - 0.3 * i / steps)
            draw.line([(x, y), (nx, ny)], fill=col, width=max(1, int(round(w * k))))
            x, y = nx, ny
            if depth > 0 and i < steps - 1:
                side = 1 if i % 2 else -1
                rest = length_m * (1.0 - i / steps)
                cord(x, y, angle + side * rng.uniform(0.6, 1.0), rest * rng.uniform(0.25, 0.45),
                     width_m * rng.uniform(0.75, 0.95), depth - 1)
                angle += side * 0.04

    for _ in range(int(rng.integers(14, 19))):
        length = rng.uniform(*SIZE_M["Juniper"])
        _rec(record, "spray", length)
        x = twig_top[0] + rng.normal(scale=0.012 * k)
        y = twig_top[1] + rng.uniform(-0.03, 0.01) * k
        angle = math.pi / 2 + rng.normal(scale=0.7)
        cord(x, y, angle, length, rng.uniform(*JUNIPER_BRANCHLET_M) * 0.5 + 0.00085, 2)
    for _ in range(int(rng.integers(1, 4))):  # a few glaucous berries
        x, y = rng.uniform(0.3, 0.7) * s, rng.uniform(0.25, 0.7) * s
        r = 0.0035 * k
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_hex("#7E8FA6", 0.05, rng))


def _paint_pine(draw, rng, s, record):
    """A ponderosa tuft: needles 10-18 cm, ~1.5 mm wide, fanning out from a twig end."""
    k = s / CARD_M["Pine"]
    pal = _PALETTE["Pine"]
    tip = (s * 0.5 + rng.normal(scale=0.005 * k), s - 0.14 * k)
    draw.line([(s * 0.5, s * 0.995), tip], fill=(96, 70, 48, 255), width=max(2, int(0.012 * k)))
    for _ in range(380):
        f = rng.uniform(0.0, 1.0)
        x0 = s * 0.5 + (tip[0] - s * 0.5) * (0.45 + 0.55 * f)
        y0 = s * 0.995 + (tip[1] - s * 0.995) * (0.45 + 0.55 * f)
        a = math.pi / 2 + rng.normal(scale=0.5 + 0.35 * (1 - f))
        length = rng.uniform(*SIZE_M["Pine"])
        _rec(record, "needle", length)
        bend = rng.normal(scale=0.15)
        pts = [(x0 + math.cos(a + bend * t) * length * k * t, y0 - math.sin(a + bend * t) * length * k * t)
               for t in np.linspace(0, 1, 6)]
        draw.line(pts, fill=_hex(rng.choice(pal), 0.08, rng), width=max(1, int(round(0.0015 * k))))
    r = 0.008 * k  # the bud
    draw.ellipse([tip[0] - r, tip[1] - r * 1.4, tip[0] + r, tip[1] + r], fill=(122, 90, 60, 255))


def _oak_leaf(draw, rng, x, y, a, length_m, k, col):
    """A Gambel oak leaf, ``length_m`` long: 5-9 rounded lobes with deep sinuses."""
    lobes = int(rng.integers(5, 9)) / 2.0
    L = length_m * k
    ca, sa = math.cos(a), math.sin(a)
    half = []
    for t in np.linspace(0.0, 1.0, 64):
        envelope = 0.3 * L * math.sin(math.pi * min(1.0, 0.05 + t)) ** 0.7
        lobe = 0.4 + 0.6 * abs(math.sin(math.pi * lobes * t)) ** 0.6  # rounded lobes, sinuses at 40 %
        half.append((t * L, envelope * lobe))
    pts = half + [(px, -py) for px, py in half[::-1]]
    poly = [(x + px * ca - py * sa, y - (px * sa + py * ca)) for px, py in pts]
    draw.polygon(poly, fill=col)
    draw.line([(x, y), (x + L * ca, y - L * sa)], fill=tuple(int(c * 0.82) for c in col[:3]) + (255,), width=1)


def _mahogany_leaf(draw, x, y, a, length_m, k, col):
    """A small obovate mountain mahogany leaf."""
    L = length_m * k
    pts = []
    for t in np.linspace(0, 2 * math.pi, 16):
        u = (1 - math.cos(t)) / 2  # 0 at the base, 1 at the tip
        px, py = u * L, math.sin(t) * L * 0.2 * (0.4 + 0.6 * u)
        pts.append((x + px * math.cos(a) - py * math.sin(a), y - (px * math.sin(a) + py * math.cos(a))))
    draw.polygon(pts, fill=col)


def _paint_broadleaf(draw, rng, s, record, mahogany: bool):
    """Oak: forked twigs of lobed leaves 5-12 cm. Mahogany: spur shoots with fascicles of
    2-3.5 cm leaves and a few feathery, curling seed plumes."""
    atlas = "Mahogany" if mahogany else "Oak"
    k = s / CARD_M[atlas]
    pal = _PALETTE[atlas]
    twig = (112, 96, 84, 255) if mahogany else (88, 66, 48, 255)
    lo, hi = SIZE_M[atlas]
    nodes = []
    margin = 0.03 * s

    def on_card(x, y, a, length_m):
        tx, ty = x + math.cos(a) * length_m * k, y - math.sin(a) * length_m * k
        return margin < tx < s - margin and margin < ty < s - margin and margin < x < s - margin

    def branch(x, y, angle, length_m, width_m, depth):
        steps = 5
        for i in range(steps):
            a = angle + rng.normal(scale=0.12)
            nx, ny = x + math.cos(a) * length_m * k / steps, y - math.sin(a) * length_m * k / steps
            draw.line([(x, y), (nx, ny)], fill=twig, width=max(1, int(width_m * k)))
            x, y = nx, ny
            nodes.append((x, y, a))
            if depth > 0 and i in (1, 3):
                branch(x, y, angle + rng.choice([-1, 1]) * rng.uniform(0.45, 0.85), length_m * 0.55,
                       width_m * 0.7, depth - 1)

    if mahogany:
        branch(s * 0.5, s * 0.995, math.pi / 2 + rng.normal(scale=0.1), 0.16, 0.003, 2)
        for x, y, a in nodes:
            for _ in range(int(rng.integers(4, 8))):  # a fascicle at each spur
                length = rng.uniform(lo, hi)
                la = a + rng.normal(scale=1.0)
                if not on_card(x, y, la, length):
                    continue
                _rec(record, "leaf", length)
                _mahogany_leaf(draw, x, y, la, length, k, _hex(rng.choice(pal), 0.06, rng))
        for _ in range(int(rng.integers(2, 4))):  # seed plumes, 5-8 cm, silvery
            x, y = nodes[int(rng.integers(len(nodes)))][:2]
            length = rng.uniform(0.05, 0.08) * k
            pts = [(x + length * t + 0.01 * k * math.sin(t * 8), y - length * 0.7 * t + 0.01 * k * math.cos(t * 8))
                   for t in np.linspace(0, 1, 18)]
            draw.line(pts, fill=(222, 216, 204, 255), width=max(1, int(0.0012 * k)))
    else:
        branch(s * 0.5, s * 0.995, math.pi / 2 + rng.normal(scale=0.1), 0.2, 0.005, 1)
        placed = 0
        for _ in range(200):
            if placed >= 24:
                break
            x, y, a = nodes[int(rng.integers(len(nodes)))]
            length = rng.uniform(lo, hi)
            la = a + rng.choice([-1, 1]) * rng.uniform(0.3, 1.1)
            if not on_card(x, y, la, length):
                continue
            _rec(record, "leaf", length)
            _oak_leaf(draw, rng, x, y, la, length, k, _hex(rng.choice(pal), 0.07, rng))
            placed += 1


def _paint(atlas: str, size: int, supersample: int, record):
    from PIL import Image, ImageDraw, ImageFilter

    if atlas not in ATLASES:
        raise ValueError(atlas)
    rng = np.random.default_rng(zlib.crc32(atlas.encode()))
    tile = size // 2 * supersample
    img = Image.new("RGBA", (tile * 2, tile * 2), (0, 0, 0, 0))
    for t in range(4):
        layer = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        if atlas == "Juniper":
            _paint_juniper(draw, rng, tile, record)
        elif atlas == "Pine":
            _paint_pine(draw, rng, tile, record)
        else:
            _paint_broadleaf(draw, rng, tile, record, atlas == "Mahogany")
        # Tile t sits at UV (t % 2, t // 2) quarters; image rows run top-down
        # while V runs bottom-up, so tile row 0 is the bottom half of the image.
        img.paste(layer, ((t % 2) * tile, (1 - t // 2) * tile))
    # Resample premultiplied so the transparent black never darkens the edges.
    img = img.convert("RGBa").resize((size, size), Image.LANCZOS).convert("RGBA")
    rgba = np.asarray(img).astype(np.float32)
    alpha = rgba[..., 3:4] / 255.0
    pre = Image.fromarray(np.clip(rgba[..., :3] * alpha, 0, 255).astype(np.uint8))
    a_img = Image.fromarray((alpha[..., 0] * 255).astype(np.uint8))
    blur = np.asarray(pre.filter(ImageFilter.GaussianBlur(24))).astype(np.float32)
    blur_a = np.asarray(a_img.filter(ImageFilter.GaussianBlur(24))).astype(np.float32)[..., None] / 255.0
    mean = (rgba[..., :3] * alpha).sum((0, 1)) / max(float(alpha.sum()), 1e-3)
    fill = np.where(blur_a > 1e-3, blur / np.maximum(blur_a, 1e-3), mean)
    rgb = np.where(alpha > 0.02, rgba[..., :3], fill)
    return np.concatenate([np.clip(rgb, 0, 255), rgba[..., 3:4]], axis=2).astype(np.uint8)


def paint_atlas(atlas: str, size: int = 1024, supersample: int = 2) -> np.ndarray:
    """RGBA uint8 array (size, size, 4): the species' 2x2 leaf-card atlas.

    Each quarter is one card, twig at the bottom of the tile. Transparent
    texels carry the blurred colour of their neighbours, so mips and filtering
    never bleed black into the edges.
    """
    return _paint(atlas, size, supersample, None)


def painted_sizes(atlas: str, size: int = 256) -> dict:
    """What the painter drew for ``atlas``, in metres: {"spray"|"needle"|"leaf": [...], ...}."""
    record = {}
    _paint(atlas, size, 2, record)
    return record


def mean_colour(img: np.ndarray) -> str:
    """Alpha-weighted mean RGB of an atlas, as hex."""
    a = np.asarray(img, float)
    w = a[..., 3] / 255.0
    rgb = (a[..., :3] * w[..., None]).sum((0, 1)) / max(float(w.sum()), 1e-6)
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in rgb)
