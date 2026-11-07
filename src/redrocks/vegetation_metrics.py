"""Measurements the realistic-vegetation spec is judged by.

Colour: CIELAB (D65) and CIE76 ΔE*ab, used to compare sampled pixels of a
capture with the same feature in a reference photo. Shape: how solid a crown is
from the side, leaf-card size, trunk size and how much ground shrubs cover
(``docs/superpowers/specs/2026-11-04-realistic-vegetation-design.md``).
"""

from __future__ import annotations

import numpy as np

# sRGB (D65) -> XYZ, and the D65 white point.
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB 0-255, shape (..., 3), to CIELAB (D65), same shape."""
    c = np.asarray(rgb, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def hex_rgb(c: str) -> np.ndarray:
    h = c.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def hex_to_lab(c: str) -> np.ndarray:
    return rgb_to_lab(hex_rgb(c))


def _lab(c) -> np.ndarray:
    return hex_to_lab(c) if isinstance(c, str) else np.asarray(c, float)


def delta_e(a, b) -> float:
    """CIE76 ΔE*ab between two colours, each a hex string or a Lab triple."""
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def median_hex(img, xy, r: int = 10) -> str:
    """Median colour of the (2r)x(2r) patch around pixel ``xy`` = (x, y), as hex."""
    a = np.asarray(img)[..., :3]
    x, y = int(xy[0]), int(xy[1])
    patch = a[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in np.median(patch, 0))


# ------------------------------------------------------------------ shapes
#
# Models are redrocks.vegetation.Model: parts "Trunk", "Crown", "Leaves" as
# Mesh (metres), leaf cards written by add_card as 4 corners and 2 triangles
# (corner 0 and 1 span the card's width at its twig end).

ALPHA_CLIP = 0.4  # M_Veg_Leaves' opacity mask clip value


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (counter-clockwise) of 2D points, by monotone chain."""
    pts = np.unique(np.asarray(points, float), axis=0)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.asarray(lower[:-1] + upper[:-1])


def _inside_hull(hull: np.ndarray, pts: np.ndarray) -> np.ndarray:
    inside = np.ones(len(pts), bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) >= -1e-9
    return inside


def side_fill(model, atlas_alpha=None, cell_m: float = 0.05, view: str = "y") -> float:
    """How solid the plant looks from the side: filled cells / cells in their convex hull.

    Looking along ``view`` ("y" or "x"), every Trunk and Crown triangle fills
    the grid cells whose centres it covers; a Leaves triangle fills a cell only
    where ``atlas_alpha`` (an (H, W) array, image rows top-down) at the cell's
    interpolated UV is at least the material's clip value. Without an atlas,
    leaf cards count as solid.
    """
    axes = (0, 2) if view == "y" else (1, 2)
    tris = []  # (positions (3, 2), uvs (3, 2) or None)
    for part in ("Trunk", "Crown", "Leaves"):
        mesh = model.parts[part]
        if not mesh.f:
            continue
        p = mesh.positions()[:, axes]
        uv = np.asarray(mesh.vt, float).reshape(-1, 2)
        for fv, ft, _ in mesh.f:
            tris.append((p[list(fv)], uv[list(ft)] if part == "Leaves" and atlas_alpha is not None else None))
    if not tris:
        return 0.0
    allp = np.concatenate([t[0] for t in tris])
    lo = allp.min(0) - cell_m
    shape = (np.ceil((allp.max(0) + cell_m - lo) / cell_m)).astype(int) + 1
    grid = np.zeros(shape, bool)
    if atlas_alpha is not None:
        alpha = np.asarray(atlas_alpha, float)
        ah, aw = alpha.shape[:2]
    for tri, uvs in tris:
        i0 = np.floor((tri.min(0) - lo) / cell_m).astype(int)
        i1 = np.ceil((tri.max(0) - lo) / cell_m).astype(int)
        gi, gj = np.meshgrid(np.arange(i0[0], i1[0] + 1), np.arange(i0[1], i1[1] + 1), indexing="ij")
        cx = lo[0] + (gi.ravel() + 0.5) * cell_m
        cy = lo[1] + (gj.ravel() + 0.5) * cell_m
        a, b, c = tri
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        w0 = ((b[1] - c[1]) * (cx - c[0]) + (c[0] - b[0]) * (cy - c[1])) / det
        w1 = ((c[1] - a[1]) * (cx - c[0]) + (a[0] - c[0]) * (cy - c[1])) / det
        w2 = 1.0 - w0 - w1
        hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if uvs is not None and hit.any():
            u = w0[hit] * uvs[0, 0] + w1[hit] * uvs[1, 0] + w2[hit] * uvs[2, 0]
            v = w0[hit] * uvs[0, 1] + w1[hit] * uvs[1, 1] + w2[hit] * uvs[2, 1]
            col = np.clip((u * aw).astype(int), 0, aw - 1)
            row = np.clip(((1.0 - v) * ah).astype(int), 0, ah - 1)
            keep = alpha[row, col] >= ALPHA_CLIP
            idx = np.flatnonzero(hit)
            hit[idx[~keep]] = False
        grid[gi.ravel()[hit], gj.ravel()[hit]] = True
    filled = np.argwhere(grid)
    if len(filled) < 3:
        return float(len(filled) > 0)
    hull = _hull(filled + 0.5)
    every = np.argwhere(np.ones_like(grid)) + 0.5
    in_hull = _inside_hull(hull, every)
    return float(grid.sum() / max(1, in_hull.sum()))


def _cards(model) -> np.ndarray:
    """(N, 4, 3) corners of every leaf card, in add_card's corner order."""
    mesh = model.parts["Leaves"]
    if not mesh.f:
        return np.zeros((0, 4, 3))
    p = mesh.positions()
    faces = mesh.face_positions()
    first, second = faces[0::2], faces[1::2]
    return np.stack([p[first[:, 0]], p[first[:, 1]], p[first[:, 2]], p[second[:, 2]]], 1)


def card_widths(model) -> np.ndarray:
    c = _cards(model)
    return np.linalg.norm(c[:, 1] - c[:, 0], axis=1)


def tip_fraction(model, radius_m: float = 1.0) -> float:
    """Share of leaf cards whose twig end is within ``radius_m`` of a branch tip."""
    c = _cards(model)
    tips = np.asarray(model.tips, float).reshape(-1, 3)
    if len(c) == 0 or len(tips) == 0:
        return 0.0
    base = (c[:, 0] + c[:, 1]) / 2
    d = np.sqrt(((base[:, None, :] - tips[None, :, :]) ** 2).sum(-1)).min(1)
    return float((d <= radius_m).mean())


def trunk_diameter_at(model, z_m: float = 1.4) -> float:
    """Trunk diameter where the plane z = ``z_m`` cuts the Trunk mesh's edges."""
    mesh = model.parts["Trunk"]
    if not mesh.f:
        return 0.0
    p = mesh.positions()
    f = mesh.face_positions()
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    za, zb = p[edges[:, 0], 2], p[edges[:, 1], 2]
    cut = (za - z_m) * (zb - z_m) < 0
    if not cut.any():
        return 0.0
    a, b = p[edges[cut, 0]], p[edges[cut, 1]]
    t = ((z_m - a[:, 2]) / (b[:, 2] - a[:, 2]))[:, None]
    pts = (a + (b - a) * t)[:, :2]
    centre = np.median(pts, 0)
    return float(2.0 * np.median(np.linalg.norm(pts - centre, axis=1)))


def crown_base_fraction(model) -> float:
    return float(model.parts["Leaves"].positions()[:, 2].min() / model.height_m)


def footprint_radius_m(model) -> float:
    pts = np.concatenate([model.parts[p].positions() for p in ("Crown", "Leaves") if model.parts[p].v])
    return float(np.percentile(np.linalg.norm(pts[:, :2], axis=1), 90))


def canopy_cover(instances: dict, models: dict, cells_xy_cm, species: tuple, res_cm: float = 50.0) -> float:
    """Share of the points ``cells_xy_cm`` (N, 2) under the footprint of a plant of ``species``.

    Footprints are discs of ``footprint_radius_m`` x the instance's scale_xy,
    painted onto a ``res_cm`` raster over the points' extent.
    """
    pts = np.asarray(cells_xy_cm, float).reshape(-1, 2)
    if len(pts) == 0:
        return 0.0
    lo = pts.min(0) - res_cm
    shape = (np.ceil((pts.max(0) + res_cm - lo) / res_cm)).astype(int) + 1
    cover = np.zeros(shape, bool)
    radius = {}
    for name, rows in instances.items():
        if name.rsplit("_", 1)[0] not in species or not rows:
            continue
        if name not in radius:
            radius[name] = footprint_radius_m(models[name]) * 100.0
        for row in rows:
            x, y, sxy = row[0], row[1], row[6]
            r = radius[name] * sxy
            i0 = np.floor((np.array([x, y]) - r - lo) / res_cm).astype(int)
            i1 = np.ceil((np.array([x, y]) + r - lo) / res_cm).astype(int)
            i0 = np.maximum(i0, 0)
            i1 = np.minimum(i1, shape - 1)
            if (i1 < i0).any():
                continue
            gi, gj = np.meshgrid(np.arange(i0[0], i1[0] + 1), np.arange(i0[1], i1[1] + 1), indexing="ij")
            cx, cy = lo[0] + gi * res_cm, lo[1] + gj * res_cm
            cover[gi, gj] |= (cx - x) ** 2 + (cy - y) ** 2 <= r * r
    idx = np.rint((pts - lo) / res_cm).astype(int)
    return float(cover[idx[:, 0], idx[:, 1]].mean())
