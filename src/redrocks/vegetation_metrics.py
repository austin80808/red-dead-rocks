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
