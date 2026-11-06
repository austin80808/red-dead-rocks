"""Signed-distance shapes for the sculpt: the sasquatch's head and the extra
body mass. Negative inside, in centimetres.

The head lives in a head frame: x forward (out of the face), y to his left,
z up, origin on the head joint (the base of the skull). `HEAD` lists every
feature with its reason, so the face can be tuned by editing numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ellipsoid(p: np.ndarray, centre, radii) -> np.ndarray:
    """Approximate signed distance to an axis-aligned ellipsoid (exact on a sphere)."""
    q = (p - np.asarray(centre, float)) / np.asarray(radii, float)
    k = np.linalg.norm(q, axis=-1)
    return (k - 1.0) * np.min(radii)


def capsule(p: np.ndarray, a, b, radius: float) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) - radius


def tapered_capsule(p: np.ndarray, a, b, ra: float, rb: float) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) - (ra + (rb - ra) * t)


def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Smooth union (polynomial), blend width k."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Smooth intersection; smooth subtraction of b is smax(a, -b, k)."""
    return -smin(-a, -b, k)


@dataclass(frozen=True)
class Feature:
    kind: str          # "ellipsoid", "capsule", "tapered"
    args: tuple
    op: str            # "union" or "cut"
    blend: float       # smooth-blend width, cm
    mirror: bool = False   # also place it at -y
    why: str = ""

    def distance(self, p: np.ndarray) -> np.ndarray:
        fn = {"ellipsoid": ellipsoid, "capsule": capsule, "tapered": tapered_capsule}[self.kind]
        d = fn(p, *self.args)
        if self.mirror:
            q = p * np.array([1.0, -1.0, 1.0])
            d = np.minimum(d, fn(q, *self.args))
        return d


# Head frame, centimetres, for a 2.55 m animal: head about 34 cm chin to crest,
# 25 cm across the cheekbones, 30 cm from the brow to the back of the skull.
HEAD = (
    Feature("ellipsoid", ((0.0, 0.0, 9.0), (12.5, 10.5, 11.0)), "union", 0.0,
            why="braincase: low and long, smaller than the face"),
    Feature("capsule", ((4.0, 0.0, 16.5), (-8.0, 0.0, 21.5), 2.5), "union", 7.0,
            why="sagittal crest, peaking at the back of the crown: the conical head"),
    Feature("ellipsoid", ((-10.0, 0.0, 10.0), (6.0, 9.5, 9.0)), "union", 5.0,
            why="occiput, flowing into the trapezius"),
    Feature("ellipsoid", ((11.6, 0.0, 5.0), (3.6, 9.2, 2.5)), "union", 3.5,
            why="supraorbital torus: the heavy brow ridge, one bar across both eyes"),
    Feature("ellipsoid", ((8.5, 7.0, -2.0), (4.5, 3.5, 3.5)), "union", 4.0, mirror=True,
            why="cheekbones (zygomatic arches): a wide face"),
    Feature("ellipsoid", ((10.5, 0.0, -6.5), (6.5, 7.5, 5.5)), "union", 5.0,
            why="muzzle: the jaws carried forward of the eyes"),
    Feature("ellipsoid", ((6.5, 0.0, -11.5), (8.0, 8.5, 4.5)), "union", 5.0,
            why="massive lower jaw with a receding chin"),
    Feature("ellipsoid", ((12.5, 0.0, 0.5), (1.8, 2.2, 3.0)), "union", 2.0,
            why="low, flat nasal bridge"),
    Feature("ellipsoid", ((15.0, 0.0, -2.6), (1.8, 4.2, 2.3)), "union", 2.5,
            why="broad flat nose, merged into the muzzle"),
    Feature("ellipsoid", ((16.3, 2.0, -3.5), (1.2, 1.25, 0.8)), "cut", 0.9, mirror=True,
            why="flared nostrils"),
    Feature("ellipsoid", ((15.6, 0.0, -6.2), (1.6, 5.2, 1.4)), "union", 1.5,
            why="upper lip: wide and thin"),
    Feature("ellipsoid", ((14.9, 0.0, -8.3), (1.6, 4.7, 1.3)), "union", 1.5,
            why="lower lip"),
    Feature("ellipsoid", ((17.0, 0.0, -7.35), (2.8, 5.0, 0.28)), "cut", 0.5,
            why="the closed mouth's line"),
    Feature("ellipsoid", ((10.4, 3.8, 1.6), (2.8, 2.4, 2.1)), "cut", 1.6, mirror=True,
            why="deep-set eye sockets under the brow"),
    Feature("ellipsoid", ((-1.0, 10.0, 2.0), (2.2, 1.0, 3.0)), "union", 1.5, mirror=True,
            why="small ears, close to the head (hidden in the hair)"),
)

# Eyeballs (head frame): centre and radius. The cornea sits just behind the socket rim.
EYE_CENTRE = (9.1, 3.8, 1.5)
EYE_RADIUS = 1.45


def head_sdf(p: np.ndarray, features=HEAD) -> np.ndarray:
    d = None
    for f in features:
        fd = f.distance(p)
        if d is None:
            d = fd
        elif f.op == "union":
            d = smin(d, fd, f.blend) if f.blend > 0 else np.minimum(d, fd)
        else:
            d = smax(d, -fd, f.blend) if f.blend > 0 else np.maximum(d, -fd)
    return d


def project_radially(points: np.ndarray, centre: np.ndarray, sdf, reach: float = 60.0,
                     steps: int = 48, refine: int = 24) -> np.ndarray:
    """Move each point along the ray from `centre` through it onto the first
    zero of `sdf` going outward (the surface the ray exits through)."""
    d = points - centre
    dist = np.linalg.norm(d, axis=1)
    u = d / np.maximum(dist, 1e-9)[:, None]
    ts = np.linspace(0.0, reach, steps + 1)
    values = np.stack([sdf(centre + t * u) for t in ts], axis=1)            # (n, steps+1)
    outside = values > 0
    first = np.argmax(outside, axis=1)
    found = outside.any(axis=1) & (first > 0)
    lo = ts[np.maximum(first - 1, 0)]
    hi = ts[first]
    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        inside = sdf(centre + mid[:, None] * u) < 0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    t = np.where(found, 0.5 * (lo + hi), dist)
    return centre + t[:, None] * u
