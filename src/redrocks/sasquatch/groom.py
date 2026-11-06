"""Grow the sasquatch's fur as hair strands and write them as .rrgroom.

Roots are scattered over the skin by area and a per-vertex density; each strand
leaves the skin at an angle, lies down along a comb direction, droops, frizzes
and clumps with its neighbours, and fades from a dark auburn root to a lighter
grey-brown tip. Everything is in the body frame (cm, Z up), the space the
skeletal mesh imports into, so the groom binds to it without a transform.

.rrgroom is read by the RedRocksEditor module's translator (see its header for
the layout).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import mesh_ops

MAGIC = b"RRGROOM1"

# Linear RGB (sRGB in the comments). A dark brown-auburn root, a reddish-brown
# middle and grey-brown tips, as in the Patterson-Gimlin film and most reports.
ROOT_COLOUR = np.array([0.026, 0.0125, 0.0075])   # sRGB  45  29  22
MID_COLOUR = np.array([0.095, 0.036, 0.016])      # sRGB  87  53  35
TIP_COLOUR = np.array([0.20, 0.140, 0.092])       # sRGB 124 105  85
GREY_COLOUR = np.array([0.30, 0.27, 0.24])        # an old animal's odd grey hairs


@dataclass
class FurField:
    """Per-vertex controls for one mesh."""

    density: np.ndarray   # strands per cm^2
    length: np.ndarray    # cm
    comb: np.ndarray      # (n, 3) unit direction the hair lies along (any length; projected to the skin)
    lift: np.ndarray      # 0..1 how far it stands off the skin at the root
    droop: np.ndarray     # 0..1 how much it falls under gravity (-Z) toward the tip
    clump: np.ndarray     # 0..1 how strongly tips gather into clumps
    tip_light: np.ndarray  # 0..1 how much the tips lighten (sun-bleached on the back and shoulders)


@dataclass
class Strands:
    counts: np.ndarray    # (S,) points per strand
    points: np.ndarray    # (P, 3)
    widths: np.ndarray    # (P,)
    colours: np.ndarray   # (P, 3) linear RGB
    root_uv: np.ndarray   # (S, 2)
    groups: np.ndarray    # (S,)
    root_face: np.ndarray  # (S,) face each root lies on (for splitting by region)

    @property
    def roots(self) -> np.ndarray:
        starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        return self.points[starts]

    def subset(self, keep: np.ndarray) -> "Strands":
        """The strands where `keep` (bool per strand) is true."""
        starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        idx = np.concatenate([np.arange(s, s + c) for s, c in zip(starts[keep], self.counts[keep])]) \
            if keep.any() else np.zeros(0, int)
        return Strands(self.counts[keep], self.points[idx], self.widths[idx], self.colours[idx],
                       self.root_uv[keep], self.groups[keep], self.root_face[keep])


def _tangent(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    t = v - (v * n).sum(axis=1, keepdims=True) * n
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.cross(n, np.array([0.0, 0.0, 1.0]))
    fallback /= np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-9)
    return np.where(ln > 1e-6, t / np.maximum(ln, 1e-9), fallback)


def grow(positions: np.ndarray, normals: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
         field: FurField, rng: np.random.Generator, points_per_strand: int = 9,
         root_width: float = 0.034, tip_width: float = 0.009) -> Strands:
    """Grow every strand. `positions`/`normals`/`uvs` are per vertex."""
    roots, face_of, bary = mesh_ops.sample_surface(positions, faces, field.density, rng)
    s_count = len(roots)
    tri = faces[face_of]

    def interp(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return (a[tri] * bary).sum(axis=1)
        return np.einsum("ki,kij->kj", bary, a[tri])

    n = interp(normals)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    comb = _tangent(interp(field.comb), n)
    length = interp(field.length) * rng.lognormal(0.0, 0.18, s_count)
    lift = interp(field.lift) * rng.uniform(0.75, 1.25, s_count)
    droop = interp(field.droop)
    clump = interp(field.clump)
    tip_light = interp(field.tip_light)
    root_uv = interp(uvs)

    # A little per-strand wander of the comb, so the coat isn't combed flat.
    side = np.cross(n, comb)
    angle = rng.normal(0.0, 0.22, s_count)
    comb = comb * np.cos(angle)[:, None] + side * np.sin(angle)[:, None]

    k = points_per_strand
    s = np.linspace(0.0, 1.0, k)                                           # (k,)
    step = (length / (k - 1))[:, None]
    pts = np.zeros((s_count, k, 3))
    pts[:, 0] = roots + n * 0.02
    down = np.array([0.0, 0.0, -1.0])
    for i in range(1, k):
        si = s[i - 1]
        stand = lift * (1.0 - si) ** 2
        d = n * stand[:, None] + comb * (1.0 - stand)[:, None] + down * (droop * si)[:, None]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        pts[:, i] = pts[:, i - 1] + d * step

    # Frizz: a slow, random curl perpendicular to the strand, growing toward the tip.
    u = side
    v = np.cross(comb, side)
    phase1 = rng.uniform(0, 2 * np.pi, s_count)
    phase2 = rng.uniform(0, 2 * np.pi, s_count)
    freq = rng.uniform(1.5, 3.5, s_count)
    amp = (0.04 * length * rng.uniform(0.5, 1.5, s_count))[:, None] * s[None, :] ** 1.2
    wave_u = np.sin(freq[:, None] * s[None, :] * 2 * np.pi + phase1[:, None]) * amp
    wave_v = np.sin(freq[:, None] * s[None, :] * 2 * np.pi + phase2[:, None]) * amp * 0.6
    pts += wave_u[..., None] * u[:, None, :] + wave_v[..., None] * v[:, None, :]

    # Clumps: about one per 2.5 cm; tips gather toward their clump's lead strand.
    lead = rng.random(s_count) < 0.12
    lead_idx = np.flatnonzero(lead)
    if len(lead_idx):
        nearest = mesh_ops.grid_nearest(roots, roots[lead_idx], cell=3.0)
        has = nearest >= 0
        g = lead_idx[np.where(has, nearest, 0)]
        pull = (clump * has)[:, None] * (s[None, :] ** 0.9)
        target = pts[g] - pts[g][:, :1] + pts[:, :1]                        # the lead's shape from this root
        tip_gather = pts[g] - pts                                           # and toward the lead itself
        pts = pts + pull[..., None] * (0.55 * (target - pts) + 0.45 * tip_gather)

    # Colour: root to tip, with per-strand variation and a few grey hairs.
    t = s[None, :, None]
    mid_mix = np.clip(t / 0.45, 0, 1)
    tip_mix = np.clip((t - 0.45) / 0.55, 0, 1) * (0.55 + 0.45 * tip_light[:, None, None])
    col = ROOT_COLOUR * (1 - mid_mix) + MID_COLOUR * mid_mix
    col = col * (1 - tip_mix) + TIP_COLOUR * tip_mix
    col = col * rng.lognormal(0.0, 0.22, s_count)[:, None, None]
    grey = rng.random(s_count) < 0.03
    col[grey] = col[grey] * 0.4 + GREY_COLOUR * 0.6 * t[0]

    width = (root_width + (tip_width - root_width) * s ** 1.5)[None, :] * np.clip(length / 8.0, 0.55, 1.0)[:, None]

    return Strands(
        counts=np.full(s_count, k, dtype=np.int32),
        points=pts.reshape(-1, 3),
        widths=width.reshape(-1),
        colours=np.clip(col, 0, 1).reshape(-1, 3),
        root_uv=root_uv,
        groups=np.zeros(s_count, dtype=np.int32),
        root_face=face_of,
    )


def write_rrgroom(strands: Strands, path: str | Path) -> None:
    s = len(strands.counts)
    p = len(strands.points)
    if s == 0 or int(strands.counts.sum()) != p:
        raise ValueError("strand counts do not match the points")
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<II", s, p))
        fh.write(strands.counts.astype("<i4").tobytes())
        fh.write(strands.points.astype("<f4").tobytes())
        fh.write(strands.widths.astype("<f4").tobytes())
        fh.write(strands.colours.astype("<f4").tobytes())
        fh.write(strands.root_uv.astype("<f4").tobytes())
        fh.write(strands.groups.astype("<i4").tobytes())


def read_rrgroom(path: str | Path) -> Strands:
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError("not an .rrgroom file")
    s, p = struct.unpack_from("<II", data, 8)
    o = 16

    def take(dtype, count):
        nonlocal o
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=o)
        o += arr.nbytes
        return arr.copy()

    counts = take("<i4", s)
    points = take("<f4", 3 * p).reshape(p, 3)
    widths = take("<f4", p)
    colours = take("<f4", 3 * p).reshape(p, 3)
    root_uv = take("<f4", 2 * s).reshape(s, 2)
    groups = take("<i4", s)
    return Strands(counts, points, widths, colours, root_uv, groups, np.zeros(s, int))
