"""Terraced seating: the bowl's rows as stepped stone terraces with plank benches.

Red Rocks' rows are not benches standing on a slope. Each row is a terrace: a
flat tread you walk along, a riser of stacked red flagstone in front of it, and
a weathered plank bench on dark steel brackets along the riser's top edge (see
the Commons photos linked in the README). This module turns the venue rows
into that, as three OBJs in the terrain's frame (centimetres, X east, Y north,
Z up, Z=0 at the terrain minimum, so no transform in Unreal):

* ``Benches.obj`` (object ``Benches``): one solid per row, its front face on
  the row's line, its flat top at the row's tread level, reaching back to the
  next row and down below the ground. Plus, where row 1 stands too high over
  the ground in front of it, a few front steps. The tops are the treads, the
  front faces the risers; the material tells them apart by their normal.
* ``Bench_Planks.obj``: two boards per row on the riser's top edge, overhanging
  it toward the stage.
* ``Bench_Brackets.obj``: steel brackets under the boards, at the same
  bearings about the stage in every row, so they line up into the radial
  lines that the photo from the top row shows converging on the stage.

**Tread levels.** The ml0w scan's surface is too smooth to show the rows
(no row-spaced pattern above 1 cm in its radial profile), so levels come from
the bowl's profile: the median ground across each row's band, smoothed by a
quadratic in the row number (rows whose band holds something tall, like the FOH
booth, are left out of the fit), then lifted by ``clearance_m``. On the real
scan the rises run 0.36 m at the bottom to 0.53 m at the top and add up to
31 m, against the official 100 ft.

**Clearance.** The scan chunks keep complex collision, and the scan surface in
the bowl is a smooth ramp a few tenths above its fitted ground. Treads lifted
``SCAN_CLEARANCE_M`` (0.45 m) over the profile keep that ramp under them, or
within 15 cm of them, in 97 % of the bowl's 0.5 m cells (measured against the
raw scan points); most of the rest is something standing in a row (2 %),
which the row splits around (``split_around``).

**The scan cut.** ``seating_fade`` writes a copy of the scan's plan-view fade
mask with the seating cut out, at ``CUT_TEXEL_M``, so the terraces replace the
scan's ramp and nothing of the scan shows through a tread. Rows stop short of
anything the scan shows standing in them (``trim_row_ends`` at their ends,
``split_around`` in the middle), so what blocks the player is something they
can see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import shapely
import shapely.prepared
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .benches import GROUND_STEP_M, _clean_polyline, _densify, _face_normal
from .scan_blend import ScanHeights
from .stage import prism
from .terrain_mesh import CM_PER_M

Ground = Callable[[float, float], float]

# Treads over the bowl profile when standing on the scan (see the module doc).
SCAN_CLEARANCE_M = 0.45
# Row lines are only de-duplicated to this tolerance: simplifying them would
# pull the risers off their arcs between vertices.
CLEAN_TOLERANCE_M = 0.01
# Each tread runs this far under the next row's riser.
TREAD_OVERLAP_M = 0.05
# Every row at least this much above the one below it.
MIN_RISE_M = 0.10
# Rows whose band median is off the smooth profile by more than this (or by
# 2.5 sigma) hold something else and are left out of the fit.
REJECT_M = 0.25
# Ground samples along each row's band.
SAMPLE_STEP_M = 1.5
SAMPLE_FRACTIONS = (0.15, 0.5, 0.85)

# Front steps: added in front of row 1 while it stands more than MAX_RISE_M
# over the ground just in front of it; each step rises FRONT_STEP_RISE_M.
MAX_RISE_M = 0.48
FRONT_STEP_RISE_M = 0.40
FRONT_STEP_DEPTH_M = 0.6
FRONT_APRON_M = 0.5  # how far in front of an edge the ground is read
MAX_FRONT_STEPS = 8

# Terrace solids reach this far below the lowest ground under them.
BURY_M = 0.3

# Planks: two boards on the riser's top edge, front edge PLANK_OVERHANG_M in
# front of the riser, back edge PLANK_BACK_M behind it, top PLANK_TOP_M over the
# tread (they rest on the brackets).
PLANK_OVERHANG_M = 0.10
PLANK_BACK_M = 0.195
PLANK_GAP_M = 0.015
PLANK_THICK_M = 0.045
PLANK_TOP_M = 0.06

# Brackets: a steel strap every BRACKET_SPACING_M (at the middle row; the same
# bearings about the stage in every row), down the riser face, under the
# boards and just proud of their top so the line shows from above.
BRACKET_SPACING_M = 2.4
BRACKET_WIDTH_M = 0.06
BRACKET_FRONT_M = 0.115
BRACKET_BACK_M = 0.205
BRACKET_DROP_M = 0.05
BRACKET_PROUD_M = 0.003
BRACKET_END_MARGIN_M = 0.4

# Rows end at their aisles: a row stops where the aisle beside its end (stage.py's
# stair gap, 1.8 m stairs and cheek wall, AISLE_CLEAR_M out from the end and
# AISLE_HALF_WIDTH_M either side) would run into a building (the stage towers)
# or into something the scan shows standing OBSTACLE_TALL_M over its ground (a
# rock wall, a tree). Never more than MAX_TRIM_FRACTION of a row from each end.
AISLE_CLEAR_M = 2.6
AISLE_HALF_WIDTH_M = 1.0
OBSTACLE_TALL_M = 1.0
MAX_TRIM_FRACTION = 0.3
TRIM_STEP_M = 0.25
# A row's trim tapers into its neighbours by at most this much per row, so one
# row cut back by a rock leaves no notch for the aisle stairs to jog around.
TRIM_TAPER_M = 0.6

# Rows split around something standing in them (a scanned object inside a
# row): each piece stops SPLIT_CLEAR_M short of it; pieces shorter than
# MIN_PIECE_M are dropped.
SPLIT_CLEAR_M = 0.5
MIN_PIECE_M = 3.0
# ... and wherever the scan (whose chunks keep collision) stands more than a
# step over the row's own tread: hidden, it would be an invisible ledge.
SPLIT_ABOVE_TREAD_M = 0.55

# Scan cut.
CUT_TEXEL_M = 0.5
CUT_MARGIN_M = 0.15  # the cut reaches this far past the terraces' (and stairs') edges


@dataclass
class Terrace:
    """One stepped terrace: a flat top from ``front`` back ``depth_m`` uphill."""

    name: str
    front: np.ndarray  # (N, 2) local metres: the riser line
    up: float  # +1: uphill is the left of travel along ``front``; -1: the right
    depth_m: float
    level_m: float  # absolute tread elevation
    row_index: int | None = None  # venue row, None for a front step

    def line(self, offset_m: float) -> np.ndarray:
        """The front line moved ``offset_m`` uphill (negative: toward the stage)."""
        return _offset(self.front, self.up * offset_m)

    def plan(self) -> Polygon:
        back = self.line(self.depth_m)
        poly = Polygon(np.vstack([self.front, back[::-1]]))
        return poly if poly.is_valid else poly.buffer(0)


@dataclass
class SeatingLayout:
    rows: list[Terrace]
    front_steps: list[Terrace]
    levels: list[float]
    info: dict = field(default_factory=dict)
    centre: tuple[float, float] | None = None

    @property
    def terraces(self) -> list[Terrace]:
        return self.front_steps + self.rows


@dataclass
class SeatingMesh:
    paths: dict[str, Path]
    triangles: dict[str, int]
    terraces: int
    front_steps: int
    brackets: int
    base_elevation_m: float
    pieces: int = 0  # terraces: rows, plus the extra pieces of rows split around things


# ---- geometry helpers ----------------------------------------------------------

def _normals(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left-of-travel mitred normals at each vertex and their mitre scale."""
    seg = np.diff(pts, axis=0)
    seg = seg / np.hypot(seg[:, 0], seg[:, 1])[:, None]
    seg_n = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    n = len(pts)
    norms = np.zeros((n, 2))
    scale = np.ones(n)
    norms[0], norms[-1] = seg_n[0], seg_n[-1]
    for i in range(1, n - 1):
        m = seg_n[i - 1] + seg_n[i]
        length = float(np.hypot(*m))
        m = m / length if length > 1e-9 else seg_n[i]
        norms[i] = m
        scale[i] = min(1.0 / max(float(m @ seg_n[i]), 1e-9), 2.0)
    return norms, scale


def _offset(pts: np.ndarray, d: float) -> np.ndarray:
    norms, scale = _normals(pts)
    return pts + norms * (d * scale)[:, None]


def _up_sides(lines: Sequence[np.ndarray]) -> list[float]:
    """+1 where the next row (uphill) is to the left of a row's travel, else -1."""
    out = []
    for i, pts in enumerate(lines):
        if len(lines) == 1:
            out.append(1.0)
            continue
        other = LineString(lines[i + 1] if i + 1 < len(lines) else lines[i - 1])
        toward = 1.0 if i + 1 < len(lines) else -1.0
        mid = len(pts) // 2
        norms, _ = _normals(pts)
        left = other.distance(shapely.Point(*(pts[mid] + norms[mid] * 0.1)))
        right = other.distance(shapely.Point(*(pts[mid] - norms[mid] * 0.1)))
        out.append(toward if left < right else -toward)
    return out


def _depths(lines: Sequence[np.ndarray]) -> list[float]:
    """How far back each row's tread runs: to the next row's riser and
    ``TREAD_OVERLAP_M`` under it, so no crack opens between treads; the last
    row, as deep as the gap below it."""
    shapes = [LineString(p) for p in lines]
    if len(shapes) == 1:
        return [1.0]
    gaps = [float(shapes[i].distance(shapes[i + 1])) for i in range(len(shapes) - 1)]
    return [g + TREAD_OVERLAP_M for g in gaps] + [gaps[-1]]


def _fronts(polylines: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [_densify(_clean_polyline(p, CLEAN_TOLERANCE_M), GROUND_STEP_M) for p in polylines]


def _median_ground(lines: Sequence[np.ndarray], ground: Ground) -> float:
    zs = [ground(float(x), float(y)) for line in lines for x, y in _densify(line, SAMPLE_STEP_M)]
    return float(np.median(zs))


# ---- row ends and aisles -----------------------------------------------------------

def seating_obstacles(top: ScanHeights | None, ground: ScanHeights | None, buildings=(), within=None):
    """What an aisle cannot run through: buildings, and every 1 m cell where the
    scan's top stands ``OBSTACLE_TALL_M`` over its ground (``within`` limits the
    cells, e.g. to around the bowl). None when there is nothing."""
    from shapely.geometry import box

    shapes = [b for b in buildings if b is not None and not b.is_empty]
    if top is not None and ground is not None and top.shape == ground.shape:
        with np.errstate(invalid="ignore"):
            tall = (top.heights - ground.heights) > OBSTACLE_TALL_M
        for i, j in zip(*np.nonzero(tall)):
            x, y = top.x0 + j * top.cell, top.y0 + i * top.cell
            if within is None or within.contains(shapely.Point(x + top.cell / 2, y + top.cell / 2)):
                shapes.append(box(x, y, x + top.cell, y + top.cell))
    return unary_union(shapes) if shapes else None


def _end_trim(line: LineString, from_start: bool, obstacles) -> float:
    """How far to pull one end of a row in so its aisle is clear."""
    length = line.length
    for s in np.arange(0.0, MAX_TRIM_FRACTION * length, TRIM_STEP_M):
        a = s if from_start else length - s
        b = a + 0.1 if from_start else a - 0.1
        p = np.array(line.interpolate(a).coords[0])
        q = np.array(line.interpolate(b).coords[0])
        out = (p - q) / max(float(np.hypot(*(p - q))), 1e-9)
        aisle = LineString([p, p + out * AISLE_CLEAR_M]).buffer(AISLE_HALF_WIDTH_M)
        if not aisle.intersects(obstacles):
            return float(s)
    return float(MAX_TRIM_FRACTION * length)


def trim_row_ends(polylines: Sequence[np.ndarray], obstacles) -> list[np.ndarray]:
    """Each row's line with its ends pulled in until the aisle beside them is clear.

    Each end's trim (metres along the row) is then tapered into the rows around
    it, at most ``TRIM_TAPER_M`` less per row away from it, so the row ends make
    a smooth aisle line. Tapering only ever trims more, so every aisle stays
    clear; no end loses more than ``MAX_TRIM_FRACTION`` of its row.
    """
    from shapely.ops import substring

    lines = [np.asarray(p, dtype=float)[:, :2] for p in polylines]
    if obstacles is None or obstacles.is_empty:
        return lines
    prepared = shapely.prepared.prep(obstacles)
    shapes = [LineString(p) for p in lines]
    trims = np.zeros((len(lines), 2))
    for i, line in enumerate(shapes):
        if prepared.intersects(line.buffer(AISLE_CLEAR_M + AISLE_HALF_WIDTH_M)):
            trims[i] = (_end_trim(line, True, obstacles), _end_trim(line, False, obstacles))
    idx = np.arange(len(lines))
    tapered = np.max(trims[None, :, :] - TRIM_TAPER_M * np.abs(idx[:, None] - idx[None, :])[:, :, None], axis=1)
    out = []
    for i, (pts, line) in enumerate(zip(lines, shapes)):
        cap = MAX_TRIM_FRACTION * line.length
        s0, s1 = (float(min(max(t, 0.0), cap)) for t in tapered[i])
        if s0 == 0.0 and s1 == 0.0:
            out.append(pts)
        else:
            out.append(np.array(substring(line, s0, line.length - s1).coords)[:, :2])
    return out


# ---- tread levels ----------------------------------------------------------------

def tread_levels(
    rows,
    polylines: Sequence[np.ndarray],
    ground: Ground | None = None,
    clearance_m: float = 0.0,
) -> tuple[list[float], dict]:
    """Absolute tread elevation of every row, and how it was fitted.

    With ``ground``: the median ground across each row's band (front line to the
    next row), fitted with a quadratic in the row number, rows off that fit
    dropped and refitted, plus ``clearance_m``. Without: the rows' own
    elevations smoothed the same way, with no clearance (clearance is only for
    the scan's collision). Either way every row rises at least ``MIN_RISE_M``.
    """
    if len(rows) != len(polylines):
        raise ValueError(f"{len(rows)} rows but {len(polylines)} polylines")
    if not rows:
        raise ValueError("no rows")
    fronts = _fronts(polylines)
    if ground is None:
        samples = np.array([r.elevation_m for r in rows], dtype=float)
        clearance_m = 0.0
    else:
        ups, depths = _up_sides(fronts), _depths(fronts)
        samples = np.array([
            _median_ground([_offset(f, u * d * frac) for frac in SAMPLE_FRACTIONS], ground)
            for f, u, d in zip(fronts, ups, depths)
        ])

    idx = np.arange(1, len(rows) + 1, dtype=float)
    deg = min(2, len(rows) - 1)
    keep = np.ones(len(rows), bool)
    fit = samples.copy()
    if deg > 0:
        for _ in range(4):
            coef = np.polyfit(idx[keep], samples[keep], deg)
            fit = np.polyval(coef, idx)
            res = samples - fit
            new = np.abs(res) < max(REJECT_M, 2.5 * float(np.std(res[keep])))
            if new.sum() <= deg + 1 or np.array_equal(new, keep):
                break
            keep = new
    levels = fit + clearance_m
    for i in range(1, len(levels)):
        levels[i] = max(levels[i], levels[i - 1] + MIN_RISE_M)
    rises = np.diff(levels) if len(levels) > 1 else np.array([0.0])
    info = {
        "source": "ground" if ground is not None else "row elevations",
        "clearance_m": clearance_m,
        "rows_ignored": [int(i) for i in idx[~keep]],
        "rise_min_m": round(float(rises.min()), 3),
        "rise_max_m": round(float(rises.max()), 3),
        "rise_mean_m": round(float(rises.mean()), 3),
        "total_rise_m": round(float(levels[-1] - levels[0]), 3),
    }
    return [float(v) for v in levels], info


# ---- layout ----------------------------------------------------------------------

def _circle_centre(lines: Sequence[np.ndarray]) -> tuple[float, float] | None:
    """Least-squares centre the rows curve about, or None if they barely curve.

    The rows are concentric circles, each with its own radius: every point
    satisfies x^2 + y^2 = 2 cx x + 2 cy y + k_row, one k per row.
    """
    pts = np.vstack(lines)
    which = np.repeat(np.arange(len(lines)), [len(p) for p in lines])
    a = np.zeros((len(pts), 2 + len(lines)))
    a[:, 0], a[:, 1] = 2 * pts[:, 0], 2 * pts[:, 1]
    a[np.arange(len(pts)), 2 + which] = 1.0
    b = (pts ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(sol[0]), float(sol[1])
    radius = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    if not np.all(np.isfinite(radius)) or np.median(radius) > 5000.0:
        return None
    # Each row must really be a circle about it (straight rows are not).
    for k in range(len(lines)):
        r = radius[which == k]
        if np.ptp(r) > 0.05 * max(float(np.median(r)), 1e-9) + 0.05:
            return None
    return cx, cy


def _ledges(t: Terrace, top: ScanHeights | None):
    """Cells under terrace ``t`` where the scan stands a step over its tread."""
    from shapely.geometry import box

    if top is None:
        return None
    plan = t.plan()
    x0, y0, x1, y1 = plan.bounds
    j0, j1 = int((x0 - top.x0) // top.cell), int((x1 - top.x0) // top.cell) + 1
    i0, i1 = int((y0 - top.y0) // top.cell), int((y1 - top.y0) // top.cell) + 1
    j0, i0 = max(j0, 0), max(i0, 0)
    sub = top.heights[i0:i1, j0:j1]
    if sub.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        high = sub > t.level_m + SPLIT_ABOVE_TREAD_M
    cells = []
    for i, j in zip(*np.nonzero(high)):
        x, y = top.x0 + (j0 + j) * top.cell, top.y0 + (i0 + i) * top.cell
        cell = box(x, y, x + top.cell, y + top.cell)
        if cell.intersects(plan):
            cells.append(cell)
    return unary_union(cells) if cells else None


def split_around(terraces: Sequence[Terrace], obstacles) -> list[Terrace]:
    """Each terrace whose plan runs into an obstacle, cut into the pieces of its
    row line that stay ``SPLIT_CLEAR_M`` clear of it (pieces under
    ``MIN_PIECE_M`` dropped). Untouched terraces pass through as they are."""
    from shapely.geometry import MultiLineString

    prepared = shapely.prepared.prep(obstacles)
    out: list[Terrace] = []
    for t in terraces:
        if not prepared.intersects(t.plan()):
            out.append(t)
            continue
        # Clear of the obstacle over the whole tread: buffer it by the tread
        # depth too, then keep the row line outside.
        keep_out = obstacles.buffer(SPLIT_CLEAR_M + t.depth_m)
        rest = LineString(t.front).difference(keep_out)
        pieces = list(rest.geoms) if isinstance(rest, MultiLineString) else ([rest] if not rest.is_empty else [])
        for k, piece in enumerate(p for p in pieces if p.length >= MIN_PIECE_M):
            pts = np.array(piece.coords)[:, :2]
            out.append(Terrace(f"{t.name}_{k + 1}", pts, t.up, t.depth_m, t.level_m, t.row_index))
    return out


def _front_steps(row1: Terrace, ground: Ground) -> list[Terrace]:
    """Steps down from row 1 toward the stage while it stands too high."""
    steps: list[Terrace] = []
    top, off = row1.level_m, 0.0
    for k in range(MAX_FRONT_STEPS):
        apron = _median_ground([row1.line(-(off + FRONT_APRON_M))], ground)
        if top - apron <= MAX_RISE_M:
            break
        off += FRONT_STEP_DEPTH_M
        top -= FRONT_STEP_RISE_M
        steps.append(Terrace(f"front_step_{k + 1}", row1.line(-off), row1.up, FRONT_STEP_DEPTH_M, top))
    return steps[::-1]  # lowest first


def seating_layout(
    rows,
    polylines: Sequence[np.ndarray],
    ground: Ground | None = None,
    floor: Ground | None = None,
    clearance_m: float = 0.0,
    top_depth_m: float | None = None,
    obstacles=None,
    top: ScanHeights | None = None,
) -> SeatingLayout:
    """Where every terrace goes and how high. ``floor`` is only used by the export.

    With ``obstacles`` (``seating_obstacles``), a row whose terrace would run
    into one is split into pieces either side of it (``split_around``); with
    ``top`` (the scan's highest point per cell), so is a row wherever the scan
    stands more than ``SPLIT_ABOVE_TREAD_M`` over its tread.
    """
    levels, info = tread_levels(rows, polylines, ground, clearance_m)
    fronts = _fronts(polylines)
    ups, depths = _up_sides(fronts), _depths(fronts)
    if top_depth_m is not None:
        depths[-1] = top_depth_m
    terraces = [
        Terrace(f"row_{r.index}", f, u, d, lv, r.index)
        for r, f, u, d, lv in zip(rows, fronts, ups, depths, levels)
    ]
    steps = _front_steps(terraces[0], ground) if ground is not None else []
    info["front_steps"] = len(steps)
    if (obstacles is not None and not obstacles.is_empty) or top is not None:
        before = len(terraces)
        split: list[Terrace] = []
        for t in terraces:
            parts = [g for g in (obstacles, _ledges(t, top)) if g is not None and not g.is_empty]
            split += split_around([t], unary_union(parts)) if parts else [t]
        terraces = split
        info["row_pieces_added"] = len(terraces) - before
    return SeatingLayout(terraces, steps, levels, info, _circle_centre(fronts))


# ---- export ----------------------------------------------------------------------

Solid = tuple[np.ndarray, list, np.ndarray, list]


def _sweep(centre: np.ndarray, depth: float, z_lo: float, z_hi: float) -> Solid:
    """A box section ``depth`` wide swept along ``centre``: flat top and bottom.

    Returns (verts_m, tris, uvs, tri_uvs), counter-clockwise from outside. UVs
    are metres: U along the line; V across on the top and bottom, up on the sides.
    """
    norms, scale = _normals(centre)
    half = depth / 2.0
    left = centre + norms * (half * scale)[:, None]
    right = centre - norms * (half * scale)[:, None]
    seg = np.hypot(*np.diff(centre, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    h = z_hi - z_lo
    n = len(centre)
    verts = []
    for i in range(n):
        verts += [(*left[i], z_lo), (*left[i], z_hi), (*right[i], z_hi), (*right[i], z_lo)]
    verts = np.array(verts)
    uvs: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []
    tuvs: list[tuple[int, int, int]] = []

    def quad(vs, us):
        a, b, c, d = vs
        ua, ub, uc, ud = us
        tris.extend([(a, b, c), (a, c, d)])
        tuvs.extend([(ua, ub, uc), (ua, uc, ud)])

    for i in range(n - 1):
        j = i + 1
        lb0, lt0, rt0, rb0 = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        lb1, lt1, rt1, rb1 = 4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3
        k = len(uvs)
        uvs += [(s[i], 0.0), (s[i], depth), (s[j], depth), (s[j], 0.0)]  # top/bottom
        uvs += [(s[i], 0.0), (s[i], h), (s[j], h), (s[j], 0.0)]  # sides
        quad((lt0, rt0, rt1, lt1), (k, k + 1, k + 2, k + 3))
        quad((lb0, lb1, rb1, rb0), (k, k + 3, k + 2, k + 1))
        quad((lb0, lt0, lt1, lb1), (k + 4, k + 5, k + 6, k + 7))
        quad((rb0, rb1, rt1, rt0), (k + 4, k + 7, k + 6, k + 5))
    k = len(uvs)
    uvs += [(0.0, 0.0), (0.0, h), (depth, h), (depth, 0.0)]
    last = 4 * (n - 1)
    quad((0, 3, 2, 1), (k, k + 3, k + 2, k + 1))
    quad((last, last + 1, last + 2, last + 3), (k, k + 1, k + 2, k + 3))

    a, b, c = (verts[[t[m] for t in tris]] for m in range(3))
    if float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) < 0:
        tris = [(t[0], t[2], t[1]) for t in tris]
        tuvs = [(t[0], t[2], t[1]) for t in tuvs]
    return verts, tris, np.array(uvs), tuvs


def _band(t: Terrace, a: float, b: float) -> np.ndarray:
    """The centre line of the strip ``a``..``b`` metres uphill of ``t.front``."""
    return t.line(0.5 * (a + b))


def _terrace_solid(t: Terrace, below: float | None, base: float, ground, floor) -> Solid:
    if ground is not None:
        lows = [
            min(s(float(x), float(y)) for s in (ground, floor) if s is not None)
            for line in (t.front, t.line(0.5 * t.depth_m), t.line(t.depth_m))
            for x, y in line
        ]
        z_lo = min(lows) - BURY_M
    else:
        z_lo = (below if below is not None else t.level_m - MAX_RISE_M) - BURY_M
    z_lo = min(z_lo, t.level_m - BURY_M)
    return _sweep(_band(t, 0.0, t.depth_m), t.depth_m, z_lo - base, t.level_m - base)


def _planks(t: Terrace, base: float) -> list[Solid]:
    w = (PLANK_OVERHANG_M + PLANK_BACK_M - PLANK_GAP_M) / 2.0
    z1 = t.level_m + PLANK_TOP_M - base
    z0 = z1 - PLANK_THICK_M
    a0 = -PLANK_OVERHANG_M
    return [
        _sweep(_band(t, a0, a0 + w), w, z0, z1),
        _sweep(_band(t, a0 + w + PLANK_GAP_M, PLANK_BACK_M), w, z0, z1),
    ]


def _bracket_stations(t: Terrace, centre, axis: float, step: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """(point on the row line, uphill unit vector) at every bracket bearing."""
    pts = t.front
    cx, cy = centre
    bear = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    bear = np.remainder(bear - axis + np.pi, 2 * np.pi) - np.pi
    if bear[-1] < bear[0]:
        pts, bear = pts[::-1], bear[::-1]
    if np.any(np.diff(bear) <= 0):
        return []
    r = float(np.median(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    margin = BRACKET_END_MARGIN_M / r
    k0 = math.ceil((bear[0] + margin) / step)
    k1 = math.floor((bear[-1] - margin) / step)
    norms, _ = _normals(t.front)
    up_ref = norms[len(norms) // 2] * t.up
    out = []
    for k in range(k0, k1 + 1):
        theta = k * step
        x = float(np.interp(theta, bear, pts[:, 0]))
        y = float(np.interp(theta, bear, pts[:, 1]))
        # Exactly on the chord at that bearing: intersect the ray with the segment.
        j = int(np.clip(np.searchsorted(bear, theta) - 1, 0, len(pts) - 2))
        d = np.array([math.cos(theta + axis), math.sin(theta + axis)])
        p0, p1 = pts[j], pts[j + 1]
        e = p1 - p0
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) > 1e-12:
            q = p0 - np.array([cx, cy])
            u = (d[0] * q[1] - d[1] * q[0]) / den
            x, y = (p0 + e * float(np.clip(-u, 0.0, 1.0)))
        radial = d if float(d @ up_ref) >= 0 else -d
        out.append((np.array([x, y]), radial))
    return out


def _brackets(t: Terrace, centre, axis: float, step: float, base: float) -> list[Solid]:
    z0 = t.level_m - BRACKET_DROP_M - base
    z1 = t.level_m + PLANK_TOP_M + BRACKET_PROUD_M - base
    out = []
    for p, n in _bracket_stations(t, centre, axis, step):
        tang = np.array([-n[1], n[0]]) * (BRACKET_WIDTH_M / 2)
        f, b = p - n * BRACKET_FRONT_M, p + n * BRACKET_BACK_M
        out.append(prism(np.array([f - tang, f + tang, b + tang, b - tang]), z0, z1))
    return out


def _write_obj(path: Path, name: str, solids: Sequence[Solid], header: str) -> int:
    lines = [f"# {header}", "o " + name]
    vs, uvs, faces = [], [], []
    normals: dict[tuple[float, float, float], int] = {}
    v_off = uv_off = 0
    for verts, tris, suv, tuvs in solids:
        vs.append(verts)
        uvs.append(suv)
        for t, tu in zip(tris, tuvs):
            nid = normals.setdefault(_face_normal(verts, t), len(normals) + 1)
            faces.append("f " + " ".join(f"{v_off + v + 1}/{uv_off + u + 1}/{nid}" for v, u in zip(t, tu)))
        v_off += len(verts)
        uv_off += len(suv)
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in np.vstack(vs) * CM_PER_M]
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in np.vstack(uvs)]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in normals]
    lines += faces
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(faces)


def export_seating(
    layout: SeatingLayout,
    out_dir: Path,
    base_elevation_m: float,
    ground: Ground | None = None,
    floor: Ground | None = None,
) -> SeatingMesh:
    """Write ``Benches.obj``, ``Bench_Planks.obj`` and ``Bench_Brackets.obj`` to ``out_dir``."""
    out_dir = Path(out_dir)
    base = base_elevation_m
    solids, below = [], None
    for t in layout.terraces:
        solids.append(_terrace_solid(t, below, base, ground, floor))
        below = t.level_m
    planks = [s for t in layout.rows for s in _planks(t, base)]
    brackets: list[Solid] = []
    if layout.centre is not None:
        cx, cy = layout.centre
        mids = [t.front[len(t.front) // 2] for t in layout.rows]
        axis = math.atan2(float(np.mean([m[1] - cy for m in mids])), float(np.mean([m[0] - cx for m in mids])))
        r_mid = float(np.median([np.hypot(m[0] - cx, m[1] - cy) for m in mids]))
        step = BRACKET_SPACING_M / r_mid
        brackets = [s for t in layout.rows for s in _brackets(t, layout.centre, axis, step, base)]

    paths = {name: out_dir / f"{name}.obj" for name in ("Benches", "Bench_Planks", "Bench_Brackets")}
    tris = {
        "Benches": _write_obj(paths["Benches"], "Benches", solids,
                              f"Red Rocks terraced seating: {len(layout.rows)} rows, "
                              f"{len(layout.front_steps)} front steps; cm, base {base:.3f} m"),
        "Bench_Planks": _write_obj(paths["Bench_Planks"], "Bench_Planks", planks,
                                   "Red Rocks bench planks, two boards per row; cm"),
    }
    if brackets:
        tris["Bench_Brackets"] = _write_obj(paths["Bench_Brackets"], "Bench_Brackets", brackets,
                                            "Red Rocks bench brackets, lined up about the stage; cm")
    else:
        paths.pop("Bench_Brackets")
    rows = len({t.row_index for t in layout.rows})
    return SeatingMesh(paths, tris, rows, len(layout.front_steps), len(brackets), base, len(layout.rows))


# ---- scan cut --------------------------------------------------------------------

def _resample(alpha: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Bilinear resample of a texel-centred grid onto another over the same extent."""
    sy, sx = alpha.shape
    ys = (np.arange(ny) + 0.5) * sy / ny - 0.5
    xs = (np.arange(nx) + 0.5) * sx / nx - 0.5
    rows = np.stack([np.interp(xs, np.arange(sx), alpha[i]) for i in range(sy)])
    return np.stack([np.interp(ys, np.arange(sy), rows[:, j]) for j in range(nx)], axis=1)


def seating_fade(
    alpha: np.ndarray,
    place: dict,
    layout: SeatingLayout,
    top: ScanHeights | None = None,
    extra=None,
) -> tuple[np.ndarray, dict]:
    """The scan's fade mask with the seating cut out, and its placement.

    ``alpha`` and ``place`` are the scan's mask (row 0 = north) and placement
    JSON from ``tools/scan_heights.py``. The result covers the same extent at
    ``CUT_TEXEL_M`` texels: 0 over the terraces and ``extra`` plans (the
    row-end stairs, the sound booth), ``CUT_MARGIN_M`` past their edges so no
    sliver of scan shows along them; the original elsewhere. Nothing of the scan shows inside
    the seating: anything standing in a row has split the row around it
    (``split_around``), so it sits in a gap, outside the cut. ``top`` is
    accepted for compatibility and unused.
    """
    width_m, height_m = place["width_cm"] / 100.0, place["height_cm"] / 100.0
    nx, ny = int(round(width_m / CUT_TEXEL_M)), int(round(height_m / CUT_TEXEL_M))
    tx, ty = width_m / nx, height_m / ny
    out = np.clip(_resample(np.asarray(alpha, float), ny, nx), 0.0, 1.0)
    east = place["x0_cm"] / 100.0 + (np.arange(nx) + 0.5) * tx
    north = place["y1_cm"] / 100.0 - (np.arange(ny) + 0.5) * ty
    e, n = np.meshgrid(east, north)

    plans = [t.plan() for t in layout.terraces] + [p for p in (extra or []) if p is not None]
    cut = shapely.contains_xy(unary_union(plans).buffer(CUT_MARGIN_M), e, n)

    out[cut] = 0.0
    meta = dict(place)
    meta["texels"] = [nx, ny]
    meta["seating_cut"] = {
        "texel_m": CUT_TEXEL_M, "margin_m": CUT_MARGIN_M,
        "cut_m2": round(float(cut.sum()) * tx * ty, 1),
    }
    return out, meta
