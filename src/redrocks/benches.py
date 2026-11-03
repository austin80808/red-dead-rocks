"""Turn the generated seating-bowl rows into bench geometry for Unreal.

Each venue row becomes one solid bench: its cross-section polyline swept into a
strip ``depth`` metres front-to-back, standing ``seat_height`` metres above the
row's tread elevation. All benches go into a single OBJ object (``Benches``).

The OBJ shares the terrain mesh's frame so it lines up with
``RedRocks_Terrain.obj`` with no transform in Unreal: centimetres, X east,
Y north, Z up, centred on the same origin, Z measured from the terrain's lowest
elevation. Pass the same base elevation the terrain used (see
``terrain_base_elevation``).

Like the terrain and monolith OBJs, every face carries a UV index: Unreal's OBJ
importer silently decimates meshes without a UV channel. U is distance along the
row in metres; V is across (tread) or up (risers) in metres.

Depth and height come from the venue data where it says something:

* seat depth defaults to 0.9 m but is clamped to 95% of the gap to the
  neighbouring row, so adjacent benches never overlap (the gap is the tread
  depth the generator actually produced);
* the seat top is 0.45 m above the row's own tread elevation, as specified. The
  row-to-row riser (about 0.44 m over the official 100 ft rise) is therefore
  already carried by the row elevations themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from shapely.geometry import LineString, Polygon

from .geo import LocalProjection
from .terrain_mesh import CM_PER_M, DEFAULT_GRID, resample_to_grid
from .venue import ROWS, Row, slice_polygon

SEAT_DEPTH_M = 0.9
SEAT_HEIGHT_M = 0.45
# Fraction of the gap to the neighbouring row a bench may occupy.
MAX_GAP_FRACTION = 0.95
# Polylines are simplified to this tolerance (metres) before sweeping.
SIMPLIFY_TOLERANCE_M = 0.1
# Mitre joins on sharp bends are capped so they cannot spike.
MAX_MITRE = 2.0


@dataclass
class BenchMesh:
    path: Path
    benches: int
    vertices: int
    triangles: int
    base_elevation_m: float


def terrain_base_elevation(dem, grid: int = DEFAULT_GRID) -> float:
    """The Z=0 elevation of ``export_terrain_obj(dem, ..., grid=grid)``.

    The terrain subtracts the minimum of its *resampled* grid, so the grid size
    must match the one used for the terrain mesh.
    """
    return float(np.nanmin(resample_to_grid(dem, grid)))


def row_polylines(
    venue,
    footprint_latlon: Sequence[tuple[float, float]],
    projection: LocalProjection,
    n_rows: int = ROWS,
) -> list[np.ndarray]:
    """Recover each venue row's cross-section polyline, in local metres (E, N).

    ``Row`` stores only a centre, length and elevation, so the chords are
    re-cut from the footprint with the same slicer ``build_venue`` used and
    matched to rows by midpoint. ``n_rows`` must equal the value given to
    ``build_venue`` (its default is ``venue.ROWS``). Returned in venue-row order.
    """
    projected = [projection.to_local(lat, lon)[:2] for lat, lon in footprint_latlon]
    polygon = Polygon(projected)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    slices = slice_polygon(polygon, n_rows)
    mids = np.array([[g.interpolate(0.5, normalized=True).x, g.interpolate(0.5, normalized=True).y] for g in slices])

    lines: list[np.ndarray] = []
    for row in venue.rows:
        d = np.hypot(mids[:, 0] - row.centre_east, mids[:, 1] - row.centre_north)
        k = int(np.argmin(d))
        if d[k] > 1.0:
            raise ValueError(
                f"row {row.index}: no footprint slice within 1 m of its centre "
                f"(n_rows={n_rows} must match build_venue)"
            )
        lines.append(np.array(slices[k].coords, dtype=float)[:, :2])
    return lines


def _clean_polyline(points: np.ndarray, tolerance_m: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)[:, :2]
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > 1e-6:
            keep.append(i)
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("bench polyline needs at least two distinct points")
    if tolerance_m > 0 and len(pts) > 2:
        simple = np.array(LineString(pts).simplify(tolerance_m, preserve_topology=False).coords)
        if len(simple) >= 2:
            pts = simple[:, :2]
    return pts


def _gaps(rows: Sequence[Row]) -> list[float]:
    """Distance from each row's centre to its nearest neighbouring row."""
    centres = np.array([[r.centre_east, r.centre_north] for r in rows])
    out = []
    for i in range(len(rows)):
        d = []
        if i > 0:
            d.append(float(np.hypot(*(centres[i] - centres[i - 1]))))
        if i < len(rows) - 1:
            d.append(float(np.hypot(*(centres[i] - centres[i + 1]))))
        out.append(min(d) if d else math.inf)
    return out


def _bench_solid(
    pts: np.ndarray, depth_m: float, z0_m: float, height_m: float
) -> tuple[np.ndarray, list[tuple[int, int, int]], np.ndarray, list[tuple[int, int, int]]]:
    """Sweep a rectangular section along ``pts``.

    Returns (vertices_m[N,3], triangle vertex ids, uvs_m[M,2], triangle uv ids),
    ids local to this bench and 0-based. Triangles wind counter-clockwise seen
    from outside.
    """
    n = len(pts)
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    tang = seg / seg_len[:, None]
    seg_norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)  # left of travel

    # Mitred normal at each station.
    norms = np.zeros((n, 2))
    scale = np.ones(n)
    norms[0], norms[-1] = seg_norm[0], seg_norm[-1]
    for i in range(1, n - 1):
        m = seg_norm[i - 1] + seg_norm[i]
        m_len = np.hypot(*m)
        m = m / m_len if m_len > 1e-9 else seg_norm[i]
        norms[i] = m
        scale[i] = min(1.0 / max(float(m @ seg_norm[i]), 1e-9), MAX_MITRE)

    half = depth_m / 2.0
    left = pts + norms * (half * scale)[:, None]
    right = pts - norms * (half * scale)[:, None]

    z_lo, z_hi = z0_m, z0_m + height_m
    # Station i: 4 section corners. L = left, R = right; B = bottom, T = top.
    verts = []
    for i in range(n):
        verts.append((*left[i], z_lo))   # 4i+0  LB
        verts.append((*left[i], z_hi))   # 4i+1  LT
        verts.append((*right[i], z_hi))  # 4i+2  RT
        verts.append((*right[i], z_lo))  # 4i+3  RB
    verts = np.array(verts)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])  # distance along the row

    uvs: list[tuple[float, float]] = []
    uv_top0 = len(uvs)
    for i in range(n):
        uvs += [(s[i], 0.0), (s[i], depth_m)]          # top: LT, RT   (V = across)
    uv_left0 = len(uvs)
    for i in range(n):
        uvs += [(s[i], 0.0), (s[i], height_m)]         # left riser: LB, LT (V = up)
    uv_right0 = len(uvs)
    for i in range(n):
        uvs += [(s[i], 0.0), (s[i], height_m)]         # right riser: RB, RT
    uv_bot0 = len(uvs)
    for i in range(n):
        uvs += [(s[i], 0.0), (s[i], depth_m)]          # bottom: LB, RB
    uv_cap0 = len(uvs)
    uvs += [(0.0, 0.0), (0.0, height_m), (depth_m, height_m), (depth_m, 0.0)]  # start cap
    uvs += [(0.0, 0.0), (0.0, height_m), (depth_m, height_m), (depth_m, 0.0)]  # end cap

    tris: list[tuple[int, int, int]] = []
    tuvs: list[tuple[int, int, int]] = []

    def quad(a, b, c, d, ua, ub, uc, ud):
        tris.append((a, b, c))
        tuvs.append((ua, ub, uc))
        tris.append((a, c, d))
        tuvs.append((ua, uc, ud))

    for i in range(n - 1):
        j = i + 1
        LB0, LT0, RT0, RB0 = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        LB1, LT1, RT1, RB1 = 4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3
        # (T, L, Z) = (travel, left, up) is right-handed, so each quad is wound
        # to give: top +Z, bottom -Z, left riser +L, right riser -L.
        quad(LT0, RT0, RT1, LT1, uv_top0 + 2 * i, uv_top0 + 2 * i + 1, uv_top0 + 2 * j + 1, uv_top0 + 2 * j)
        quad(LB0, LB1, RB1, RB0, uv_bot0 + 2 * i, uv_bot0 + 2 * j, uv_bot0 + 2 * j + 1, uv_bot0 + 2 * i + 1)
        quad(LB0, LT0, LT1, LB1, uv_left0 + 2 * i, uv_left0 + 2 * i + 1, uv_left0 + 2 * j + 1, uv_left0 + 2 * j)
        quad(RB0, RB1, RT1, RT0, uv_right0 + 2 * i, uv_right0 + 2 * j, uv_right0 + 2 * j + 1, uv_right0 + 2 * i + 1)

    # End caps.
    last = 4 * (n - 1)
    quad(0, 3, 2, 1, uv_cap0, uv_cap0 + 3, uv_cap0 + 2, uv_cap0 + 1)
    quad(last, last + 1, last + 2, last + 3, uv_cap0 + 4, uv_cap0 + 5, uv_cap0 + 6, uv_cap0 + 7)

    # Safety net: a positive signed volume means counter-clockwise from outside.
    # The quads above are already wound outward, so this should never flip.
    a = verts[[t[0] for t in tris]]
    b = verts[[t[1] for t in tris]]
    c = verts[[t[2] for t in tris]]
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0
    if volume < 0:
        tris = [(t[0], t[2], t[1]) for t in tris]
        tuvs = [(t[0], t[2], t[1]) for t in tuvs]

    return verts, tris, np.array(uvs), tuvs


def export_benches_obj(
    rows: Sequence[Row],
    polylines: Sequence[np.ndarray],
    out_path: Path,
    base_elevation_m: float,
    depth_m: float = SEAT_DEPTH_M,
    seat_height_m: float = SEAT_HEIGHT_M,
    simplify_m: float = SIMPLIFY_TOLERANCE_M,
    fit_to_row_gap: bool = True,
) -> BenchMesh:
    """Write every row as a bench in one OBJ object, ``Benches``.

    Args:
        rows: venue rows (``venue.rows``); each supplies its tread elevation
            (``elevation_m``) and centre.
        polylines: one (N, 2) array of local (east, north) metres per row, in
            the same order; see ``row_polylines``. Local metres are relative to
            the same origin as the terrain.
        out_path: where to write ``Benches.obj``.
        base_elevation_m: Z=0 elevation, the terrain's minimum
            (``terrain_base_elevation``).
        depth_m: seat depth, front to back.
        seat_height_m: seat top above the row's tread elevation.
        simplify_m: polyline simplification tolerance.
        fit_to_row_gap: clamp depth so neighbouring benches do not overlap.
    """
    if len(rows) != len(polylines):
        raise ValueError(f"{len(rows)} rows but {len(polylines)} polylines")
    if not rows:
        raise ValueError("no rows to export")
    if depth_m <= 0 or seat_height_m <= 0:
        raise ValueError("depth and seat height must be positive")

    gaps = _gaps(rows)

    all_v: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    faces: list[str] = []
    v_off = 0
    uv_off = 0
    n_tris = 0

    for row, line, gap in zip(rows, polylines, gaps):
        pts = _clean_polyline(line, simplify_m)
        depth = min(depth_m, MAX_GAP_FRACTION * gap) if fit_to_row_gap else depth_m
        z0 = row.elevation_m - base_elevation_m
        verts, tris, uvs, tuvs = _bench_solid(pts, depth, z0, seat_height_m)

        all_v.append(verts)
        all_uv.append(uvs)
        for t, tu in zip(tris, tuvs):
            faces.append(
                "f " + " ".join(f"{v_off + v + 1}/{uv_off + u + 1}" for v, u in zip(t, tu))
            )
        v_off += len(verts)
        uv_off += len(uvs)
        n_tris += len(tris)

    verts_cm = np.vstack(all_v) * CM_PER_M
    lines = [
        "# Red Rocks seating benches generated from the venue rows",
        f"# {len(rows)} benches, {n_tris} triangles, centimetres, base {base_elevation_m:.3f} m",
        "o Benches",
    ]
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in verts_cm]
    # U = metres along the row, V = metres across / up. Every face uses these.
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in np.vstack(all_uv)]
    lines += faces

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    return BenchMesh(
        path=out_path,
        benches=len(rows),
        vertices=len(verts_cm),
        triangles=n_tris,
        base_elevation_m=base_elevation_m,
    )
