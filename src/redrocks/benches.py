"""Turn the generated seating-bowl rows into bench geometry for Unreal.

Each venue row becomes one solid bench: its (curved) row polyline swept into a
strip ``depth`` metres front-to-back, standing ``seat_height`` metres above the
ground. All benches go into a single OBJ object (``Benches``).

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
* when a ``ground`` function is given (the ml0w scan's ground,
  ``scan_blend.ScanGround``, or the terrain mesh, ``terrain_mesh.TerrainSurface``)
  the polyline is subdivided every ``ground_step_m`` and each station is
  lowered or raised to the ground there: the seat top is 0.45 m above the
  ground under the row's centreline, and the base drops to the lowest ground
  under the section so no edge floats. An optional ``floor`` (the walkable
  terrain, which sits under the scan) pulls the base down to it as well, so a
  bench stands on the scan and still meets the terrain where the scan is
  thin. Without ``ground`` a bench is flat at its row's ``elevation_m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
# With a ground function, polylines are subdivided to at most this spacing so
# the bench can follow the terrain between its vertices.
GROUND_STEP_M = 2.5
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
    """Each venue row's plan-view polyline, in local metres (E, N), row order.

    ``build_venue`` stores the polyline on every row it makes (curved arcs
    around the stage), and those are returned as they are. Rows without one
    (built by hand, or by older code that kept only a centre) fall back to
    re-cutting the footprint with the straight-chord slicer and matching by
    midpoint; ``n_rows`` must then equal the value given to ``build_venue``.
    """
    if all(row.polyline is not None for row in venue.rows):
        return [np.asarray(row.polyline, dtype=float)[:, :2] for row in venue.rows]

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


def _densify(points: np.ndarray, max_step_m: float) -> np.ndarray:
    """Subdivide segments longer than ``max_step_m`` into equal parts."""
    out = [points[0]]
    for p, q in zip(points[:-1], points[1:]):
        k = max(1, int(math.ceil(float(np.hypot(*(q - p))) / max_step_m)))
        for j in range(1, k + 1):
            out.append(p + (q - p) * (j / k))
    return np.array(out)


def _gaps(polylines: Sequence[np.ndarray]) -> list[float]:
    """Distance from each row's polyline to its nearest neighbouring row's.

    Measured between the lines themselves, not their midpoints: on concentric
    arcs the midpoints sit at different angles and say little about the gap.
    """
    shapes = [LineString(p) for p in polylines]
    out = []
    for i, line in enumerate(shapes):
        d = []
        if i > 0:
            d.append(float(line.distance(shapes[i - 1])))
        if i < len(shapes) - 1:
            d.append(float(line.distance(shapes[i + 1])))
        out.append(min(d) if d else math.inf)
    return out


def _offsets(pts: np.ndarray, depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    tang = seg / seg_len[:, None]
    seg_norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)  # left of travel

    # Mitred normal at each station.
    n = len(pts)
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
    return left, right, seg_len


def _follow_ground(
    pts: np.ndarray,
    depth_m: float,
    ground: Callable[[float, float], float],
    base_elevation_m: float,
    seat_height_m: float,
    floor: Callable[[float, float], float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-station (bottom, top) heights, in metres above the base elevation.

    The top is ``seat_height_m`` above the ground under the centreline, so the
    seat surface is the same height off the ground along the whole row. The
    bottom is the lowest ground under the section's left, centre and right
    corners, so a bench on a cross-slope is buried on its uphill side rather
    than floating on its downhill side; with ``floor``, the lowest of the
    ground and the floor there.
    """
    left, right, _ = _offsets(pts, depth_m)
    under_centre = np.array([ground(float(x), float(y)) for x, y in pts])
    bottom = under_centre.copy()
    for surface in (ground, floor) if floor is not None else (ground,):
        for line in (left, pts, right):
            bottom = np.minimum(bottom, [surface(float(x), float(y)) for x, y in line])
    return bottom - base_elevation_m, under_centre + seat_height_m - base_elevation_m


def _bench_solid(
    pts: np.ndarray, depth_m: float, z_lo, z_hi
) -> tuple[np.ndarray, list[tuple[int, int, int]], np.ndarray, list[tuple[int, int, int]]]:
    """Sweep a rectangular section along ``pts``.

    ``z_lo`` and ``z_hi`` are the section's bottom and top heights in metres,
    each a scalar or one value per station.

    Returns (vertices_m[N,3], triangle vertex ids, uvs_m[M,2], triangle uv ids),
    ids local to this bench and 0-based. Triangles wind counter-clockwise seen
    from outside.
    """
    n = len(pts)
    left, right, seg_len = _offsets(pts, depth_m)
    z_lo = np.broadcast_to(np.asarray(z_lo, dtype=float), (n,))
    z_hi = np.broadcast_to(np.asarray(z_hi, dtype=float), (n,))
    height_m = float(np.max(z_hi - z_lo))  # only sizes the end caps' UVs

    # Station i: 4 section corners. L = left, R = right; B = bottom, T = top.
    verts = []
    for i in range(n):
        verts.append((*left[i], z_lo[i]))   # 4i+0  LB
        verts.append((*left[i], z_hi[i]))   # 4i+1  LT
        verts.append((*right[i], z_hi[i]))  # 4i+2  RT
        verts.append((*right[i], z_lo[i]))  # 4i+3  RB
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


def _face_normal(verts: np.ndarray, tri) -> tuple[float, float, float]:
    """Unit normal of a counter-clockwise triangle, rounded so equal normals share a ``vn``."""
    a, b, c = verts[list(tri)]
    n = np.cross(b - a, c - a)
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(round(float(x), 4) + 0.0 for x in n / length)


def export_benches_obj(
    rows: Sequence[Row],
    polylines: Sequence[np.ndarray],
    out_path: Path,
    base_elevation_m: float,
    depth_m: float = SEAT_DEPTH_M,
    seat_height_m: float = SEAT_HEIGHT_M,
    simplify_m: float = SIMPLIFY_TOLERANCE_M,
    fit_to_row_gap: bool = True,
    ground: Callable[[float, float], float] | None = None,
    ground_step_m: float = GROUND_STEP_M,
    floor: Callable[[float, float], float] | None = None,
) -> BenchMesh:
    """Write every row as a bench in one OBJ object, ``Benches``.

    Args:
        rows: venue rows (``venue.rows``); each supplies its tread elevation
            (``elevation_m``), used when ``ground`` is not given.
        polylines: one (N, 2) array of local (east, north) metres per row, in
            the same order; see ``row_polylines``. Local metres are relative to
            the same origin as the terrain.
        out_path: where to write ``Benches.obj``.
        base_elevation_m: Z=0 elevation, the terrain's minimum
            (``terrain_base_elevation``).
        depth_m: seat depth, front to back.
        seat_height_m: seat top above the tread elevation (or above the ground
            under the row's centreline when ``ground`` is given).
        simplify_m: polyline simplification tolerance.
        fit_to_row_gap: clamp depth so neighbouring benches do not overlap.
        ground: ``ground(east, north)`` -> absolute elevation in metres, e.g. a
            ``scan_blend.ScanGround`` or ``terrain_mesh.TerrainSurface``. When
            given, each bench is subdivided every ``ground_step_m`` and follows
            the ground along its length instead of sitting flat at
            ``row.elevation_m``.
        ground_step_m: maximum station spacing when following the ground.
        floor: optional surface the bench bottoms also reach down to (the
            walkable terrain under the scan). Only used with ``ground``.
    """
    if len(rows) != len(polylines):
        raise ValueError(f"{len(rows)} rows but {len(polylines)} polylines")
    if not rows:
        raise ValueError("no rows to export")
    if depth_m <= 0 or seat_height_m <= 0:
        raise ValueError("depth and seat height must be positive")

    gaps = _gaps([_clean_polyline(line, simplify_m) for line in polylines])

    all_v: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    faces: list[str] = []
    normals: dict[tuple[float, float, float], int] = {}
    v_off = 0
    uv_off = 0
    n_tris = 0

    for row, line, gap in zip(rows, polylines, gaps):
        pts = _clean_polyline(line, simplify_m)
        depth = min(depth_m, MAX_GAP_FRACTION * gap) if fit_to_row_gap else depth_m
        if ground is None:
            z_lo = row.elevation_m - base_elevation_m
            z_hi = z_lo + seat_height_m
        else:
            pts = _densify(pts, ground_step_m)
            z_lo, z_hi = _follow_ground(pts, depth, ground, base_elevation_m, seat_height_m, floor)
        verts, tris, uvs, tuvs = _bench_solid(pts, depth, z_lo, z_hi)

        all_v.append(verts)
        all_uv.append(uvs)
        for t, tu in zip(tris, tuvs):
            n = normals.setdefault(_face_normal(verts, t), len(normals) + 1)
            faces.append(
                "f " + " ".join(f"{v_off + v + 1}/{uv_off + u + 1}/{n}" for v, u in zip(t, tu))
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
    # Flat normals per face. Left to the importer, a corner's normal averages
    # the seat top with the riser under it, weighted by area; on the scan the
    # risers reach down to the terrain, so the seat tops came out tilted past
    # M_Benches's "top" threshold and lost their planks.
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in normals]
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
