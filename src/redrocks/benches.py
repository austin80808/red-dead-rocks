"""Row geometry the terraced seating (``redrocks.seating``) is built from.

``row_polylines`` gives each venue row's plan-view line; ``terrain_base_elevation``
the Z = 0 datum every OBJ shares with ``RedRocks_Terrain.obj`` (centimetres,
X east, Y north, Z up, Z measured from the terrain's lowest elevation). The
small polyline and normal helpers are shared with the seating and its tests.
Every face of every OBJ carries a UV index: Unreal's OBJ importer silently
decimates meshes without a UV channel.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from shapely.geometry import LineString, Polygon

from .geo import LocalProjection
from .terrain_mesh import DEFAULT_GRID, resample_to_grid
from .venue import ROWS, slice_polygon

# Row lines are subdivided to at most this spacing, so the seating's buried
# bottoms read the ground between their vertices.
GROUND_STEP_M = 2.5


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


def _face_normal(verts: np.ndarray, tri) -> tuple[float, float, float]:
    """Unit normal of a counter-clockwise triangle, rounded so equal normals share a ``vn``."""
    a, b, c = verts[list(tri)]
    n = np.cross(b - a, c - a)
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(round(float(x), 4) + 0.0 for x in n / length)


