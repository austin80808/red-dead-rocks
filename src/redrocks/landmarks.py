"""Named 3D points from the build, used to fit game cameras to real photos.

Every point is a terrain-frame ``(east m, north m, elevation m)`` triple read
from ``out/manifest.json``, the build's OBJs and the scan's top surface, so a
camera fitted to a photo lands on the same geometry the level is built from.
``to_level_cm`` turns them into Unreal level-frame centimetres.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scan_blend import ScanHeights

ROWS = (8, 20, 35, 50, 70)
MONOLITHS = {
    "creation_rock_summit": "Creation_Rock.obj",
    "ship_rock_summit": "Ship_Rock.obj",
    "stage_rock_summit": "Stage_Rock.obj",
}
# Level Z origin: the build's base elevation (m), Unreal z = (elevation - this) * 100.
LEVEL_Z_ZERO_M = 1742.098

Point = tuple[float, float, float]


def _obj_vertices(path: Path) -> np.ndarray:
    """Vertices of an OBJ as an (n, 3) array in the file's units (cm)."""
    rows = [l.split()[1:4] for l in path.read_text().splitlines() if l.startswith("v ")]
    return np.array(rows, dtype=float)


def _hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (Andrew's monotone chain) of (n, 2) points."""
    pts = sorted(map(tuple, np.unique(points, axis=0)))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _inside(hull: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask of (x, y) inside a counter-clockwise convex hull."""
    inside = np.ones(x.shape, dtype=bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        inside &= (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= 0
    return inside


def _summit(mono: dict, obj: Path, scan: ScanHeights) -> Point:
    """Highest scan_top cell inside the monolith's footprint hull, else the manifest's top."""
    v = _obj_vertices(obj)
    hull = _hull(v[:, :2] / 100.0)
    east, north = scan.centres()
    mask = _inside(hull, east, north) & np.isfinite(scan.heights)
    if mask.any():
        heights = np.where(mask, scan.heights, -np.inf)
        i, j = np.unravel_index(int(np.argmax(heights)), heights.shape)
        return (float(east[i, j]), float(north[i, j]), float(scan.heights[i, j]))
    return (mono["centroid_east"], mono["centroid_north"], mono["top_elevation_m"])


def _booth(out_dir: Path, toward: Point) -> tuple[Point, Point]:
    """(roof top centre, middle of the roof's edge facing ``toward``) from Booth.obj."""
    v = _obj_vertices(out_dir / "Booth.obj")
    top = v[:, 2].max()
    roof = v[np.isclose(v[:, 2], top)][:, :2] / 100.0
    elevation = top / 100.0 + LEVEL_Z_ZERO_M
    # The roof slab is the hull of the topmost vertices.
    hull = _hull(roof)
    centre = hull.mean(axis=0)
    direction = np.array(toward[:2]) - centre
    direction /= np.linalg.norm(direction)
    nearest = hull[np.argsort(-(hull - centre) @ direction)[:2]]
    edge = nearest.mean(axis=0)
    return (
        (float(centre[0]), float(centre[1]), float(elevation)),
        (float(edge[0]), float(edge[1]), float(elevation)),
    )


def catalog(out_dir: Path, scan_top: Path) -> dict[str, Point]:
    """Name -> terrain-frame (east m, north m, elevation m) for the photo views.

    Names: ``creation_rock_summit``, ``ship_rock_summit``, ``stage_rock_summit``;
    ``aisle_start_row{N}`` and ``aisle_end_row{N}`` for N in 8, 20, 35, 50, 70;
    ``booth_roof`` (top centre of Booth.obj), ``booth_front_centre`` (middle of the
    roof's stage-side edge, at roof top), ``walkway_centre``, ``stage_front_centre``.

    The aisle points are the two ends of one row's aisle step. ``start`` is the NORTH
    side and ``end`` the SOUTH side: at every row the start has the larger north
    coordinate (row 1: start north 16.3 m, end 5.5 m; row 70: start -6.5 m, end -42.4 m).
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    scan = ScanHeights.load(Path(scan_top))
    points: dict[str, Point] = {}

    by_prefix = {m["obj"]: m for m in manifest["monoliths"]}
    for name, obj in MONOLITHS.items():
        points[name] = _summit(by_prefix[obj], out_dir / obj, scan)

    stage = manifest["stage"]
    treads = manifest["benches"]["tread_levels_m"]
    for n in ROWS:
        for side in ("start", "end"):
            steps = np.array(stage["aisles"][side], dtype=float)
            k = int(np.argmin(np.abs(steps[:, 2] - treads[n - 1])))
            points[f"aisle_{side}_row{n}"] = tuple(float(c) for c in steps[k])

    front = (stage["front_edge_east_m"], stage["front_edge_north_m"], stage["deck_top_elevation_m"])
    points["stage_front_centre"] = front
    points["booth_roof"], points["booth_front_centre"] = _booth(out_dir, front)
    points["walkway_centre"] = (
        stage["walkway_east_m"], stage["walkway_north_m"], stage["walkway_elevation_m"]
    )
    return points


def to_level_cm(points: dict[str, tuple]) -> dict[str, Point]:
    """Terrain frame (m) -> Unreal level frame (cm), as ``unreal/level_frame.to_level_cm``.

    x = east * 100, y = -north * 100, z = (elevation - 1742.098) * 100.
    """
    return {
        name: (e * 100.0, -n * 100.0, (z - LEVEL_Z_ZERO_M) * 100.0)
        for name, (e, n, z) in points.items()
    }
