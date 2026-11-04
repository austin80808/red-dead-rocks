"""Roads, paths and parking lots from OSM, draped on the terrain mesh.

Three OBJs come out of here: ``Roads.obj``, ``Paths.obj`` and ``Parking.obj``.
All share the terrain mesh's frame so they need no transform in Unreal:
centimetres, X east, Y north, Z up, centred on the same origin, Z measured from
the terrain's lowest elevation.

Draping. The terrain OBJ is a regular grid split into two triangles per cell
along the NW-SE diagonal. ``TerrainSurface`` reproduces that exact triangulation
(not a smooth bilinear surface), so a vertex sampled here sits on the mesh the
engine renders. Ribbons are densified to ``DENSIFY_M`` between samples so they
follow the slope between vertices, then lifted by a few centimetres per layer
(parking < roads < paths) to avoid z-fighting with the terrain and each other.

Widths by ``highway=*`` (metres, full carriageway width, not half):

======================================  =====
motorway, trunk                         10.0
primary                                  8.0
secondary                                7.0
tertiary, unclassified, residential      6.0
service (``service=parking_aisle``)      4.0 (3.0)
living_street                            4.5
track                                    3.5
cycleway                                 2.5
footway, pedestrian, bridleway           2.0
path                                     1.5
steps                                    1.5
======================================  =====

``*_link`` ways use their parent class width. A numeric ``width=*`` tag (metres)
overrides the class, clamped to 0.5-30 m. ``tunnel=*`` ways are skipped (they
are underground) and closed ``area=yes`` ways are skipped (they are plazas, not
lines). Bridges are draped on the terrain like any other way; there is no
deck geometry yet.

Roads are everything in ``ROAD_WIDTHS_M``; paths are everything in
``PATH_WIDTHS_M``. Other ``highway`` values (``construction``, ``proposed``,
``bus_stop`` ...) are ignored.

UVs: every face carries ``vt``. For ribbons, U is distance along the way
divided by the way's width (so a square texture tile stays square) and V runs
0..1 across the width. For parking patches, UVs are planar in metres / 10.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon, box
from shapely.ops import triangulate

from .dem import Dem
from .geo import metres_per_degree
from .osm import OsmData, Way
from .terrain_mesh import CM_PER_M, DEFAULT_GRID, resample_to_grid

ROAD_WIDTHS_M: dict[str, float] = {
    "motorway": 10.0,
    "trunk": 10.0,
    "primary": 8.0,
    "secondary": 7.0,
    "tertiary": 6.0,
    "unclassified": 6.0,
    "residential": 6.0,
    "living_street": 4.5,
    "service": 4.0,
    "track": 3.5,
}
PATH_WIDTHS_M: dict[str, float] = {
    "cycleway": 2.5,
    "footway": 2.0,
    "pedestrian": 2.0,
    "bridleway": 2.0,
    "path": 1.5,
    "steps": 1.5,
}
PARKING_AISLE_WIDTH_M = 3.0
MIN_WIDTH_M = 0.5
MAX_WIDTH_M = 30.0

# Height above the terrain surface, metres. Layered so overlaps do not fight.
PARKING_LIFT_M = 0.03
ROAD_LIFT_M = 0.05
PATH_LIFT_M = 0.07

# Max distance between sampled points along a way / inside a parking patch.
DENSIFY_M = 2.0
PARKING_SPACING_M = 3.0
# Mitre joins on sharp bends are capped so they cannot spike.
MAX_MITRE = 2.0
PARKING_UV_TILE_M = 10.0


@dataclass
class ParkMesh:
    name: str
    path: Path
    features: int
    vertices: int
    triangles: int


class TerrainSurface:
    """The terrain mesh as a height function, in the terrain mesh's frame.

    Local metres: X east, Y north, centred on the DEM bbox centre (what
    ``export_terrain_obj`` uses). Heights are metres above the grid minimum.
    """

    def __init__(self, dem: Dem, grid: int = DEFAULT_GRID) -> None:
        if grid < 2:
            raise ValueError("grid must be at least 2")
        self.dem = dem
        self.heights = resample_to_grid(dem, grid)
        self.base = float(np.nanmin(self.heights))
        lat, lon = dem.bbox.centre
        self.lat0, self.lon0 = lat, lon
        self.m_per_deg_lon, self.m_per_deg_lat = metres_per_degree(lat)
        self.half_x = dem.bbox.width_deg * self.m_per_deg_lon / 2.0
        self.half_y = dem.bbox.height_deg * self.m_per_deg_lat / 2.0

    @property
    def base_elevation_m(self) -> float:
        return self.base

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (
            (lon - self.lon0) * self.m_per_deg_lon,
            (lat - self.lat0) * self.m_per_deg_lat,
        )

    def height(self, x, y) -> np.ndarray:
        """Terrain height in metres above Z=0 at local (x, y) metres.

        Evaluates the same two triangles per cell as the exported OBJ; points
        outside the bbox clamp to the edge.
        """
        h = self.heights
        rows, cols = h.shape
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        fc = np.clip((x + self.half_x) / (2 * self.half_x) * (cols - 1), 0, cols - 1)
        fr = np.clip((self.half_y - y) / (2 * self.half_y) * (rows - 1), 0, rows - 1)
        c0 = np.minimum(fc.astype(int), cols - 2)
        r0 = np.minimum(fr.astype(int), rows - 2)
        tx = fc - c0
        ty = fr - r0
        a = h[r0, c0]
        b = h[r0, c0 + 1]
        d = h[r0 + 1, c0]
        e = h[r0 + 1, c0 + 1]
        # Cells split a-e: triangles (a, d, e) where ty >= tx, (a, e, b) otherwise.
        lower = a + (e - d) * tx + (d - a) * ty
        upper = a + (b - a) * tx + (e - b) * ty
        return np.where(ty >= tx, lower, upper) - self.base

    def bounds_polygon(self) -> Polygon:
        return box(-self.half_x, -self.half_y, self.half_x, self.half_y)


def way_xy(osm: OsmData, way: Way, surface: TerrainSurface) -> np.ndarray:
    """A way's nodes as an (N, 2) array of local metres (missing nodes dropped)."""
    pts = [surface.to_xy(lat, lon) for lat, lon in osm.way_coords(way)]
    return np.array(pts, dtype=float).reshape(-1, 2)


def _parse_width(value: str | None) -> float | None:
    if not value:
        return None
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*(m)?\s*$", value)
    if not m:
        return None
    w = float(m.group(1).replace(",", "."))
    return min(max(w, MIN_WIDTH_M), MAX_WIDTH_M) if w > 0 else None


def way_width_m(way: Way) -> float | None:
    """Ribbon width for a highway way, or None when it is not a road or path."""
    hw = way.tags.get("highway")
    if not hw:
        return None
    cls = hw[:-5] if hw.endswith("_link") else hw
    table = ROAD_WIDTHS_M if cls in ROAD_WIDTHS_M else PATH_WIDTHS_M
    if cls not in table:
        return None
    explicit = _parse_width(way.tags.get("width"))
    if explicit is not None:
        return explicit
    if cls == "service" and way.tags.get("service") == "parking_aisle":
        return PARKING_AISLE_WIDTH_M
    return table[cls]


def _is_buildable(way: Way) -> bool:
    if way.tags.get("tunnel") not in (None, "no"):
        return False
    if way.is_closed and way.tags.get("area") == "yes":
        return False
    return True


def is_road(way: Way) -> bool:
    hw = way.tags.get("highway", "")
    cls = hw[:-5] if hw.endswith("_link") else hw
    return cls in ROAD_WIDTHS_M and _is_buildable(way)


def is_path(way: Way) -> bool:
    return way.tags.get("highway", "") in PATH_WIDTHS_M and _is_buildable(way)


def _densify(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert points so no segment exceeds ``step``; returns (points, distance)."""
    out = [points[0]]
    dist = [0.0]
    total = 0.0
    for p, q in zip(points[:-1], points[1:]):
        seg = float(np.hypot(*(q - p)))
        n = max(1, math.ceil(seg / step))
        for i in range(1, n + 1):
            out.append(p + (q - p) * (i / n))
            dist.append(total + seg * i / n)
        total += seg
    return np.array(out), np.array(dist)


def _dedupe(points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    keep = [0]
    for i in range(1, len(points)):
        if np.hypot(*(points[i] - points[keep[-1]])) > eps:
            keep.append(i)
    return points[keep]


def _clip_to_bounds(points: np.ndarray, bounds: Polygon) -> list[np.ndarray]:
    line = LineString(points)
    clipped = line.intersection(bounds)
    if clipped.is_empty:
        return []
    if isinstance(clipped, LineString):
        parts = [clipped]
    elif isinstance(clipped, MultiLineString):
        parts = list(clipped.geoms)
    else:  # GeometryCollection: keep only the line pieces
        parts = [g for g in getattr(clipped, "geoms", []) if isinstance(g, LineString)]
    return [np.array(p.coords)[:, :2] for p in parts if len(p.coords) >= 2]


def ribbon(
    points: np.ndarray,
    width_m: float,
    surface: TerrainSurface,
    lift_m: float,
    densify_m: float = DENSIFY_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sweep a polyline into a draped ribbon.

    Returns (vertices (M, 3) metres, uvs (M, 2), triangles (T, 3) indices into
    the vertex array), or None if the line is degenerate. Vertices alternate
    right, left per sample along the line. Triangles are counter-clockwise from
    above.
    """
    pts = _dedupe(np.asarray(points, dtype=float))
    if len(pts) < 2:
        return None
    pts, dist = _densify(pts, densify_m)
    n = len(pts)

    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    tangent = seg / seg_len[:, None]
    left_seg = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    normals = np.empty((n, 2))
    scale = np.ones(n)
    normals[0] = left_seg[0]
    normals[-1] = left_seg[-1]
    for i in range(1, n - 1):
        avg = left_seg[i - 1] + left_seg[i]
        norm = float(np.hypot(*avg))
        if norm < 1e-9:  # 180-degree reversal: keep the incoming side
            normals[i] = left_seg[i - 1]
            continue
        avg /= norm
        # Mitre length factor 1/cos(half angle) = 1/(avg . left_seg).
        scale[i] = min(1.0 / max(float(avg @ left_seg[i]), 1e-6), MAX_MITRE)
        normals[i] = avg

    half = width_m / 2.0
    right = pts - normals * (half * scale)[:, None]
    left = pts + normals * (half * scale)[:, None]

    verts = np.empty((2 * n, 3))
    verts[0::2, :2] = right
    verts[1::2, :2] = left
    verts[:, 2] = surface.height(verts[:, 0], verts[:, 1]) + lift_m

    uvs = np.empty((2 * n, 2))
    uvs[0::2, 0] = dist / width_m
    uvs[1::2, 0] = dist / width_m
    uvs[0::2, 1] = 0.0
    uvs[1::2, 1] = 1.0

    i = np.arange(n - 1) * 2
    # right_i, right_i+1, left_i+1 / right_i, left_i+1, left_i  (CCW from above)
    tris = np.concatenate(
        [np.stack([i, i + 2, i + 3], axis=1), np.stack([i, i + 3, i + 1], axis=1)]
    )
    return verts, uvs, tris


def parking_patch(
    ring: np.ndarray,
    surface: TerrainSurface,
    lift_m: float = PARKING_LIFT_M,
    spacing_m: float = PARKING_SPACING_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Triangulate a parking polygon into a patch that follows the terrain.

    Boundary and interior grid points are Delaunay-triangulated and triangles
    whose centroid falls outside the polygon are dropped. Returns
    (vertices (M, 3) metres, uvs (M, 2), triangles (T, 3)) or None if the ring
    is degenerate.
    """
    if len(ring) < 4:
        return None
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon" or poly.area < 1.0:
        return None

    boundary, _ = _densify(np.array(poly.exterior.coords), spacing_m)
    minx, miny, maxx, maxy = poly.bounds
    gx = np.arange(minx + spacing_m / 2, maxx, spacing_m)
    gy = np.arange(miny + spacing_m / 2, maxy, spacing_m)
    inner = poly.buffer(-spacing_m / 3.0)
    interior = [
        (x, y) for x in gx for y in gy if not inner.is_empty and inner.contains(Point(x, y))
    ]
    pts = np.unique(np.round(np.vstack([boundary[:-1]] + ([np.array(interior)] if interior else [])), 4), axis=0)
    if len(pts) < 3:
        return None

    index = {(round(x, 4), round(y, 4)): k for k, (x, y) in enumerate(pts)}
    tris = []
    for t in triangulate(MultiPoint([tuple(p) for p in pts])):
        c = t.centroid
        if not poly.contains(c):
            continue
        ids = [index[(round(x, 4), round(y, 4))] for x, y in list(t.exterior.coords)[:3]]
        a, b, d = (pts[i] for i in ids)
        cross = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
        if abs(cross) < 1e-9:
            continue
        tris.append(ids if cross > 0 else [ids[0], ids[2], ids[1]])
    if not tris:
        return None

    verts = np.column_stack([pts, surface.height(pts[:, 0], pts[:, 1]) + lift_m])
    uvs = pts / PARKING_UV_TILE_M
    return verts, uvs, np.array(tris)


def _write_obj(path: Path, obj_name: str, parts, header: str) -> tuple[int, int]:
    """Write ribbon/patch parts to one OBJ. Returns (vertices, triangles)."""
    verts = np.vstack([p[0] for p in parts]) * CM_PER_M
    uvs = np.vstack([p[1] for p in parts])
    offsets = np.cumsum([0] + [len(p[0]) for p in parts])[:-1]
    tris = np.vstack([p[2] + off + 1 for p, off in zip(parts, offsets)])

    lines = [f"# {header}", f"# {len(tris)} triangles, centimetres", f"o {obj_name}"]
    lines += [f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in verts]
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in uvs]
    # Vertex and UV indices are identical, so every face has a UV.
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in tris]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(verts), len(tris)


def _export_ribbons(osm, surface, out_path, name, selector, lift_m) -> ParkMesh | None:
    parts, features = [], 0
    bounds = surface.bounds_polygon()
    for way in osm.ways:
        if not selector(way):
            continue
        width = way_width_m(way)
        xy = way_xy(osm, way, surface)
        if width is None or len(xy) < 2:
            continue
        built = False
        for piece in _clip_to_bounds(xy, bounds):
            r = ribbon(piece, width, surface, lift_m)
            if r is not None:
                parts.append(r)
                built = True
        features += built
    if not parts:
        return None
    v, t = _write_obj(
        Path(out_path), name, parts, f"Red Rocks {name.lower()} from OpenStreetMap (ODbL 1.0)"
    )
    return ParkMesh(name, Path(out_path), features, v, t)


def export_roads_obj(osm: OsmData, surface: TerrainSurface, out_path: Path) -> ParkMesh | None:
    """Write ``Roads.obj``. Returns None (and writes nothing) when there are no roads."""
    return _export_ribbons(osm, surface, out_path, "Roads", is_road, ROAD_LIFT_M)


def export_paths_obj(osm: OsmData, surface: TerrainSurface, out_path: Path) -> ParkMesh | None:
    """Write ``Paths.obj``. Returns None (and writes nothing) when there are no paths."""
    return _export_ribbons(osm, surface, out_path, "Paths", is_path, PATH_LIFT_M)


def export_parking_obj(osm: OsmData, surface: TerrainSurface, out_path: Path) -> ParkMesh | None:
    """Write ``Parking.obj`` from closed ``amenity=parking`` ways."""
    parts, features = [], 0
    bounds = surface.bounds_polygon()
    for way in osm.by_kind("parking"):
        if not way.is_closed:
            continue
        xy = way_xy(osm, way, surface)
        if len(xy) < 4:
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = poly.intersection(bounds)
        if poly.is_empty:
            continue
        built = False
        for g in getattr(poly, "geoms", [poly]):
            if g.geom_type != "Polygon":
                continue
            patch = parking_patch(np.array(g.exterior.coords), surface)
            if patch is not None:
                parts.append(patch)
                built = True
        features += built
    if not parts:
        return None
    v, t = _write_obj(
        Path(out_path), "Parking", parts, "Red Rocks parking lots from OpenStreetMap (ODbL 1.0)"
    )
    return ParkMesh("Parking", Path(out_path), features, v, t)


def export_park(
    osm: OsmData, dem: Dem, out_dir: Path, grid: int = DEFAULT_GRID
) -> dict[str, ParkMesh]:
    """Write Roads.obj, Paths.obj and Parking.obj; skips any with no features.

    ``grid`` must equal the terrain mesh grid so the Z datum and surface match.
    """
    out_dir = Path(out_dir)
    surface = TerrainSurface(dem, grid)
    results = {
        "roads": export_roads_obj(osm, surface, out_dir / "Roads.obj"),
        "paths": export_paths_obj(osm, surface, out_dir / "Paths.obj"),
        "parking": export_parking_obj(osm, surface, out_dir / "Parking.obj"),
    }
    return {k: v for k, v in results.items() if v is not None}
