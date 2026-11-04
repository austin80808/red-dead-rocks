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
from shapely.geometry.polygon import orient
from shapely.ops import triangulate

from .dem import Dem
from .geo import metres_per_degree
from .monoliths import CREATION_ROCK_WAY_ID, SHIP_ROCK_WAY_ID, STAGE_ROCK_WAY_ID
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

# Buildings: heights by OSM tags, buried foundation for hillside slopes, and UV scale.
BUILDING_HEIGHTS_M: dict[str, float] = {
    "attraction": 6.5,
    "school": 5.5,
    "roof": 3.6,
    "toilets": 3.2,
    "container": 2.8,
    "yes": 4.5,
}
BUILDING_LEVEL_HEIGHT_M = 3.8
BUILDING_FOUNDATION_M = 1.2
BUILDING_PARAPET_M = 0.55
BUILDING_PARAPET_INSET_M = 0.45
BUILDING_UV_TILE_M = 5.0
# OSM building=roof ways inside the amphitheatre bowl (stage roof & FOH booth) are
# already modelled in full detail by Stage.obj, so skip them in Buildings.obj.
BOWL_ROOF_KEEP_OUT_RADIUS_M = 60.0

# Barriers / walls: (width_m, height_m) by barrier=* tag.
BARRIER_DIMENSIONS_M: dict[str, tuple[float, float]] = {
    "retaining_wall": (0.45, 1.50),
    "wall": (0.45, 1.50),
    "city_wall": (0.60, 2.00),
    "stones": (0.60, 0.95),
    "fence": (0.18, 1.25),
    "guard_rail": (0.25, 0.85),
    "handrail": (0.12, 1.00),
}
DEFAULT_BARRIER_DIM_M = (0.25, 1.20)
BARRIER_BURY_M = 0.35

# Secondary rock outcrops across the park (Creation, Ship and Stage Rock are separate monoliths).
ROCK_BURY_M = 1.5
ROCK_TAPER = 0.12
ROCK_SEGMENTS = 3
MONOLITH_WAY_IDS = (CREATION_ROCK_WAY_ID, SHIP_ROCK_WAY_ID, STAGE_ROCK_WAY_ID)
MONOLITH_NAMES = ("Creation Rock", "Ship Rock", "Stage Rock")


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


def _earclip(pts: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple 2D counter-clockwise polygon ring of N vertices into N-2 triangles."""
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    def cross2(a, b, c) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def point_in_tri(p, a, b, c) -> bool:
        c1 = cross2(a, b, p)
        c2 = cross2(b, c, p)
        c3 = cross2(c, a, p)
        return c1 >= -1e-9 and c2 >= -1e-9 and c3 >= -1e-9

    idx = list(range(n))
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < n * n:
        guard += 1
        m = len(idx)
        clipped = False
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross2(a, b, c) <= 1e-9:
                continue
            if any(
                j not in (i0, i1, i2) and point_in_tri(pts[j], a, b, c)
                for j in idx
            ):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            # Fallback for nearly collinear vertices: clip the sharpest convex corner.
            best_k = max(
                range(m),
                key=lambda k: cross2(pts[idx[(k - 1) % m]], pts[idx[k]], pts[idx[(k + 1) % m]]),
            )
            tris.append((idx[(best_k - 1) % m], idx[best_k], idx[(best_k + 1) % m]))
            idx.pop(best_k)
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def is_building(way: Way) -> bool:
    """True when the way is a closed building polygon."""
    b = way.tags.get("building")
    return bool(way.is_closed and b and b != "no")


def building_height_m(way: Way) -> float:
    """Wall height (metres above uphill ground) derived from OSM tags."""
    explicit = _parse_width(way.tags.get("height"))
    if explicit is not None:
        return min(max(explicit, 2.5), 30.0)
    levels_str = way.tags.get("building:levels")
    if levels_str:
        m = re.match(r"\s*(\d+(?:[.,]\d+)?)", levels_str)
        if m:
            levels = float(m.group(1).replace(",", "."))
            if levels > 0:
                return min(max(levels * BUILDING_LEVEL_HEIGHT_M, 3.0), 25.0)
    t = way.tags
    if t.get("tourism") == "attraction" or "Trading Post" in t.get("name", ""):
        return BUILDING_HEIGHTS_M["attraction"]
    if t.get("amenity") == "toilets":
        return BUILDING_HEIGHTS_M["toilets"]
    b = t.get("building", "yes")
    return BUILDING_HEIGHTS_M.get(b, BUILDING_HEIGHTS_M["yes"])


def building_solid(
    ring: np.ndarray,
    height_m: float,
    surface: TerrainSurface,
    parapet_m: float = BUILDING_PARAPET_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extrude a building footprint on the terrain with a buried foundation and parapet rim.

    Returns (vertices_m[M, 3], uvs[M, 2], triangles[T, 3]), or None if degenerate.
    """
    if len(ring) < 4:
        return None
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon" or poly.area < 1.0:
        return None
    poly = orient(poly, sign=1.0)
    pts = _dedupe(np.array(poly.exterior.coords)[:, :2][:-1], eps=1e-2)
    if len(pts) < 3:
        return None

    ground_z = surface.height(pts[:, 0], pts[:, 1])
    z_base = float(np.min(ground_z)) - BUILDING_FOUNDATION_M
    z_roof = float(np.max(ground_z)) + height_m
    z_top = z_roof + max(0.0, parapet_m)

    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []

    n = len(pts)
    h = z_top - z_base
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        p, q = pts[i], pts[j]
        seg_len = float(np.hypot(*(q - p)))
        base = len(verts)
        verts += [
            (p[0], p[1], z_base),
            (q[0], q[1], z_base),
            (q[0], q[1], z_top),
            (p[0], p[1], z_top),
        ]
        uvs += [
            (s, 0.0),
            (s + seg_len, 0.0),
            (s + seg_len, h),
            (s, h),
        ]
        s += seg_len
        tris.append((base + 0, base + 1, base + 2))
        tris.append((base + 0, base + 2, base + 3))

    # Top cap (+Z) and bottom cap (-Z) via ear-clipping.
    cap_tris = _earclip(pts)
    top0 = len(verts)
    for x, y in pts:
        verts.append((float(x), float(y), z_top))
        uvs.append((float(x) / BUILDING_UV_TILE_M, float(y) / BUILDING_UV_TILE_M))
    for a, b, c in cap_tris:
        tris.append((top0 + a, top0 + b, top0 + c))

    bot0 = len(verts)
    for x, y in pts:
        verts.append((float(x), float(y), z_base))
        uvs.append((float(x) / BUILDING_UV_TILE_M, float(y) / BUILDING_UV_TILE_M))
    for a, b, c in cap_tris:
        tris.append((bot0 + a, bot0 + c, bot0 + b))

    return np.array(verts, dtype=float), np.array(uvs, dtype=float), np.array(tris, dtype=int)


def export_buildings_obj(osm: OsmData, surface: TerrainSurface, out_path: Path) -> ParkMesh | None:
    """Write ``Buildings.obj`` from closed ``building=*`` ways."""
    parts, features = [], 0
    bounds = surface.bounds_polygon()
    for way in osm.ways:
        if not is_building(way):
            continue
        xy = way_xy(osm, way, surface)
        if len(xy) < 4:
            continue
        # Skip the OSM roof polygons inside the seating bowl (stage roof & FOH booth),
        # since Stage.obj models those structures directly.
        if way.tags.get("building") == "roof" and way.tags.get("layer") == "1":
            cx, cy = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
            if math.hypot(cx, cy) < BOWL_ROOF_KEEP_OUT_RADIUS_M:
                continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = poly.intersection(bounds)
        if poly.is_empty:
            continue
        height = building_height_m(way)
        built = False
        for g in getattr(poly, "geoms", [poly]):
            if g.geom_type != "Polygon":
                continue
            solid = building_solid(np.array(g.exterior.coords), height, surface)
            if solid is not None:
                parts.append(solid)
                built = True
        features += built
    if not parts:
        return None
    v, t = _write_obj(
        Path(out_path), "Buildings", parts, "Red Rocks buildings from OpenStreetMap (ODbL 1.0)"
    )
    return ParkMesh("Buildings", Path(out_path), features, v, t)


def is_barrier(way: Way) -> bool:
    """True when the way is a physical wall, retaining wall, fence or rail."""
    b = way.tags.get("barrier")
    return bool(b and b != "no")


def barrier_dimensions_m(way: Way) -> tuple[float, float]:
    """(width_m, height_m) for a barrier way."""
    b = way.tags.get("barrier", "")
    w_default, h_default = BARRIER_DIMENSIONS_M.get(b, DEFAULT_BARRIER_DIM_M)
    w = _parse_width(way.tags.get("width")) or w_default
    h = _parse_width(way.tags.get("height")) or h_default
    return min(max(w, 0.08), 3.0), min(max(h, 0.30), 6.0)


def barrier_solid(
    points: np.ndarray,
    width_m: float,
    height_m: float,
    surface: TerrainSurface,
    densify_m: float = DENSIFY_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sweep a polyline into a terrain-following wall/fence solid.

    Returns (vertices_m[M, 3], uvs[M, 2], triangles[T, 3]) or None if degenerate.
    """
    pts = _dedupe(np.asarray(points, dtype=float)[:, :2])
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
        if norm < 1e-9:
            normals[i] = left_seg[i - 1]
            continue
        avg /= norm
        scale[i] = min(1.0 / max(float(avg @ left_seg[i]), 1e-6), MAX_MITRE)
        normals[i] = avg

    half = width_m / 2.0
    left = pts + normals * (half * scale)[:, None]
    right = pts - normals * (half * scale)[:, None]

    h_centre = surface.height(pts[:, 0], pts[:, 1])
    h_left = surface.height(left[:, 0], left[:, 1])
    h_right = surface.height(right[:, 0], right[:, 1])
    z_lo = np.minimum(np.minimum(h_left, h_right), h_centre) - BARRIER_BURY_M
    z_hi = h_centre + height_m

    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []

    def add_quad(p0, p1, p2, p3, u0, u1, u2, u3):
        base = len(verts)
        verts.extend([p0, p1, p2, p3])
        uvs.extend([u0, u1, u2, u3])
        tris.append((base + 0, base + 1, base + 2))
        tris.append((base + 0, base + 2, base + 3))

    full_h = height_m + BARRIER_BURY_M
    for i in range(n - 1):
        j = i + 1
        s0, s1 = float(dist[i]), float(dist[j])
        LB0 = (float(left[i, 0]), float(left[i, 1]), float(z_lo[i]))
        LT0 = (float(left[i, 0]), float(left[i, 1]), float(z_hi[i]))
        RT0 = (float(right[i, 0]), float(right[i, 1]), float(z_hi[i]))
        RB0 = (float(right[i, 0]), float(right[i, 1]), float(z_lo[i]))
        LB1 = (float(left[j, 0]), float(left[j, 1]), float(z_lo[j]))
        LT1 = (float(left[j, 0]), float(left[j, 1]), float(z_hi[j]))
        RT1 = (float(right[j, 0]), float(right[j, 1]), float(z_hi[j]))
        RB1 = (float(right[j, 0]), float(right[j, 1]), float(z_lo[j]))
        # Top (+Z), Bottom (-Z), Left (+L), Right (-L).
        add_quad(LT0, RT0, RT1, LT1, (s0, 0.0), (s0, width_m), (s1, width_m), (s1, 0.0))
        add_quad(LB0, LB1, RB1, RB0, (s0, 0.0), (s1, 0.0), (s1, width_m), (s0, width_m))
        add_quad(LB0, LT0, LT1, LB1, (s0, 0.0), (s0, full_h), (s1, full_h), (s1, 0.0))
        add_quad(RB0, RB1, RT1, RT0, (s0, 0.0), (s1, 0.0), (s1, full_h), (s0, full_h))

    # Start and end caps.
    LB0 = (float(left[0, 0]), float(left[0, 1]), float(z_lo[0]))
    LT0 = (float(left[0, 0]), float(left[0, 1]), float(z_hi[0]))
    RT0 = (float(right[0, 0]), float(right[0, 1]), float(z_hi[0]))
    RB0 = (float(right[0, 0]), float(right[0, 1]), float(z_lo[0]))
    add_quad(LB0, RB0, RT0, LT0, (0.0, 0.0), (width_m, 0.0), (width_m, full_h), (0.0, full_h))

    LBN = (float(left[-1, 0]), float(left[-1, 1]), float(z_lo[-1]))
    LTN = (float(left[-1, 0]), float(left[-1, 1]), float(z_hi[-1]))
    RTN = (float(right[-1, 0]), float(right[-1, 1]), float(z_hi[-1]))
    RBN = (float(right[-1, 0]), float(right[-1, 1]), float(z_lo[-1]))
    add_quad(LBN, LTN, RTN, RBN, (0.0, 0.0), (0.0, full_h), (width_m, full_h), (width_m, 0.0))

    return np.array(verts, dtype=float), np.array(uvs, dtype=float), np.array(tris, dtype=int)


def export_walls_obj(osm: OsmData, surface: TerrainSurface, out_path: Path) -> ParkMesh | None:
    """Write ``Walls.obj`` from ``barrier=*`` ways (retaining walls, walls, fences, rails)."""
    parts, features = [], 0
    bounds = surface.bounds_polygon()
    for way in osm.ways:
        if not is_barrier(way):
            continue
        xy = way_xy(osm, way, surface)
        if len(xy) < 2:
            continue
        w, h = barrier_dimensions_m(way)
        built = False
        for piece in _clip_to_bounds(xy, bounds):
            solid = barrier_solid(piece, w, h, surface)
            if solid is not None:
                parts.append(solid)
                built = True
        features += built
    if not parts:
        return None
    v, t = _write_obj(
        Path(out_path), "Walls", parts, "Red Rocks walls and barriers from OpenStreetMap (ODbL 1.0)"
    )
    return ParkMesh("Walls", Path(out_path), features, v, t)


def is_park_rock(way: Way) -> bool:
    """True when the way is a closed rock/cliff outcrop other than the three monoliths.

    Creation Rock, Ship Rock and Stage Rock are their own meshes (monoliths.py), which
    Unreal hides in game where the ml0w scan has the real rocks; extruding them here
    too would leave a second, unhideable copy poking through the scan.
    """
    return (
        way.is_closed
        and way.kind() == "rock"
        and way.id not in MONOLITH_WAY_IDS
        and way.tags.get("name") not in MONOLITH_NAMES
    )


def rock_height_m(way: Way, area_m2: float) -> float:
    """Outcrop height (metres above terrain) for a surveyed OSM rock/cliff polygon."""
    explicit = _parse_width(way.tags.get("height"))
    if explicit is not None:
        return min(max(explicit, 1.5), 60.0)
    span = math.sqrt(max(area_m2, 1.0))
    if "name" in way.tags:
        return min(max(0.45 * span, 8.0), 28.0)
    nat = way.tags.get("natural", "bare_rock")
    if nat == "cliff":
        return min(max(0.40 * span, 6.0), 22.0)
    if nat == "stone":
        return min(max(0.28 * span, 1.5), 8.0)
    return min(max(0.32 * span, 2.5), 18.0)


def rock_solid(
    ring: np.ndarray,
    height_m: float,
    surface: TerrainSurface,
    segments: int = ROCK_SEGMENTS,
    taper: float = ROCK_TAPER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extrude a sandstone rock outcrop on the terrain with battered tiers and UVs.

    Returns (vertices_m[M, 3], uvs[M, 2], triangles[T, 3]) or None if degenerate.
    """
    if len(ring) < 4:
        return None
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon" or poly.area < 1.0:
        return None
    poly = orient(poly, sign=1.0)
    pts = _dedupe(np.array(poly.exterior.coords)[:, :2][:-1], eps=1e-2)
    if len(pts) < 3:
        return None

    n = len(pts)
    centre = np.array([float(poly.centroid.x), float(poly.centroid.y)])
    h_ring = surface.height(pts[:, 0], pts[:, 1])
    z_base = float(np.min(h_ring)) - ROCK_BURY_M
    # Follow local terrain slope while rising to a hogback crest height_m above ground.
    z_crest = 0.65 * h_ring + 0.35 * float(np.max(h_ring)) + height_m

    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []

    ring_levels = []
    z_levels = []
    for lvl in range(segments + 1):
        frac = lvl / segments
        shrink = taper * (frac ** 1.4)
        xy_lvl = pts * (1.0 - shrink) + centre * shrink
        z_lvl = z_base + frac * (z_crest - z_base)
        ring_levels.append(xy_lvl)
        z_levels.append(z_lvl)

    # Side walls across all vertical segments.
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        seg_len = float(np.hypot(*(pts[j] - pts[i])))
        for lvl in range(segments):
            xy0, z0 = ring_levels[lvl], z_levels[lvl]
            xy1, z1 = ring_levels[lvl + 1], z_levels[lvl + 1]
            v0 = (lvl / segments) * height_m
            v1 = ((lvl + 1) / segments) * height_m
            base = len(verts)
            verts += [
                (float(xy0[i, 0]), float(xy0[i, 1]), float(z0[i])),
                (float(xy0[j, 0]), float(xy0[j, 1]), float(z0[j])),
                (float(xy1[j, 0]), float(xy1[j, 1]), float(z1[j])),
                (float(xy1[i, 0]), float(xy1[i, 1]), float(z1[i])),
            ]
            uvs += [
                (s, v0),
                (s + seg_len, v0),
                (s + seg_len, v1),
                (s, v1),
            ]
            tris.append((base + 0, base + 1, base + 2))
            tris.append((base + 0, base + 2, base + 3))
        s += seg_len

    # Top cap (+Z) and bottom cap (-Z).
    cap_tris = _earclip(pts)
    top0 = len(verts)
    top_xy, top_z = ring_levels[-1], z_levels[-1]
    for i in range(n):
        verts.append((float(top_xy[i, 0]), float(top_xy[i, 1]), float(top_z[i])))
        uvs.append((float(top_xy[i, 0]) / PARKING_UV_TILE_M, float(top_xy[i, 1]) / PARKING_UV_TILE_M))
    for a, b, c in cap_tris:
        tris.append((top0 + a, top0 + b, top0 + c))

    bot0 = len(verts)
    for i in range(n):
        verts.append((float(pts[i, 0]), float(pts[i, 1]), z_base))
        uvs.append((float(pts[i, 0]) / PARKING_UV_TILE_M, float(pts[i, 1]) / PARKING_UV_TILE_M))
    for a, b, c in cap_tris:
        tris.append((bot0 + a, bot0 + c, bot0 + b))

    return np.array(verts, dtype=float), np.array(uvs, dtype=float), np.array(tris, dtype=int)


def export_rocks_obj(osm: OsmData, surface: TerrainSurface, out_path: Path) -> ParkMesh | None:
    """Write ``Rocks.obj`` from surveyed OSM ``natural=bare_rock|stone|cliff`` ways, except the three monoliths."""
    parts, features = [], 0
    bounds = surface.bounds_polygon()
    for way in osm.ways:
        if not is_park_rock(way):
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
            if g.geom_type != "Polygon" or g.area < 1.0:
                continue
            h = rock_height_m(way, float(g.area))
            solid = rock_solid(np.array(g.exterior.coords), h, surface)
            if solid is not None:
                parts.append(solid)
                built = True
        features += built
    if not parts:
        return None
    v, t = _write_obj(
        Path(out_path), "Rocks", parts, "Red Rocks park outcrops from OpenStreetMap (ODbL 1.0)"
    )
    return ParkMesh("Rocks", Path(out_path), features, v, t)


def export_park(
    osm: OsmData, dem: Dem, out_dir: Path, grid: int = DEFAULT_GRID
) -> dict[str, ParkMesh]:
    """Write Roads.obj, Paths.obj, Parking.obj, Buildings.obj, Walls.obj and Rocks.obj; skips any with no features.

    ``grid`` must equal the terrain mesh grid so the Z datum and surface match.
    """
    out_dir = Path(out_dir)
    surface = TerrainSurface(dem, grid)
    results = {
        "roads": export_roads_obj(osm, surface, out_dir / "Roads.obj"),
        "paths": export_paths_obj(osm, surface, out_dir / "Paths.obj"),
        "parking": export_parking_obj(osm, surface, out_dir / "Parking.obj"),
        "buildings": export_buildings_obj(osm, surface, out_dir / "Buildings.obj"),
        "walls": export_walls_obj(osm, surface, out_dir / "Walls.obj"),
        "rocks": export_rocks_obj(osm, surface, out_dir / "Rocks.obj"),
    }
    return {k: v for k, v in results.items() if v is not None}
