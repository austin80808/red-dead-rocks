"""Downtown Denver's skyline from OpenStreetMap, at its true place from the stage.

From the top rows of Red Rocks you look over the stage and the plains to
downtown Denver (the Commons postcard,
https://commons.wikimedia.org/wiki/File:Red_Rocks_Amphitheater.JPG). Its towers
are about 20.6 km from the stage in a straight line, at a bearing of about 65
degrees (the 15 miles, ~24 km, usually quoted is the drive).

``export_denver`` writes ``Denver.obj``:

* **Buildings.** Every ``building=*`` and ``building:part=*`` way and
  multipolygon relation in ``DOWNTOWN`` (Union Station and LoDo to Capitol
  Hill), fetched once from the Overpass API and cached (``osm.fetch_overpass``).
  A building's height is its ``height`` tag, else ``building:levels`` x 3.5 m,
  else 10 m; a part starts at ``min_height`` (else ``building:min_level`` x
  3.5 m). Downtown is mapped in 3D: where a building's outline has
  ``building:part`` ways inside it (Republic Plaza has 29), the parts are used
  instead of the outline, unless they stop short of its height (only the
  podium mapped in parts), when the outline stays too. Anything whose top is
  under 12 m goes: from the top row a 12 m building is about one pixel tall.
* **Few triangles.** Most parts are columns, fins and crown pieces a few square
  metres in plan. Pieces whose bottoms and tops round to the same 3 m are merged
  into one footprint, footprints are simplified to 1.5 m (a pixel at 20 km is
  over 10 m), footprints under 10 m^2 go unless they are spires on a roof, and a
  prism wholly inside another goes.
* **Ground.** Each building stands on the lowest point under its outline on a
  coarse USGS 3DEP DEM of downtown, fetched the way ``dem.py`` fetches the
  park's and cached (``dem.cached_dem``). Parts share their outline's ground,
  so a crown sits on its tower.
* **Frame.** The terrain frame, like every OBJ: centimetres, X east, Y north,
  Z up, Z = 0 at the terrain's lowest elevation (the manifest's base). East and
  north come from the venue's ``LocalProjection`` about the project origin, a
  tangent plane, so bearings and distances from the bowl are the true ones.
* **Earth curvature.** The level is flat, but at 20 km the ground has fallen
  33 m below the tangent plane, and the air bends sight lines back down by
  about 13 % of that. Every vertex drops by ``curvature_drop_m``:
  d^2 / 2R x (1 - k), R = 6,371 km, k = 0.13, d its distance from the origin
  (29 m at downtown, 43 m at 25 km). The terrain skirt
  (``unreal/setup_environment.py``) drops every ring past the terrain's edge
  the same way, so the plains and the city agree; the 3.6 km terrain mesh
  itself stays flat (the drop is 0.2 m at its edge).

Prisms are closed, wind counter-clockwise seen from outside, and carry UVs
(metres) and flat normals on every face, like the seating's OBJs.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

from .dem import BBox, Dem, aspect_rows, cached_dem
from .geo import LocalProjection
from .osm import OsmData, Way, fetch_overpass
from .park import _earclip
from .seating import _write_obj

# Union Station and LoDo (west) to Capitol Hill (east), Coors Field (north) to
# Speer Boulevard's towers (south): every building over 45 m in the area.
DOWNTOWN = BBox(west=-105.016, south=39.728, east=-104.962, north=39.765)

LEVEL_HEIGHT_M = 3.5      # building:levels and building:min_level
DEFAULT_HEIGHT_M = 10.0   # a building with neither height nor levels
MIN_HEIGHT_M = 12.0       # tops lower than this are not built
PARTS_SHORT_M = 2.0       # parts topping out lower than their outline by more keep the outline

HEIGHT_BIN_M = 3.0        # pieces whose bottoms and tops round alike merge
CLOSE_M = 0.3             # gaps this wide between merged pieces close
SIMPLIFY_M = 1.5          # footprint simplification
MIN_AREA_M2 = 10.0        # smaller footprints go ...
SPIRE_AREA_M2 = 1.0       # ... unless they are spires: this big,
SPIRE_HEIGHT_M = 10.0     # this tall,
SPIRE_ROOF_M = 20.0       # standing at least this high on a building
HIDDEN_SLACK_M = 0.5      # a prism within this of another, inside it in plan and height, goes

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_K = 0.13       # standard terrestrial refraction coefficient
GROUND_DEM_SIZE = 257     # columns of the coarse 3DEP raster over DOWNTOWN: ~18 m pixels

OBJ_NAME = "Denver"


@dataclass
class Block:
    """One extrusion: a footprint (terrain-frame metres) between two elevations."""

    plan: Polygon
    bottom_m: float   # metres above sea level
    top_m: float
    ground_m: float   # the ground it stands on (its building's)
    name: str = ""
    osm: str = ""     # "way/123" or "relation/456"

    @property
    def height_m(self) -> float:
        return self.top_m - self.bottom_m


@dataclass
class DenverMesh:
    path: Path
    buildings: int          # OSM buildings with something 12 m or taller
    blocks: list[Block]     # what was extruded
    triangles: int
    base_elevation_m: float
    centre: tuple[float, float]      # volume-weighted centre of the blocks, east/north m
    bearing_deg: float               # from ``origin`` (the stage front centre)
    distance_m: float
    origin: tuple[float, float]
    tallest: Block
    ground_range_m: tuple[float, float]
    bbox: BBox


# ---- tags -----------------------------------------------------------------------

_FEET = 0.3048


def parse_length_m(value: str | None) -> float | None:
    """A length tag in metres: ``218``, ``218 m``, ``21,5``, ``715 ft``, ``715'``, ``10'6"``.

    Lists (``218;220``) take the first value. None when it is not a positive length.
    """
    if not value:
        return None
    text = value.split(";")[0].strip().lower()
    feet = re.fullmatch(r"(\d+(?:\.\d+)?)\s*'\s*(?:(\d+(?:\.\d+)?)\s*\")?", text)
    if feet:
        metres = float(feet.group(1)) * _FEET + float(feet.group(2) or 0.0) * _FEET / 12.0
        return metres if metres > 0 else None
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(m|metres?|meters?|ft|feet)?", text)
    if not m:
        return None
    number = float(m.group(1).replace(",", "."))
    if m.group(2) in ("ft", "feet"):
        number *= _FEET
    return number if number > 0 else None


def _levels(value: str | None) -> float | None:
    if not value:
        return None
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", value)
    return float(m.group(1).replace(",", ".")) if m else None


def building_height_m(tags: dict[str, str]) -> float:
    """Top of a building or part above its ground: ``height``, else levels x 3.5 m, else 10 m."""
    height = parse_length_m(tags.get("height"))
    if height is not None:
        return height
    levels = _levels(tags.get("building:levels"))
    if levels:
        return levels * LEVEL_HEIGHT_M
    return DEFAULT_HEIGHT_M


def building_min_height_m(tags: dict[str, str]) -> float:
    """Bottom of a part above its ground: ``min_height``, else ``building:min_level`` x 3.5 m, else 0."""
    low = parse_length_m(tags.get("min_height"))
    if low is not None:
        return low
    levels = _levels(tags.get("building:min_level"))
    return levels * LEVEL_HEIGHT_M if levels else 0.0


def _is_building(tags: dict[str, str]) -> bool:
    return tags.get("building", "no") != "no" and tags.get("location") != "underground"


def _is_part(tags: dict[str, str]) -> bool:
    return tags.get("building:part", "no") != "no" and tags.get("location") != "underground"


# ---- fetching -------------------------------------------------------------------

def overpass_query(bbox: BBox = DOWNTOWN) -> str:
    """Every building and building part in ``bbox``, with their nodes and outer ways."""
    box = f"{bbox.south},{bbox.west},{bbox.north},{bbox.east}"
    return (
        f"[out:xml][timeout:240][bbox:{box}];\n"
        "(\n"
        '  way["building"];\n'
        '  way["building:part"];\n'
        '  relation["type"="multipolygon"]["building"];\n'
        '  relation["type"="multipolygon"]["building:part"];\n'
        ");\n"
        "(._;>;);\n"
        "out body;\n"
    )


def overpass_cache_path(cache_dir, bbox: BBox = DOWNTOWN) -> Path:
    """Where ``fetch_downtown`` keeps the Overpass answer for ``bbox``."""
    return Path(cache_dir) / (
        f"overpass_buildings_{bbox.south:.4f}_{bbox.west:.4f}_{bbox.north:.4f}_{bbox.east:.4f}.osm"
    )


def fetch_downtown(cache_dir, bbox: BBox = DOWNTOWN) -> tuple[OsmData, Dem]:
    """(buildings from Overpass, a coarse 3DEP DEM) for ``bbox``, both cached in ``cache_dir``."""
    osm = fetch_overpass(overpass_query(bbox), cache=overpass_cache_path(cache_dir, bbox))
    ground = cached_dem(bbox, GROUND_DEM_SIZE, cache_dir)
    return osm, ground


# ---- buildings --------------------------------------------------------------------

@dataclass
class _Feature:
    osm: str
    part: bool
    tags: dict[str, str]
    rings: list[list[tuple[float, float]]]  # outer rings, (lat, lon)
    plans: list[Polygon]                    # the same, projected


def _rings_of_relation(osm: OsmData, ways: dict[int, Way], members) -> list[list[tuple[float, float]]]:
    """Outer rings of a multipolygon, assembled from its member ways, as (lat, lon)."""
    lines = []
    for kind, ref, role in members:
        if kind != "way" or role not in ("outer", "") or ref not in ways:
            continue
        coords = osm.way_coords(ways[ref])
        if len(coords) >= 2:
            lines.append(LineString([(lon, lat) for lat, lon in coords]))
    if not lines:
        return []
    merged = linemerge(lines)
    rings = []
    for poly in polygonize(list(getattr(merged, "geoms", [merged]))):
        rings.append([(lat, lon) for lon, lat in poly.exterior.coords])
    return rings


def _features(osm: OsmData, projection: LocalProjection, bbox: BBox) -> list[_Feature]:
    """Building outlines and parts whose middle is in ``bbox``, projected to the terrain frame."""
    cache: dict[tuple[float, float], tuple[float, float]] = {}

    def project(lat: float, lon: float) -> tuple[float, float]:
        key = (lat, lon)
        if key not in cache:
            cache[key] = projection.to_local(lat, lon)[:2]
        return cache[key]

    raw: list[tuple[str, dict, list]] = []
    for way in osm.ways:
        if (_is_building(way.tags) or _is_part(way.tags)) and way.is_closed:
            coords = osm.way_coords(way)
            if len(coords) >= 4:
                raw.append((f"way/{way.id}", way.tags, [coords]))
    by_id = {w.id: w for w in osm.ways}
    for rel in osm.relations:
        if rel.tags.get("type") == "multipolygon" and (_is_building(rel.tags) or _is_part(rel.tags)):
            rings = _rings_of_relation(osm, by_id, rel.members)
            if rings:
                raw.append((f"relation/{rel.id}", rel.tags, rings))

    out = []
    for ident, tags, rings in raw:
        lat = float(np.mean([p[0] for r in rings for p in r]))
        lon = float(np.mean([p[1] for r in rings for p in r]))
        if not (bbox.south <= lat <= bbox.north and bbox.west <= lon <= bbox.east):
            continue
        plans = []
        for ring in rings:
            poly = Polygon([project(a, b) for a, b in ring])
            if not poly.is_valid:
                poly = poly.buffer(0)
            plans += [g for g in getattr(poly, "geoms", [poly]) if g.geom_type == "Polygon" and g.area > 0.0]
        if plans:
            out.append(_Feature(ident, _is_part(tags) and not _is_building(tags), tags, rings, plans))
    return out


def building_blocks(osm: OsmData, projection: LocalProjection, ground: Dem,
                    bbox: BBox = DOWNTOWN) -> tuple[list[Block], int]:
    """(blocks 12 m or taller, how many OSM buildings they come from).

    An outline with ``building:part`` features inside it is replaced by them,
    unless they stop short of its height (some outlines carry the tower's
    height while only their podium is mapped in parts: then the outline stays
    too). Every part stands on its outline's ground.
    """
    feats = _features(osm, projection, bbox)
    outlines = [f for f in feats if not f.part]
    parts = [f for f in feats if f.part]
    plan_of = [unary_union(f.plans) for f in outlines]
    tree = STRtree(plan_of) if plan_of else None
    owner: list[int | None] = []
    parts_top: dict[int, float] = {}
    for p in parts:
        point = unary_union(p.plans).representative_point()
        hits = [int(i) for i in (tree.query(point) if tree is not None else [])
                if plan_of[int(i)].contains(point)]
        i = min(hits, key=lambda i: plan_of[i].area) if hits else None
        owner.append(i)
        if i is not None:
            parts_top[i] = max(parts_top.get(i, 0.0), building_height_m(p.tags))
    has_parts = {i for i, top in parts_top.items()
                 if top + PARTS_SHORT_M >= building_height_m(outlines[i].tags)}

    def ground_of(f: _Feature) -> float:
        return min(ground.sample(lat, lon) for ring in f.rings for lat, lon in ring)

    grounds: dict[int, float] = {}
    blocks: list[Block] = []
    owners: set[str] = set()

    def add(f: _Feature, base: float, name: str, owner_key: str) -> None:
        top, low = building_height_m(f.tags), building_min_height_m(f.tags)
        if top < MIN_HEIGHT_M or top <= low:
            return
        for plan in f.plans:
            blocks.append(Block(plan, base + low, base + top, base, name, f.osm))
        owners.add(owner_key)

    for i, f in enumerate(outlines):
        if i in has_parts:
            continue
        add(f, ground_of(f), f.tags.get("name", ""), f.osm)
    for p, i in zip(parts, owner):
        if i is None:
            add(p, ground_of(p), p.tags.get("name", ""), p.osm)
            continue
        if i not in grounds:
            grounds[i] = ground_of(outlines[i])
        add(p, grounds[i], outlines[i].tags.get("name", "") or p.tags.get("name", ""), outlines[i].osm)
    return blocks, len(owners)


def merge_blocks(blocks: Sequence[Block], bin_m: float = HEIGHT_BIN_M,
                 simplify_m: float = SIMPLIFY_M, min_area_m2: float = MIN_AREA_M2) -> list[Block]:
    """Fewer, simpler blocks with the same skyline.

    Blocks whose bottoms and tops fall in the same ``bin_m`` bins are unioned
    (touching ones become one footprint; each piece keeps the lowest bottom and
    highest top of what it was made from, so the tallest keeps its height);
    footprints lose their holes and are simplified; small ones go unless they
    are spires on a roof; a block inside another in plan and height goes.
    """
    bins: dict[tuple[int, int], list[Block]] = defaultdict(list)
    for b in blocks:
        bins[(round(b.bottom_m / bin_m), round(b.top_m / bin_m))].append(b)
    merged: list[Block] = []
    for items in bins.values():
        tree = STRtree([b.plan for b in items])
        union = unary_union([b.plan for b in items])
        union = union.buffer(CLOSE_M, join_style="mitre").buffer(-CLOSE_M, join_style="mitre")
        for piece in getattr(union, "geoms", [union]):
            if piece.geom_type != "Polygon" or piece.is_empty:
                continue
            members = [items[int(i)] for i in tree.query(piece) if items[int(i)].plan.intersects(piece)]
            if not members:
                continue
            plan = Polygon(piece.exterior).simplify(simplify_m, preserve_topology=True)
            if plan.is_empty or plan.geom_type != "Polygon" or not plan.is_valid:
                continue
            tallest = max(members, key=lambda b: b.top_m)
            block = Block(plan, min(b.bottom_m for b in members), tallest.top_m,
                          min(b.ground_m for b in members),
                          next((b.name for b in sorted(members, key=lambda b: -b.top_m) if b.name), ""),
                          tallest.osm)
            spire = (plan.area >= SPIRE_AREA_M2 and block.height_m >= SPIRE_HEIGHT_M
                     and block.bottom_m - block.ground_m >= SPIRE_ROOF_M)
            if plan.area >= min_area_m2 or spire:
                merged.append(block)
    return _drop_hidden(merged)


def _drop_hidden(blocks: list[Block]) -> list[Block]:
    """Blocks that are not wholly inside another block (in plan and in height)."""
    if not blocks:
        return []
    grown = [b.plan.buffer(HIDDEN_SLACK_M, join_style="mitre") for b in blocks]
    tree = STRtree([b.plan for b in blocks])
    keep = []
    for k, b in enumerate(blocks):
        hidden = False
        for j in tree.query(b.plan):
            j = int(j)
            o = blocks[j]
            if j == k or o.bottom_m > b.bottom_m + 1e-6 or o.top_m < b.top_m - 1e-6:
                continue
            same = abs(o.bottom_m - b.bottom_m) < 1e-6 and abs(o.top_m - b.top_m) < 1e-6
            if grown[j].contains(b.plan) and not (same and j > k and grown[k].contains(o.plan)):
                hidden = True
                break
        if not hidden:
            keep.append(b)
    return keep


# ---- geometry ---------------------------------------------------------------------

def curvature_drop_m(distance_m, radius_m: float = EARTH_RADIUS_M, k: float = REFRACTION_K):
    """How far the ground at ``distance_m`` sits below a flat level, as seen through air.

    The Earth falls d^2 / 2R below the tangent plane; refraction bends the
    sight line back over a (1 - k) share of that.
    """
    d = np.asarray(distance_m, dtype=float)
    drop = d * d / (2.0 * radius_m) * (1.0 - k)
    return float(drop) if drop.ndim == 0 else drop


def prism(plan: Polygon, z0_m: float, z1_m: float,
          drop: Callable[[np.ndarray], np.ndarray] | None = None):
    """Closed vertical prism over a simple polygon: (verts[N,3], tris, uvs[M,2], tri_uvs).

    ``z0_m``/``z1_m`` are the bottom and top in the OBJ's metres; ``drop(xy)``
    lowers every vertex by the given amount (both rings alike, so the walls stay
    vertical and the prism keeps its height). Counter-clockwise seen from
    outside; UVs are metres, along the perimeter and up on the walls.
    """
    if z1_m <= z0_m:
        raise ValueError("prism needs z1 > z0")
    ring = np.asarray(orient(plan, sign=1.0).exterior.coords, dtype=float)[:-1, :2]
    keep = [0]
    for i in range(1, len(ring)):
        if np.hypot(*(ring[i] - ring[keep[-1]])) > 1e-3:
            keep.append(i)
    ring = ring[keep]
    if len(ring) > 3 and np.hypot(*(ring[-1] - ring[0])) <= 1e-3:
        ring = ring[:-1]
    n = len(ring)
    if n < 3:
        raise ValueError("prism plan needs at least 3 distinct corners")
    lower = np.zeros(n) if drop is None else np.asarray(drop(ring), dtype=float)
    verts = np.vstack([np.column_stack([ring, z0_m - lower]), np.column_stack([ring, z1_m - lower])])

    cap_uv = ring - ring.min(axis=0)
    uvs: list[tuple[float, float]] = [tuple(p) for p in cap_uv]
    tris: list[tuple[int, int, int]] = []
    tuvs: list[tuple[int, int, int]] = []
    for a, b, c in _earclip(ring):
        tris.append((n + a, n + b, n + c))  # top, +Z
        tuvs.append((a, b, c))
        tris.append((a, c, b))              # bottom, -Z
        tuvs.append((a, c, b))
    s, h = 0.0, z1_m - z0_m
    for i in range(n):
        j = (i + 1) % n
        length = float(np.hypot(*(ring[j] - ring[i])))
        k = len(uvs)
        uvs += [(s, 0.0), (s + length, 0.0), (s + length, h), (s, h)]
        s += length
        tris += [(i, j, n + j), (i, n + j, n + i)]
        tuvs += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return verts, tris, np.array(uvs), tuvs


def bearing_distance(frm: Sequence[float], to: Sequence[float]) -> tuple[float, float]:
    """(compass bearing in degrees, distance in metres) between two terrain-frame points."""
    de, dn = float(to[0]) - float(frm[0]), float(to[1]) - float(frm[1])
    return math.degrees(math.atan2(de, dn)) % 360.0, math.hypot(de, dn)


def visible_height_m(eye_z: float, distances, heights, tower_d: float,
                     base_z: float, top_z: float) -> float:
    """Metres of a tower seen over the ground in front of it.

    ``distances``/``heights`` sample the ground along the sight line from the
    eye (horizontal metres, elevations in the eye's frame: apply the curvature
    drop first where it belongs). Ground at or beyond the tower does not hide
    it. 0 when the tower is wholly hidden.
    """
    d = np.asarray(distances, dtype=float)
    h = np.asarray(heights, dtype=float)
    near = (d > 0.0) & (d < tower_d) & np.isfinite(h)
    graze = float(np.max((h[near] - eye_z) / d[near])) if near.any() else -math.inf
    hidden_to = eye_z + graze * tower_d
    return max(0.0, top_z - max(base_z, hidden_to))


# ---- export -----------------------------------------------------------------------

def export_denver(out_path: Path, osm: OsmData, ground: Dem, projection: LocalProjection,
                  base_elevation_m: float, origin: Sequence[float] = (0.0, 0.0),
                  bbox: BBox = DOWNTOWN) -> DenverMesh:
    """Write ``Denver.obj`` from fetched buildings and ground (``fetch_downtown``).

    ``origin`` is where bearings and distances are measured from: the stage
    front centre (terrain frame, metres).
    """
    raw, buildings = building_blocks(osm, projection, ground, bbox)
    blocks = merge_blocks(raw)
    if not blocks:
        raise ValueError(f"no buildings {MIN_HEIGHT_M:.0f} m or taller in {bbox}")

    def drop(xy: np.ndarray) -> np.ndarray:
        return curvature_drop_m(np.hypot(xy[:, 0], xy[:, 1]))

    solids = [prism(b.plan, b.bottom_m - base_elevation_m, b.top_m - base_elevation_m, drop)
              for b in blocks]
    out_path = Path(out_path)
    triangles = _write_obj(out_path, OBJ_NAME, solids, (
        "Downtown Denver from OpenStreetMap (ODbL 1.0) on USGS 3DEP ground; "
        f"centimetres, terrain frame, base {base_elevation_m:.3f} m; every vertex dropped "
        f"d^2/2R*(1-k) for Earth curvature, R {EARTH_RADIUS_M / 1000:.0f} km, k {REFRACTION_K}"))

    weights = np.array([b.plan.area * b.height_m for b in blocks])
    cx = float(np.average([b.plan.centroid.x for b in blocks], weights=weights))
    cy = float(np.average([b.plan.centroid.y for b in blocks], weights=weights))
    bearing, distance = bearing_distance(origin, (cx, cy))
    grounds = [b.ground_m for b in blocks]
    return DenverMesh(
        path=out_path, buildings=buildings, blocks=blocks, triangles=triangles,
        base_elevation_m=base_elevation_m, centre=(cx, cy), bearing_deg=bearing,
        distance_m=distance, origin=(float(origin[0]), float(origin[1])),
        tallest=max(blocks, key=lambda b: b.top_m - b.ground_m),
        ground_range_m=(min(grounds), max(grounds)), bbox=bbox,
    )


def _tallest_info(mesh: DenverMesh) -> dict:
    t = mesh.tallest
    bearing, distance = bearing_distance(mesh.origin, (t.plan.centroid.x, t.plan.centroid.y))
    return {
        "name": t.name,
        "osm": t.osm,
        "height_m": round(t.top_m - t.ground_m, 2),
        "top_elevation_m": round(t.top_m, 2),
        "bearing_deg": round(bearing, 2),
        "distance_km": round(distance / 1000.0, 3),
    }


def manifest_block(mesh: DenverMesh) -> dict:
    """The manifest's ``denver`` entry."""
    b = mesh.bbox
    return {
        "obj": mesh.path.name,
        "buildings": mesh.buildings,
        "prisms": len(mesh.blocks),
        "triangles": mesh.triangles,
        "bearing_deg": round(mesh.bearing_deg, 2),
        "distance_km": round(mesh.distance_m / 1000.0, 3),
        "measured_from": "stage front centre (stage.front_edge_*); to the skyline's "
                         "volume-weighted centre",
        "centre_east_m": round(mesh.centre[0], 1),
        "centre_north_m": round(mesh.centre[1], 1),
        "tallest": _tallest_info(mesh),
        "min_height_m": MIN_HEIGHT_M,
        "ground_elevation_m": [round(v, 1) for v in mesh.ground_range_m],
        "ground": f"USGS 3DEP, {GROUND_DEM_SIZE}x{aspect_rows(b, GROUND_DEM_SIZE)} over the bbox",
        "bbox": {"west": b.west, "south": b.south, "east": b.east, "north": b.north},
        "base_elevation_m": round(mesh.base_elevation_m, 3),
        "curvature": {
            "radius_km": EARTH_RADIUS_M / 1000.0,
            "refraction_k": REFRACTION_K,
            "drop_m": round(curvature_drop_m(math.hypot(*mesh.centre)), 2),
            "applies_to": "every Denver.obj vertex, by its distance from the origin; the terrain "
                          "skirt drops the same (unreal/setup_environment.py), the terrain mesh is flat",
        },
    }


def summary(mesh: DenverMesh) -> str:
    t = _tallest_info(mesh)
    return (
        f"  {mesh.buildings} buildings 12 m+ -> {len(mesh.blocks)} prisms, "
        f"{mesh.triangles:,} triangles -> {mesh.path.name}\n"
        f"  skyline {mesh.distance_m / 1000.0:.2f} km from the stage front at bearing "
        f"{mesh.bearing_deg:.1f} deg; ground {mesh.ground_range_m[0]:.0f}-{mesh.ground_range_m[1]:.0f} m, "
        f"dropped {curvature_drop_m(math.hypot(*mesh.centre)):.1f} m for curvature\n"
        f"  tallest: {t['name'] or t['osm']} {t['height_m']:.0f} m, "
        f"{t['distance_km']:.2f} km at {t['bearing_deg']:.1f} deg"
    )
