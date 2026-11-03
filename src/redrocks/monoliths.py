"""Extrude the flanking sandstone monoliths from their OSM rock footprints.

The two monoliths are real and their footprints are surveyed in OSM, so the
plan-view silhouette is authentic. Height comes from the official figures
(Creation Rock 300 ft, Ship Rock 200 ft above the bowl floor) rather than the
DEM, which reads up to 40 ft high because it samples the ridge behind each rock.

Both beds dip ~30 degrees toward the west. That dip is a property of the strata,
not a lean of the rock: the monoliths are vertical columns and the dip shows up
as diagonal banding on their faces. It is therefore carried as material metadata
for texturing, and deliberately not applied as a geometric shear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .geo import LocalProjection, metres_per_degree

FT = 0.3048

# Official heights above the bowl floor (redrocksonline.com architecture page).
CREATION_ROCK_HEIGHT_FT = 300.0  # north side
SHIP_ROCK_HEIGHT_FT = 200.0  # south side
STAGE_ROCK_HEIGHT_FT = 60.0  # east, smaller

# Both monoliths dip ~30 degrees toward the west.
STRATA_DIP_DEGREES = 30.0

# Slices used to turn the tilted extrusion into a mesh.
HEIGHT_SEGMENTS = 24


@dataclass
class Monolith:
    name: str
    footprint: Polygon
    base_elevation_m: float
    height_m: float
    # Strata dip, for the material. Not a geometric lean.
    strata_dip_degrees: float = STRATA_DIP_DEGREES

    @property
    def top_elevation_m(self) -> float:
        return self.base_elevation_m + self.height_m

    def summary(self) -> str:
        minx, miny, maxx, maxy = self.footprint.bounds
        return (
            f"{self.name}\n"
            f"  footprint   {maxx - minx:.1f} x {maxy - miny:.1f} m "
            f"({self.footprint.area:,.0f} m2)\n"
            f"  height      {self.height_m:.1f} m ({self.height_m / FT:.0f} ft)\n"
            f"  base        {self.base_elevation_m:.1f} m\n"
            f"  top         {self.top_elevation_m:.1f} m\n"
            f"  strata dip  {self.strata_dip_degrees:.0f} deg west (material only)\n"
            f"  centroid    E{self.footprint.centroid.x:.1f} N{self.footprint.centroid.y:.1f}"
        )


def _rock_polygon(osm, way_id: int, projection: LocalProjection) -> Polygon:
    way = next((w for w in osm.ways if w.id == way_id), None)
    if way is None:
        raise LookupError(f"OSM way {way_id} not found")

    ring = [projection.to_local(lat, lon)[:2] for lat, lon in osm.way_coords(way)]
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0.0:
        raise ValueError(f"OSM way {way_id} has no usable footprint")
    return polygon


def _bowl_floor(osm, dem, projection: LocalProjection, lat0: float, lon0: float) -> float:
    """Lowest point of the amphitheatre footprint: the reference for 'above floor'."""
    venue = next(w for w in osm.ways if w.tags.get("amenity") == "theatre")
    m_per_deg_lon, m_per_deg_lat = metres_per_degree(lat0)
    lowest = math.inf
    for lat, lon in osm.way_coords(venue):
        lowest = min(lowest, dem.sample(lat, lon))
    return lowest


def build_monoliths(
    osm,
    dem,
    projection: LocalProjection,
    lat0: float,
    lon0: float,
    creation_way: int = 91483333,
    ship_way: int = 91483376,
) -> list[Monolith]:
    """Build the two flanking monoliths from their surveyed OSM footprints."""
    floor = _bowl_floor(osm, dem, projection, lat0, lon0)

    a = _rock_polygon(osm, creation_way, projection)
    b = _rock_polygon(osm, ship_way, projection)

    # Creation Rock is the northern monolith and the taller of the two; assign by
    # position so a change in OSM ids cannot silently swap the heights.
    north, south = (a, b) if a.centroid.y > b.centroid.y else (b, a)

    return [
        Monolith(
            name="Creation Rock (north)",
            footprint=north,
            base_elevation_m=floor,
            height_m=CREATION_ROCK_HEIGHT_FT * FT,
        ),
        Monolith(
            name="Ship Rock (south)",
            footprint=south,
            base_elevation_m=floor,
            height_m=SHIP_ROCK_HEIGHT_FT * FT,
        ),
    ]


def monolith_to_obj(monolith: Monolith, out_path: Path, segments: int = HEIGHT_SEGMENTS) -> Path:
    """Write a closed vertical extrusion of the footprint as a Wavefront OBJ."""
    # OSM rings can be either winding; normalise to CCW so face normals always
    # come out pointing away from the rock.
    ring = list(orient(monolith.footprint, sign=1.0).exterior.coords)[:-1]
    if len(ring) < 3 or monolith.footprint.area <= 0.0:
        raise ValueError(f"{monolith.name} footprint has no usable area")

    height_units = monolith.height_m * 100.0  # UE works in centimetres
    scale = 100.0

    vertices: list[tuple[float, float, float]] = []
    for level in range(segments + 1):
        fraction = level / segments
        z = fraction * height_units
        for x, y in ring:
            vertices.append((x * scale, y * scale, z))

    n = len(ring)
    faces: list[tuple[int, ...]] = []

    # Side walls.
    for level in range(segments):
        for i in range(n):
            j = (i + 1) % n
            a = level * n + i
            b = level * n + j
            c = (level + 1) * n + j
            d = (level + 1) * n + i
            faces.append((a + 1, b + 1, c + 1, d + 1))

    # Cap top and bottom with a fan from the first vertex.
    top = segments * n
    for i in range(1, n - 1):
        faces.append((top + 1, top + i + 1, top + i + 2))
        faces.append((1, i + 2, i + 1))

    lines = [f"# {monolith.name}", f"# height {monolith.height_m:.2f} m above bowl floor"]
    lines.append(f"o {monolith.name.split(' (')[0].replace(' ', '_')}")
    for x, y, z in vertices:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    for face in faces:
        lines.append("f " + " ".join(str(i) for i in face))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
