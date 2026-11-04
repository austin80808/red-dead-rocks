"""Build the amphitheatre seating bowl from the real OSM footprint.

Rows are derived by slicing the venue polygon with arcs centred on the stage,
so the bowl follows the actual sandstone shape and each row curves around the
stage the way the real benches do. Seat counts are scaled to the venue's
official capacity.

The DEM cannot resolve individual rows: one row rises 0.44 m while a DEM pixel
spans 1.19 m, so rows are authored here and only the bowl's overall relief comes
from terrain data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge

from .geo import LocalProjection, metres_per_degree

# Official Red Rocks facts (redrocksonline.com Fast Facts, verified 2026-11).
CAPACITY = 9525
ROWS = 70
BENCH_LENGTH_M = 2.5 * 1609.344
OFFICIAL_RISE_M = 100 * 0.3048

# 2.5 miles of bench across 9,525 seats.
SEAT_PITCH_M = BENCH_LENGTH_M / CAPACITY


@dataclass
class Row:
    index: int  # 1 = stage row, 70 = top
    seats: int
    length_m: float
    elevation_m: float
    centre_east: float
    centre_north: float
    # The row's plan-view centreline in local (east, north) metres, ordered
    # along the row. None for rows built by hand (tests, legacy callers).
    polyline: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class Venue:
    rows: list[Row]

    @property
    def total_seats(self) -> int:
        return sum(r.seats for r in self.rows)

    @property
    def rise_m(self) -> float:
        return self.rows[-1].elevation_m - self.rows[0].elevation_m

    @property
    def rake_degrees(self) -> float:
        if len(self.rows) < 2 or self.rise_m == 0:
            return 0.0
        # Plan distance between the first and last rows' centres. Rows curve and
        # the bowl is not aligned to north, so use both axes.
        horizontal = math.hypot(
            self.rows[0].centre_east - self.rows[-1].centre_east,
            self.rows[0].centre_north - self.rows[-1].centre_north,
        ) or 1.0
        return math.degrees(math.atan(self.rise_m / horizontal))


def _principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A ring repeats its first vertex at the end; leaving it in biases the mean
    # toward that corner and rotates the axis away from the true long side.
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]

    centre = points.mean(axis=0)
    centred = points - centre
    _, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    major = eigenvectors[:, -1]
    normal = np.array([-major[1], major[0]])
    return centre, major, normal


def slice_polygon(polygon: Polygon, n: int) -> list[LineString]:
    """Cut `polygon` into `n` parallel cross-sections along its long axis."""
    if polygon.is_empty or polygon.area <= 0.0:
        raise ValueError("venue footprint has no area")

    coords = np.array(polygon.exterior.coords)
    centre, major, normal = _principal_axis(coords)

    # Offsets are measured from the centre, so subtract its projection first;
    # otherwise every cut is shifted by the centre's distance along the axis.
    along = (coords - centre) @ major
    t_min, t_max = float(along.min()), float(along.max())
    reach = (t_max - t_min) * 2.0 + 200.0

    slices: list[LineString] = []
    for i in range(n):
        t = t_min + (t_max - t_min) * (i / (n - 1)) if n > 1 else 0.0
        anchor = centre + major * t
        cut = LineString([anchor - normal * reach, anchor + normal * reach])
        piece = polygon.intersection(cut)

        if piece.is_empty:
            continue
        if isinstance(piece, MultiLineString):
            piece = max(piece.geoms, key=lambda g: g.length)
        if isinstance(piece, LineString) and piece.length > 1.0:
            slices.append(piece)

    return slices


# Arc points per quarter turn when approximating a circle: 64 keeps the chord
# error under 1 cm out to 100 m radius.
ARC_QUAD_SEGS = 64
# Pieces of a row shorter than this are not benches.
MIN_ROW_LENGTH_M = 5.0
# How far beyond the footprint's low end the stage centre sits (see stage_centre).
STAGE_SETBACK_M = 30.0


def stage_centre(
    polygon: Polygon,
    elevation_at: Callable[[float, float], float],
    setback_m: float = STAGE_SETBACK_M,
) -> tuple[float, float]:
    """The point the rows curve around: the stage, off the low end of the footprint.

    The seating bowl falls toward the stage, so of the two ends of the
    footprint's principal axis the lower one (by ``elevation_at(east, north)``)
    is the stage end. The stage platform sits beyond the first row, so the
    centre is pushed ``setback_m`` further out along the axis.
    """
    coords = np.array(polygon.exterior.coords)
    centre, major, _ = _principal_axis(coords)
    along = (coords - centre) @ major
    ends = [(float(along.min()), -1.0), (float(along.max()), 1.0)]
    t, sign = min(
        ends, key=lambda e: elevation_at(*(float(v) for v in centre + major * e[0]))
    )
    point = centre + major * (t + sign * setback_m)
    return float(point[0]), float(point[1])


def _pieces(piece) -> list[LineString]:
    """The line parts of an intersection result, touching parts merged."""
    if piece.is_empty:
        return []
    if isinstance(piece, LineString):
        return [piece]
    lines = [g for g in getattr(piece, "geoms", []) if isinstance(g, (LineString, MultiLineString))]
    if not lines:
        return []
    merged = linemerge(lines)
    return list(merged.geoms) if isinstance(merged, MultiLineString) else [merged]


def _arc_at(
    polygon: Polygon,
    centre: tuple[float, float],
    radius: float,
    near_angle: float | None = None,
) -> tuple[np.ndarray, float] | None:
    """The part of the circle of `radius` about `centre` that lies in `polygon`.

    A concave footprint can split the circle into several pieces. With
    `near_angle` (radians about `centre`) the piece whose midpoint is closest to
    that bearing is chosen, so consecutive rows stay on the same part of the
    bowl; without it, the longest piece. Returns the points, running
    counter-clockwise about `centre`, and the piece's midpoint bearing, or None
    if the circle misses the polygon.
    """
    cx, cy = centre
    circle = Point(cx, cy).buffer(radius, quad_segs=ARC_QUAD_SEGS).exterior
    pieces = _pieces(polygon.intersection(circle))
    if not pieces:
        return None

    def bearing(line: LineString) -> float:
        m = line.interpolate(0.5, normalized=True)
        return math.atan2(m.y - cy, m.x - cx)

    if near_angle is None:
        line = max(pieces, key=lambda g: g.length)
    else:
        line = min(pieces, key=lambda g: abs(math.remainder(bearing(g) - near_angle, math.tau)))

    pts = np.array(line.coords)[:, :2]
    a0 = math.atan2(pts[0, 1] - cy, pts[0, 0] - cx)
    a1 = math.atan2(pts[-1, 1] - cy, pts[-1, 0] - cx)
    if math.remainder(a1 - a0, math.tau) < 0:
        pts = pts[::-1]
    return pts, bearing(line)


def _arcs_at(
    polygon: Polygon, centre: tuple[float, float], radii: Sequence[float]
) -> list[tuple[float, np.ndarray] | None]:
    """An arc for each radius (ascending), each following on from the last.

    Entries are (radius, points), or None where the arc is missing or shorter
    than ``MIN_ROW_LENGTH_M``.
    """
    out: list[tuple[float, np.ndarray] | None] = []
    bearing: float | None = None
    for r in radii:
        hit = _arc_at(polygon, centre, float(r), bearing) if r > 0.0 else None
        if hit is None or LineString(hit[0]).length < MIN_ROW_LENGTH_M:
            out.append(None)
            continue
        bearing = hit[1]
        out.append((float(r), hit[0]))
    return out


def slice_polygon_arcs(
    polygon: Polygon, n: int, centre: tuple[float, float]
) -> list[LineString]:
    """Cut `polygon` into `n` concentric arcs centred on `centre`.

    The radius range is the span over which an arc still has a usable length
    (``MIN_ROW_LENGTH_M``) inside the polygon: nearer the stage than that the
    circle only grazes the footprint's corner, farther out it only clips its
    far tip. `n` radii are spread evenly across that span. Each arc is clipped
    to the polygon; where a concave footprint splits a circle, the piece on the
    same part of the bowl as the row before is kept. Returned stage-first
    (increasing radius), each running counter-clockwise about `centre`.
    """
    if polygon.is_empty or polygon.area <= 0.0:
        raise ValueError("venue footprint has no area")

    cx, cy = centre
    ring = np.array(polygon.exterior.coords)
    dist = np.hypot(ring[:, 0] - cx, ring[:, 1] - cy)
    r_min, r_max = float(dist.min()), float(dist.max())
    if r_max - r_min < 1e-6:
        raise ValueError("venue footprint is degenerate around the stage centre")

    # Scan for the usable span at roughly 0.25 m steps.
    scan = np.linspace(r_min, r_max, max(int((r_max - r_min) / 0.25), 8))
    good = [hit[0] for hit in _arcs_at(polygon, centre, scan) if hit is not None]
    if not good:
        return []
    r_lo, r_hi = good[0], good[-1]

    radii = [r_lo + (r_hi - r_lo) * (i / (n - 1)) if n > 1 else 0.5 * (r_lo + r_hi) for i in range(n)]
    return [LineString(hit[1]) for hit in _arcs_at(polygon, centre, radii) if hit is not None]


def build_venue(
    footprint_latlon: list[tuple[float, float]],
    dem,
    projection: LocalProjection,
    lat0: float,
    lon0: float,
    n_rows: int = ROWS,
    capacity: int = CAPACITY,
) -> Venue:
    """Derive curved bench rows from the OSM footprint and terrain."""
    projected = [projection.to_local(lat, lon)[:2] for lat, lon in footprint_latlon]
    polygon = Polygon(projected)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    m_per_deg_lon, m_per_deg_lat = metres_per_degree(lat0)

    def ground(east: float, north: float) -> float:
        return dem.sample(lat0 + north / m_per_deg_lat, lon0 + east / m_per_deg_lon)

    centre = stage_centre(polygon, ground)
    slices = slice_polygon_arcs(polygon, n_rows, centre)
    if not slices:
        raise ValueError("could not slice the venue footprint into rows")

    rows: list[Row] = []
    for i, cap in enumerate(slices):
        midpoint = cap.interpolate(0.5, normalized=True)
        # A row's tread elevation is the mean ground along the whole arc: a
        # single sample at the midpoint would be thrown by one noisy DEM pixel.
        samples = [cap.interpolate(f, normalized=True) for f in np.linspace(0.0, 1.0, 17)]
        elevation = float(np.mean([ground(float(p.x), float(p.y)) for p in samples]))

        rows.append(
            Row(
                index=i + 1,
                seats=0,  # filled below
                length_m=float(cap.length),
                elevation_m=elevation,
                centre_east=float(midpoint.x),
                centre_north=float(midpoint.y),
                polyline=np.array(cap.coords, dtype=float)[:, :2],
            )
        )

    # Row 1 must be the stage end, which sits lowest.
    if rows[0].elevation_m > rows[-1].elevation_m:
        rows.reverse()
        for i, row in enumerate(rows):
            row.index = i + 1

    _distribute_seats(rows, capacity)
    return Venue(rows=rows)


def _distribute_seats(rows: list[Row], capacity: int) -> None:
    """Assign seats in proportion to row length, summing exactly to capacity."""
    total_length = sum(r.length_m for r in rows)
    if total_length <= 0:
        raise ValueError("venue rows have no length")

    exact = [capacity * r.length_m / total_length for r in rows]
    floors = [int(math.floor(v)) for v in exact]

    remainder = capacity - sum(floors)
    order = sorted(range(len(exact)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1

    for row, seats in zip(rows, floors):
        row.seats = seats


def summary(venue: Venue) -> str:
    first, last = venue.rows[0], venue.rows[-1]
    widest = max(venue.rows, key=lambda r: r.length_m)
    return (
        f"rows        {len(venue.rows)}  (official {ROWS})\n"
        f"seats       {venue.total_seats}  (official {CAPACITY})\n"
        f"rise        {venue.rise_m:.2f} m  (official {OFFICIAL_RISE_M:.2f} m)\n"
        f"rake        {venue.rake_degrees:.1f} deg\n"
        f"widest row  {widest.index}  {widest.length_m:.1f} m  {widest.seats} seats\n"
        f"row 1       elev {first.elevation_m:.1f} m  {first.length_m:.1f} m\n"
        f"row {last.index:<3d}     elev {last.elevation_m:.1f} m  {last.length_m:.1f} m\n"
        f"seat pitch  {SEAT_PITCH_M:.3f} m ({SEAT_PITCH_M / 0.3048 * 12:.1f} in)"
    )
