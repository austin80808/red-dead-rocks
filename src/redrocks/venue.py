"""Build the amphitheatre seating bowl from the real OSM footprint.

Rows are derived by slicing the venue polygon, so the bowl follows the actual
sandstone shape rather than an idealised fan. Seat counts are scaled to the
venue's official capacity.

The DEM cannot resolve individual rows: one row rises 0.44 m while a DEM pixel
spans 1.19 m, so rows are authored here and only the bowl's overall relief comes
from terrain data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon

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
        horizontal = self.rows[0].centre_north - self.rows[-1].centre_north
        horizontal = abs(horizontal) or 1.0
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


def build_venue(
    footprint_latlon: list[tuple[float, float]],
    dem,
    projection: LocalProjection,
    lat0: float,
    lon0: float,
    n_rows: int = ROWS,
    capacity: int = CAPACITY,
) -> Venue:
    """Derive bench rows from the OSM footprint and terrain."""
    projected = [projection.to_local(lat, lon)[:2] for lat, lon in footprint_latlon]
    polygon = Polygon(projected)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    slices = slice_polygon(polygon, n_rows)
    if not slices:
        raise ValueError("could not slice the venue footprint into rows")

    m_per_deg_lon, m_per_deg_lat = metres_per_degree(lat0)

    rows: list[Row] = []
    for i, cap in enumerate(slices):
        midpoint = cap.interpolate(0.5, normalized=True)
        east, north = float(midpoint.x), float(midpoint.y)

        lat = lat0 + north / m_per_deg_lat
        lon = lon0 + east / m_per_deg_lon
        elevation = dem.sample(lat, lon)

        rows.append(
            Row(
                index=i + 1,
                seats=0,  # filled below
                length_m=float(cap.length),
                elevation_m=elevation,
                centre_east=east,
                centre_north=north,
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
