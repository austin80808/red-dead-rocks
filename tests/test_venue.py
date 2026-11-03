import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from redrocks.venue import (
    BENCH_LENGTH_M,
    CAPACITY,
    OFFICIAL_RISE_M,
    ROWS,
    SEAT_PITCH_M,
    Row,
    Venue,
    _distribute_seats,
    slice_polygon,
)


class FlatDem:
    def __init__(self, elevation=1900.0):
        self.elevation = elevation

    def sample(self, lat, lon):
        return self.elevation


class RampDem:
    """Elevation rises with latitude, so the north end is the top of the bowl."""

    def __init__(self, base=1935.0, per_unit=0.5, lat0=0.0):
        self.base = base
        self.per_unit = per_unit
        self.lat0 = lat0

    def sample(self, lat, lon):
        return self.base + (lat - self.lat0) * self.per_unit


class IdentityProjection:
    """Treat lat/lon degrees as metres so geometry tests stay readable."""

    def to_local(self, lat, lon, h=0.0):
        return (lon, lat, h)


def stadium_footprint(half_width=30.0, half_length=60.0):
    """A metre-scale rectangle centred on the origin, long axis running north."""
    return [
        (-half_length, -half_width),
        (-half_length, half_width),
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
    ]


def test_official_constants_are_self_consistent():
    assert SEAT_PITCH_M == pytest.approx(BENCH_LENGTH_M / CAPACITY)
    # Red Rocks benches are famously shoulder to shoulder, ~16.6 inches.
    assert SEAT_PITCH_M == pytest.approx(16.63 / 39.3701, rel=1e-3)
    assert OFFICIAL_RISE_M == pytest.approx(30.48)


def test_slice_polygon_returns_requested_count():
    poly = Polygon([(0, 0), (0, 10), (40, 10), (40, 0)])
    assert len(slice_polygon(poly, 10)) == 10


def test_slice_polygon_runs_across_the_long_axis():
    poly = Polygon([(0, 0), (0, 10), (40, 10), (40, 0)])
    slices = slice_polygon(poly, 8)
    # The major axis is X (40 long), so each cut spans the 10 m short axis.
    for cut in slices:
        assert cut.length == pytest.approx(10.0, rel=1e-6)


def test_slice_polygon_handles_single_row():
    poly = Polygon([(0, 0), (0, 10), (40, 10), (40, 0)])
    assert len(slice_polygon(poly, 1)) == 1


def test_distribute_seats_sums_exactly():
    rows = [Row(i + 1, 0, 10.0 + i, 1900.0, 0.0, 0.0) for i in range(20)]
    _distribute_seats(rows, 1000)
    assert sum(r.seats for r in rows) == 1000


def test_distribute_seats_proportional_to_length():
    rows = [Row(1, 0, 10.0, 0.0, 0.0, 0.0), Row(2, 0, 30.0, 0.0, 0.0, 0.0)]
    _distribute_seats(rows, 400)
    assert rows[0].seats == 100
    assert rows[1].seats == 300


def test_distribute_seats_rejects_zero_length():
    rows = [Row(1, 0, 0.0, 0.0, 0.0, 0.0)]
    with pytest.raises(ValueError, match="no length"):
        _distribute_seats(rows, 100)


def test_build_venue_uses_capacity():
    from redrocks.venue import build_venue

    poly = stadium_footprint()
    venue = build_venue(poly, RampDem(), IdentityProjection(), 0.0, 0.0)
    assert venue.total_seats == CAPACITY


def test_build_venue_row_one_is_lowest():
    from redrocks.venue import build_venue

    poly = stadium_footprint()
    venue = build_venue(poly, RampDem(), IdentityProjection(), 0.0, 0.0)
    elevations = [r.elevation_m for r in venue.rows]
    assert elevations == sorted(elevations)


def test_build_venue_rejects_degenerate_footprint():
    from redrocks.venue import build_venue

    tiny = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    with pytest.raises(ValueError):
        build_venue(tiny, FlatDem(), IdentityProjection(), 0.0, 0.0)


def test_venue_rise_and_rake():
    rows = [
        Row(1, 100, 50.0, 1935.0, 0.0, 0.0),
        Row(2, 100, 50.0, 1935.0 + OFFICIAL_RISE_M, 0.0, -30.0),
    ]
    venue = Venue(rows=rows)
    assert venue.rise_m == pytest.approx(OFFICIAL_RISE_M)
    assert venue.rake_degrees == pytest.approx(math.degrees(math.atan(OFFICIAL_RISE_M / 30.0)))
