"""Row polylines and the terrain datum that the seating (redrocks.seating) builds on."""

import numpy as np
import pytest

from redrocks.benches import row_polylines, terrain_base_elevation
from redrocks.dem import BBox, Dem
from redrocks.geo import LocalProjection
from redrocks.venue import Row, build_venue


def test_terrain_base_matches_terrain_min():
    bbox = BBox(west=-105.21, south=39.66, east=-105.20, north=39.67)
    data = np.linspace(1900.0, 1950.0, 64, dtype=np.float32).reshape(8, 8)
    assert terrain_base_elevation(Dem(data=data, bbox=bbox), grid=17) == pytest.approx(1900.0)


class _SlopeDem:
    def sample(self, lat, lon):
        return 1800.0 + (lat - 39.6650) * 11000.0


def test_row_polylines_match_venue_rows():
    lat0, lon0 = 39.6655, -105.2055
    proj = LocalProjection(lat0, lon0, 1800.0)
    # A ~120 m x 60 m footprint, the long side north-south.
    d = 0.0005
    ring = [
        (lat0 - 3 * d, lon0 - d), (lat0 - 3 * d, lon0 + d),
        (lat0 + 3 * d, lon0 + d), (lat0 + 3 * d, lon0 - d), (lat0 - 3 * d, lon0 - d),
    ]
    venue = build_venue(ring, _SlopeDem(), proj, lat0, lon0, n_rows=20, capacity=1000)
    lines = row_polylines(venue, ring, proj, n_rows=20)
    assert len(lines) == len(venue.rows) == 20
    for row, line in zip(venue.rows, lines):
        # build_venue's own (curved) arcs come back as they are.
        assert np.array_equal(line, row.polyline)
        assert float(np.hypot(*np.diff(line, axis=0).T).sum()) == pytest.approx(row.length_m, abs=0.01)
        if row.length_m > 20.0:
            assert row.length_m > np.hypot(*(line[-1] - line[0])) + 0.01  # bowed, not a chord


def test_row_polylines_falls_back_to_straight_chords_for_bare_rows():
    """Rows without a stored polyline are re-cut with the straight slicer."""
    from shapely.geometry import Polygon

    from redrocks.venue import slice_polygon

    lat0, lon0 = 39.6655, -105.2055
    proj = LocalProjection(lat0, lon0, 1800.0)
    d = 0.0005
    ring = [
        (lat0 - 3 * d, lon0 - d), (lat0 - 3 * d, lon0 + d),
        (lat0 + 3 * d, lon0 + d), (lat0 + 3 * d, lon0 - d), (lat0 - 3 * d, lon0 - d),
    ]
    poly = Polygon([proj.to_local(a, b)[:2] for a, b in ring])
    rows = []
    for i, chord in enumerate(slice_polygon(poly, 20)):
        m = chord.interpolate(0.5, normalized=True)
        rows.append(Row(i + 1, 50, chord.length, 1800.0 + i, m.x, m.y))  # no polyline

    class Bare:
        pass

    venue = Bare()
    venue.rows = rows
    lines = row_polylines(venue, ring, proj, n_rows=20)
    for row, line in zip(rows, lines):
        assert np.hypot(*(line[-1] - line[0])) == pytest.approx(row.length_m, abs=0.01)
    with pytest.raises(ValueError):
        row_polylines(venue, ring, proj, n_rows=7)


