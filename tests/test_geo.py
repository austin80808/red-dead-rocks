import math

import pytest

from redrocks.geo import (
    WGS84_A,
    WGS84_E2,
    LocalProjection,
    geodetic_to_ecef,
    lonlat_to_tile,
    metres_per_degree,
)


def test_ecef_matches_known_equator_point():
    x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert x == pytest.approx(6378137.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert z == pytest.approx(0.0, abs=1e-6)


def test_ecef_north_pole_is_polar_radius():
    _, _, z = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert z == pytest.approx(6356752.314245, abs=1e-3)


def test_projection_origin_is_zero():
    proj = LocalProjection(39.6655, -105.2055, 1938.0)
    east, north, up = proj.to_local(39.6655, -105.2055, 1938.0)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(0.0, abs=1e-6)
    assert up == pytest.approx(0.0, abs=1e-6)


def test_projection_east_is_positive_x():
    proj = LocalProjection(39.6655, -105.2055)
    east, north, _ = proj.to_local(39.6655, -105.2045)
    assert east > 0
    assert abs(north) < 1.0


def test_projection_north_is_positive_y():
    proj = LocalProjection(39.6655, -105.2055)
    east, north, _ = proj.to_local(39.6665, -105.2055)
    assert north > 0
    assert abs(east) < 1.0


def test_projection_distance_matches_geodesy():
    """100 m north of the origin should measure 100 m in the tangent plane."""
    proj = LocalProjection(39.6655, -105.2055)
    _, m_per_deg_lat = metres_per_degree(39.6655)
    lat = 39.6655 + 100.0 / m_per_deg_lat
    _, north, _ = proj.to_local(lat, -105.2055)
    assert north == pytest.approx(100.0, rel=1e-4)


def test_metres_per_degree_at_equator():
    m_lon, m_lat = metres_per_degree(0.0)
    assert m_lon == pytest.approx(111319.49, rel=1e-4)
    assert m_lat == pytest.approx(110574.39, rel=1e-4)


def test_metres_per_degree_longitude_shrinks_with_latitude():
    m_lon_equator, _ = metres_per_degree(0.0)
    m_lon_redrocks, _ = metres_per_degree(39.6655)
    assert m_lon_redrocks < m_lon_equator
    # Ellipsoidal, so it tracks cos(lat) to within the flattening (~0.14%).
    assert m_lon_redrocks == pytest.approx(m_lon_equator * math.cos(math.radians(39.6655)), rel=2e-3)


def test_metres_per_degree_longitude_uses_ellipsoid():
    """Longitude spacing is (pi/180) * N * cos(lat) with N the prime vertical radius."""
    lat = 39.6655
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(math.radians(lat)) ** 2)
    expected = (math.pi / 180.0) * n * math.cos(math.radians(lat))
    m_lon, _ = metres_per_degree(lat)
    assert m_lon == pytest.approx(expected, rel=1e-12)


def test_tile_coords_at_zoom_zero_centre():
    x, y = lonlat_to_tile(0.0, 0.0, 0)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)


def test_tile_coords_red_rocks_zoom_16():
    x, y = lonlat_to_tile(-105.2055, 39.6655, 16)
    assert x == pytest.approx(13615.9, abs=0.5)
    assert y == pytest.approx(24889.9, abs=0.5)
