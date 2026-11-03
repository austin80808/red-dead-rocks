import numpy as np
import pytest

from redrocks.dem import BBox, Dem, square_bbox

BBOX = BBox(west=-105.21, south=39.66, east=-105.20, north=39.67)


def test_centre_of_bbox():
    lat, lon = BBOX.centre
    assert lat == pytest.approx(39.665)
    assert lon == pytest.approx(-105.205)


def test_sample_returns_corner_values():
    # Row 0 is north, column 0 is west.
    data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    dem = Dem(data=data, bbox=BBOX)

    assert dem.sample(39.67, -105.21) == pytest.approx(10.0)
    assert dem.sample(39.67, -105.20) == pytest.approx(20.0)
    assert dem.sample(39.66, -105.21) == pytest.approx(30.0)
    assert dem.sample(39.66, -105.20) == pytest.approx(40.0)


def test_sample_interpolates_bilinearly():
    data = np.array([[0.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    dem = Dem(data=data, bbox=BBOX)
    # Halfway between the west and east edges, on the south edge.
    assert dem.sample(39.66, -105.205) == pytest.approx(50.0)


def test_sample_clamps_outside_bbox():
    data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    dem = Dem(data=data, bbox=BBOX)
    assert dem.sample(39.99, -105.99) == pytest.approx(10.0)
    assert dem.sample(39.00, -104.00) == pytest.approx(40.0)


def test_min_max_elevation():
    data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    dem = Dem(data=data, bbox=BBOX)
    assert dem.min_elevation == pytest.approx(10.0)
    assert dem.max_elevation == pytest.approx(40.0)


def test_square_bbox_is_roughly_square_in_metres():
    from redrocks.geo import metres_per_degree

    box = square_bbox(39.6655, -105.2055, 2400.0)
    m_per_deg_lon, m_per_deg_lat = metres_per_degree(39.6655)

    width_m = box.width_deg * m_per_deg_lon
    height_m = box.height_deg * m_per_deg_lat

    assert width_m == pytest.approx(2400.0, rel=1e-3)
    assert height_m == pytest.approx(2400.0, rel=1e-3)


def test_square_bbox_centred():
    box = square_bbox(39.6655, -105.2055, 1000.0)
    lat, lon = box.centre
    assert lat == pytest.approx(39.6655)
    assert lon == pytest.approx(-105.2055)
