import io
import types

import numpy as np
import pytest
from PIL import Image

from redrocks import dem as dem_mod
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


def test_cached_dem_asks_for_square_pixels_and_keeps_the_returned_extent(monkeypatch, tmp_path):
    asked = {}
    bbox = BBox(west=-105.0, south=39.70, east=-104.9, north=39.75)
    rows = dem_mod.aspect_rows(bbox, 200)
    assert rows == 100

    def fake_json(url, params, timeout):
        asked.update(params)
        return {"href": "raster", "extent": {"xmin": -105.0, "ymin": 39.7, "xmax": -104.9, "ymax": 39.75}}

    def fake_get(url, timeout, params=None):
        buf = io.BytesIO()
        Image.fromarray(np.full((rows, 200), 1600.0, dtype=np.float32), mode="F").save(buf, format="TIFF")
        return types.SimpleNamespace(content=buf.getvalue())

    monkeypatch.setattr(dem_mod, "_get_json", fake_json)
    monkeypatch.setattr(dem_mod, "_get", fake_get)
    got = dem_mod.cached_dem(bbox, 200, tmp_path)
    assert asked["size"] == "200,100"
    assert got.data.shape == (100, 200) and got.bbox == bbox
    again = dem_mod.cached_dem(bbox, 200, tmp_path)  # from the file now
    assert again.bbox == bbox and again.data.shape == (100, 200)


def test_fetch_dem_can_keep_the_extent_the_service_widened(monkeypatch):
    """3DEP keeps square pixels: a square raster of a non-square bbox comes back taller."""
    bbox = BBox(west=-105.2265, south=39.6493, east=-105.1845, north=39.6817)

    def fake_json(url, params, timeout):
        return {"href": "r", "extent": {"xmin": bbox.west, "ymin": 39.6445, "xmax": bbox.east, "ymax": 39.6865}}

    def fake_get(url, timeout, params=None):
        buf = io.BytesIO()
        Image.fromarray(np.zeros((8, 8), dtype=np.float32), mode="F").save(buf, format="TIFF")
        return types.SimpleNamespace(content=buf.getvalue())

    monkeypatch.setattr(dem_mod, "_get_json", fake_json)
    monkeypatch.setattr(dem_mod, "_get", fake_get)
    assert dem_mod.fetch_dem(bbox, size=8).bbox == bbox  # the park build's behaviour, unchanged
    assert dem_mod.fetch_dem(bbox, size=8, keep_extent=True).bbox.north == 39.6865
