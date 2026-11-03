import math

import pytest
from shapely.geometry import Polygon

from redrocks.monoliths import (
    CREATION_ROCK_HEIGHT_FT,
    FT,
    SHIP_ROCK_HEIGHT_FT,
    STRATA_DIP_DEGREES,
    Monolith,
    monolith_to_obj,
)


def square(size=100.0, cx=0.0, cy=0.0):
    h = size / 2
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


def make_monolith(**kwargs):
    defaults = dict(
        name="Test Rock",
        footprint=square(),
        base_elevation_m=1932.5,
        height_m=100.0,
    )
    defaults.update(kwargs)
    return Monolith(**defaults)


def test_official_heights():
    assert CREATION_ROCK_HEIGHT_FT == 300.0
    assert SHIP_ROCK_HEIGHT_FT == 200.0
    assert CREATION_ROCK_HEIGHT_FT * FT == pytest.approx(91.44)
    assert SHIP_ROCK_HEIGHT_FT * FT == pytest.approx(60.96)


def test_top_elevation():
    rock = make_monolith(base_elevation_m=1932.4, height_m=60.96)
    assert rock.top_elevation_m == pytest.approx(1993.36)


def test_default_dip_is_30_degrees():
    assert STRATA_DIP_DEGREES == 30.0
    assert make_monolith().strata_dip_degrees == 30.0


def test_extrusion_is_vertical_not_leaning(tmp_path):
    """The monoliths are columns; the dip lives in the material, not the geometry."""
    rock = make_monolith(footprint=square(100.0), height_m=100.0)
    path = monolith_to_obj(rock, tmp_path / "rock.obj", segments=4)
    text = path.read_text()

    coords = [
        tuple(float(v) for v in line.split()[1:4])
        for line in text.splitlines()
        if line.startswith("v ")
    ]
    # Every level must share the same XY silhouette.
    base = sorted((round(x, 6), round(y, 6)) for x, y, z in coords if z == 0.0)
    for z in sorted({c[2] for c in coords}):
        level = sorted((round(x, 6), round(y, 6)) for x, y, zz in coords if zz == z)
        assert level == base


def test_obj_has_expected_vertex_count(tmp_path):
    rock = make_monolith()
    path = monolith_to_obj(rock, tmp_path / "rock.obj", segments=4)
    text = path.read_text()

    vertices = [l for l in text.splitlines() if l.startswith("v ")]
    faces = [l for l in text.splitlines() if l.startswith("f ")]
    # 5 levels (0..4) of 4 footprint corners.
    assert len(vertices) == 20
    # 4 levels x 4 walls + 2 caps of 2 triangles.
    assert len(faces) == 16 + 4


def test_obj_is_scaled_to_centimetres(tmp_path):
    rock = make_monolith(footprint=square(200.0), height_m=100.0)
    path = monolith_to_obj(rock, tmp_path / "rock.obj", segments=2)
    text = path.read_text()

    coords = [
        tuple(float(v) for v in line.split()[1:4])
        for line in text.splitlines()
        if line.startswith("v ")
    ]
    xs = [c[0] for c in coords]
    zs = [c[2] for c in coords]

    # 200 m footprint spans 20,000 cm; 100 m height is 10,000 cm.
    assert max(xs) - min(xs) == pytest.approx(20000.0, rel=1e-6)
    assert max(zs) == pytest.approx(10000.0, rel=1e-6)
    assert min(zs) == pytest.approx(0.0)


def test_obj_rejects_degenerate_footprint(tmp_path):
    rock = make_monolith(footprint=Polygon([(0, 0), (1, 1), (2, 2)]))
    with pytest.raises(ValueError, match="no usable area"):
        monolith_to_obj(rock, tmp_path / "rock.obj")


def test_obj_face_indices_are_one_based_and_in_range(tmp_path):
    rock = make_monolith()
    path = monolith_to_obj(rock, tmp_path / "rock.obj", segments=3)
    lines = path.read_text().splitlines()

    n_vertices = sum(1 for l in lines if l.startswith("v "))
    for line in lines:
        if not line.startswith("f "):
            continue
        for idx in line.split()[1:]:
            i = int(idx)
            assert 1 <= i <= n_vertices
