import math

import pytest
from shapely.geometry import Polygon

from redrocks.monoliths import (
    CREATION_ROCK_HEIGHT_FT,
    FT,
    SHIP_ROCK_HEIGHT_FT,
    STAGE_ROCK_HEIGHT_FT,
    STAGE_ROCK_WAY_ID,
    STRATA_DIP_DEGREES,
    Monolith,
    build_monoliths,
    build_stage_rock,
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
        for tok in line.split()[1:]:
            i = int(tok.split("/")[0])
            assert 1 <= i <= n_vertices


def test_obj_declares_uvs(tmp_path):
    """Unreal's OBJ importer collapses meshes that have no UV channel."""
    rock = make_monolith()
    text = monolith_to_obj(rock, tmp_path / "r.obj", segments=4).read_text().splitlines()

    n_verts = sum(1 for l in text if l.startswith("v "))
    n_uvs = sum(1 for l in text if l.startswith("vt "))
    assert n_verts > 0
    assert n_uvs > 0


def test_every_face_references_a_valid_uv(tmp_path):
    rock = make_monolith()
    text = monolith_to_obj(rock, tmp_path / "r.obj", segments=4).read_text().splitlines()

    n_uvs = sum(1 for l in text if l.startswith("vt "))
    for line in text:
        if not line.startswith("f "):
            continue
        for token in line.split()[1:]:
            parts = token.split("/")
            assert len(parts) >= 2, f"face token {token} has no UV index"
            uv_index = int(parts[1])
            assert 1 <= uv_index <= n_uvs


def test_stage_rock_official_height_and_build():
    from redrocks.geo import LocalProjection
    from redrocks.osm import OsmData, Way

    assert STAGE_ROCK_HEIGHT_FT == 60.0
    assert STAGE_ROCK_HEIGHT_FT * FT == pytest.approx(18.288)

    lat0, lon0 = 39.6655, -105.2055
    proj = LocalProjection(lat0, lon0, 1930.0)

    def ring_nodes(start_id, dlat, dlon):
        pts = [
            (lat0 + dlat - 0.0002, lon0 + dlon - 0.0002),
            (lat0 + dlat - 0.0002, lon0 + dlon + 0.0002),
            (lat0 + dlat + 0.0002, lon0 + dlon + 0.0002),
            (lat0 + dlat + 0.0002, lon0 + dlon - 0.0002),
        ]
        nodes = {start_id + i: p for i, p in enumerate(pts)}
        ids = [start_id, start_id + 1, start_id + 2, start_id + 3, start_id]
        return nodes, ids

    n_v, ids_v = ring_nodes(1, 0.0, 0.0)
    n_c, ids_c = ring_nodes(10, 0.001, 0.0)
    n_s, ids_s = ring_nodes(20, -0.001, 0.0)
    n_st, ids_st = ring_nodes(30, 0.0, 0.0006)
    nodes = {**n_v, **n_c, **n_s, **n_st}

    class FlatDem:
        def sample(self, lat, lon):
            return 1932.0

    osm_two = OsmData(
        nodes=nodes,
        ways=[
            Way(id=1, node_ids=ids_v, tags={"amenity": "theatre"}),
            Way(id=91483333, node_ids=ids_c, tags={"natural": "bare_rock", "name": "Creation Rock"}),
            Way(id=91483376, node_ids=ids_s, tags={"natural": "bare_rock", "name": "Ship Rock"}),
        ],
    )
    assert build_stage_rock(osm_two, FlatDem(), proj, lat0, lon0) is None
    assert len(build_monoliths(osm_two, FlatDem(), proj, lat0, lon0)) == 2

    osm_three = OsmData(
        nodes=nodes,
        ways=osm_two.ways + [
            Way(id=STAGE_ROCK_WAY_ID, node_ids=ids_st, tags={"natural": "bare_rock", "name": "Stage Rock"}),
        ],
    )
    sr = build_stage_rock(osm_three, FlatDem(), proj, lat0, lon0)
    assert sr is not None and sr.name == "Stage Rock (east)"
    assert sr.height_m == pytest.approx(60.0 * FT)
    all_rocks = build_monoliths(osm_three, FlatDem(), proj, lat0, lon0)
    assert [r.name for r in all_rocks] == [
        "Creation Rock (north)",
        "Ship Rock (south)",
        "Stage Rock (east)",
    ]
