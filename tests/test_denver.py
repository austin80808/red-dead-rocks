"""Downtown Denver's skyline (redrocks.denver): where it lands, its prisms, the
build step offline, the Unreal import lists, and the line-of-sight helpers."""

import importlib.util
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from redrocks import cli, denver
from redrocks import dem as dem_mod
from redrocks import osm as osm_mod
from redrocks.dem import Dem
from redrocks.geo import LocalProjection
from redrocks.osm import parse_osm

ROOT = Path(__file__).resolve().parent.parent
ORIGIN = (39.6655, -105.2055)            # the build's origin, the bowl's centre
REPUBLIC_PLAZA = (39.7437, -104.9878)    # 370 17th St, 218 m


# ---- an independent geodesic: Vincenty's inverse on WGS84 ---------------------------

def vincenty(lat1, lon1, lat2, lon2):
    """(initial bearing deg, distance m) on the WGS84 ellipsoid."""
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    L = math.radians(lon2 - lon1)
    U1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    U2 = math.atan((1 - f) * math.tan(math.radians(lat2)))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        cs = sU1 * sU2 + cU1 * cU2 * cl
        sigma = math.atan2(ss, cs)
        sa = cU1 * cU2 * sl / ss
        c2a = 1 - sa * sa
        c2sm = cs - 2 * sU1 * sU2 / c2a
        C = f / 16 * c2a * (4 + f * (4 - 3 * c2a))
        prev, lam = lam, L + (1 - C) * f * sa * (sigma + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm ** 2)))
        if abs(lam - prev) < 1e-12:
            break
    u2 = c2a * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds = B * ss * (c2sm + B / 4 * (cs * (-1 + 2 * c2sm ** 2) - B / 6 * c2sm * (-3 + 4 * ss ** 2) * (-3 + 4 * c2sm ** 2)))
    dist = b * A * (sigma - ds)
    bearing = math.degrees(math.atan2(cU2 * math.sin(lam), cU1 * sU2 - sU1 * cU2 * math.cos(lam))) % 360
    return bearing, dist


@pytest.mark.parametrize("stage", [ORIGIN, (39.66561, -105.20524)])
def test_republic_plaza_lands_at_its_true_bearing_and_distance(stage):
    """Within 1 % of the geodesic from the stage. It is ~20.6 km at ~65 degrees:
    the "15 miles / 24 km" usually quoted is the drive, not the line of sight."""
    proj = LocalProjection(*ORIGIN)
    s = proj.to_local(*stage)[:2]
    p = proj.to_local(*REPUBLIC_PLAZA)[:2]
    bearing, dist = denver.bearing_distance(s, p)
    true_bearing, true_dist = vincenty(*stage, *REPUBLIC_PLAZA)
    assert dist == pytest.approx(true_dist, rel=0.01)
    assert bearing == pytest.approx(true_bearing, rel=0.01)
    assert dist == pytest.approx(true_dist, abs=5.0)       # in fact to metres
    assert bearing == pytest.approx(true_bearing, abs=0.02)
    assert 20300 < dist < 20800 and 64.0 < bearing < 66.0  # ENE, over the stage


def test_curvature_drop():
    assert denver.curvature_drop_m(25000.0) == pytest.approx(42.67, abs=0.05)  # 42-43 m at 25 km
    assert denver.curvature_drop_m(24000.0) == pytest.approx(39.33, abs=0.05)
    assert denver.curvature_drop_m(20600.0) == pytest.approx(28.97, abs=0.05)  # downtown
    assert denver.curvature_drop_m(0.0) == 0.0
    np.testing.assert_allclose(denver.curvature_drop_m(np.array([1000.0, 2000.0])),
                               [0.0683, 0.2731], atol=1e-3)
    # Without refraction it is the plain sagitta d^2 / 2R.
    assert denver.curvature_drop_m(25000.0, k=0.0) == pytest.approx(25000.0 ** 2 / (2 * 6371000.0))


def test_skirt_and_denver_drop_alike():
    env = _load_unreal("setup_environment")
    for d in (2000.0, 20600.0, 120000.0):
        assert env.curvature_drop_cm(d * 100.0) == pytest.approx(denver.curvature_drop_m(d) * 100.0)


# ---- tags -------------------------------------------------------------------------------

@pytest.mark.parametrize("value,metres", [
    ("218", 218.0), ("218 m", 218.0), ("218m", 218.0), ("21,5", 21.5), ("218.5", 218.5),
    ("715 ft", 715 * 0.3048), ("715'", 715 * 0.3048), ("10'6\"", 10.5 * 0.3048), ("218;220", 218.0),
    ("", None), (None, None), ("tall", None), ("0", None),
])
def test_parse_length(value, metres):
    got = denver.parse_length_m(value)
    assert got == (None if metres is None else pytest.approx(metres))


def test_heights_from_tags():
    assert denver.building_height_m({"height": "218", "building:levels": "56"}) == 218.0
    assert denver.building_height_m({"building:levels": "10"}) == 35.0   # 3.5 m a level
    assert denver.building_height_m({"building": "yes"}) == 10.0          # the fallback
    assert denver.building_min_height_m({"min_height": "150"}) == 150.0
    assert denver.building_min_height_m({"building:min_level": "4"}) == 14.0
    assert denver.building_min_height_m({}) == 0.0


# ---- prisms -----------------------------------------------------------------------------

def _closed_and_outward(verts, tris):
    edges = Counter()
    for t in tris:
        for a, b in zip(t, t[1:] + t[:1]):
            edges[(a, b)] += 1
    closed = all(edges[(b, a)] == 1 for (a, b) in edges) and all(c == 1 for c in edges.values())
    vol = sum(np.dot(verts[a], np.cross(verts[b], verts[c])) for a, b, c in tris) / 6.0
    return closed, vol


@pytest.mark.parametrize("plan", [
    Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)]),           # L, counter-clockwise
    Polygon([(0, 0), (0, 40), (25, 40), (25, 0)]),                               # clockwise square
    Polygon([(0, 0), (20, -5), (40, 0), (35, 20), (20, 8), (5, 20)]),            # concave
])
def test_prisms_are_closed_and_face_outward(plan):
    def drop(xy):
        return denver.curvature_drop_m(np.hypot(xy[:, 0] + 18000.0, xy[:, 1] + 9000.0))

    verts, tris, uvs, tuvs = denver.prism(plan, -150.0, 68.0, drop)
    closed, vol = _closed_and_outward(verts, tris)
    assert closed
    assert vol == pytest.approx(plan.area * 218.0, rel=1e-6)
    assert len(tris) == 4 * (len(plan.exterior.coords) - 1) - 4
    assert len(tris) == len(tuvs) and all(0 <= i < len(uvs) for t in tuvs for i in t)
    # The walls stay vertical and every corner keeps the full height.
    n = len(verts) // 2
    np.testing.assert_allclose(verts[n:, :2], verts[:n, :2])
    np.testing.assert_allclose(verts[n:, 2] - verts[:n, 2], 218.0)


def test_prism_rejects_a_flat_one():
    with pytest.raises(ValueError):
        denver.prism(Polygon([(0, 0), (1, 0), (1, 1)]), 5.0, 5.0)


# ---- buildings from a small Overpass answer ------------------------------------------------

def _square(lat, lon, half_m, first_id):
    """Node lines and ids of a closed square ``half_m`` metres about (lat, lon)."""
    dlat, dlon = half_m / 111000.0, half_m / 85500.0
    corners = [(lat - dlat, lon - dlon), (lat - dlat, lon + dlon), (lat + dlat, lon + dlon), (lat + dlat, lon - dlon)]
    ids = list(range(first_id, first_id + 4))
    nodes = "\n".join(f'  <node id="{i}" lat="{a:.7f}" lon="{b:.7f}"/>' for i, (a, b) in zip(ids, corners))
    return nodes, ids + ids[:1]


def _way(way_id, refs, tags):
    nds = "".join(f'<nd ref="{r}"/>' for r in refs)
    tag = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
    return f'  <way id="{way_id}">{nds}{tag}</way>'


def fixture_xml() -> bytes:
    parts = []
    n1, r1 = _square(*REPUBLIC_PLAZA, 22.0, 100)            # Republic Plaza, 218 m
    n2, r2 = _square(39.7449, -104.9873, 25.0, 200)         # outline 100 m, with parts
    n3, r3 = _square(39.7449, -104.9873, 10.0, 300)         # part: the tower, 0-130 m
    n4, r4 = _square(39.7449, -104.9873, 2.0, 400)          # part: a 25 m spire on it
    n5, r5 = _square(39.7480, -104.9950, 15.0, 500)         # levels only: 10 x 3.5 m
    n6, r6 = _square(39.7470, -104.9930, 12.0, 600)         # 9 m: too short
    n7, r7 = _square(39.7465, -104.9910, 12.0, 700)         # no height: 10 m, too short
    n8, r8 = _square(39.7000, -104.9000, 20.0, 800)         # tall but outside DOWNTOWN
    n9, r9 = _square(39.7520, -105.0000, 20.0, 900)         # a relation's outer, in two ways
    n10, r10 = _square(39.7400, -104.9800, 20.0, 1000)      # a 70 m outline whose parts stop at 20 m
    n11, r11 = _square(39.7400, -104.9800, 15.0, 1100)
    parts += [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11]
    ways = [
        _way(1, r1, {"building": "office", "height": "218", "name": "Republic Plaza Building"}),
        _way(2, r2, {"building": "yes", "height": "100", "name": "Parted Tower"}),
        _way(3, r3, {"building:part": "yes", "height": "130"}),
        _way(4, r4, {"building:part": "yes", "height": "155", "min_height": "130"}),
        _way(5, r5, {"building": "apartments", "building:levels": "10"}),
        _way(6, r6, {"building": "retail", "height": "9"}),
        _way(7, r7, {"building": "yes"}),
        _way(8, r8, {"building": "yes", "height": "80"}),
        _way(9, r9[:3], {}),
        _way(10, r9[2:], {}),
        _way(12, r10, {"building": "hotel", "height": "70", "name": "Podium Tower"}),
        _way(13, r11, {"building:part": "yes", "height": "20"}),
    ]
    rel = ('  <relation id="11"><member type="way" ref="9" role="outer"/>'
           '<member type="way" ref="10" role="outer"/>'
           '<tag k="type" v="multipolygon"/><tag k="building" v="yes"/><tag k="height" v="30"/></relation>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="Overpass API">\n'
            + "\n".join(parts + ways + [rel]) + "\n</osm>\n").encode()


def flat_ground(elevation=1600.0):
    return Dem(data=np.full((8, 8), elevation, dtype=np.float32), bbox=denver.DOWNTOWN)


def test_parse_osm_reads_relations():
    osm = parse_osm(fixture_xml())
    assert len(osm.relations) == 1
    rel = osm.relations[0]
    assert rel.tags["building"] == "yes" and [m[1] for m in rel.members] == [9, 10]
    assert all(m[0] == "way" and m[2] == "outer" for m in rel.members)


def test_buildings_heights_parts_and_relations():
    proj = LocalProjection(*ORIGIN)
    blocks, buildings = denver.building_blocks(parse_osm(fixture_xml()), proj, flat_ground())
    # Republic Plaza, the parted tower (two parts), the levels building, the relation
    # and the podium tower.
    assert buildings == 5
    by_name = {}
    for b in blocks:
        by_name.setdefault(b.name or b.osm, []).append(b)
    rp = by_name["Republic Plaza Building"][0]
    assert rp.top_m - rp.ground_m == 218.0  # exactly its tag
    assert rp.bottom_m == rp.ground_m == pytest.approx(1600.0, abs=1e-3)
    parted = sorted(by_name["Parted Tower"], key=lambda b: b.top_m)
    assert [(b.bottom_m - b.ground_m, b.top_m - b.ground_m) for b in parted] == [(0.0, 130.0), (130.0, 155.0)]
    assert parted[0].ground_m == parted[1].ground_m  # parts stand on their outline's ground
    assert not any(b.osm == "way/2" for b in blocks)  # the outline gives way to its parts
    assert any(b.top_m == pytest.approx(1635.0, abs=1e-3) for b in blocks)  # 10 levels
    rel = [b for b in blocks if b.osm == "relation/11"]
    assert len(rel) == 1 and rel[0].top_m == pytest.approx(1630.0, abs=1e-3) and rel[0].plan.area > 1000.0
    assert not any(b.osm in ("way/6", "way/7", "way/8") for b in blocks)
    # Parts that stop short of their outline's height leave the outline standing.
    podium = sorted(by_name["Podium Tower"], key=lambda b: b.top_m)
    assert [b.osm for b in podium] == ["way/13", "way/12"] and podium[-1].top_m - podium[-1].ground_m == 70.0
    merged = denver.merge_blocks(blocks)
    assert [b.top_m - b.ground_m for b in merged if b.name == "Podium Tower"] == [70.0]  # the podium is inside


def test_merge_keeps_the_tallest_height_and_drops_hidden_and_tiny():
    p = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])
    blocks = [
        denver.Block(p, 1600.0, 1818.0, 1600.0, "Tower", "way/1"),
        denver.Block(Polygon([(30, 0), (60, 0), (60, 30), (30, 30)]), 1600.0, 1817.0, 1600.0, "", "way/2"),
        denver.Block(Polygon([(5, 5), (25, 5), (25, 25), (5, 25)]), 1600.0, 1700.0, 1600.0, "", "way/3"),
        denver.Block(Polygon([(100, 0), (102, 0), (102, 2), (100, 2)]), 1600.0, 1640.0, 1600.0, "", "way/4"),
        denver.Block(Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]), 1818.0, 1840.0, 1600.0, "Tower", "way/5"),
    ]
    merged = denver.merge_blocks(blocks)
    tall = [b for b in merged if b.top_m == 1818.0]
    assert len(tall) == 1 and tall[0].plan.area == pytest.approx(1800.0, rel=0.01)  # the two touching towers
    assert tall[0].name == "Tower"
    assert not any(b.top_m == 1700.0 for b in merged)      # inside the tower
    assert not any(b.top_m == 1640.0 for b in merged)      # 4 m^2 at the ground: invisible
    assert any(b.top_m == 1840.0 for b in merged)          # a 22 m spire on the roof stays


def test_export_writes_a_closed_obj_in_the_terrain_frame(tmp_path):
    proj = LocalProjection(*ORIGIN)
    base = 1742.098
    mesh = denver.export_denver(tmp_path / "Denver.obj", parse_osm(fixture_xml()), flat_ground(), proj, base,
                                origin=(21.9, 12.1))
    text = (tmp_path / "Denver.obj").read_text().splitlines()
    verts = np.array([[float(t) for t in l.split()[1:4]] for l in text if l.startswith("v ")]) / 100.0
    faces = [[int(t.split("/")[0]) - 1 for t in l.split()[1:]] for l in text if l.startswith("f ")]
    assert all(len(l.split()[1].split("/")) == 3 for l in text if l.startswith("f "))  # v/vt/vn
    assert len(faces) == mesh.triangles
    closed, vol = _closed_and_outward(verts, faces)
    assert closed and vol > 0
    # Republic Plaza: ~20.5 km out, at its tagged height, dropped for curvature.
    rp = next(b for b in mesh.blocks if b.name == "Republic Plaza Building")
    cx, cy = rp.plan.centroid.x, rp.plan.centroid.y
    near = np.hypot(verts[:, 0] - cx, verts[:, 1] - cy) < 40.0
    drop = denver.curvature_drop_m(np.hypot(verts[near, 0], verts[near, 1]))  # each vertex its own
    flat = verts[near, 2] + drop
    top, bottom = 1600.0 + 218.0 - base, 1600.0 - base
    assert np.all(np.isclose(flat, top, atol=0.01) | np.isclose(flat, bottom, atol=0.01))
    assert np.isclose(flat, top, atol=0.01).sum() == np.isclose(flat, bottom, atol=0.01).sum()
    assert np.all((28.0 < drop) & (drop < 30.0))
    info = denver.manifest_block(mesh)
    assert info["tallest"]["name"] == "Republic Plaza Building" and info["tallest"]["height_m"] == 218.0
    assert info["tallest"]["distance_km"] == pytest.approx(20.55, abs=0.1)
    assert 64.0 < info["tallest"]["bearing_deg"] < 66.0


def test_visible_height_over_a_ridge():
    d = np.array([100.0, 1000.0, 5000.0, 30000.0])
    eye = 1966.0
    # Ground well under the sight lines; the 1.9 km hill at 30 km is behind the tower.
    h = np.array([1900.0, 1900.0, 1700.0, 1900.0])
    assert denver.visible_height_m(eye, d, h, 20000.0, 1570.0, 1790.0) == pytest.approx(220.0)
    # A ridge 1 km out that the line to 1700 m at 20 km just grazes: 90 m of the tower show.
    h2 = np.array([1900.0, eye + (1700.0 - eye) / 20.0, 1700.0, 1900.0])
    assert denver.visible_height_m(eye, d, h2, 20000.0, 1570.0, 1790.0) == pytest.approx(90.0)
    # A ridge 8 m under the eye 1 km out hides everything under 1806 m at 20 km.
    h3 = np.array([1900.0, 1958.0, 1700.0, 1900.0])
    assert denver.visible_height_m(eye, d, h3, 20000.0, 1570.0, 1790.0) == 0.0


# ---- the build step, offline ------------------------------------------------------------------

def _seed_cache(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    denver.overpass_cache_path(cache).write_bytes(fixture_xml())
    path = dem_mod.dem_cache_path(denver.DOWNTOWN, denver.GROUND_DEM_SIZE, cache)
    b = denver.DOWNTOWN
    np.savez_compressed(path, data=np.full((40, 60), 1600.0, dtype=np.float32),
                        bbox=np.array([b.west, b.south, b.east, b.north]))


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the build must not touch the network here")

    monkeypatch.setattr(osm_mod.requests, "post", refuse)
    monkeypatch.setattr(dem_mod.requests, "get", refuse)


@pytest.fixture
def offline_build(monkeypatch):
    def fake_dem(bbox, size=2017):
        data = np.linspace(1900.0, 1950.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        return Dem(data=data, bbox=bbox)

    monkeypatch.setattr(cli, "fetch_dem", fake_dem)
    monkeypatch.setattr(cli, "fetch_osm", lambda bbox: osm_mod.OsmData(nodes={}, ways=[]))
    monkeypatch.setattr(cli, "build_monoliths", lambda *a, **k: [])


def _build(out, *extra):
    return cli.main(["build", "--out", str(out), "--landscape-size", "33", "--terrain-grid", "9", *extra])


def test_build_writes_denver_from_the_cache_offline(offline_build, no_network, tmp_path):
    cache = tmp_path / "cache"
    _seed_cache(cache)
    assert _build(tmp_path / "out", "--denver", "--denver-cache", str(cache)) == 0
    out = tmp_path / "out"
    assert (out / "Denver.obj").is_file()
    d = json.loads((out / "manifest.json").read_text())["denver"]
    assert d["obj"] == "Denver.obj" and d["buildings"] == 5 and d["triangles"] > 0
    assert d["tallest"]["name"] == "Republic Plaza Building" and d["tallest"]["height_m"] == 218.0
    assert 64.0 < d["tallest"]["bearing_deg"] < 66.0 and 20.3 < d["tallest"]["distance_km"] < 20.8
    assert d["curvature"]["refraction_k"] == 0.13 and d["curvature"]["radius_km"] == 6371.0
    assert d["base_elevation_m"] == pytest.approx(1900.0, abs=1.0)


def test_build_without_denver_leaves_no_stale_obj(offline_build, no_network, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "Denver.obj").write_text("stale")
    assert _build(out, "--no-denver") == 0
    assert not (out / "Denver.obj").exists()
    assert "denver" not in json.loads((out / "manifest.json").read_text())


def test_build_goes_on_when_downtown_cannot_be_fetched(offline_build, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(osm_mod, "OVERPASS_ATTEMPTS", 1)

    def down(*args, **kwargs):
        raise osm_mod.requests.ConnectionError("offline")

    monkeypatch.setattr(osm_mod.requests, "post", down)
    assert _build(tmp_path / "out", "--denver", "--denver-cache", str(tmp_path / "empty")) == 0
    assert "skipping Denver" in capsys.readouterr().out
    assert "denver" not in json.loads((tmp_path / "out" / "manifest.json").read_text())


def test_denver_is_on_by_default_outside_the_tests():
    assert cli.DEFAULT_DENVER is False  # conftest keeps test builds hermetic
    src = (ROOT / "src" / "redrocks" / "cli.py").read_text()
    assert "DEFAULT_DENVER = True" in src


# ---- fetching: the Overpass cache -------------------------------------------

def test_overpass_answer_is_cached_and_read_back(monkeypatch, tmp_path):
    calls = []

    class Answer:
        content = fixture_xml()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(osm_mod.requests, "post", lambda *a, **k: calls.append(k) or Answer())
    cache = tmp_path / "q.osm"
    first = osm_mod.fetch_overpass("[out:xml];way;out;", cache=cache)
    second = osm_mod.fetch_overpass("[out:xml];way;out;", cache=cache)
    assert len(calls) == 1 and cache.read_bytes() == fixture_xml()
    assert len(first.ways) == len(second.ways) == 12


def test_overpass_runtime_errors_are_not_cached(monkeypatch, tmp_path):
    class Failed:
        content = (b'<?xml version="1.0"?><osm><remark> runtime error: Query timed out in "query" '
                   b'at line 3 after 181 seconds. </remark></osm>')

        def raise_for_status(self):
            pass

    monkeypatch.setattr(osm_mod, "OVERPASS_ATTEMPTS", 1)
    monkeypatch.setattr(osm_mod.requests, "post", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError, match="timed out"):
        osm_mod.fetch_overpass("[out:xml];way;out;", cache=tmp_path / "q.osm")
    assert not (tmp_path / "q.osm").exists()


# ---- Unreal side -------------------------------------------------------------------------------

def _load_unreal(name):
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location(name + "_denver_test", ROOT / "unreal" / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


def test_gameplay_imports_denver_far_away_and_untouchable():
    gp = _load_unreal("setup_gameplay")
    assert "Denver" in gp.NEW_MESHES and "Denver" in gp.NO_COLLIDE and "Denver" not in gp.COLLIDE
    assert gp.FAR_AWAY == ["Denver"]
    assert gp.MESH_MATERIAL["Denver"] == "M_DistantCity"
    colour, roughness = gp.PLAIN["M_DistantCity"]
    assert tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4)) == (170, 170, 165)
    s = gp.FAR_AWAY_SETTINGS
    assert s["cast_shadow"] is False and s["ld_max_draw_distance"] == 0.0
    assert s["never_distance_cull"] is True and s["allow_cull_distance_volume"] is False


def test_verify_walkable_expects_denvers_triangles():
    src = (ROOT / "unreal" / "verify_walkable.py").read_text()
    assert 'manifest["denver"]["triangles"]' in src


# ---- the check tool's skirt evaluator -------------------------------------------------------------

def test_check_tool_reads_the_skirt_mesh_exactly():
    spec = importlib.util.spec_from_file_location("check_denver_test", ROOT / "tools" / "check_denver.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    env = tool.load_env()
    n, size = 9, 3600.0
    verts = []
    for r in range(n):  # export_terrain_obj's order: row 0 north, west to east
        for c in range(n):
            x = (-size / 2 + size * c / (n - 1)) * 100.0
            y = (size / 2 - size * r / (n - 1)) * 100.0
            verts.append((x, y, 3000.0 + 0.02 * x - 0.01 * y))
    border = env.grid_border(verts)
    sverts, _, stris = env.skirt_mesh(border, 1742.0)
    skirt = tool.Skirt(border, sverts, len(env.RING_OUT_CM), 1742.0)
    v = np.asarray(sverts) / 100.0
    rng = np.random.default_rng(3)
    for t in rng.choice(len(stris), 200, replace=False):
        a, b, c = (v[i] for i in stris[t])
        w = rng.dirichlet([1.0, 1.0, 1.0])
        p = w[0] * a + w[1] * b + w[2] * c
        assert skirt(p[0], p[1]) == pytest.approx(p[2] + 1742.0, abs=0.02)
