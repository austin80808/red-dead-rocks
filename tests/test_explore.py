"""The explorable-realm planner: cart route, sled hill, spots. No network."""

import json
import math

import numpy as np

from redrocks import explore
from redrocks.osm import OsmData, Way
from redrocks.vegetation import sled_run_cells


def _xy(lat, lon):
    # A toy frame: 1 degree = 1000 m both ways.
    return (lon * 1000.0, lat * 1000.0)


def _hill(x, y):
    """A cone 300 m high at the origin, falling 0.25 per metre outward, floor at 1700 m."""
    return 1700.0 + max(0.0, 300.0 - 0.25 * math.hypot(x, y))


def _osm():
    # A road from near the top of the hill (100 m out) down to "town" at 1500 m east,
    # plus a short spur that goes nowhere higher.
    nodes = {}
    road = []
    for i, x in enumerate(np.linspace(100.0, 1500.0, 15)):
        nodes[i + 1] = (0.0, x / 1000.0)  # (lat, lon) -> (x, 0)
        road.append(i + 1)
    nodes[100] = (0.2, 0.1)  # spur end, 200 m north of node 1
    ways = [
        Way(1, road, {"highway": "unclassified", "name": "Park Road"}),
        Way(2, [1, 100], {"highway": "footway"}),
    ]
    return OsmData(nodes=nodes, ways=ways)


def test_cart_route_runs_downhill_to_town():
    osm = _osm()
    route = explore.cart_route(osm, _xy, _hill, goal_xy=(1500.0, 0.0))
    assert route.points[0][:2] == (100.0, 0.0)  # highest road node
    assert route.points[-1][:2] == (1500.0, 0.0)
    assert route.length_m == pytest_approx(1400.0)
    assert route.drop_m == pytest_approx(_hill(100, 0) - _hill(1500, 0))


def test_footways_are_not_cart_roads():
    nodes, adj = explore.road_graph(_osm(), _xy)
    assert 100 not in nodes


def test_shortest_path_unreachable_is_empty():
    adj = {1: [(2, 1.0, None)], 2: [(1, 1.0, None)], 3: []}
    assert explore.shortest_path(adj, 1, 3) == []
    assert explore.shortest_path(adj, 1, 2) == [1, 2]


def test_densify_keeps_ends_and_spacing():
    r = explore.Route([(0.0, 0.0, 10.0), (100.0, 0.0, 0.0)])
    d = explore.densify(r, step_m=20.0)
    assert d.points[0] == (0.0, 0.0, 10.0)
    assert d.points[-1] == (100.0, 0.0, 0.0)
    assert len(d.points) == 6
    assert d.points[1][2] == pytest_approx(8.0)


def test_fall_line_goes_straight_down_the_cone():
    line = explore.fall_line(_hill, 400.0, 0.0, step=10.0, max_len=200.0)
    assert len(line) > 10
    assert all(b[2] < a[2] for a, b in zip(line, line[1:]))
    assert all(abs(p[1]) < 1e-6 for p in line)  # straight out along +X
    assert line[-1][0] > 590.0


def test_sled_hill_finds_a_run_off_the_road():
    osm = _osm()
    hill = explore.sled_hill(osm, _xy, _hill, half_extent=1600.0)
    assert hill is not None
    (x, y, z), yaw, run, drop, line = hill
    assert run >= explore.SLED_MIN_RUN_M
    assert drop / run == pytest_approx(0.25, rel=0.05)
    # Not on the road (which runs along y = 0, x > 0).
    assert abs(y) > explore.SLED_KEEP_OUT_M or x < 0
    assert line[0][:2] == (x, y)


def test_yaw_matches_unreal_axes():
    assert explore.yaw_deg(1, 0) == 0.0  # east
    assert explore.yaw_deg(0, 1) == 90.0  # north


def test_plan_lists_spots_in_order():
    osm = _osm()
    spawn = {"east_m": 0.0, "north_m": 0.0, "ground_elevation_m": 2000.0, "yaw_deg": 90.0}
    monoliths = [{"name": "Creation Rock (north)", "centroid_east": 0.0, "centroid_north": 150.0,
                  "base_elevation_m": 1960.0, "top_elevation_m": 2050.0}]
    plan = explore.plan(osm, _xy, _hill, 1600.0, spawn, None, monoliths, town=(0.0, 1.5))
    keys = [s["key"] for s in plan["spots"]]
    assert keys[:3] == ["stage", "climb", "summit"]
    assert "carts" in keys and "sleds" in keys and "ridge" in keys
    assert keys[-1] == "morrison"
    assert plan["carts"]["destination"] == "Morrison"
    assert plan["carts"]["route"][0][:2] == [100.0, 0.0]
    summit = next(s for s in plan["spots"] if s["key"] == "summit")
    assert summit["place"] == "summit"
    json.dumps(plan)  # manifest-safe


def test_vegetation_clears_the_sled_run(tmp_path):
    manifest = {"explore": {"sleds": {"run": [[0.0, 0.0, 10.0], [100.0, 0.0, 0.0]]}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    cells = sled_run_cells(path)
    # 8 m cells along y = 0 from 0 to 100 m, two cells either side.
    assert (0, 0) in cells and (12, 0) in cells and (6, 2) in cells and (6, -2) in cells
    assert (6, 4) not in cells
    assert sled_run_cells(tmp_path / "missing.json") == set()


def pytest_approx(v, rel=1e-6):
    import pytest

    return pytest.approx(v, rel=rel)


def _creek_osm():
    nodes = {1000 + i: (0.0, x / 1000.0) for i, x in enumerate(np.linspace(-3000.0, 3000.0, 61))}
    return OsmData(nodes=nodes, ways=[Way(9, list(nodes), {"waterway": "stream", "name": "Bear Creek"})])


def _valley(x, y):
    # Falls to the east, with a bump the water must not climb.
    return 2000.0 - 0.05 * x + (5.0 if 100.0 < x < 200.0 else 0.0)


def test_creek_route_is_clipped_downstream_and_never_climbs():
    route = explore.creek_route(_creek_osm(), _xy, _valley, half_extent=1000.0)
    xs = [p[0] for p in route.points]
    assert min(xs) >= -1000.0 + explore.CREEK_EDGE_MARGIN_M - 1e-6
    assert max(xs) <= 1000.0 - explore.CREEK_EDGE_MARGIN_M + 1e-6
    assert xs[0] < xs[-1]  # flows east, downhill
    zs = [p[2] for p in route.points]
    assert all(b <= a + 1e-9 for a, b in zip(zs, zs[1:]))


def test_water_ribbon_is_ccw_and_has_uvs(tmp_path):
    route = explore.Route([(0.0, 0.0, 10.0), (20.0, 0.0, 9.0), (40.0, 0.0, 8.0)])
    tris = explore.export_water_obj(route, 0.0, tmp_path / "Water.obj", width_m=6.0)
    assert tris == 4
    text = (tmp_path / "Water.obj").read_text().splitlines()
    verts = [tuple(map(float, l.split()[1:])) for l in text if l.startswith("v ")]
    faces = [[int(t.split("/")[0]) for t in l.split()[1:]] for l in text if l.startswith("f ")]
    assert len([l for l in text if l.startswith("vt ")]) == len(verts)
    for f in faces:
        (ax, ay, _), (bx, by, _), (cx, cy, _) = (verts[i - 1] for i in f)
        assert (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0  # counter-clockwise from above
    assert all(abs(v[1]) == 300.0 for v in verts)  # 3 m either side, in cm


def test_plan_adds_tubing_before_morrison():
    osm = _osm()
    osm.ways.append(_creek_osm().ways[0])
    osm.nodes.update(_creek_osm().nodes)
    plan = explore.plan(osm, _xy, _hill, 1600.0, None, None, [], town=(0.0, 1.5))
    keys = [s["key"] for s in plan["spots"]]
    assert keys.index("tubing") == keys.index("morrison") - 1
    assert plan["tubes"]["name"] == "Bear Creek" and plan["tubes"]["route_length_m"] > 1000


def test_carve_creek_cuts_a_bed_and_keeps_the_datum():
    from redrocks.dem import BBox, Dem

    data = np.full((101, 101), 1800.0, dtype=np.float32)
    data[0, 0] = 1700.0  # the datum, far from the creek
    dem = Dem(data=data, bbox=BBox(west=0.0, south=0.0, east=0.01, north=0.01))
    route = explore.Route([(-40.0, 0.0, 1795.0), (40.0, 0.0, 1795.0)])
    carved = explore.carve_creek(dem, route, half_x=50.0, half_y=50.0, floor_m=1700.0)
    mid = carved.data[50, 50]
    assert mid == pytest_approx(1795.0 - explore.CREEK_BED_DEPTH_M, rel=1e-6)
    assert carved.data[50, 5] == 1800.0 or carved.data[50, 5] >= carved.data[50, 50]
    assert carved.data[10, 50] == 1800.0  # 40 m off the creek: untouched
    assert carved.data.min() == 1700.0
