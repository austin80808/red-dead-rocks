import json

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from redrocks import cli
from redrocks.dem import Dem, square_bbox
from redrocks.geo import metres_per_degree
from redrocks.osm import parse_osm
from redrocks.park import (
    PARKING_AISLE_WIDTH_M,
    PATH_LIFT_M,
    PATH_WIDTHS_M,
    ROAD_LIFT_M,
    ROAD_WIDTHS_M,
    TerrainSurface,
    export_park,
    export_parking_obj,
    export_paths_obj,
    export_roads_obj,
    is_path,
    is_road,
    ribbon,
    way_width_m,
)
from redrocks.terrain_mesh import export_terrain_obj

LAT, LON = 39.6655, -105.2055
GRID = 41
BBOX = square_bbox(LAT, LON, 1000.0)


def make_dem(slope=0.1, n=64):
    """A plane rising to the east: `slope` metres per metre."""
    cols = np.linspace(0, 1000.0 * slope, n)
    data = np.tile(cols, (n, 1)).astype(np.float32) + 1900.0
    return Dem(data=data, bbox=BBOX)


def at(east_m, north_m):
    """lat/lon of a point given metres from the bbox centre."""
    mlon, mlat = metres_per_degree(LAT)
    return LAT + north_m / mlat, LON + east_m / mlon


def osm_xml(ways):
    """Synthetic OSM data. ways: list of (tags, [(east, north), ...]).

    A "_close" key in tags closes the ring.
    """
    nodes, way_xml, nid = [], [], 1
    for wid, (tags, pts) in enumerate(ways, start=100):
        tags = dict(tags)
        refs = []
        for e, n in pts:
            lat, lon = at(e, n)
            nodes.append(f'<node id="{nid}" lat="{lat:.8f}" lon="{lon:.8f}"/>')
            refs.append(nid)
            nid += 1
        if tags.pop("_close", False):
            refs.append(refs[0])
        nd = "".join(f'<nd ref="{r}"/>' for r in refs)
        tg = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        way_xml.append(f'<way id="{wid}">{nd}{tg}</way>')
    return parse_osm(f"<osm>{''.join(nodes)}{''.join(way_xml)}</osm>".encode())


def load(path):
    verts, uvs, faces = [], [], []
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif tok[0] == "vt":
            uvs.append([float(x) for x in tok[1:3]])
        elif tok[0] == "f":
            faces.append([tuple(int(i) for i in t.split("/")) for t in tok[1:]])
    return np.array(verts), np.array(uvs), faces


def tri_up(verts, faces):
    """z component of every face normal (positive = faces up)."""
    out = []
    for f in faces:
        a, b, c = (verts[i[0] - 1] for i in f)
        out.append(np.cross(b - a, c - a)[2])
    return np.array(out)


@pytest.fixture
def surface():
    return TerrainSurface(make_dem(), GRID)


# --- classification ---------------------------------------------------------


def test_width_by_class():
    for hw, w in {**ROAD_WIDTHS_M, **PATH_WIDTHS_M}.items():
        way = osm_xml([({"highway": hw}, [(0, 0), (10, 0)])]).ways[0]
        assert way_width_m(way) == w


def test_link_uses_parent_width():
    way = osm_xml([({"highway": "primary_link"}, [(0, 0), (10, 0)])]).ways[0]
    assert way_width_m(way) == ROAD_WIDTHS_M["primary"] and is_road(way)


def test_parking_aisle_and_explicit_width():
    aisle = osm_xml([({"highway": "service", "service": "parking_aisle"}, [(0, 0), (9, 0)])]).ways[0]
    assert way_width_m(aisle) == PARKING_AISLE_WIDTH_M
    wide = osm_xml([({"highway": "residential", "width": "7.5"}, [(0, 0), (9, 0)])]).ways[0]
    assert way_width_m(wide) == 7.5
    junk = osm_xml([({"highway": "residential", "width": "wide"}, [(0, 0), (9, 0)])]).ways[0]
    assert way_width_m(junk) == ROAD_WIDTHS_M["residential"]


def test_roads_and_paths_are_disjoint_and_filtered():
    data = osm_xml(
        [
            ({"highway": "residential"}, [(0, 0), (10, 0)]),
            ({"highway": "footway"}, [(0, 5), (10, 5)]),
            ({"highway": "construction"}, [(0, 8), (10, 8)]),
            ({"highway": "service", "tunnel": "yes"}, [(0, 9), (10, 9)]),
            ({"highway": "footway", "area": "yes", "_close": True}, [(0, 0), (5, 0), (5, 5)]),
            ({"building": "yes"}, [(0, 0), (5, 0)]),
        ]
    )
    assert [w.id for w in data.ways if is_road(w)] == [100]
    assert [w.id for w in data.ways if is_path(w)] == [101]


# --- terrain surface --------------------------------------------------------


def test_surface_matches_exported_terrain_mesh(tmp_path):
    dem = make_dem()
    dem.data[20:30, 20:30] += 15.0  # a bump so cells are not planar
    mesh = export_terrain_obj(dem, tmp_path / "t.obj", grid=GRID)
    verts, _, faces = load(tmp_path / "t.obj")
    surface = TerrainSurface(dem, GRID)
    assert surface.base == pytest.approx(mesh.min_elevation_m)
    # Every mesh vertex is reproduced, and triangle centroids lie on the surface.
    h = surface.height(verts[:, 0] / 100.0, verts[:, 1] / 100.0) * 100.0
    assert np.allclose(h, verts[:, 2], atol=0.05)
    for f in faces[::37]:
        c = np.mean([verts[i[0] - 1] for i in f], axis=0)
        assert surface.height(c[0] / 100.0, c[1] / 100.0) * 100.0 == pytest.approx(c[2], abs=0.05)


def test_surface_clamps_outside_bbox(surface):
    assert np.isfinite(surface.height(1e5, -1e5))


# --- ribbons ----------------------------------------------------------------


def test_ribbon_width_and_drape(surface):
    pts = np.array([[-100.0, 0.0], [100.0, 0.0]])
    verts, uvs, tris = ribbon(pts, 6.0, surface, ROAD_LIFT_M)
    right, left = verts[0::2], verts[1::2]
    assert np.allclose(np.hypot(*(left - right)[:, :2].T), 6.0)
    ground = surface.height(verts[:, 0], verts[:, 1])
    assert np.allclose(verts[:, 2], ground + ROAD_LIFT_M)
    assert np.max(np.diff(verts[0::2, 0])) <= 2.0 + 1e-9  # densified
    assert len(tris) == 2 * (len(verts) // 2 - 1)


def test_ribbon_faces_up_and_u_runs_along_length(surface):
    pts = np.array([[0.0, -50.0], [0.0, 0.0], [40.0, 30.0]])  # bends north then north-east
    verts, uvs, tris = ribbon(pts, 4.0, surface, ROAD_LIFT_M)
    flat = np.column_stack([verts[:, :2], np.zeros(len(verts))])
    area_z = [np.cross(flat[b] - flat[a], flat[c] - flat[a])[2] for a, b, c in tris]
    assert min(area_z) > 0  # counter-clockwise from above
    u = uvs[0::2, 0]
    assert np.all(np.diff(u) > 0)
    assert u[-1] == pytest.approx((50.0 + np.hypot(40, 30)) / 4.0)  # U = distance / width
    assert set(np.unique(uvs[:, 1])) == {0.0, 1.0}


def test_ribbon_degenerate_returns_none(surface):
    assert ribbon(np.array([[1.0, 1.0], [1.0, 1.0]]), 4.0, surface, 0.05) is None


# --- OBJ exports ------------------------------------------------------------


def test_roads_obj_has_uvs_and_only_roads(tmp_path, surface):
    data = osm_xml(
        [
            ({"highway": "secondary"}, [(-200, 0), (0, 50), (200, 0)]),
            ({"highway": "footway"}, [(-100, -100), (100, -100)]),
        ]
    )
    mesh = export_roads_obj(data, surface, tmp_path / "Roads.obj")
    assert mesh.features == 1 and mesh.triangles > 0
    verts, uvs, faces = load(tmp_path / "Roads.obj")
    assert len(faces) == mesh.triangles and len(verts) == len(uvs) == mesh.vertices
    assert all(len(f) == 3 and all(len(t) == 2 for t in f) for f in faces)  # v/vt on every face
    assert tri_up(verts, faces).min() > 0
    assert verts[:, 0].max() - verts[:, 0].min() > 39000  # centimetres
    assert verts[:, 1].min() > -5000  # the footway at north=-100 m is not in here
    assert "o Roads" in (tmp_path / "Roads.obj").read_text()


def test_paths_obj_is_lifted_above_roads_layer(tmp_path, surface):
    data = osm_xml([({"highway": "path"}, [(-50, 20), (50, 20)])])
    mesh = export_paths_obj(data, surface, tmp_path / "Paths.obj")
    verts, _, _ = load(tmp_path / "Paths.obj")
    ground = surface.height(verts[:, 0] / 100, verts[:, 1] / 100) * 100
    assert np.allclose(verts[:, 2] - ground, PATH_LIFT_M * 100, atol=0.02)
    assert PATH_LIFT_M > ROAD_LIFT_M
    assert mesh.name == "Paths"


def test_way_crossing_bbox_edge_is_clipped(tmp_path, surface):
    data = osm_xml([({"highway": "residential"}, [(-300, 0), (900, 0)])])  # bbox half-width 500 m
    export_roads_obj(data, surface, tmp_path / "Roads.obj")
    verts, _, _ = load(tmp_path / "Roads.obj")
    assert verts[:, 0].max() <= surface.half_x * 100 + 1.0


def test_tunnel_only_yields_no_file(tmp_path, surface):
    data = osm_xml([({"highway": "service", "tunnel": "yes"}, [(0, 0), (50, 0)])])
    assert export_roads_obj(data, surface, tmp_path / "Roads.obj") is None
    assert not (tmp_path / "Roads.obj").exists()


def test_parking_patch_follows_terrain_and_covers_polygon(tmp_path, surface):
    ring = [(0, 0), (60, 0), (60, 40), (20, 40), (20, 20), (0, 20)]  # L-shaped lot
    data = osm_xml([({"amenity": "parking", "_close": True}, ring)])
    mesh = export_parking_obj(data, surface, tmp_path / "Parking.obj")
    assert mesh.features == 1
    verts, uvs, faces = load(tmp_path / "Parking.obj")
    assert len(uvs) == len(verts) and all(len(t) == 2 for f in faces for t in f)
    assert tri_up(verts, faces).min() > 0
    assert verts[:, 2].max() - verts[:, 2].min() > 400.0  # draped on the 10% slope, not flat

    xy = np.array([surface.to_xy(*at(e, n)) for e, n in ring])
    poly = Polygon(xy)
    area = 0.0
    for f in faces:
        a, b, c = (verts[i[0] - 1][:2] / 100.0 for i in f)
        area += abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    assert area == pytest.approx(poly.area, rel=0.05)
    grown = poly.buffer(0.5)
    for f in faces:
        centroid = np.mean([verts[i[0] - 1][:2] / 100.0 for i in f], axis=0)
        assert grown.contains(Point(centroid))


def test_open_parking_way_is_ignored(tmp_path, surface):
    data = osm_xml([({"amenity": "parking"}, [(0, 0), (30, 0), (30, 30)])])
    assert export_parking_obj(data, surface, tmp_path / "Parking.obj") is None


def test_export_park_shares_terrain_datum(tmp_path):
    dem = make_dem()
    data = osm_xml(
        [
            ({"highway": "residential"}, [(-200, 0), (200, 0)]),
            ({"highway": "footway"}, [(-100, -80), (100, -80)]),
            ({"amenity": "parking", "_close": True}, [(0, 100), (50, 100), (50, 140), (0, 140)]),
        ]
    )
    export_terrain_obj(dem, tmp_path / "RedRocks_Terrain.obj", grid=GRID)
    result = export_park(data, dem, tmp_path, grid=GRID)
    assert set(result) == {"roads", "paths", "parking"}
    for name in ("Roads", "Paths", "Parking"):
        assert (tmp_path / f"{name}.obj").exists()
    # The plane rises 0.1 m/m eastward from Z=0 at the west edge (-500 m), so a
    # road point at east=-200 m is 30 m = 3000 cm up, plus the lift.
    rv, _, _ = load(tmp_path / "Roads.obj")
    west = rv[np.argmin(rv[:, 0])]
    assert west[2] == pytest.approx(3000.0 + ROAD_LIFT_M * 100, abs=30.0)


# --- CLI wiring -------------------------------------------------------------


def test_build_records_park_in_manifest(monkeypatch, tmp_path):
    def fake_dem(bbox, size=2017):
        return Dem(data=make_dem().data, bbox=bbox)

    centre = (cli.RED_ROCKS_LAT, cli.RED_ROCKS_LON)
    fake_osm = osm_xml_at(
        centre,
        [
            ({"highway": "residential"}, [(-100, 0), (100, 0)]),
            ({"amenity": "parking", "_close": True}, [(0, 50), (30, 50), (30, 80), (0, 80)]),
        ],
    )
    monkeypatch.setattr(cli, "fetch_dem", fake_dem)
    monkeypatch.setattr(cli, "fetch_osm", lambda bbox: fake_osm)
    monkeypatch.setattr(cli, "build_monoliths", lambda *a, **k: [])

    assert cli.main(["build", "--out", str(tmp_path), "--landscape-size", "33", "--terrain-grid", "21"]) == 0
    park = json.loads((tmp_path / "manifest.json").read_text())["park"]
    assert set(park) == {"roads", "parking"}  # no paths in the fixture, so no Paths.obj
    assert park["roads"]["obj"] == "Roads.obj" and park["roads"]["triangles"] > 0
    assert (tmp_path / "Roads.obj").exists() and (tmp_path / "Parking.obj").exists()
    assert not (tmp_path / "Paths.obj").exists()


def osm_xml_at(centre, ways):
    """Like osm_xml but placed around the CLI's real centre."""
    mlon, mlat = metres_per_degree(centre[0])
    nodes, way_xml, nid = [], [], 1
    for wid, (tags, pts) in enumerate(ways, start=100):
        tags = dict(tags)
        refs = []
        for e, n in pts:
            nodes.append(f'<node id="{nid}" lat="{centre[0] + n / mlat:.8f}" lon="{centre[1] + e / mlon:.8f}"/>')
            refs.append(nid)
            nid += 1
        if tags.pop("_close", False):
            refs.append(refs[0])
        nd = "".join(f'<nd ref="{r}"/>' for r in refs)
        tg = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        way_xml.append(f'<way id="{wid}">{nd}{tg}</way>')
    return parse_osm(f"<osm>{''.join(nodes)}{''.join(way_xml)}</osm>".encode())
