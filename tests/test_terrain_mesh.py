from collections import Counter

import numpy as np
import pytest

from redrocks.dem import BBox, Dem
from redrocks.terrain_mesh import CM_PER_M, export_terrain_obj, resample_to_grid

BBOX = BBox(west=-105.21, south=39.66, east=-105.20, north=39.67)


def make_dem(n=8, lo=1900.0, hi=1950.0):
    data = np.linspace(lo, hi, n * n, dtype=np.float32).reshape(n, n)
    return Dem(data=data, bbox=BBOX)


def load_obj(path):
    verts, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            # Faces may be "v", "v/vt" or "v/vt/vn"; take the vertex index only.
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(verts), faces


def test_resample_shape():
    dem = make_dem(8)
    assert resample_to_grid(dem, 17).shape == (17, 17)


def test_resample_preserves_range():
    dem = make_dem(8, lo=1900.0, hi=1950.0)
    out = resample_to_grid(dem, 17)
    assert out.min() == pytest.approx(1900.0)
    assert out.max() == pytest.approx(1950.0)


def test_vertex_count(tmp_path):
    mesh = export_terrain_obj(make_dem(8), tmp_path / "t.obj", grid=9)
    verts, faces = load_obj(mesh.path)
    assert len(verts) == 9 * 9
    assert mesh.triangles == 8 * 8 * 2
    assert len(faces) == mesh.triangles


def test_height_is_relative_and_scaled_to_cm(tmp_path):
    dem = make_dem(8, lo=1900.0, hi=1950.0)
    mesh = export_terrain_obj(dem, tmp_path / "t.obj", grid=9)
    verts, _ = load_obj(mesh.path)

    # Z is height above the minimum, in centimetres: 0 .. 5000.
    assert verts[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert verts[:, 2].max() == pytest.approx(50.0 * CM_PER_M, rel=1e-6)


def test_mesh_is_centred_on_origin(tmp_path):
    mesh = export_terrain_obj(make_dem(8), tmp_path / "t.obj", grid=9)
    verts, _ = load_obj(mesh.path)
    assert verts[:, 0].mean() == pytest.approx(0.0, abs=1.0)
    assert verts[:, 1].mean() == pytest.approx(0.0, abs=1.0)


def test_extent_matches_bbox(tmp_path):
    from redrocks.geo import metres_per_degree

    dem = make_dem(8)
    mesh = export_terrain_obj(dem, tmp_path / "t.obj", grid=9)
    verts, _ = load_obj(mesh.path)

    lat, _ = BBOX.centre
    m_lon, m_lat = metres_per_degree(lat)
    expected_w = BBOX.width_deg * m_lon * CM_PER_M
    expected_h = BBOX.height_deg * m_lat * CM_PER_M

    assert verts[:, 0].max() - verts[:, 0].min() == pytest.approx(expected_w, rel=1e-6)
    assert verts[:, 1].max() - verts[:, 1].min() == pytest.approx(expected_h, rel=1e-6)


def test_normals_point_up(tmp_path):
    """Positive signed volume about the base plane means upward-facing normals."""
    dem = make_dem(8, lo=1900.0, hi=1950.0)
    mesh = export_terrain_obj(dem, tmp_path / "t.obj", grid=9)
    verts, faces = load_obj(mesh.path)

    # Every triangle's normal must have a positive Z component.
    for f in faces:
        p0, p1, p2 = verts[f[0]], verts[f[1]], verts[f[2]]
        n = np.cross(p1 - p0, p2 - p0)
        assert n[2] > 0.0


def test_mesh_is_watertight_across_the_surface(tmp_path):
    """Interior edges are shared by exactly two triangles."""
    mesh = export_terrain_obj(make_dem(8), tmp_path / "t.obj", grid=9)
    _, faces = load_obj(mesh.path)

    edges = Counter()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[tuple(sorted((a, b)))] += 1

    interior = [c for c in edges.values() if c == 2]
    boundary = [c for c in edges.values() if c == 1]
    assert len(interior) > len(boundary)
    assert all(c in (1, 2) for c in edges.values())


def test_grid_one_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        export_terrain_obj(make_dem(8), tmp_path / "t.obj", grid=1)


def test_obj_declares_uvs_for_every_vertex(tmp_path):
    """Unreal's OBJ importer collapses meshes that have no UV channel."""
    mesh = export_terrain_obj(make_dem(8), tmp_path / "t.obj", grid=9)
    text = mesh.path.read_text().splitlines()

    n_verts = sum(1 for l in text if l.startswith("v "))
    n_uvs = sum(1 for l in text if l.startswith("vt "))
    assert n_verts > 0
    assert n_uvs == n_verts


def test_every_face_references_a_valid_uv(tmp_path):
    mesh = export_terrain_obj(make_dem(8), tmp_path / "t.obj", grid=9)
    text = mesh.path.read_text().splitlines()

    n_uvs = sum(1 for l in text if l.startswith("vt "))
    for line in text:
        if not line.startswith("f "):
            continue
        for token in line.split()[1:]:
            parts = token.split("/")
            assert len(parts) >= 2, f"face token {token} has no UV index"
            uv_index = int(parts[1])
            assert 1 <= uv_index <= n_uvs
