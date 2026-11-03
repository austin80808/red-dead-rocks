"""Mesh integrity checks: watertightness and outward-facing normals."""

from collections import Counter

import numpy as np
import pytest
from shapely.geometry import Polygon

from redrocks.monoliths import Monolith, monolith_to_obj


def square(size=100.0):
    h = size / 2
    return Polygon([(-h, -h), (h, -h), (h, h), (-h, h)])


def l_shape():
    return Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)])


def load_obj(path):
    text = path.read_text().splitlines()
    verts = np.array([[float(v) for v in l.split()[1:4]] for l in text if l.startswith("v ")])
    faces = [[int(i) - 1 for i in l.split()[1:]] for l in text if l.startswith("f ")]
    return verts, faces


def edge_counts(faces):
    edges = Counter()
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            edges[tuple(sorted((a, b)))] += 1
    return edges


def signed_volume(verts, faces):
    total = 0.0
    for face in faces:
        p0 = verts[face[0]]
        for i in range(1, len(face) - 1):
            total += np.dot(p0, np.cross(verts[face[i]], verts[face[i + 1]])) / 6.0
    return total


@pytest.mark.parametrize("footprint", [square(), l_shape()])
def test_mesh_is_watertight(tmp_path, footprint):
    rock = Monolith("R", footprint, 0.0, 50.0)
    verts, faces = load_obj(monolith_to_obj(rock, tmp_path / "r.obj", segments=6))
    assert all(c == 2 for c in edge_counts(faces).values())


@pytest.mark.parametrize("footprint", [square(), l_shape()])
def test_normals_face_outward(tmp_path, footprint):
    """Positive signed volume means the winding is outward-facing."""
    rock = Monolith("R", footprint, 0.0, 50.0)
    verts, faces = load_obj(monolith_to_obj(rock, tmp_path / "r.obj", segments=6))
    assert signed_volume(verts, faces) > 0.0


@pytest.mark.parametrize("footprint", [square(), l_shape()])
def test_volume_matches_footprint_times_height(tmp_path, footprint):
    """A vertical extrusion has volume exactly area x height, in cm^3."""
    height_m = 50.0
    rock = Monolith("R", footprint, 0.0, height_m)
    verts, faces = load_obj(monolith_to_obj(rock, tmp_path / "r.obj", segments=6))

    expected_m3 = footprint.area * height_m
    actual_m3 = abs(signed_volume(verts, faces)) / 1e6
    assert actual_m3 == pytest.approx(expected_m3, rel=1e-6)


@pytest.mark.parametrize("footprint", [square(), l_shape()])
def test_footprint_spans_match_polygon(tmp_path, footprint):
    rock = Monolith("R", footprint, 0.0, 40.0)
    verts, faces = load_obj(monolith_to_obj(rock, tmp_path / "r.obj", segments=4))

    minx, miny, maxx, maxy = footprint.bounds
    assert (verts[:, 0].max() - verts[:, 0].min()) / 100.0 == pytest.approx(maxx - minx, rel=1e-6)
    assert (verts[:, 1].max() - verts[:, 1].min()) / 100.0 == pytest.approx(maxy - miny, rel=1e-6)
    assert verts[:, 2].max() / 100.0 == pytest.approx(40.0, rel=1e-9)


def test_more_segments_do_not_change_volume(tmp_path):
    rock = Monolith("R", l_shape(), 0.0, 30.0)
    volumes = []
    for segments in (2, 8, 24):
        verts, faces = load_obj(monolith_to_obj(rock, tmp_path / f"r{segments}.obj", segments=segments))
        volumes.append(abs(signed_volume(verts, faces)))
    assert volumes[0] == pytest.approx(volumes[1], rel=1e-9)
    assert volumes[1] == pytest.approx(volumes[2], rel=1e-9)
