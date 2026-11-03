import json

import numpy as np
import pytest

from redrocks import cli
from redrocks.dem import Dem
from redrocks.osm import OsmData


@pytest.fixture
def offline(monkeypatch):
    """Replace the two network fetches with small in-memory data."""

    def fake_dem(bbox, size=2017):
        data = np.linspace(1900.0, 1950.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        return Dem(data=data, bbox=bbox)

    monkeypatch.setattr(cli, "fetch_dem", fake_dem)
    monkeypatch.setattr(cli, "fetch_osm", lambda bbox: OsmData(nodes={}, ways=[]))
    # The monoliths need the real surveyed OSM footprints; this test only cares
    # about the terrain step, so stub them out.
    monkeypatch.setattr(cli, "build_monoliths", lambda *args, **kwargs: [])


def run_build(out, *extra):
    return cli.main(["build", "--out", str(out), "--landscape-size", "33", *extra])


def test_build_writes_terrain_mesh(offline, tmp_path):
    assert run_build(tmp_path, "--terrain-grid", "9") == 0
    obj = tmp_path / "RedRocks_Terrain.obj"
    assert obj.exists()
    text = obj.read_text()
    assert sum(1 for line in text.splitlines() if line.startswith("v ")) == 81
    assert sum(1 for line in text.splitlines() if line.startswith("f ")) == 8 * 8 * 2


def test_manifest_records_terrain_mesh(offline, tmp_path):
    run_build(tmp_path, "--terrain-grid", "9")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    mesh = manifest["terrain_mesh"]
    assert mesh["obj"] == "RedRocks_Terrain.obj"
    assert mesh["grid"] == 9
    assert mesh["triangles"] == 8 * 8 * 2
    assert mesh["width_m"] > 0 and mesh["height_m"] > 0
    assert mesh["min_elevation_m"] == pytest.approx(1900.0)
    assert mesh["max_elevation_m"] == pytest.approx(1950.0)


def test_build_prints_terrain_summary(offline, tmp_path, capsys):
    run_build(tmp_path, "--terrain-grid", "9")
    assert "terrain  9x9" in capsys.readouterr().out


def test_default_terrain_grid_is_377(monkeypatch):
    captured = {}

    def fake_build(args):
        captured["grid"] = args.terrain_grid
        return 0

    monkeypatch.setattr(cli, "build", fake_build)
    assert cli.main(["build"]) == 0
    assert captured["grid"] == 377


def test_terrain_grid_zero_skips_mesh(offline, tmp_path):
    assert run_build(tmp_path, "--terrain-grid", "0") == 0
    assert not (tmp_path / "RedRocks_Terrain.obj").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "terrain_mesh" not in manifest
