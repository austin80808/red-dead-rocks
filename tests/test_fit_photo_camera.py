"""tools/fit_photo_camera.py: fit every board view's camera from its landmarks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from redrocks.photo_camera import Camera, project

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fit_photo_camera", REPO / "tools" / "fit_photo_camera.py")
fpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpc)

SIZE = (1280, 960)
TRUE = Camera(x=12000, y=-3000, z=2500, yaw=-150, pitch=-8, roll=1.5, fov=63)
GUESS = Camera(x=11000, y=-2000, z=2000, yaw=-140, pitch=0, fov=70)
# Eight landmarks 50-400 m in front of TRUE, 0-90 m up (level cm).
POINTS = {
    f"mark_{i}": (
        TRUE.x + d * np.cos(np.radians(TRUE.yaw + a)),
        TRUE.y + d * np.sin(np.radians(TRUE.yaw + a)),
        z,
    )
    for i, (d, a, z) in enumerate(
        [(5000, -20, 0), (9000, 15, 1500), (15000, -5, 3000), (22000, 20, 9000),
         (30000, -18, 4500), (40000, 8, 7000), (12000, 25, 600), (26000, -25, 2000)]
    )
}


def _view(name: str, pixels: dict) -> dict:
    return {
        "name": name,
        "photo": {"id": name, "title": "t", "page": "p", "file_url": "u", "author": "a",
                  "licence": "CC BY-SA 4.0", "size": list(SIZE)},
        "aspect": [4, 3],
        "camera_guess": GUESS.to_dict(),
        "camera": None,
        "fit_error": None,
        "sun": {"azimuth_deg": 130.0, "elevation_deg": 58.0},
        "landmarks": [{"name": n, "photo_xy": [float(x), float(y)]} for n, (x, y) in pixels.items()],
        "samples": [],
        "skyline_ignore": [],
    }


@pytest.fixture()
def board(tmp_path, monkeypatch):
    names = list(POINTS)
    px, _ = project(TRUE, np.array([POINTS[n] for n in names]), SIZE)
    good = dict(zip(names, px.tolist()))
    wrong = dict(good)
    # One landmark clicked on the wrong feature: its pixel belongs to another landmark.
    wrong["mark_3"], wrong["mark_5"] = good["mark_5"], good["mark_3"]
    path = tmp_path / "views.json"
    path.write_text(json.dumps({"about": "test", "views": [_view("synthetic", good), _view("wrong", wrong)]}))
    monkeypatch.setattr(fpc, "load_points", lambda out_dir: POINTS)
    return path


def test_fits_the_good_view_and_refuses_the_wrong_one(board, capsys):
    code = fpc.main(["--views", str(board)])
    out = capsys.readouterr().out
    views = {v["name"]: v for v in json.loads(board.read_text())["views"]}

    got = Camera.from_dict(views["synthetic"]["camera"])
    for a in ("x", "y", "z"):
        assert abs(getattr(got, a) - getattr(TRUE, a)) < 50  # 0.5 m
    for a in ("yaw", "pitch", "roll", "fov"):
        assert abs(getattr(got, a) - getattr(TRUE, a)) < 0.5
    assert views["synthetic"]["fit_error"] < 0.001
    assert views["synthetic"]["fit_error"] == round(views["synthetic"]["fit_error"], 4)

    assert "REFUSED wrong:" in out
    assert views["wrong"]["camera"] is None and views["wrong"]["fit_error"] is None
    assert code == 1


def test_only_fits_the_named_view_and_exits_0(board, capsys):
    code = fpc.main(["--views", str(board), "--only", "synthetic"])
    views = {v["name"]: v for v in json.loads(board.read_text())["views"]}
    assert code == 0
    assert views["synthetic"]["camera"] is not None
    assert views["wrong"]["camera"] is None
    assert "REFUSED" not in capsys.readouterr().out


def test_a_refused_view_keeps_its_old_camera(board, capsys):
    data = json.loads(board.read_text())
    old = {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 4.0, "pitch": 5.0, "roll": 0.0, "fov": 60.0}
    data["views"][1]["camera"], data["views"][1]["fit_error"] = old, 0.01
    board.write_text(json.dumps(data))
    assert fpc.main(["--views", str(board), "--only", "wrong"]) == 1
    view = json.loads(board.read_text())["views"][1]
    assert view["camera"] == old and view["fit_error"] == 0.01
    assert "REFUSED wrong:" in capsys.readouterr().out


def test_a_landmark_the_build_lacks_is_refused_by_name(board, capsys):
    data = json.loads(board.read_text())
    data["views"][0]["landmarks"][0]["name"] = "eiffel_tower"
    board.write_text(json.dumps(data))
    assert fpc.main(["--views", str(board), "--only", "synthetic"]) == 1
    assert "REFUSED synthetic: the build has no landmark eiffel_tower" in capsys.readouterr().out
