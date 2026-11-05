"""tools/make_reference_compare.py: before | after | photo composites and ΔE."""

import importlib.util
import json
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_reference_compare", REPO / "tools" / "make_reference_compare.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def test_views_json_credits_every_photo_and_names_are_unique():
    views = rc.load_views(REPO / "docs/media/vegetation-real/views.json")
    assert [v["name"] for v in views] == ["trail", "ponderosa", "junipers", "hillsides"]
    for v in views:
        assert v["photo"]["page"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert v["photo"]["file_url"].startswith(("https://upload.wikimedia.org/", "https://thumb.wikimedia.org/"))
        assert v["photo"]["licence"] and v["photo"]["author"] and v["photo"]["title"]
        assert set(v["camera"]) == {"x", "y", "z", "pitch", "yaw", "fov"}
        assert len(v["aspect"]) == 2 and min(v["aspect"]) > 0


def test_compose_is_three_panels_with_a_caption_strip():
    out = rc.compose(*(Image.new("RGB", (1600, 900), c) for c in ("red", "green", "blue")),
                     ("a", "b", "c"), (16, 9), panel_w=320)
    assert out.width == 3 * 320 and out.height > 180  # 180 px panels plus the caption strip
    # Panels keep their order: before, after, photo.
    assert out.getpixel((160, out.height - 10))[0] > 200
    assert out.getpixel((480, out.height - 10))[1] > 100
    assert out.getpixel((800, out.height - 10))[2] > 200


def test_compose_crops_a_landscape_shot_to_a_portrait_view():
    out = rc.compose(*(Image.new("RGB", (1600, 900), "white") for _ in range(3)), ("a", "b", "c"), (3, 4),
                     panel_w=300)
    assert out.width == 900 and out.height >= 400


def _view(name="trail"):
    return {"name": name,
            "photo": {"id": "refX", "title": "A photo", "page": "https://commons.wikimedia.org/wiki/File:A.jpg",
                      "file_url": "https://upload.wikimedia.org/a.jpg", "author": "Someone", "licence": "CC BY 2.0"},
            "camera": {"x": 1, "y": 2, "z": 3, "pitch": 0, "yaw": 0, "fov": 70}, "aspect": [16, 9],
            "samples": [{"label": "juniper", "photo_xy": [10, 10], "game_xy": [20, 20], "target": "#505B46"}]}


def test_main_writes_composites_and_delta_e(tmp_path, monkeypatch):
    shots = tmp_path / "shots"
    shots.mkdir()
    Image.new("RGB", (160, 90), (0x50, 0x5B, 0x46)).save(shots / "before_trail.png")
    Image.new("RGB", (160, 90), (0x50, 0x5B, 0x46)).save(shots / "after_trail.png")
    photo = tmp_path / "refX.jpg"
    Image.new("RGB", (160, 90), (0x50, 0x5B, 0x46)).save(photo)
    monkeypatch.setattr(rc, "fetch_photo", lambda p, cache: photo)
    vf = tmp_path / "views.json"
    vf.write_text(json.dumps({"views": [_view()]}))
    out = tmp_path / "out"
    assert rc.main(["--shots", str(shots), "--out", str(out), "--cache", str(tmp_path), "--views", str(vf)]) == 0
    assert (out / "trail.jpg").exists()
    de = json.loads((out / "delta_e.json").read_text())
    assert set(de["trail"]) == {"juniper"}
    assert de["trail"]["juniper"]["delta_e"] < 3.0


def test_main_skips_a_view_with_no_shots(tmp_path, monkeypatch):
    shots = tmp_path / "shots"
    shots.mkdir()
    vf = tmp_path / "views.json"
    vf.write_text(json.dumps({"views": [_view()]}))
    monkeypatch.setattr(rc, "fetch_photo", lambda p, cache: tmp_path / "missing.jpg")
    assert rc.main(["--shots", str(shots), "--out", str(tmp_path / "out"), "--cache", str(tmp_path),
                    "--views", str(vf)]) == 1


def test_fetch_photo_uses_the_cache(tmp_path, monkeypatch):
    cached = tmp_path / "refX.jpg"
    cached.write_bytes(b"jpeg")

    def boom(*a, **k):
        raise AssertionError("should not download a cached photo")
    monkeypatch.setattr(rc.requests, "get", boom)
    assert rc.fetch_photo(_view()["photo"], tmp_path) == cached
