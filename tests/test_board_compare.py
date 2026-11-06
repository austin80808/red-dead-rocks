"""tools/make_board_compare.py and redrocks.board_images: before | after | photo
composites, the board GIF and scores, the guards that stop it before it writes
anything, and the calibration (projection) check."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from redrocks import board_images
from redrocks.photo_camera import Camera, project

REPO = Path(__file__).resolve().parent.parent
SKY, SAND, GREEN = (90, 140, 220), (170, 90, 60), (60, 140, 60)


def _tool(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", REPO / "tools" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _tool("make_board_compare")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test reaches Commons: a photo missing from the cache fails to fetch."""
    def no_network(*a, **k):
        raise requests.ConnectionError("network is off in tests")
    monkeypatch.setattr(board_images.requests, "get", no_network)


def scene(size, ridge=0.5, rock=SAND):
    """Sky above a flat ridge at ``ridge`` of the height, ``rock`` below."""
    w, h = size
    a = np.empty((h, w, 3), np.uint8)
    a[:] = rock
    a[: int(round(h * ridge))] = SKY
    return Image.fromarray(a)


def view(name, aspect, photo_size, **over):
    w, h = photo_size
    v = {"name": name,
         "photo": {"id": name, "title": "Red Rocks %s.jpg" % name,
                   "page": "https://commons.wikimedia.org/wiki/File:%s.jpg" % name,
                   "file_url": "https://upload.wikimedia.org/wikipedia/commons/%s.jpg" % name,
                   "author": "Someone", "licence": "CC BY-SA 4.0"},
         "aspect": list(aspect), "camera": None, "sun": {"azimuth_deg": 130.0, "elevation_deg": 58.0},
         "landmarks": [],
         "samples": [{"label": "sky", "kind": "sky", "photo_box": [0, 0, w, h // 5], "game_box": [0, 0, w, h // 5]},
                     {"label": "rock", "kind": "rock_lit", "photo_box": [0, h * 4 // 5, w, h],
                      "game_box": [0, h * 4 // 5, w, h]}],
         "skyline_ignore": []}
    v.update(over)
    return v


def board(tmp_path, views, photos, shots):
    """views.json, a pre-filled photo cache and the shots folder."""
    sd, cache = tmp_path / "shots", tmp_path / "refs"
    sd.mkdir()
    cache.mkdir()
    for name, img in photos.items():
        img.save(cache / ("%s.jpg" % name), quality=95)
    for name, img in shots.items():
        img.save(sd / name)
    vf = tmp_path / "views.json"
    vf.write_text(json.dumps({"about": "test board", "views": views}))
    return sd, cache, vf


def run(sd, cache, vf, out, *extra):
    return bc.main(["--shots", str(sd), "--pass", "p0", "--views", str(vf), "--cache", str(cache),
                    "--out", str(out), *extra])


def two_views(tmp_path, **shot_over):
    views = [view("postcard", (4, 3), (160, 120)), view("plaza", (3, 2), (150, 100))]
    photos = {"postcard": scene((160, 120)), "plaza": scene((150, 100))}
    shots = {"before_postcard.png": scene((1600, 1200), rock=GREEN),  # green tells the panels apart
             "after_postcard.png": scene((1600, 1200), ridge=0.6, rock=(120, 60, 40)),
             "before_plaza.png": scene((1600, 1067), rock=GREEN),
             "after_plaza.png": scene((1600, 1067), rock=(168, 92, 62))}
    shots.update(shot_over)
    return board(tmp_path, views, photos, {k: v for k, v in shots.items() if v is not None})


def test_composites_scores_and_the_board_gif_for_two_views(tmp_path, capsys):
    sd, cache, vf = two_views(tmp_path)
    out = tmp_path / "p0"
    assert run(sd, cache, vf, out) == 0
    comp = Image.open(out / "postcard.jpg").convert("RGB")
    assert comp.size == (3 * 640, board_images.CAPTION_H + 480)
    low = comp.height - 20
    before, after, photo = (comp.getpixel((x, low)) for x in (320, 960, 1600))
    assert before[1] > 120 and after[0] < 140 and photo[0] > 150  # before | after | photo
    assert Image.open(out / "plaza.jpg").size == (3 * 640, board_images.CAPTION_H + 427)

    gif = Image.open(out / "board.gif")
    assert gif.n_frames == 2 and gif.info["duration"] == 2500 and gif.width == 2 * bc.GIF_PANEL_W
    frame = gif.convert("RGB")
    rock = board_images.CAPTION_H + bc.GIF_PANEL_W * 3 // 4 - 10  # low in the 4:3 panels
    assert frame.getpixel((bc.GIF_PANEL_W // 2, rock))[0] < 140  # the after panel, next to ...
    assert frame.getpixel((bc.GIF_PANEL_W * 3 // 2, rock))[0] > 150  # ... the photo

    scores = json.loads((out / "scores.json").read_text())
    assert set(scores) == {"postcard", "plaza", "worst"}
    assert scores["postcard"]["skyline_error"] == pytest.approx(0.1, abs=0.01)  # ridge 12 px low in 120
    assert scores["postcard"]["skyline_share"] == pytest.approx(1.0)
    assert scores["postcard"]["regions"]["sky"]["delta_e"] < 3.0
    assert scores["plaza"]["regions"]["rock"]["delta_e"] < 5.0
    assert len(scores["worst"]) == 3 and scores["worst"][0][:2] == ["postcard", "rock"]
    printed = capsys.readouterr().out
    assert "skyline 0.100 of the height over 100 % of the columns" in printed
    assert "worst regions" in printed and "postcard" in printed.split("worst regions")[1]


def test_worst_n_lists_that_many_regions(tmp_path, capsys):
    sd, cache, vf = two_views(tmp_path)
    assert run(sd, cache, vf, tmp_path / "p0", "--worst", "1") == 0
    scores = json.loads((tmp_path / "p0" / "scores.json").read_text())
    assert [w[:2] for w in scores["worst"]] == [["postcard", "rock"]]
    after = capsys.readouterr().out.split("worst regions:")[1]
    ranked = [line for line in after.splitlines() if line.strip()[:1].isdigit()]
    assert len(ranked) == 1 and "postcard" in ranked[0] and "rock" in ranked[0]


def test_no_before_puts_the_photo_next_to_the_after_shot(tmp_path):
    sd, cache, vf = board(tmp_path, [view("postcard", (4, 3), (160, 120))], {"postcard": scene((160, 120))},
                          {"after_postcard.png": scene((1600, 1200), rock=GREEN)})
    out = tmp_path / "p0"
    assert run(sd, cache, vf, out, "--no-before") == 0
    comp = Image.open(out / "postcard.jpg").convert("RGB")
    assert comp.size == (2 * 640, board_images.CAPTION_H + 480)
    low = comp.height - 20
    assert comp.getpixel((320, low))[0] > 150 and comp.getpixel((960, low))[1] > 120  # photo | after
    assert (out / "board.gif").exists() and (out / "scores.json").exists()


def test_an_unreachable_photo_stops_before_writing_anything(tmp_path, capsys):
    sd, cache, vf = two_views(tmp_path)
    (cache / "plaza.jpg").unlink()  # not cached, and the network is off
    out = tmp_path / "p0"
    assert run(sd, cache, vf, out) == 1
    assert "could not fetch photo for plaza: network is off in tests" in capsys.readouterr().err
    assert not out.exists()


def test_a_corrupt_cached_photo_is_a_fetch_failure_too(tmp_path, capsys):
    sd, cache, vf = two_views(tmp_path)
    (cache / "postcard.jpg").write_bytes(b"<html>rate limited</html>")
    assert run(sd, cache, vf, tmp_path / "p0") == 1
    assert "could not fetch photo for postcard:" in capsys.readouterr().err
    assert not (tmp_path / "p0").exists()


def test_a_black_shot_is_refused_by_name_before_writing_anything(tmp_path, capsys):
    sd, cache, vf = two_views(tmp_path, **{"after_plaza.png": Image.new("RGB", (1600, 1067), (4, 4, 4))})
    out = tmp_path / "p0"
    assert run(sd, cache, vf, out) == 1
    assert "%s is black: Lumen or streaming not settled" % (sd / "after_plaza.png") in capsys.readouterr().err
    assert not out.exists()


def test_a_missing_shot_stops_before_writing_anything(tmp_path, capsys):
    sd, cache, vf = two_views(tmp_path, **{"after_plaza.png": None})
    out = tmp_path / "p0"
    assert run(sd, cache, vf, out) == 1
    assert "missing shot %s" % (sd / "after_plaza.png") in capsys.readouterr().err
    assert not out.exists()


def test_mean_luminance_is_rec709_on_0_to_255():
    assert board_images.mean_luminance(Image.new("RGB", (8, 8), (10, 10, 10))) == pytest.approx(10.0)
    assert board_images.mean_luminance(Image.new("RGB", (8, 8), (0, 0, 255))) == pytest.approx(255 * 0.0722)


def _dots(a, centres, radius, color=(255, 0, 255)):
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in centres:
        a[(xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= radius ** 2] = color
    return a


def test_marker_pixels_finds_three_magenta_dots_within_a_pixel():
    a = np.asarray(scene((400, 300))).copy()
    centres = [(50.3, 40.7), (210.0, 150.5), (333.6, 260.2)]
    _dots(a, centres, 5.0)
    _dots(a, [(100.0, 100.0)], 3.0, (200, 40, 200))  # dull magenta: r and b must be over 200
    _dots(a, [(300.0, 20.0)], 3.0, (230, 80, 230))   # pink: too much green
    found = board_images.marker_pixels(Image.fromarray(a))
    assert found.shape == (3, 2)
    for (cx, cy), (fx, fy) in zip(sorted(centres), found):  # sorted left to right
        assert abs(fx - cx) <= 1.0 and abs(fy - cy) <= 1.0
    assert board_images.marker_pixels(scene((40, 30))).shape == (0, 2)


CAM = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "fov": 90.0}
MARKS = {"stage_rock_summit": (10000.0, -3000.0, 1500.0), "booth_roof": (8000.0, 2500.0, -800.0),
         "walkway_centre": (12000.0, 500.0, 300.0), "aisle_end_row70": (9000.0, -500.0, -2000.0)}


def calib_board(tmp_path, shift, drop=0):
    """A calibration shot with a magenta dot where project() puts each landmark,
    every dot moved right by ``shift`` of the width."""
    v = view("postcard", (4, 3), (160, 120), camera=CAM,
             landmarks=[{"name": n, "photo_xy": [0, 0]} for n in MARKS])
    size = (1600, 1200)
    px, front = project(Camera.from_dict(CAM), np.array(list(MARKS.values())), size)
    assert front.all() and (px[:, 0] > 0).all() and (px[:, 0] < size[0]).all()
    dots = [(x + shift * size[0], y) for x, y in px[drop:]]
    shot = Image.fromarray(_dots(np.asarray(scene(size)).copy(), dots, 4.0))
    sd, cache, vf = board(tmp_path, [v], {}, {"calib_postcard.png": shot})
    (sd / "landmarks_cm.json").write_text(json.dumps({"landmarks": {k: list(p) for k, p in MARKS.items()}}))
    return sd, vf


@pytest.mark.parametrize("shift,code", [(0.02, 1), (0.002, 0)])
def test_the_calibration_check_catches_a_projection_mismatch(tmp_path, capsys, shift, code):
    sd, vf = calib_board(tmp_path, shift)
    assert bc.main(["--shots", str(sd), "--pass", "p0", "--views", str(vf), "--calib"]) == code
    out = capsys.readouterr().out
    assert ("projection convention mismatch" in out) == bool(code)
    assert "4 of 4 markers" in out


def test_calibration_with_too_few_markers_fails_by_name(tmp_path, capsys):
    sd, vf = calib_board(tmp_path, 0.0, drop=3)
    assert bc.main(["--shots", str(sd), "--views", str(vf), "--calib"]) == 1
    assert "postcard" in (out := capsys.readouterr().out) and "found 1 of 4 markers" in out


def test_calibration_needs_the_marker_positions_the_shoot_used(tmp_path, capsys):
    sd, vf = calib_board(tmp_path, 0.0)
    (sd / "landmarks_cm.json").unlink()
    assert bc.main(["--shots", str(sd), "--views", str(vf), "--calib"]) == 1
    assert "--write-landmarks" in capsys.readouterr().err


def test_write_landmarks_puts_the_catalog_in_level_cm(tmp_path, monkeypatch, capsys):
    from redrocks import landmarks
    seen = {}

    def fake_catalog(out_dir, scan_top):
        seen["args"] = (Path(out_dir), Path(scan_top))
        return {"booth_roof": (7.5, 2.0, 1943.098)}

    monkeypatch.setattr(landmarks, "catalog", fake_catalog)
    sd = tmp_path / "board"
    assert bc.main(["--write-landmarks", "--shots", str(sd), "--build", str(tmp_path / "out"),
                    "--scan-top", str(tmp_path / "top.npz")]) == 0
    data = json.loads((sd / "landmarks_cm.json").read_text())
    assert data["landmarks"] == {"booth_roof": [750.0, -200.0, 20100.0]}  # level frame: y = -north
    assert seen["args"] == (tmp_path / "out", tmp_path / "top.npz")
    assert "landmarks_cm.json" in capsys.readouterr().out


def test_the_vegetation_tool_uses_the_shared_helpers():
    rc = _tool("make_reference_compare")
    assert rc.fetch_photo is board_images.fetch_photo and rc.compose is board_images.compose
    assert rc._cover is board_images.cover and rc._photo_caption is board_images.photo_caption


def test_a_long_title_is_shortened_but_the_credit_never_is():
    photo = {"title": "Aerial view of Red Rocks Amphitheatre, January 2013.jpg", "author": "Stoermerjp",
             "licence": "CC BY-SA 3.0"}
    assert board_images.photo_caption(photo) == (
        "Aerial view of Red Rocks Amphitheatre, January 2013.jpg — Stoermerjp, CC BY-SA 3.0")
    short = board_images.photo_caption(photo, max_px=300)
    assert short.endswith("… — Stoermerjp, CC BY-SA 3.0") and board_images.text_width(short) <= 300
    assert short.startswith("Aerial")
