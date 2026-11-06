"""redrocks.board_metrics: skyline and colour scores against a photo."""

import numpy as np
import pytest
from PIL import Image

from redrocks.board_metrics import (fit_to, region_hex, score_view, skyline,
                                    skyline_error, sky_mask, worst_regions)

BLUE, SAND = (90, 140, 220), (170, 90, 60)


def scene(w=1600, h=1200, shift=0, rock_cols=0, sky_cols=None, flag=None):
    """Blue sky above a ridge at 200 + 50 sin(x/100) (+ shift), sandstone below."""
    x = np.arange(w)
    truth = (200 + 50 * np.sin((x - w / 2) / 100)).astype(int) + shift
    rows = np.arange(h)[:, None]
    img = np.empty((h, w, 3), np.uint8)
    img[:] = SAND
    img[rows < truth[None, :]] = BLUE
    if rock_cols:
        img[:, :rock_cols] = SAND
    if sky_cols is not None:
        img[:, sky_cols:] = SAND
    if flag:
        x0, y0, x1, y1 = flag
        img[y0:y1, x0:x1] = SAND
    return img, truth


def test_skyline_finds_a_drawn_ridge():
    img, truth = scene()
    assert np.abs(skyline(sky_mask(img)) - (truth - 1)).max() <= 1


def test_columns_without_sky_are_left_out_not_zero():
    a, _ = scene(rock_cols=400)
    b, _ = scene(rock_cols=400)
    assert skyline(sky_mask(a))[:400].tolist() == [-1] * 400
    assert skyline_error(Image.fromarray(a), Image.fromarray(b), []) == pytest.approx(0.0, abs=1e-9)


def test_too_little_shared_sky_is_none():
    a, _ = scene(sky_cols=160)  # sky in 10 % of columns
    b, _ = scene()
    assert skyline_error(Image.fromarray(a), Image.fromarray(b), []) is None


def test_sky_must_connect_to_the_top_row():
    img, _ = scene(w=100, h=400)
    img[300:310, 10:20] = BLUE  # blue pond in the rock
    assert not sky_mask(img)[300:310, 10:20].any()
    assert sky_mask(img)[:100].all()


def test_skyline_error_is_a_fraction_of_height_and_ignores_boxes():
    photo, _ = scene()
    game, _ = scene(shift=24)
    assert skyline_error(Image.fromarray(game), Image.fromarray(photo), []) == pytest.approx(0.02, abs=1e-3)
    flagged, _ = scene(shift=24, flag=(600, 100, 700, 400))
    assert skyline_error(Image.fromarray(flagged), Image.fromarray(photo), []) != pytest.approx(0.02, abs=1e-3)
    assert skyline_error(Image.fromarray(flagged), Image.fromarray(photo),
                         [[600, 0, 700, 400]]) == pytest.approx(0.02, abs=1e-3)


def test_a_wrong_aspect_shot_scores_like_the_right_one():
    photo = Image.fromarray(scene(1600, 1200)[0])
    # same vertical field of view, wider: 2133x1200 squeezed into 1600x900
    wide = Image.fromarray(scene(2133, 1200)[0]).resize((1600, 900))
    tall = Image.fromarray(scene(1600, 1200)[0])
    assert fit_to(wide, photo.size).size == photo.size
    assert skyline_error(wide, photo, []) == pytest.approx(skyline_error(tall, photo, []), abs=0.005)


def test_region_kinds_and_worst_regions():
    rgb = np.empty((20, 20, 3), np.uint8)
    rgb[:] = BLUE
    rgb[:3, 7:10] = (255, 255, 255)  # a small white cloud corner, still sky
    rgb[10:, :] = SAND
    assert region_hex(rgb, [0, 0, 10, 10], "sky") == "#5A8CDC"
    assert region_hex(rgb, [0, 10, 10, 20], "tread") == "#AA5A3C"
    green = rgb.copy()
    green[12:14, 0:3] = (60, 110, 50)
    green[10:, 3:] = (150, 150, 150)  # not green
    green[10:12, 0:10] = (120, 90, 60)
    assert region_hex(green, [0, 10, 10, 20], "plants") == "#3C6E32"
    with pytest.raises(ValueError):
        region_hex(rgb, [0, 0, 5, 5], "lava")
    assert worst_regions({"a": {"regions": {"x": {"delta_e": 3.0}, "y": {"delta_e": 22.0}}},
                          "b": {"regions": {"z": {"delta_e": 11.0}}}}) == [("a", "y", 22.0), ("b", "z", 11.0), ("a", "x", 3.0)]


def test_score_view_scores_each_sample_and_the_skyline():
    photo = Image.fromarray(scene(400, 300)[0])
    game = Image.fromarray(scene(400, 300)[0])
    view = {"samples": [{"label": "sky", "kind": "sky", "photo_box": [0, 0, 20, 20], "game_box": [0, 0, 20, 20]},
                        {"label": "rock", "kind": "rock_lit", "photo_box": [0, 250, 20, 290], "game_box": [0, 250, 20, 290]}],
            "skyline_ignore": []}
    s = score_view(view, game, photo)
    assert s["regions"]["sky"] == {"kind": "sky", "photo": "#5A8CDC", "game": "#5A8CDC", "delta_e": 0.0}
    assert s["regions"]["rock"]["delta_e"] == pytest.approx(0.0, abs=1e-6)
    assert s["skyline_error"] == pytest.approx(0.0, abs=1e-9)
