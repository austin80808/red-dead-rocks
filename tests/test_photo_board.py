"""The photo board's views file (src/redrocks/photo_board.py, docs/media/photo-board/views.json)."""

from copy import deepcopy
from pathlib import Path

import pytest

from redrocks.landmarks import NAMES
from redrocks.photo_board import load_board, validate_board

ROOT = Path(__file__).resolve().parent.parent
BOARD = ROOT / "docs" / "media" / "photo-board" / "views.json"
VIEW_NAMES = ["postcard", "up_the_rows", "across_the_rows", "stage_front", "monolith_plaza", "aerial"]


@pytest.fixture()
def views():
    return load_board(BOARD)


def test_the_committed_board_is_valid(views):
    assert [v["name"] for v in views] == VIEW_NAMES
    assert validate_board(views, set(NAMES)) == []


def test_validation_names_the_view_and_the_problem(views):
    bad = deepcopy(views)
    bad[0]["landmarks"][0]["name"] = "eiffel_tower"
    del bad[1]["photo"]["licence"]
    problems = validate_board(bad, set(NAMES))
    assert "postcard: unknown landmark eiffel_tower" in problems
    assert "up_the_rows: photo has no licence" in problems


def test_validation_catches_thin_views_and_a_wrong_aspect(views):
    bad = deepcopy(views)
    bad[0]["landmarks"] = bad[0]["landmarks"][:5]
    bad[3]["samples"][0]["kind"] = "lava"
    bad[4]["aspect"] = [16, 9]
    del bad[5]["sun"]["elevation_deg"]
    problems = validate_board(bad, set(NAMES))
    assert "postcard: has 5 landmarks, needs at least 6" in problems
    assert any(p.startswith("stage_front: sample") and "unknown kind lava" in p for p in problems)
    assert any(p.startswith("monolith_plaza: aspect 16:9") for p in problems)
    assert "aerial: sun has no elevation_deg" in problems


def test_a_view_with_a_fit_note_still_needs_a_landmark(views):
    noted = [v for v in views if v.get("fit_note")]
    for view in noted:
        assert view["fit_note"].strip() and view["landmarks"]
    bad = deepcopy(views)
    bad[0]["landmarks"] = bad[0]["landmarks"][:2]
    bad[0]["fit_note"] = "only two of the build's points show"
    assert validate_board(bad, set(NAMES)) == []
    bad[0]["landmarks"] = []
    assert "postcard: has a fit_note but no landmarks to check its camera against" in validate_board(bad, set(NAMES))


def test_every_pixel_is_inside_its_photo(views):
    for view in views:
        w, h = view["photo"]["size"]
        for mark in view["landmarks"]:
            x, y = mark["photo_xy"]
            assert 0 <= x <= w and 0 <= y <= h, (view["name"], mark["name"])
        bad = deepcopy(view)
        bad["landmarks"][0]["photo_xy"] = [w + 50, 10]
        assert f"{view['name']}: landmark {bad['landmarks'][0]['name']} is outside the photo" in (
            validate_board([bad], set(NAMES))
        )


def test_every_view_credits_a_freely_licensed_commons_photo(views):
    for view in views:
        photo = view["photo"]
        assert photo["page"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert photo["licence"].startswith("CC BY")
        assert photo["id"] == view["name"]  # cached as out/refs/<name>.jpg
