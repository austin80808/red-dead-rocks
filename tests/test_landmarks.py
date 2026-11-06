"""Named landmarks from the build (src/redrocks/landmarks.py), against the real build."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from redrocks import landmarks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SCAN_TOP = ROOT / "assets" / "scan" / "scan_top.npz"

pytestmark = pytest.mark.skipif(
    not (OUT / "manifest.json").exists() or not SCAN_TOP.exists(),
    reason="needs a build in out/ (out/manifest.json) and assets/scan/scan_top.npz",
)

spec = importlib.util.spec_from_file_location("level_frame", ROOT / "unreal" / "level_frame.py")
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)

ROWS = (8, 20, 35, 50, 70)
NAMES = (
    ["creation_rock_summit", "ship_rock_summit", "stage_rock_summit", "booth_roof",
     "walkway_centre", "stage_front_centre", "booth_front_centre"]
    + [f"aisle_{s}_row{n}" for n in ROWS for s in ("start", "end")]
)


@pytest.fixture(scope="module")
def cat():
    return landmarks.catalog(OUT, SCAN_TOP)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((OUT / "manifest.json").read_text())


def test_every_name_present(cat):
    assert set(NAMES) <= set(cat)
    assert set(cat) == set(landmarks.NAMES)


def test_every_row_has_aisle_ends_on_its_tread(cat, manifest):
    treads = manifest["benches"]["tread_levels_m"]
    for n in landmarks.ROWS:
        for side in ("start", "end"):
            assert abs(cat[f"aisle_{side}_row{n}"][2] - treads[n - 1]) < 0.6, (side, n)
        assert cat[f"aisle_start_row{n}"][1] > cat[f"aisle_end_row{n}"][1], n


def test_stage_roof_and_deck_corners(cat, manifest):
    stage = manifest["stage"]
    dims = {d["name"]: d["value_m"] for d in stage["dimensions"]}
    deck = stage["deck_top_elevation_m"]
    axis = np.array(stage["axis_east_north"])
    front = np.array(cat["stage_front_centre"][:2])
    roof_top = deck + dims["roof_grid_above_deck"] + dims["roof_thickness"]
    for end in ("front", "back"):
        n, s = (np.array(cat[f"stage_roof_{end}_{side}"]) for side in ("north", "south"))
        assert n[2] == pytest.approx(roof_top, abs=0.05) and s[2] == pytest.approx(roof_top, abs=0.05)
        assert n[1] > s[1]
        # The front corners are the 45° chamfers' inner points (roof_chamfer cut off each side).
        width = dims["stage_width"] - (2 * dims["roof_chamfer"] if end == "front" else 0.0)
        assert np.linalg.norm(n[:2] - s[:2]) == pytest.approx(width, abs=0.05)
        setback = ((n[:2] + s[:2]) / 2 - front) @ axis  # along the axis toward the audience
        assert setback == pytest.approx(0.0 if end == "front" else -dims["roof_depth"], abs=0.1)
    for side in ("north", "south"):
        d = np.array(cat[f"stage_deck_front_{side}"])
        r = np.array(cat[f"stage_roof_front_{side}"])
        assert d[2] == pytest.approx(deck, abs=0.01)
        # The roof's front edge is over the deck's lip; its corner landmark is the
        # chamfer's inner point, roof_chamfer in from the deck's corner.
        assert np.linalg.norm(d[:2] - r[:2]) == pytest.approx(dims["roof_chamfer"], abs=0.05)
    mid = (np.array(cat["stage_deck_front_north"][:2]) + np.array(cat["stage_deck_front_south"][:2])) / 2
    assert np.linalg.norm(mid - front) < 0.05


def test_creation_rock_summit_near_manifest(cat, manifest):
    top = next(m for m in manifest["monoliths"] if m["obj"] == "Creation_Rock.obj")
    assert abs(cat["creation_rock_summit"][2] - top["top_elevation_m"]) < 15.0


def test_aisle_end_row70_on_tread(cat, manifest):
    assert abs(cat["aisle_end_row70"][2] - manifest["benches"]["tread_levels_m"][69]) < 0.6


def test_aisle_start_is_north_of_end(cat):
    for n in ROWS:
        assert cat[f"aisle_start_row{n}"][1] > cat[f"aisle_end_row{n}"][1]


def test_booth_front_is_stage_side_of_roof(cat):
    assert cat["booth_front_centre"][2] == cat["booth_roof"][2]
    assert cat["booth_front_centre"][0] > cat["booth_roof"][0]


def test_to_level_cm_matches_level_frame(cat):
    pts = {"a": (10.5, -20.25, 1800.0), "b": (-61.239, -25.122, 1964.058), "c": (0.0, 3.0, 1742.098)}
    got = landmarks.to_level_cm(pts)
    for name, (e, n, z) in pts.items():
        x, y = lf.to_level_cm(e, n)
        assert got[name] == pytest.approx((x, y, (z - 1742.098) * 100.0))
    assert set(landmarks.to_level_cm(cat)) == set(cat)
