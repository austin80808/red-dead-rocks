"""Named landmarks from the build (src/redrocks/landmarks.py), against the real build."""

import importlib.util
import json
from pathlib import Path

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
