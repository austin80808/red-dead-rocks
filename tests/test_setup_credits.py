"""Offline tests for unreal/setup_credits.py: the credit lines come from data."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UNREAL = ROOT / "unreal"
SCAN_MD = (ROOT / "assets" / "scan" / "ATTRIBUTION.md").read_text(encoding="utf-8")
ROCKS_MD = (ROOT / "assets" / "rocks" / "ATTRIBUTION.md").read_text(encoding="utf-8")
MANIFEST = {"attribution": [
    "Elevation: USGS 3DEP, public domain",
    "Map data (c) OpenStreetMap contributors, ODbL 1.0",
]}


def _load(name):
    """Import an unreal/ script with a stub `unreal` that has no editor API."""
    stub = types.ModuleType("unreal")
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = stub
    try:
        spec = importlib.util.spec_from_file_location(name + "_under_test", UNREAL / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


@pytest.fixture(scope="module")
def sc():
    return _load("setup_credits")


def test_scan_credit_from_attribution_md(sc):
    e = sc.parse_scan_attribution(SCAN_MD)
    assert e["who"] == "“Red Rocks, Colorado” by ml0w"
    assert e["licence"] == "CC BY 4.0"
    assert e["link"] == "sketchfab.com/3d-models/red-rocks-colorado-9dec580ed1254c13be687a93bae55ae1"


def test_rocks_credit_from_attribution_md(sc):
    e = sc.parse_rocks_attribution(ROCKS_MD)
    assert (e["who"], e["licence"], e["link"]) == ("Poly Haven", "CC0", "polyhaven.com")


@pytest.mark.parametrize("line,role,who,licence,link", [
    ("Elevation: USGS 3DEP, public domain", "Elevation", "USGS 3DEP", "Public domain", ""),
    ("Map data (c) OpenStreetMap contributors, ODbL 1.0", "Map data",
     "© OpenStreetMap contributors", "ODbL 1.0", "openstreetmap.org/copyright"),
    ("Just some credit", "", "Just some credit", "", ""),
])
def test_manifest_lines(sc, line, role, who, licence, link):
    assert sc.parse_manifest_line(line) == {
        "role": role, "who": who, "licence": licence, "link": link}


def test_entries_cover_every_required_credit(sc):
    entries = sc.credit_entries(MANIFEST, SCAN_MD, ROCKS_MD)
    text = " ".join(" ".join(e.values()) for e in entries)
    for needed in ("ml0w", "Red Rocks, Colorado", "CC BY 4.0",
                   "sketchfab.com/3d-models/red-rocks-colorado-9dec580ed1254c13be687a93bae55ae1",
                   "OpenStreetMap contributors", "ODbL 1.0", "USGS 3DEP", "Public domain",
                   "Poly Haven", "CC0"):
        assert needed in text, needed
    assert [e["role"] for e in entries][0] == "Photogrammetry scan"


def test_real_manifest_if_built(sc):
    path = ROOT.parent.parent.parent / "out" / "manifest.json"  # worktree -> checkout/out
    for candidate in (ROOT / "out" / "manifest.json", path):
        if candidate.exists():
            manifest = json.loads(candidate.read_text())
            entries = sc.credit_entries(manifest, SCAN_MD, ROCKS_MD)
            assert len(entries) == 2 + len(manifest.get("attribution", []))
            return
    pytest.skip("no out/manifest.json (run `redrocks build`)")


def test_overlay_matches_in_world_credits(sc):
    """Drift check: everything setup_gameplay.py puts on the in-world sign
    is on the overlay too (author, licences, links)."""
    gp = _load("setup_gameplay")
    sign = gp.credits_text(MANIFEST)
    overlay = " ".join(" ".join(e.values()) for e in sc.credit_entries(MANIFEST, SCAN_MD, ROCKS_MD))
    for fragment in ("ml0w", "CC BY 4.0", "OpenStreetMap contributors", "ODbL 1.0",
                     "USGS 3DEP", "Poly Haven", "CC0"):
        assert fragment in sign and fragment in overlay, fragment
    assert gp.SCAN_CREDIT.splitlines()[-1].strip() in overlay  # same Sketchfab link


def test_broken_scan_attribution_fails_loudly(sc):
    with pytest.raises(ValueError):
        sc.parse_scan_attribution("# Scan\n\nno fields here")


def test_does_not_run_without_unreal(sc):
    assert sc.log == []
