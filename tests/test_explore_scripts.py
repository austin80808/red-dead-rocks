"""Offline maths in unreal/setup_explore.py and unreal/playtest_explore.py."""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

UNREAL = Path(__file__).resolve().parent.parent / "unreal"


def _load(name):
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


def test_to_cm_subtracts_the_datum_and_lifts():
    se = _load("setup_explore")
    assert se.to_cm(1.0, -2.0, 1800.0, 1750.0) == (100.0, -200.0, 5000.0 + se.LIFT_CM)
    assert se.to_cm(0.0, 0.0, 1750.0, 1750.0, 0.0) == (0.0, 0.0, 0.0)


def test_row_offsets_are_side_by_side_and_centred():
    se = _load("setup_explore")
    offs = se.row_offsets(3, 200.0, 0.0)  # facing east: the row runs north-south
    assert [round(x, 6) for x, _ in offs] == [0.0, 0.0, 0.0]
    assert [round(y, 6) for _, y in offs] == [-200.0, 0.0, 200.0]
    offs = se.row_offsets(2, 100.0, 90.0)  # facing north: the row runs east-west
    assert [round(y, 6) for _, y in offs] == [0.0, 0.0]
    assert sorted(round(x, 6) for x, _ in offs) == [-50.0, 50.0]


def test_steer_toward_turns_the_short_way_and_advances():
    pt = _load("playtest_explore")
    route = [(0.0, 0.0), (10000.0, 0.0), (10000.0, 10000.0)]
    steer, idx = pt.steer_toward(0.0, (0.0, 0.0), route, 0)
    assert idx == 1 and steer == pytest.approx(0.0)
    steer, idx = pt.steer_toward(0.0, (9000.0, 0.0), route, 1)  # near the corner: aim at the next leg
    assert idx == 2 and steer > 0.9  # +Y is a positive yaw turn
    steer, _ = pt.steer_toward(90.0, (0.0, 0.0), [(0.0, 0.0), (10000.0, -100.0)], 0)
    assert steer == -1.0
