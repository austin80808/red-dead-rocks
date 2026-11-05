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


# The aisle's sharpest sideways jog (the walk's steps 57-63, cm): the stair
# shifts 1.2 m toward the rows, and past step 59 the flight below runs on to a
# 0.45 m side wall and a 1 m drop. The walk fell off the stairs here (2026-11-04).
JOG = [(-939.0, -3307.0), (-984.0, -3350.0), (-1033.0, -3388.0), (-1143.0, -3343.0),
       (-1193.0, -3380.0), (-1286.0, -3356.0), (-1336.0, -3393.0)]


def _off_path(path, x, y):
    """Signed distance (cm) from the path: + to the right of the walking direction."""
    best = None
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        px, py = ax + t * dx, ay + t * dy
        d = math.hypot(x - px, y - py)
        if best is None or d < abs(best):
            side = dx * (y - ay) - dy * (x - ax)  # > 0: left of the leg
            best = -d if side > 0 else d
    return best


def test_follow_path_aims_ahead_along_the_line_and_never_goes_back():
    pt = _load("playtest_explore")
    line = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)]
    s, carrot, i = pt.follow_path(line, (0.0, 0.0), 0.0, lookahead=70.0)
    assert s == pytest.approx(0.0) and carrot == pytest.approx((70.0, 0.0)) and i == 0
    # 30 cm before the corner the carrot is already round it, on the next leg.
    s, carrot, i = pt.follow_path(line, (970.0, 0.0), s, lookahead=70.0)
    assert s == pytest.approx(970.0) and carrot == pytest.approx((1000.0, 40.0))
    # Knocked back behind where it got to: progress holds, it does not turn round.
    s2, carrot2, _ = pt.follow_path(line, (500.0, 0.0), s, lookahead=70.0)
    assert s2 == pytest.approx(970.0) and carrot2 == pytest.approx(carrot)
    # Off the end: the carrot stops at the last point.
    s, carrot, i = pt.follow_path(line, (1000.0, 1000.0), s, lookahead=70.0)
    assert carrot == pytest.approx((1000.0, 1000.0)) and i == 1


def test_follow_path_does_not_jump_to_a_far_leg_that_passes_close():
    pt = _load("playtest_explore")
    # A hairpin: the leg back passes 50 cm from the start, 20 m further on.
    hairpin = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 50.0), (-500.0, 50.0)]
    s, _, i = pt.follow_path(hairpin, (100.0, 30.0), 0.0, lookahead=70.0)
    assert i == 0 and s == pytest.approx(100.0)


def _walk_jog(pt, control_dt, pace):
    """Walk the jog with Unreal-like physics at 60 Hz (ground friction 8 turns the
    velocity, as CharacterMovement's CalcVelocity does) but steering only every
    ``control_dt`` s, as the play-test's Slate tick does: about 3 Hz in the editor
    (logged 2026-11-04). Returns (reached the top, worst cm left, worst cm right)."""
    x, y = JOG[0]
    vx = vy = 0.0
    s, dt, worst_left, worst_right = 0.0, 1.0 / 60.0, 0.0, 0.0
    want = (0.0, 0.0)
    steer_in = 0.0
    for _ in range(60 * 60):
        steer_in -= dt
        if steer_in <= 0.0:
            steer_in = control_dt
            s, (cx, cy), _ = pt.follow_path(JOG, (x, y), s)
            speed = 520.0 * pace(control_dt)
            d = math.hypot(cx - x, cy - y) or 1.0
            want = (speed * (cx - x) / d, speed * (cy - y) / d)
        k = min(1.0, 8.0 * dt)
        vx += (want[0] - vx) * k
        vy += (want[1] - vy) * k
        x, y = x + vx * dt, y + vy * dt
        off = _off_path(JOG, x, y)
        worst_left, worst_right = max(worst_left, -off), max(worst_right, off)
        if math.hypot(x - JOG[-1][0], y - JOG[-1][1]) < 30.0:
            return True, worst_left, worst_right
    return False, worst_left, worst_right


def test_full_pace_at_the_editors_3_hz_steering_runs_off_the_stairs():
    """What the walk did in Unreal: 520 cm/s steered at 3 Hz is 1.7 m per steer."""
    pt = _load("playtest_explore")
    _, left, right = _walk_jog(pt, 0.33, lambda dt: 1.0)
    assert max(left, right) > 40.0


@pytest.mark.parametrize("control_dt", [1.0 / 60.0, 0.1, 0.33, 0.5])
def test_the_walk_pace_keeps_it_on_the_aisle_jog_at_any_tick_rate(control_dt):
    """Paced to the tick, the walker never runs more than 25 cm to the left of the
    line (the stair's outer side, where the drop is) nor 40 cm to the right (the rows)."""
    pt = _load("playtest_explore")
    reached, left, right = _walk_jog(pt, control_dt, pt.walk_pace)
    assert reached, "never got to the top of the jog"
    assert left < 25.0, "ran %.0f cm toward the drop" % left
    assert right < 40.0, "ran %.0f cm onto the rows" % right
