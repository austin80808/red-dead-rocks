import numpy as np
import pytest

from redrocks.photo_camera import Camera, FitRefused, fit_camera, project

SIZE = (1280, 960)
TRUE = Camera(x=12000, y=-3000, z=2500, yaw=-150, pitch=-8, roll=1.5, fov=63)
GUESS = Camera(x=11000, y=-2000, z=2000, yaw=-140, pitch=0, fov=70)


def _points_in_front(cam: Camera) -> np.ndarray:
    """Eight landmarks 50-400 m ahead of ``cam``, z 0-9000 cm, spread across the view."""
    yaw = np.radians(cam.yaw)
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([-np.sin(yaw), np.cos(yaw)])
    spec = [  # (distance m, sideways m, z cm)
        (50, -15, 300), (80, 25, 1500), (120, -40, 6000), (160, 60, 800),
        (220, -90, 9000), (280, 110, 2500), (340, -60, 4500), (400, 140, 0),
    ]
    base = np.array([cam.x, cam.y])
    return np.array(
        [[*(base + d * 100 * fwd + s * 100 * right), z] for d, s, z in spec], dtype=float
    )


PTS = _points_in_front(TRUE)
PX, _ = project(TRUE, PTS, SIZE)


def test_a_point_straight_ahead_lands_in_the_centre():
    cam = Camera(x=0, y=0, z=0, yaw=0, pitch=0, fov=90)
    px, front = project(cam, np.array([[1000.0, 0, 0]]), (1600, 1200))
    assert front.all() and px[0] == pytest.approx([800, 600], abs=1e-6)


def test_level_plus_y_is_screen_right_and_plus_z_is_up():
    cam = Camera(x=0, y=0, z=0, yaw=0, pitch=0, fov=90)
    px, _ = project(cam, np.array([[1000.0, 500, 0], [1000.0, 0, 500]]), (1600, 1200))
    assert px[0][0] > 800 and px[1][1] < 600      # Unreal is left-handed: +Y right of +X


def test_fov_is_horizontal():
    cam = Camera(x=0, y=0, z=0, yaw=0, pitch=0, fov=90)
    px, _ = project(cam, np.array([[1000.0, 1000, 0]]), (1600, 1200))   # 45° right = image edge
    assert px[0][0] == pytest.approx(1600, abs=1e-6)


def test_positive_pitch_looks_up_and_yaw_turns_toward_plus_y():
    px, _ = project(Camera(0, 0, 0, yaw=0, pitch=10, fov=90), np.array([[1000.0, 0, 0]]), (1600, 1200))
    assert px[0][1] > 600          # target is below the view centre when looking up
    px, _ = project(Camera(0, 0, 0, yaw=90, pitch=0, fov=90), np.array([[0, 1000.0, 0]]), (1600, 1200))
    assert px[0] == pytest.approx([800, 600], abs=1e-6)


def test_camera_dict_round_trip_with_optional_roll():
    cam = Camera.from_dict({"x": 1, "y": 2, "z": 3, "yaw": 4, "pitch": 5, "fov": 60})
    assert cam.roll == 0.0 and Camera.from_dict(cam.to_dict()) == cam


def test_fit_recovers_a_synthetic_camera():
    got, err = fit_camera(PTS, PX, SIZE, GUESS)
    assert abs(got.x - TRUE.x) < 50 and abs(got.y - TRUE.y) < 50 and abs(got.z - TRUE.z) < 50   # 0.5 m
    for a in ("yaw", "pitch", "roll", "fov"):
        assert abs(getattr(got, a) - getattr(TRUE, a)) < 0.5
    assert err < 0.001


def test_fit_refuses_noisy_landmarks():
    rng = np.random.default_rng(7)
    noisy = PX + rng.choice([-1.0, 1.0], PX.shape) * 0.03 * SIZE[0]   # ±3 % of the width
    with pytest.raises(FitRefused, match="1.5 %"):
        fit_camera(PTS, noisy, SIZE, GUESS)


def test_fit_refuses_fewer_than_four_points_and_points_behind_the_camera():
    with pytest.raises(FitRefused, match="at least 4"):
        fit_camera(PTS[:3], PX[:3], SIZE, GUESS)
    # a landmark 100 m behind `true` at its pixel: refused, not fitted mirrored
    fwd = np.array([np.cos(np.radians(TRUE.yaw)), np.sin(np.radians(TRUE.yaw)), 0.0])
    behind = PTS.copy()
    behind[3] = np.array([TRUE.x, TRUE.y, TRUE.z]) - 10000 * fwd
    with pytest.raises(FitRefused, match="behind"):
        fit_camera(behind, PX, SIZE, GUESS)


def _unreal_axes(yaw, pitch, roll):
    """Forward, right, up straight from Unreal's FRotationMatrix."""
    sp, cp = np.sin(np.radians(pitch)), np.cos(np.radians(pitch))
    sy, cy = np.sin(np.radians(yaw)), np.cos(np.radians(yaw))
    sr, cr = np.sin(np.radians(roll)), np.cos(np.radians(roll))
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return fwd, right, up


def test_roll_plus_ten_matches_unreal_example():
    cam = Camera(0, 0, 0, yaw=0, pitch=0, roll=10, fov=90)
    px, _ = project(cam, np.array([[1000.0, 0, 500]]), (1600, 1200))
    assert px[0][0] == pytest.approx(730.5, abs=0.1)


@pytest.mark.parametrize("rot", [(30, 20, 10), (-150, -8, 1.5), (200, 45, -35), (75, -60, 80)])
def test_axes_match_unreal_rotation_matrix(rot):
    from redrocks.photo_camera import _axes

    yaw, pitch, roll = rot
    got = _axes(Camera(0, 0, 0, yaw=yaw, pitch=pitch, roll=roll))
    for g, want in zip(got, _unreal_axes(yaw, pitch, roll)):
        assert g == pytest.approx(want, abs=1e-9)
