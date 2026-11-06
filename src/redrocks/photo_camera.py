"""Pinhole camera in Unreal's level frame, and a least-squares fit from landmarks.

Conventions follow Unreal: left-handed, +X forward, +Y right, +Z up, units in
centimetres, angles in degrees. Rotator yaw turns about Z, pitch about Y
(positive pitch looks up) and roll about X.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

MAX_ERROR_FRACTION = 0.015  # mean reprojection error, as a fraction of image width
_NEAR = 1e-3  # cm; depth floor so a point crossing the camera plane cannot blow up the fit
_PARAM_NAMES = ("x", "y", "z", "yaw", "pitch", "roll", "fov")
_STEPS = (1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01)  # central-difference steps: cm, then degrees


class FitRefused(ValueError):
    """Raised when landmarks cannot give a trustworthy camera."""


@dataclass
class Camera:
    """A camera in the level frame.

    Attributes:
        x: Position X in cm.
        y: Position Y in cm.
        z: Position Z in cm.
        yaw: Rotator yaw in degrees.
        pitch: Rotator pitch in degrees (positive looks up).
        roll: Rotator roll in degrees.
        fov: Horizontal field of view in degrees.
    """

    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float = 0.0
    fov: float = 70.0

    def to_dict(self) -> dict:
        """Return the camera as a plain dict with keys x y z yaw pitch roll fov."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        """Build a camera from a dict; ``roll`` and ``fov`` default when absent."""
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            z=float(d["z"]),
            yaw=float(d["yaw"]),
            pitch=float(d["pitch"]),
            roll=float(d.get("roll", 0.0)),
            fov=float(d.get("fov", 70.0)),
        )


def _axes(cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the camera's forward, right and up unit vectors."""
    yaw, pitch, roll = (math.radians(a) for a in (cam.yaw, cam.pitch, cam.roll))
    f = np.array(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    )
    r = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    u = np.cross(f, r)  # +Z for a level camera
    cr, sr = math.cos(roll), math.sin(roll)
    return f, r * cr + u * sr, u * cr - r * sr


def _project(cam: Camera, pts: np.ndarray, size: tuple[int, int], floor: bool):
    width, height = size
    f, r, u = _axes(cam)
    d = np.asarray(pts, dtype=float).reshape(-1, 3) - np.array([cam.x, cam.y, cam.z])
    depth = d @ f
    front = depth > 0
    focal = (width / 2.0) / math.tan(math.radians(cam.fov) / 2.0)
    div = np.maximum(depth, _NEAR) if floor else depth
    px = np.stack(
        [width / 2.0 + focal * (d @ r) / div, height / 2.0 - focal * (d @ u) / div], axis=1
    )
    return px, front


def project(
    cam: Camera, points_cm: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project level-frame points into an image.

    Args:
        cam: The camera.
        points_cm: Array of shape (N, 3), level coordinates in cm.
        size: Image (width, height) in pixels.

    Returns:
        Pixels of shape (N, 2) (x right, y down, origin top-left) and a bool
        array of shape (N,) that is True where the point is in front of the camera.
        Pixels for points behind the camera are mirrored and not meaningful.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return _project(cam, points_cm, size, floor=False)


def _vec(cam: Camera) -> np.ndarray:
    return np.array([getattr(cam, n) for n in _PARAM_NAMES], dtype=float)


def _cam(vec: np.ndarray) -> Camera:
    return Camera(**dict(zip(_PARAM_NAMES, (float(v) for v in vec))))


def _residuals(vec, pts, pixels, size):
    px, _ = _project(_cam(vec), pts, size, floor=True)
    return (px - pixels).ravel()


def _jacobian(vec, pts, pixels, size):
    cols = []
    for i, step in enumerate(_STEPS):
        hi, lo = vec.copy(), vec.copy()
        hi[i] += step
        lo[i] -= step
        cols.append((_residuals(hi, pts, pixels, size) - _residuals(lo, pts, pixels, size)) / (2 * step))
    return np.stack(cols, axis=1)


def _mean_error(vec, pts, pixels, size) -> float:
    px, _ = _project(_cam(vec), pts, size, floor=True)
    return float(np.linalg.norm(px - pixels, axis=1).mean() / size[0])


def fit_camera(
    points_cm: np.ndarray,
    pixels: np.ndarray,
    size: tuple[int, int],
    guess: Camera,
    iterations: int = 60,
) -> tuple[Camera, float]:
    """Fit a camera to landmarks by Levenberg-Marquardt on reprojection error.

    All seven parameters (position, yaw, pitch, roll, fov) are solved, using a
    central-difference Jacobian (1 cm, 0.01 degree steps) starting at ``guess``.

    Args:
        points_cm: Landmark positions, shape (N, 3), level cm.
        pixels: Where each landmark appears in the photo, shape (N, 2).
        size: Image (width, height) in pixels.
        guess: Starting camera.
        iterations: Maximum LM iterations.

    Returns:
        The fitted camera and the mean reprojection error as a fraction of image width.

    Raises:
        FitRefused: With fewer than 4 points, if a landmark is behind the guess or
            the fitted camera, or if the mean error exceeds 1.5 % of the width.
    """
    pts = np.asarray(points_cm, dtype=float).reshape(-1, 3)
    pix = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(pts) != len(pix):
        raise FitRefused("points and pixels must be the same length")
    if len(pts) < 4:
        raise FitRefused(f"need at least 4 landmarks, got {len(pts)}")
    if not _project(guess, pts, size, floor=True)[1].all():
        raise FitRefused("a landmark is behind the camera at the starting guess")

    vec = _vec(guess)
    res = _residuals(vec, pts, pix, size)
    cost = float(res @ res)
    lam = 1e-3
    for _ in range(iterations):
        jac = _jacobian(vec, pts, pix, size)
        gram = jac.T @ jac
        grad = jac.T @ res
        improved = False
        for _try in range(12):
            damped = gram + lam * np.diag(np.maximum(np.diag(gram), 1e-12))
            try:
                step = np.linalg.solve(damped, -grad)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            cand = vec + step
            cand[6] = float(np.clip(cand[6], 5.0, 170.0))
            cres = _residuals(cand, pts, pix, size)
            ccost = float(cres @ cres)
            if np.isfinite(ccost) and ccost < cost:
                done = abs(cost - ccost) <= 1e-14 * max(cost, 1.0)
                vec, res, cost = cand, cres, ccost
                lam = max(lam / 5, 1e-12)
                improved = True
                if done:
                    lam = None
                break
            lam *= 10
        if lam is None or not improved or cost < 1e-16:
            break

    fitted = _cam(vec)
    if not _project(fitted, pts, size, floor=True)[1].all():
        raise FitRefused("a landmark is behind the camera after fitting; check the landmark matches")
    err = _mean_error(vec, pts, pix, size)
    if err > MAX_ERROR_FRACTION:
        raise FitRefused(
            f"landmarks disagree: mean error {err * 100:.1f} % of the width is over 1.5 % "
            "(wrong matches or too few clean ones)"
        )
    return fitted, err
