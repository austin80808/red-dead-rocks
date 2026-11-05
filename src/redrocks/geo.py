"""Geodesy helpers: local tangent-plane projection and Web Mercator tile math.

Everything here is in a right-handed terrain frame: east -> +X, north -> +Y,
up -> +Z, like a map. Unreal is left-handed (X forward, Y right, Z up), so the
level puts north on -Y; unreal/level_frame.py is the one place that converts.
"""

from __future__ import annotations

import math

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float = 0.0) -> tuple[float, float, float]:
    """WGS84 geodetic degrees + ellipsoidal height (m) to ECEF metres."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    cos_lat = math.cos(lat)
    return (
        (n + h) * cos_lat * math.cos(lon),
        (n + h) * cos_lat * math.sin(lon),
        (n * (1.0 - WGS84_E2) + h) * sin_lat,
    )


class LocalProjection:
    """East-North-Up tangent plane anchored at an origin, returning metres.

    Accurate to well under a centimetre across a few kilometres, which is far
    beyond what a landscape needs.
    """

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0) -> None:
        self.lat0 = lat0
        self.lon0 = lon0
        self.h0 = h0
        self._origin = geodetic_to_ecef(lat0, lon0, h0)

        lat = math.radians(lat0)
        lon = math.radians(lon0)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)

        self._east = (-sin_lon, cos_lon, 0.0)
        self._north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
        self._up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)

    def to_local(self, lat: float, lon: float, h: float = 0.0) -> tuple[float, float, float]:
        """Return (east, north, up) in metres relative to the origin."""
        x, y, z = geodetic_to_ecef(lat, lon, h)
        dx = x - self._origin[0]
        dy = y - self._origin[1]
        dz = z - self._origin[2]

        def dot(v: tuple[float, float, float]) -> float:
            return dx * v[0] + dy * v[1] + dz * v[2]

        return dot(self._east), dot(self._north), dot(self._up)


def metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """Approximate (metres per degree longitude, metres per degree latitude)."""
    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    w = 1.0 - WGS84_E2 * sin_lat * sin_lat
    m_per_deg_lat = (math.pi / 180.0) * WGS84_A * (1.0 - WGS84_E2) / (w ** 1.5)
    m_per_deg_lon = (math.pi / 180.0) * WGS84_A * math.cos(lat) / math.sqrt(w)
    return m_per_deg_lon, m_per_deg_lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Fractional Web Mercator tile coordinates (XYZ scheme)."""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y
