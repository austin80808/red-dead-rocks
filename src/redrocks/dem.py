"""USGS 3DEP bare-earth DEM: fetch a bbox as a float raster and sample it."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image

SERVICE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
NODATA = -9999.0
USER_AGENT = "redrocks-ue/0.1 (terrain prototype)"


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    @property
    def width_deg(self) -> float:
        return self.east - self.west

    @property
    def height_deg(self) -> float:
        return self.north - self.south

    @property
    def centre(self) -> tuple[float, float]:
        return (self.south + self.north) / 2.0, (self.west + self.east) / 2.0

    def as_query(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"


@dataclass
class Dem:
    """Elevation raster in metres. Row 0 is the north edge, column 0 the west."""

    data: np.ndarray
    bbox: BBox

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # (rows, cols)

    @property
    def min_elevation(self) -> float:
        return float(np.nanmin(self.data))

    @property
    def max_elevation(self) -> float:
        return float(np.nanmax(self.data))

    def sample(self, lat: float, lon: float) -> float:
        """Bilinear elevation lookup. Clamps to the raster edge."""
        rows, cols = self.data.shape
        fx = (lon - self.bbox.west) / self.bbox.width_deg * (cols - 1)
        fy = (self.bbox.north - lat) / self.bbox.height_deg * (rows - 1)
        fx = min(max(fx, 0.0), cols - 1.0)
        fy = min(max(fy, 0.0), rows - 1.0)

        x0, y0 = int(fx), int(fy)
        x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
        tx, ty = fx - x0, fy - y0

        top = self.data[y0, x0] * (1 - tx) + self.data[y0, x1] * tx
        bottom = self.data[y1, x0] * (1 - tx) + self.data[y1, x1] * tx
        return float(top * (1 - ty) + bottom * ty)


def fetch_dem(bbox: BBox, size: int = 2017, timeout: float = 180.0) -> Dem:
    """Download a bare-earth DEM for `bbox` resampled to `size` x `size`."""
    params = {
        "bbox": bbox.as_query(),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{size},{size}",
        "format": "tiff",
        "pixelType": "F32",
        "f": "json",
    }

    payload = _get_json(f"{SERVICE}/exportImage", params, timeout)

    if "href" not in payload:
        raise RuntimeError(f"3DEP exportImage did not return a raster: {payload}")

    raster = _get(payload["href"], timeout)

    image = Image.open(io.BytesIO(raster.content))
    data = np.array(image, dtype=np.float32)

    if data.shape != (size, size):
        raise RuntimeError(f"expected {size}x{size} raster, got {data.shape}")

    # 3DEP voids come back as a large negative sentinel or as NaN.
    data[data <= NODATA + 1.0] = np.nan
    if np.isnan(data).any():
        data = _fill_nan(data)

    return Dem(data=data, bbox=bbox)


ATTEMPTS = 8


def _get(url: str, timeout: float, params: dict[str, str] | None = None) -> requests.Response:
    """GET with exponential backoff; the 3DEP service 502s intermittently."""
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout
            )
            response.raise_for_status()
            return response
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as exc:
            last = exc
            if attempt == ATTEMPTS - 1:
                break
            delay = min(3.0 * (2.0 ** attempt), 45.0)
            print(f"  3DEP {getattr(exc, 'response', None) and exc.response.status_code} "
                  f"retry {attempt + 1}/{ATTEMPTS - 1} in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"3DEP request failed after {ATTEMPTS} attempts: {last}")


def _get_json(url: str, params: dict[str, str], timeout: float) -> dict:
    return _get(url, timeout, params).json()


def _fill_nan(data: np.ndarray) -> np.ndarray:
    """Replace voids with the nearest valid neighbour via iterative dilation."""
    filled = data.copy()
    for _ in range(64):
        holes = np.isnan(filled)
        if not holes.any():
            break
        padded = np.pad(filled, 1, mode="edge")
        neighbours = np.stack(
            [
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ]
        )
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(neighbours, axis=0)
        filled[holes] = mean[holes]
    return filled


def square_bbox(lat: float, lon: float, side_m: float) -> BBox:
    """A bbox centred on (lat, lon) that is `side_m` metres on each side."""
    from .geo import metres_per_degree

    m_per_deg_lon, m_per_deg_lat = metres_per_degree(lat)
    half_lat = (side_m / 2.0) / m_per_deg_lat
    half_lon = (side_m / 2.0) / m_per_deg_lon
    return BBox(
        west=lon - half_lon,
        south=lat - half_lat,
        east=lon + half_lon,
        north=lat + half_lat,
    )
