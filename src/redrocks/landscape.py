"""Export a DEM as an Unreal Engine landscape heightmap.

Unreal's landscape heightmap is a 16-bit grayscale PNG where the full 0..65535
range maps linearly onto the landscape's Z extent. The Z scale that makes the
terrain come out at true proportions is:

    Z_scale = (max_elevation - min_elevation) / 512 * 100

because Unreal's default landscape is 100 units tall per 512 units of the
heightmap's normalised range. Feed that number into the Landscape tool's
"Z Scale" field and the terrain is 1:1 with reality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .dem import Dem

# Unreal requires (component_size * sections_per_component) + 1 vertices per side.
VALID_SIZES = (127, 253, 505, 1009, 2017, 4033, 8129)


@dataclass
class LandscapeExport:
    png_path: Path
    size: int
    min_elevation: float
    max_elevation: float
    z_scale: float
    metres_per_pixel: float

    def summary(self) -> str:
        return (
            f"heightmap  {self.png_path.name}  {self.size}x{self.size}\n"
            f"elevation  {self.min_elevation:.1f} m .. {self.max_elevation:.1f} m "
            f"({self.max_elevation - self.min_elevation:.1f} m relief)\n"
            f"Z scale    {self.z_scale:.4f}   <- paste into Landscape > Z Scale\n"
            f"resolution {self.metres_per_pixel:.2f} m per pixel"
        )


def nearest_valid_size(target: int) -> int:
    return min(VALID_SIZES, key=lambda s: abs(s - target))


def resample(data: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resample a 2D float array to size x size."""
    rows, cols = data.shape
    if (rows, cols) == (size, size):
        return data

    ys = np.linspace(0, rows - 1, size)
    xs = np.linspace(0, cols - 1, size)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, rows - 1)
    x1 = np.minimum(x0 + 1, cols - 1)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]

    top = data[np.ix_(y0, x0)] * (1 - tx) + data[np.ix_(y0, x1)] * tx
    bottom = data[np.ix_(y1, x0)] * (1 - tx) + data[np.ix_(y1, x1)] * tx
    return top * (1 - ty) + bottom * ty


def export_landscape(dem: Dem, out_path: Path, size: int = 2017) -> LandscapeExport:
    """Write a 16-bit heightmap PNG plus the Z scale needed to match reality."""
    size = nearest_valid_size(size)
    heights = resample(dem.data, size)

    lo = float(np.nanmin(heights))
    hi = float(np.nanmax(heights))
    if hi - lo < 1e-6:
        raise ValueError("DEM has no relief; refusing to write a flat heightmap")

    normalised = (heights - lo) / (hi - lo)
    # Unreal reads the PNG as unsigned 16-bit; 65535 is the top of the range.
    encoded = np.clip(np.round(normalised * 65535.0), 0, 65535).astype(np.uint16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pillow infers I;16 from a uint16 array; passing mode= explicitly rejects it.
    Image.fromarray(encoded).save(out_path)

    relief = hi - lo
    z_scale = relief / 512.0 * 100.0

    return LandscapeExport(
        png_path=out_path,
        size=size,
        min_elevation=lo,
        max_elevation=hi,
        z_scale=z_scale,
        metres_per_pixel=_metres_per_pixel(dem, size),
    )


def _metres_per_pixel(dem: Dem, size: int) -> float:
    from .geo import metres_per_degree

    lat, _ = dem.bbox.centre
    m_per_deg_lon, _ = metres_per_degree(lat)
    return (dem.bbox.width_deg * m_per_deg_lon) / size
