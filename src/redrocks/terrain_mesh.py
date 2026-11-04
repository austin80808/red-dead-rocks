"""Export the DEM as a static-mesh terrain instead of an Unreal Landscape.

Unreal's Landscape tool cannot be driven from Python in vanilla UE, and its
heightmap import is GUI-only. A plain triangle mesh has none of those limits: it
imports through AssetImportTask like any other OBJ, works in a commandlet, and
carries real vertex normals so it lights correctly.

The trade-off is no landscape-specific features (splat layers, grass, LOD
switching driven by the landscape component). For a walkable venue that is an
acceptable cost, and the mesh can be swapped for a Landscape later if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dem import Dem
from .geo import metres_per_degree

# Unreal works in centimetres.
CM_PER_M = 100.0

# One UV tile per 1 km of terrain, so tiling materials read at a sane scale.
UV_TILE_CM = 100000.0

# 505 x 505 keeps the mesh under ~500k triangles, which imports comfortably on a
# 24 GB machine while still resolving the bowl's 30 m rise.
DEFAULT_GRID = 505


@dataclass
class TerrainMesh:
    path: Path
    grid: int
    width_m: float
    height_m: float
    min_elevation_m: float
    max_elevation_m: float
    triangles: int


def resample_to_grid(dem: Dem, grid: int) -> np.ndarray:
    """Bilinearly resample the DEM to a square grid."""
    rows, cols = dem.data.shape
    ys = np.linspace(0, rows - 1, grid)
    xs = np.linspace(0, cols - 1, grid)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, rows - 1)
    x1 = np.minimum(x0 + 1, cols - 1)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]

    top = dem.data[np.ix_(y0, x0)] * (1 - tx) + dem.data[np.ix_(y0, x1)] * tx
    bottom = dem.data[np.ix_(y1, x0)] * (1 - tx) + dem.data[np.ix_(y1, x1)] * tx
    return top * (1 - ty) + bottom * ty


class TerrainSurface:
    """The exported terrain mesh as a function: ``surface(east, north)`` -> metres.

    Reproduces ``export_terrain_obj`` exactly, including its triangulation (each
    cell is split along the diagonal from its north-west to its south-east
    corner), so anything placed at ``surface(e, n)`` sits on the mesh as
    imported rather than on the finer DEM. Queries outside the extent clamp to
    the edge. Heights are absolute elevations, not mesh Z.
    """

    def __init__(self, dem: Dem, grid: int = DEFAULT_GRID) -> None:
        if grid < 2:
            raise ValueError("grid must be at least 2 to form any triangles")
        self.heights = resample_to_grid(dem, grid)
        lat, _ = dem.bbox.centre
        m_per_deg_lon, m_per_deg_lat = metres_per_degree(lat)
        self.half_x = dem.bbox.width_deg * m_per_deg_lon / 2.0
        self.half_y = dem.bbox.height_deg * m_per_deg_lat / 2.0
        self.grid = grid

    @property
    def base_elevation_m(self) -> float:
        """Elevation of mesh Z = 0 (the lowest vertex)."""
        return float(np.nanmin(self.heights))

    def __call__(self, east: float, north: float) -> float:
        n = self.grid
        fc = (east + self.half_x) / (2.0 * self.half_x) * (n - 1)
        fr = (self.half_y - north) / (2.0 * self.half_y) * (n - 1)
        fc = min(max(fc, 0.0), n - 1.0)
        fr = min(max(fr, 0.0), n - 1.0)
        c = min(int(fc), n - 2)
        r = min(int(fr), n - 2)
        u, v = fc - c, fr - r  # u east, v south, within the cell
        h = self.heights
        a, b, d, e = h[r, c], h[r, c + 1], h[r + 1, c], h[r + 1, c + 1]
        if v >= u:  # triangle a, d, e
            return float(a + (d - a) * v + (e - d) * u)
        return float(a + (b - a) * u + (e - b) * v)  # triangle a, e, b


def export_terrain_obj(dem: Dem, out_path: Path, grid: int = DEFAULT_GRID) -> TerrainMesh:
    """Write the DEM as an OBJ terrain mesh centred on the origin.

    X runs east, Y runs north, Z is up, all in centimetres, so the mesh lands in
    Unreal with no axis conversion.
    """
    if grid < 2:
        raise ValueError("grid must be at least 2 to form any triangles")

    heights = resample_to_grid(dem, grid)
    rows, cols = heights.shape

    lat, _ = dem.bbox.centre
    m_per_deg_lon, m_per_deg_lat = metres_per_degree(lat)
    width_m = dem.bbox.width_deg * m_per_deg_lon
    depth_m = dem.bbox.height_deg * m_per_deg_lat

    base = float(np.nanmin(heights))
    half_x = width_m / 2.0
    half_y = depth_m / 2.0

    # Vertex grid: row 0 is the north edge, so Y decreases with row index.
    xs = np.linspace(-half_x, half_x, cols) * CM_PER_M
    ys = np.linspace(half_y, -half_y, rows) * CM_PER_M
    zs = (heights - base) * CM_PER_M

    xx, yy = np.meshgrid(xs, ys)

    lines = [
        "# Red Rocks terrain generated from USGS 3DEP DEM",
        "# Elevation data: USGS 3DEP, public domain",
        f"# grid {grid}x{grid}, {width_m:.1f} x {depth_m:.1f} m, base {base:.1f} m",
        "o RedRocks_Terrain",
    ]

    # OBJ indices are 1-based.
    for r in range(rows):
        for c in range(cols):
            lines.append(f"v {xx[r, c]:.2f} {yy[r, c]:.2f} {zs[r, c]:.2f}")

    # Unreal's OBJ importer requires a UV channel; without one it silently
    # collapses the mesh (it fails an ensure in InterchangeOBJTranslator and
    # keeps a fraction of the triangles). Planar UVs in metres per tile are
    # enough, and give a usable tiling texture scale.
    for r in range(rows):
        for c in range(cols):
            lines.append(f"vt {xx[r, c] / UV_TILE_CM:.6f} {yy[r, c] / UV_TILE_CM:.6f}")

    def vid(r: int, c: int) -> int:
        return r * cols + c + 1

    for r in range(rows - 1):
        for c in range(cols - 1):
            a = vid(r, c)
            b = vid(r, c + 1)
            d = vid(r + 1, c)
            e = vid(r + 1, c + 1)
            # Counter-clockwise seen from above (+Z) so normals point up.
            # Vertex and UV indices are numerically identical here.
            lines.append(f"f {a}/{a} {d}/{d} {e}/{e}")
            lines.append(f"f {a}/{a} {e}/{e} {b}/{b}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    return TerrainMesh(
        path=out_path,
        grid=grid,
        width_m=width_m,
        height_m=depth_m,
        min_elevation_m=float(np.nanmin(heights)),
        max_elevation_m=float(np.nanmax(heights)),
        triangles=(rows - 1) * (cols - 1) * 2,
    )


def summary(mesh: TerrainMesh) -> str:
    return (
        f"terrain  {mesh.grid}x{mesh.grid}  {mesh.triangles:,} triangles\n"
        f"extent   {mesh.width_m:.1f} x {mesh.height_m:.1f} m\n"
        f"relief   {mesh.max_elevation_m - mesh.min_elevation_m:.1f} m"
    )
