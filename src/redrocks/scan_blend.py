"""Conform the terrain to the ml0w scan so the two meet without a seam (issue #17).

The scan (``unreal/import_scan.py``) is placed over the 3DEP terrain, but the
two surfaces differ by a median of about 4 m and by 15 m or more around the
rock fins, where the fitted scan and the 6 m terrain grid disagree. Where the
terrain is higher it buries the scan (orange patches through the photo); where
it is lower the scan's edge is a floating shelf over a gap.

This module fixes the geometry from the terrain's side:

* **Inside the scan's footprint** the terrain mesh's vertices take the scan's
  own surface, ``MARGIN_M`` lower, then are pushed down wherever a triangle
  would still rise into the scan (``lower_envelope``). The scan always draws
  on top, and the walkable terrain follows what the player sees.
* **Outside the footprint** it eases back to the DEM over ``FALLOFF_M``, so the
  scan's edge meets terrain at its own height instead of a step.
* **The benches and stage are left alone** (``keep_out``): they are generated
  against the DEM (and the player spawns on the deck), so the terrain under
  them stays the DEM, easing into the scan blend over a few metres.

The scan side of the seam (its ragged edge dithering out over ``FADE_M``) is
the material in ``unreal/import_scan.py``, driven by the fade mask that
``tools/scan_heights.py`` writes from the same footprint as here.

The scan's heights come in as a ``ScanHeights`` raster, written by
``tools/scan_heights.py`` from the USD and the placement constants in
``unreal/import_scan.py``. Everything here is plain numpy, so it runs without
the USD libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dem import Dem
from .geo import metres_per_degree
from .terrain_mesh import DEFAULT_GRID

# ---- constants and how they were chosen --------------------------------------
# Raster cell of the scan heights. The scan has ~11 vertices per m^2, so 2 m
# cells all have vertices except under the steepest walls (holes are filled).
CELL_M = 2.0
# Terrain sits at least this far under every scan point. 0.4 m is enough for
# the depth buffer at the distances the scan is seen from and small enough that
# a player walking the terrain is not visibly sunk into the scan.
MARGIN_M = 0.4
# Back to the DEM over this distance outside the footprint. Edge mismatches are
# up to ~15 m (95th percentile about 12 m); over 60 m a smoothstep's steepest
# added slope is 1.5 * 12 / 60 = 0.3, about 17 degrees, which reads as natural
# ground rather than a ramp.
FALLOFF_M = 60.0
# Notches and holes in the footprint narrower than about twice this are closed
# so the fade does not ring every tiny gap in the scan's coverage.
CLOSE_M = 6.0
# The scan's material dithers from opaque to invisible over the last FADE_M
# inside its footprint. 30 m is about half the falloff, so the scan is fully
# gone where the terrain still matches it exactly and the colour change is
# spread over ~90 m in total. Read by tools/scan_heights.py for the fade mask.
FADE_M = 30.0
# Keep-out: the benches and stage blockout are built on the DEM (each bench's
# bottom is the lowest DEM under it; the deck is a 1.5 m slab), so the terrain
# under them stays the DEM. Within KEEP_OUT_M of their footprints nothing
# moves; the scan blend returns over the next KEEP_OUT_TAPER_M. Both are short
# on purpose: just north of the bowl the DEM's rock ridge stands 10 m or more
# above the scan, and every metre of taper there is a strip of terrain showing
# through the scan. 1 m covers the 5 % gaps between bench rows and the corridor
# from the deck to row 1 (the play-test's walk), 6 m keeps that edge a slope a
# player can climb rather than a cliff.
KEEP_OUT_M = 1.0
KEEP_OUT_TAPER_M = 6.0
# -----------------------------------------------------------------------------


@dataclass
class ScanHeights:
    """The scan's floor on a regular grid in the terrain's local frame.

    ``heights[i, j]`` is the lowest scan point (absolute elevation, metres) in
    the cell whose south-west corner is ``(x0 + j * cell, y0 + i * cell)``
    (east, north metres from the bbox centre); NaN where the scan has no point.
    Row 0 is the SOUTH edge.
    """

    heights: np.ndarray
    x0: float
    y0: float
    cell: float = CELL_M

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(east, north) of every cell centre, each shaped like ``heights``."""
        ny, nx = self.heights.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, heights=self.heights.astype(np.float32), x0=self.x0, y0=self.y0, cell=self.cell
        )

    @classmethod
    def load(cls, path: Path) -> "ScanHeights":
        with np.load(path) as d:
            return cls(
                heights=d["heights"].astype(float),
                x0=float(d["x0"]),
                y0=float(d["y0"]),
                cell=float(d["cell"]),
            )

    @classmethod
    def from_points(cls, east, north, elevation, cell: float = CELL_M) -> "ScanHeights":
        """Rasterise scan points: each cell keeps its LOWEST point (the floor).

    The floor, not the top: under a tree or beside a wall the terrain has to
    stay below the lowest scan surface there or it shows through the side.
    """
        east, north, elevation = (np.asarray(a, float) for a in (east, north, elevation))
        x0 = np.floor(east.min() / cell) * cell
        y0 = np.floor(north.min() / cell) * cell
        nx = int((east.max() - x0) // cell) + 1
        ny = int((north.max() - y0) // cell) + 1
        ix = np.clip(((east - x0) // cell).astype(int), 0, nx - 1)
        iy = np.clip(((north - y0) // cell).astype(int), 0, ny - 1)
        low = np.full((ny, nx), np.inf)
        np.minimum.at(low, (iy, ix), elevation)
        low[~np.isfinite(low)] = np.nan
        return cls(heights=low, x0=float(x0), y0=float(y0), cell=cell)

    def padded(self, margin_m: float) -> "ScanHeights":
        """The same raster with NaN cells added on every side."""
        p = int(np.ceil(margin_m / self.cell))
        h = np.pad(self.heights, p, constant_values=np.nan)
        return ScanHeights(h, self.x0 - p * self.cell, self.y0 - p * self.cell, self.cell)


# ---- raster helpers (numpy only) ---------------------------------------------
def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """``out[i, j] = a[i - dy, j - dx]``; cells shifted in from outside get ``fill``."""
    out = np.full_like(a, fill)
    ny, nx = a.shape
    out[max(dy, 0):ny + min(dy, 0), max(dx, 0):nx + min(dx, 0)] = a[
        max(-dy, 0):ny + min(-dy, 0), max(-dx, 0):nx + min(-dx, 0)
    ]
    return out


def _disc(r: int):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for dy, dx in _disc(r):
        out |= _shift(mask, dy, dx, False)
    return out


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    return ~dilate(~mask, r)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """``mask`` plus every region of ``~mask`` not connected to the border."""
    outside = np.zeros_like(mask)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside &= ~mask
    while True:
        grown = outside.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown |= _shift(outside, dy, dx, False)
        grown &= ~mask
        if (grown == outside).all():
            return ~outside
        outside = grown


def distance_to(mask: np.ndarray, cell: float, limit_m: float = np.inf) -> np.ndarray:
    """Chamfer distance (m) from each cell to the nearest ``mask`` cell.

    Within 8 % of Euclidean (diagonal steps are exact, knight's moves are not).
    Cells further than ``limit_m`` keep +inf.
    """
    d = np.where(mask, 0.0, np.inf)
    steps = [(dy, dx, cell * float(np.hypot(dy, dx))) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    for _ in range(int(np.ceil(limit_m / cell)) + 2 if np.isfinite(limit_m) else max(mask.shape)):
        new = d
        for dy, dx, w in steps:
            new = np.minimum(new, _shift(d, dy, dx, np.inf) + w)
        if np.array_equal(new, d):
            break
        d = new
    if np.isfinite(limit_m):
        d[d > limit_m] = np.inf
    return d


def extend(values: np.ndarray, iterations: int) -> np.ndarray:
    """Fill NaN cells from their finite neighbours, one ring per iteration.

    Each new cell takes the mean of its finite 8-neighbours, so the field is
    carried outward smoothly; cells further than ``iterations`` stay NaN.
    """
    v = values.copy()
    for _ in range(iterations):
        nan = np.isnan(v)
        if not nan.any():
            break
        total = np.zeros_like(v)
        count = np.zeros_like(v)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    s = _shift(v, dy, dx, np.nan)
                    ok = ~np.isnan(s)
                    total[ok] += s[ok]
                    count[ok] += 1
        grow = nan & (count > 0)
        if not grow.any():
            break
        v[grow] = total[grow] / count[grow]
    return v


def smoothstep(edge0: float, edge1: float, x):
    t = np.clip((np.asarray(x, float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---- the blend ---------------------------------------------------------------
def footprint(scan: ScanHeights, close_m: float = CLOSE_M) -> np.ndarray:
    """Cells the scan covers, with notches and holes narrower than ~2 * close_m closed."""
    covered = ~np.isnan(scan.heights)
    r = max(1, int(round(close_m / scan.cell)))
    # Pad so closing near the raster edge does not erode real coverage.
    padded = np.pad(covered, r + 1)
    closed = erode(dilate(padded, r), r)[r + 1:-(r + 1), r + 1:-(r + 1)]
    return fill_holes(closed | covered)


def fade_alpha(scan: ScanHeights, fade_m: float = FADE_M, close_m: float = CLOSE_M) -> np.ndarray:
    """Scan opacity per cell: 0 at and outside the footprint edge, 1 from ``fade_m`` in."""
    inside = footprint(scan, close_m)
    # Distance from the outside, so the raster border counts as outside too.
    outside = np.pad(~inside, 1, constant_values=True)
    d = distance_to(outside, scan.cell, limit_m=fade_m + scan.cell)[1:-1, 1:-1]
    # A cell centre is half a cell in from its edge; measure from the edge.
    return np.where(inside, smoothstep(0.0, fade_m, d - scan.cell / 2.0), 0.0)


def _node_grid(dem: Dem, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """East and north (1-D) of the terrain mesh's vertex columns and rows."""
    lat, _ = dem.bbox.centre
    m_lon, m_lat = metres_per_degree(lat)
    half_x = dem.bbox.width_deg * m_lon / 2.0
    half_y = dem.bbox.height_deg * m_lat / 2.0
    return np.linspace(-half_x, half_x, grid), np.linspace(half_y, -half_y, grid)


def _mesh_cells(xs: np.ndarray, ys: np.ndarray, east: np.ndarray, north: np.ndarray):
    """Mesh cell (row, col) and in-cell fractions (u east, v south) of points."""
    n = len(xs)
    fc = np.clip((east - xs[0]) / (xs[1] - xs[0]), 0.0, n - 1.0)
    fr = np.clip((north - ys[0]) / (ys[1] - ys[0]), 0.0, n - 1.0)
    c = np.minimum(fc.astype(int), n - 2)
    r = np.minimum(fr.astype(int), n - 2)
    return r, c, fc - c, fr - r


def _bilinear(raster: ScanHeights, values: np.ndarray, east: np.ndarray, north: np.ndarray, outside):
    """Sample ``values`` (cell-centred on ``raster``) at points; ``outside`` beyond it."""
    ny, nx = values.shape
    fx = (east - raster.x0) / raster.cell - 0.5
    fy = (north - raster.y0) / raster.cell - 0.5
    inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx = np.clip(fx - x0, 0, 1)
    ty = np.clip(fy - y0, 0, 1)
    v = (
        values[y0, x0] * (1 - tx) * (1 - ty)
        + values[y0, x0 + 1] * tx * (1 - ty)
        + values[y0 + 1, x0] * (1 - tx) * ty
        + values[y0 + 1, x0 + 1] * tx * ty
    )
    return np.where(inside, v, outside)


def dem_local_grid(dem: Dem) -> tuple[np.ndarray, np.ndarray]:
    """(east, north) metres of every DEM pixel, in the terrain mesh's frame.

    Matches ``terrain_mesh.export_terrain_obj``: column 0 at -half width, row 0
    at +half height (north), linear in between.
    """
    rows, cols = dem.shape
    xs, _ = _node_grid(dem, cols)
    _, ys = _node_grid(dem, rows)
    return np.meshgrid(xs, ys)


def harmonic_fill(values: np.ndarray, fixed: np.ndarray, tol: float = 0.005, max_iter: int = 4000) -> np.ndarray:
    """Solve Laplace's equation for the cells not ``fixed``; fixed cells are the boundary.

    Jacobi iteration from ``values`` as the first guess, until no cell moves
    more than ``tol`` (m) in an iteration. The result is the smoothest surface
    that meets every fixed cell: no ridges, no pits, no overshoot.
    """
    v = values.astype(float).copy()
    free = ~fixed
    if not free.any():
        return v
    for _ in range(max_iter):
        up = _shift(v, 1, 0, np.nan)
        down = _shift(v, -1, 0, np.nan)
        left = _shift(v, 0, 1, np.nan)
        right = _shift(v, 0, -1, np.nan)
        stack = np.stack([up, down, left, right])
        mean = np.nanmean(np.where(np.isnan(stack), np.nan, stack), axis=0)
        new = np.where(free & ~np.isnan(mean), mean, v)
        change = np.nanmax(np.abs(new - v)[free]) if free.any() else 0.0
        v = new
        if change < tol:
            break
    return v


def _dem_at(dem: Dem, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Bilinear DEM elevation at local (east, north) metres, clamped to its edge."""
    rows, cols = dem.shape
    xs, _ = _node_grid(dem, cols)
    _, ys = _node_grid(dem, rows)
    fc = np.clip((east - xs[0]) / (xs[1] - xs[0]), 0.0, cols - 1.0)
    fr = np.clip((north - ys[0]) / (ys[1] - ys[0]), 0.0, rows - 1.0)
    c0 = np.minimum(fc.astype(int), cols - 2)
    r0 = np.minimum(fr.astype(int), rows - 2)
    tc, tr = fc - c0, fr - r0
    d = dem.data.astype(float)
    top = d[r0, c0] * (1 - tc) + d[r0, c0 + 1] * tc
    bot = d[r0 + 1, c0] * (1 - tc) + d[r0 + 1, c0 + 1] * tc
    return top * (1 - tr) + bot * tr


def mesh_height(nodes: np.ndarray, xs, ys, east, north) -> np.ndarray:
    """Height of the triangulated vertex grid ``nodes`` at points.

    Same triangulation as ``terrain_mesh.export_terrain_obj`` (and
    ``TerrainSurface``): each cell split along its NW-SE diagonal.
    """
    r, c, u, v = _mesh_cells(xs, ys, east, north)
    a, b = nodes[r, c], nodes[r, c + 1]
    d, e = nodes[r + 1, c], nodes[r + 1, c + 1]
    return np.where(v >= u, a + (d - a) * v + (e - d) * u, a + (b - a) * u + (e - b) * v)


def lower_envelope(nodes: np.ndarray, xs, ys, east, north, floor, margin_m: float = MARGIN_M) -> np.ndarray:
    """Lower vertices until the triangulated surface is ``margin_m`` under every floor point.

    Each triangle that rises above a point is dropped as a whole (its three
    vertices) by the worst excess in it, so one pass fixes every point and
    lowering never raises anything else. The result may sit a little deeper
    than strictly needed in concave spots; it never sits higher.
    """
    out = nodes.copy()
    for _ in range(3):  # one pass suffices; the rest only confirm
        excess = mesh_height(out, xs, ys, east, north) - (floor - margin_m)
        bad = excess > 1e-6
        if not bad.any():
            break
        r, c, u, v = _mesh_cells(xs, ys, east[bad], north[bad])
        lower_tri = v[:] >= u[:]
        third_r = np.where(lower_tri, r + 1, r)  # SW (r+1, c) or NE (r, c+1)
        third_c = np.where(lower_tri, c, c + 1)
        drop = np.zeros_like(out)
        for rr, cc in ((r, c), (r + 1, c + 1), (third_r, third_c)):
            np.maximum.at(drop, (rr, cc), excess[bad])
        out = out - drop
    return out


def conform_dem(
    dem: Dem,
    scan: ScanHeights,
    grid: int = DEFAULT_GRID,
    keep_out=None,
    margin_m: float = MARGIN_M,
    falloff_m: float = FALLOFF_M,
    close_m: float = CLOSE_M,
    keep_out_m: float = KEEP_OUT_M,
    keep_out_taper_m: float = KEEP_OUT_TAPER_M,
) -> Dem:
    """A copy of ``dem`` whose ``grid``-vertex terrain mesh sits just under the scan.

    Inside the scan's footprint the mesh vertices are set to the scan floor
    minus ``margin_m``, then lowered (``lower_envelope``) until no triangle
    rises within ``margin_m`` of any scan point. Across the ``falloff_m`` band
    outside it the surface is the smooth (harmonic) fill between the scan's
    edge and the DEM, so nothing at the edge (a rock fin, a gully) is dragged
    outward. ``keep_out`` is an optional shapely geometry (local east/north
    metres) where the DEM is kept: untouched within ``keep_out_m`` of it, fully
    blended ``keep_out_taper_m`` further out.
    """
    xs, ys = _node_grid(dem, grid)
    # Pad far enough for the falloff and for every vertex of a triangle that
    # holds a scan point (one mesh cell out) to land on the raster.
    pad = max(falloff_m, 2.0 * abs(xs[1] - xs[0])) + 3 * scan.cell
    s = scan.padded(pad)
    inside = footprint(s, close_m)
    ce, cn = s.centres()
    # The scan floor, carried over holes inside the footprint (and outward, as
    # the starting guess for the fill below).
    floor = extend(np.where(inside, s.heights, np.nan), int(np.ceil(pad / s.cell)) + 1)

    d_out = distance_to(inside, s.cell, limit_m=falloff_m + s.cell)
    band = ~inside & (d_out <= falloff_m)
    surface = np.where(inside, floor - margin_m, np.nan)
    surface = np.where(band, np.where(np.isnan(floor), _dem_at(dem, ce, cn), floor - margin_m), surface)
    surface = np.where(inside | band, surface, _dem_at(dem, ce, cn))
    surface = harmonic_fill(surface, fixed=~band)
    weight = np.where(inside | band, 1.0, 0.0)
    if keep_out is not None:
        import shapely

        dist = shapely.distance(keep_out, shapely.points(ce.ravel(), cn.ravel())).reshape(ce.shape)
        weight = weight * smoothstep(keep_out_m, keep_out_m + keep_out_taper_m, dist)

    # Target heights at the mesh vertices, then pushed under every scan point.
    ne, nn = np.meshgrid(xs, ys)
    nodes = _bilinear(s, surface, ne, nn, np.nan)
    covered = ~np.isnan(s.heights)
    filled = np.where(np.isnan(nodes), np.inf, nodes)  # vertices off the raster never bind
    nodes = np.where(np.isnan(nodes), np.nan, lower_envelope(filled, xs, ys, ce[covered], cn[covered],
                                                            s.heights[covered], margin_m))

    # Back onto the DEM's pixels: target by the mesh's own interpolation of the
    # vertices, weight from the raster. Where there is no target, keep the DEM.
    east, north = dem_local_grid(dem)
    target = mesh_height(nodes, xs, ys, east, north)
    w = _bilinear(s, weight, east, north, 0.0)
    w = np.where(np.isnan(target), 0.0, w)
    data = dem.data.astype(float)
    blended = data + w * (np.where(np.isnan(target), data, target) - data)
    return Dem(data=blended.astype(dem.data.dtype), bbox=dem.bbox)


def blend_summary(dem: Dem, blended: Dem) -> dict:
    """How much the terrain moved: changed area and the spread of the change."""
    delta = blended.data.astype(float) - dem.data.astype(float)
    moved = np.abs(delta) > 0.01
    lat, _ = dem.bbox.centre
    m_lon, m_lat = metres_per_degree(lat)
    rows, cols = dem.shape
    px_area = (dem.bbox.width_deg * m_lon / (cols - 1)) * (dem.bbox.height_deg * m_lat / (rows - 1))
    out = {"changed_area_m2": round(float(moved.sum() * px_area), 1)}
    if moved.any():
        p = np.percentile(delta[moved], [5, 50, 95])
        out.update(
            raised_max_m=round(float(delta.max()), 2),
            lowered_max_m=round(float(-delta.min()), 2),
            delta_p5_p50_p95_m=[round(float(x), 2) for x in p],
        )
    return out
