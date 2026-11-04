"""Blockout of the Red Rocks stage, shell, retaining walls and top walkway.

None of this exists in free geodata, so it is built from a handful of boxes and
wedges (vertical prisms with a convex plan) and positioned from the data the
rest of the package already has: row 1 of the generated seating bowl (the stage
end) and the last row (the top of the bowl).

The OBJ shares the terrain and bench frame, so it needs no transform in Unreal:
centimetres, X east, Y north, Z up, centred on the same origin, Z measured from
the terrain's lowest elevation (pass the same ``base_elevation_m`` the benches
used; see ``benches.terrain_base_elevation``). Faces wind counter-clockwise seen
from outside, every face carries a UV index (Unreal's OBJ importer decimates
meshes without ``vt``) and the single ``Stage`` object is watertight.

Layout
------
The stage axis ``a`` runs horizontally from row 1's centre to the last row's
centre, i.e. from the stage toward the audience. The stage faces along ``a``.
Its front edge sits ``STAGE_FRONT_SETBACK_M`` downhill of row 1's centre, and
the deck extends away from the audience (``-a``). Lateral ``l`` is
perpendicular to ``a``. The deck top is row 1's DEM elevation minus
``STAGE_DROP_M``.

Dimension sources
-----------------
Every number is in ``DIMENSIONS`` with its source. ``kind`` is one of:

* ``published``: stated by a cited source;
* ``derived``: computed from published figures or the OSM/DEM data;
* ``estimate``: an honest rough estimate, not measured on site. **The venue does not publish the stage
  width/depth, and no source found gives them**, so those are rough estimates, not
  measured, and must be replaced if a tech pack or survey turns up.

Sources:

* IA Stage, "Red Rocks Amphitheater" (rigging contractor project page,
  https://www.iastage.com/red-rocks-amphitheater/): SkyDeck I-beam grid at
  40'2" above the stage floor; Y-columns meet at 36'10"; trusses 12 ft behind
  the downstage roof edge then 25 ft apart (3 trusses); footers 6 ft below the
  stage floor.
* Structure magazine, "New Stage Roof Rocks Iconic Venue"
  (https://www.structuremag.org/article/new-stage-roof-rocks-iconic-venue/):
  column count reduced from twelve to four.
* Red Rocks official history (https://www.redrocksonline.com/our-story/history/):
  about 100 ft rise from stage to the top row (already in ``venue.py``).
* OSM ``amenity=theatre`` footprint and the USGS 3DEP DEM, via ``venue.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .terrain_mesh import CM_PER_M
from .venue import Row

FT = 0.3048
# Structure magazine: the roof's columns were reduced from twelve to four.
COLUMN_COUNT = 4


@dataclass(frozen=True)
class Dimension:
    name: str
    value_m: float
    kind: str  # "published" | "derived" | "estimate"
    source: str


# SkyDeck I-beam grid at 40'2" above the stage floor (IA Stage).
ROOF_TOP_ABOVE_DECK_M = (40 + 2 / 12.0) * FT

# Roof truss stations: 12 ft behind the downstage edge, then 25 ft apart, three trusses.
_TRUSS_3_FROM_FRONT_M = (12 + 25 + 25) * FT

DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("stage_width", 24.4, "estimate",
              "Rough estimate, not measured (about 80 ft); no source found publishes it. Not OSM-derived."),
    Dimension("stage_depth", 21.3, "estimate",
              "Roof trusses reach 62 ft (19 m) behind the downstage edge (IA Stage) plus ~8 ft of "
              "backstage deck, a rough estimate, not measured: 70 ft."),
    Dimension("stage_front_setback", 3.0, "estimate",
              "Rough estimate, not measured: pit/aisle between the stage lip and bench row 1."),
    Dimension("stage_drop", 1.2, "estimate",
              "Rough estimate, not measured (about 4 ft): deck top below row 1's elevation, per issue #20."),
    Dimension("deck_thickness", 1.5, "estimate",
              "Blockout slab thickness; footers are 6 ft below the deck (IA Stage), the rest is buried."),
    Dimension("roof_top_above_deck", ROOF_TOP_ABOVE_DECK_M, "published",
              "SkyDeck I-beam grid at 40'2\" above the stage floor (IA Stage)."),
    Dimension("roof_thickness", 1.5, "estimate",
              "Estimate: roof truss + decking depth; IA Stage gives bay spacing, not truss depth."),
    Dimension("column_size", 0.9, "estimate",
              "Rough estimate, not measured: round/splayed column base about 3 ft across, modelled square."),
    Dimension("column_front_station", 12 * FT, "published",
              "Truss 1 is ~12 ft behind the downstage roof edge (IA Stage); front columns sit there."),
    Dimension("column_rear_station", _TRUSS_3_FROM_FRONT_M, "derived",
              "Truss 3: 12 + 25 + 25 ft behind the downstage roof edge (IA Stage); rear columns sit there."),
    Dimension("shell_thickness", 0.6, "estimate", "Estimate: back wall of the shell."),
    Dimension("shell_wing_length", 8.0, "estimate", "Rough estimate, not measured: splayed side walls."),
    Dimension("shell_wing_height", 6.0, "estimate", "Rough estimate, not measured."),
    Dimension("shell_wing_splay_degrees", 15.0, "estimate", "Estimate: outward splay of the side walls."),
    Dimension("retaining_wall_gap", 2.0, "estimate", "Estimate: service gap between shell back wall and retaining wall."),
    Dimension("retaining_wall_thickness", 0.9, "estimate", "Estimate: typical 3 ft concrete retaining wall, not measured."),
    Dimension("retaining_wall_height", 6.0, "estimate",
              "Rough estimate, not measured: height above the deck; the rock slope behind rises above this."),
    Dimension("retaining_wall_buried", 2.0, "estimate", "Estimate: depth below the deck so the wall meets terrain."),
    Dimension("retaining_wall_overhang", 3.0, "estimate", "Estimate: wall extends this far past each side of the stage."),
    Dimension("retaining_gusset_length", 6.0, "estimate", "Estimate: triangular return at each wall end."),
    Dimension("walkway_width", 4.0, "estimate", "Rough estimate, not measured: top-of-bowl promenade."),
    Dimension("walkway_thickness", 0.4, "estimate", "Blockout slab thickness."),
    Dimension("walkway_gap", 0.5, "estimate", "Estimate: clearance between the last bench and the walkway."),
)

_D = {d.name: d for d in DIMENSIONS}


def dim(name: str) -> float:
    return _D[name].value_m


STAGE_WIDTH_M = dim("stage_width")
STAGE_DEPTH_M = dim("stage_depth")
STAGE_FRONT_SETBACK_M = dim("stage_front_setback")
STAGE_DROP_M = dim("stage_drop")


@dataclass
class Part:
    """A vertical prism: convex plan polygon (world E, N metres) from ``z0`` to ``z1`` (metres above base)."""

    name: str
    plan: np.ndarray
    z0: float
    z1: float


@dataclass
class StageMesh:
    path: Path
    parts: dict[str, tuple[int, int]]  # name -> (first triangle, end triangle) in the OBJ
    vertices: int
    triangles: int
    base_elevation_m: float
    deck_top_elevation_m: float
    walkway_elevation_m: float
    front_edge_east: float = 0.0
    front_edge_north: float = 0.0
    axis: tuple[float, float] = field(default=(0.0, 1.0))


def stage_axis(rows: Sequence[Row]) -> np.ndarray:
    """Unit horizontal vector from row 1's centre toward the last row's centre."""
    if len(rows) < 2:
        raise ValueError("need at least two rows to find the stage axis")
    d = np.array([rows[-1].centre_east - rows[0].centre_east,
                  rows[-1].centre_north - rows[0].centre_north])
    n = float(np.hypot(*d))
    if n < 1e-6:
        raise ValueError("first and last rows coincide")
    return d / n


def _rect(frame_o, a, l, x0, x1, y0, y1) -> np.ndarray:
    """Plan corners of a rectangle in stage coordinates (x lateral, y along a)."""
    return np.array([frame_o + x * l + y * a for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])


def _poly(frame_o, a, l, pts) -> np.ndarray:
    return np.array([frame_o + x * l + y * a for x, y in pts])


DECK_CLEARANCE_M = 0.05  # deck top above the highest terrain under it
DECK_SAMPLE_M = 1.0  # terrain sample spacing over the deck footprint
# Only the front of the deck has to clear the terrain: the rear is cut into the
# hillside, which is what the retaining wall behind the shell holds back.
DECK_CLEAR_FRACTION = 0.5


def deck_top_elevation(
    rows: Sequence[Row], ground: Callable[[float, float], float] | None = None
) -> float:
    """Deck top elevation (m above sea level).

    Row 1's elevation minus ``STAGE_DROP_M``, raised if needed so the front
    ``DECK_CLEAR_FRACTION`` of the deck clears the terrain under it: the DEM bulges at the stage end, and a deck
    below the terrain is invisible and buries the player's spawn.
    """
    top = rows[0].elevation_m - STAGE_DROP_M
    if ground is None:
        return top
    a = stage_axis(rows)
    l = np.array([a[1], -a[0]])
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - a * STAGE_FRONT_SETBACK_M
    hw = STAGE_WIDTH_M / 2.0
    highest = max(
        ground(*(origin + x * l + y * a))
        for x in np.arange(-hw, hw + 1e-9, DECK_SAMPLE_M)
        for y in np.arange(-STAGE_DEPTH_M * DECK_CLEAR_FRACTION, 1e-9, DECK_SAMPLE_M)
    )
    return max(top, highest + DECK_CLEARANCE_M)


def stage_parts(
    rows: Sequence[Row],
    polylines: Sequence[np.ndarray],
    base_elevation_m: float,
    ground: Callable[[float, float], float] | None = None,
) -> list[Part]:
    """Lay out every blockout part from the seating-bowl rows.

    ``ground(east, north)`` (metres) is the terrain the deck must clear; without
    it the deck sits ``STAGE_DROP_M`` below row 1 regardless of the terrain.
    """
    if len(rows) != len(polylines):
        raise ValueError(f"{len(rows)} rows but {len(polylines)} polylines")
    a = stage_axis(rows)
    l = np.array([a[1], -a[0]])
    first, last = rows[0], rows[-1]
    origin = np.array([first.centre_east, first.centre_north]) - a * STAGE_FRONT_SETBACK_M

    W, D = STAGE_WIDTH_M, STAGE_DEPTH_M
    deck_top = deck_top_elevation(rows, ground) - base_elevation_m
    deck_bot = deck_top - dim("deck_thickness")
    roof_top = deck_top + ROOF_TOP_ABOVE_DECK_M
    roof_bot = roof_top - dim("roof_thickness")
    hw = W / 2.0

    parts: list[Part] = []
    parts.append(Part("deck", _rect(origin, a, l, -hw, hw, -D, 0.0), deck_bot, deck_top))
    parts.append(Part("roof", _rect(origin, a, l, -hw, hw, -D, 0.0), roof_bot, roof_top))

    # Four columns, in from each side, under trusses 1 and 3.
    cs = dim("column_size")
    for sx in (-1, 1):
        for station in (dim("column_front_station"), dim("column_rear_station")):
            cx = sx * (hw - cs)
            y = -station
            parts.append(Part(
                f"column_{'e' if sx > 0 else 'w'}_{'front' if station == dim('column_front_station') else 'rear'}",
                _rect(origin, a, l, cx - cs / 2, cx + cs / 2, y - cs / 2, y + cs / 2),
                deck_top, roof_bot))

    # Shell: back wall plus a splayed wing wall on each side (parallelogram wedge in plan).
    t = dim("shell_thickness")
    parts.append(Part("shell_back", _rect(origin, a, l, -hw, hw, -D, -D + t), deck_top, roof_bot))
    wl = dim("shell_wing_length")
    off = wl * math.tan(math.radians(dim("shell_wing_splay_degrees")))
    wh = dim("shell_wing_height")
    for sx, nm in ((1, "e"), (-1, "w")):
        pts = [(sx * hw, -D + t), (sx * (hw + t), -D + t),
               (sx * (hw + t + off), -D + t + wl), (sx * (hw + off), -D + t + wl)]
        parts.append(Part(f"shell_wing_{nm}", _poly(origin, a, l, pts), deck_top, deck_top + wh))

    # Retaining wall behind the shell, with a triangular return at each end.
    rw_y1 = -D - dim("retaining_wall_gap")
    rw_y0 = rw_y1 - dim("retaining_wall_thickness")
    rw_hw = hw + dim("retaining_wall_overhang")
    rz0 = deck_top - dim("retaining_wall_buried")
    rz1 = deck_top + dim("retaining_wall_height")
    parts.append(Part("retaining_wall", _rect(origin, a, l, -rw_hw, rw_hw, rw_y0, rw_y1), rz0, rz1))
    g = dim("retaining_gusset_length")
    for sx, nm in ((1, "e"), (-1, "w")):
        pts = [(sx * rw_hw, rw_y0), (sx * rw_hw, rw_y1), (sx * (rw_hw + g), rw_y1)]
        parts.append(Part(f"retaining_gusset_{nm}", _poly(origin, a, l, pts), rz0, rz1))

    # Top-of-bowl walkway behind the last row, along its chord.
    chord = np.asarray(polylines[-1], dtype=float)[:, :2]
    p0, p1 = chord[0], chord[-1]
    c = p1 - p0
    c = c / float(np.hypot(*c))
    n = np.array([-c[1], c[0]])
    if n @ a < 0:
        n = -n
    bench_half = 0.45  # half of benches.SEAT_DEPTH_M
    n0 = bench_half + dim("walkway_gap")
    n1 = n0 + dim("walkway_width")
    wz1 = last.elevation_m - base_elevation_m
    walk = np.array([p0 + n * n0, p1 + n * n0, p1 + n * n1, p0 + n * n1])
    parts.append(Part("walkway", walk, wz1 - dim("walkway_thickness"), wz1))
    return parts


def _ccw(plan: np.ndarray) -> np.ndarray:
    x, y = plan[:, 0], plan[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area == 0:
        raise ValueError("degenerate plan polygon")
    return plan if area > 0 else plan[::-1].copy()


def _is_convex(plan: np.ndarray) -> bool:
    n = len(plan)
    for i in range(n):
        e1 = plan[(i + 1) % n] - plan[i]
        e2 = plan[(i + 2) % n] - plan[(i + 1) % n]
        if e1[0] * e2[1] - e1[1] * e2[0] < -1e-9:
            return False
    return True


def prism(plan: np.ndarray, z0: float, z1: float):
    """Vertical prism over a convex plan. Returns (verts[N,3], tris, uvs[M,2], tri_uvs).

    Triangles wind counter-clockwise seen from outside. UVs are in metres:
    side faces use perimeter distance along U and height along V; caps use the
    plan measured along/across the polygon's first edge.
    """
    if z1 <= z0:
        raise ValueError("prism needs z1 > z0")
    plan = _ccw(np.asarray(plan, dtype=float)[:, :2])
    if len(plan) < 3 or not _is_convex(plan):
        raise ValueError("prism plan must be a convex polygon with at least 3 corners")
    n = len(plan)
    verts = np.array([(*p, z0) for p in plan] + [(*p, z1) for p in plan])  # bottom 0..n-1, top n..2n-1

    e = plan[1] - plan[0]
    u_axis = e / float(np.hypot(*e))
    v_axis = np.array([-u_axis[1], u_axis[0]])
    cap_uv = (plan - plan[0]) @ np.stack([u_axis, v_axis], axis=1)

    uvs: list[tuple[float, float]] = [tuple(p) for p in cap_uv]  # 0..n-1 cap uvs
    tris: list[tuple[int, int, int]] = []
    tuvs: list[tuple[int, int, int]] = []

    for i in range(1, n - 1):
        tris.append((n + 0, n + i, n + i + 1))      # top, +Z
        tuvs.append((0, i, i + 1))
        tris.append((0, i + 1, i))                  # bottom, -Z
        tuvs.append((0, i + 1, i))

    s = 0.0
    h = z1 - z0
    for i in range(n):
        j = (i + 1) % n
        length = float(np.hypot(*(plan[j] - plan[i])))
        base = len(uvs)
        uvs += [(s, 0.0), (s + length, 0.0), (s + length, h), (s, h)]
        s += length
        tris.append((i, j, n + j))
        tuvs.append((base, base + 1, base + 2))
        tris.append((i, n + j, n + i))
        tuvs.append((base, base + 2, base + 3))
    return verts, tris, np.array(uvs), tuvs


def export_stage_obj(
    rows: Sequence[Row],
    polylines: Sequence[np.ndarray],
    out_path: Path,
    base_elevation_m: float,
    ground: Callable[[float, float], float] | None = None,
) -> StageMesh:
    """Write the blockout as one OBJ object, ``Stage``.

    Args:
        rows: venue rows (``venue.rows``), row 1 = stage end, last = top.
        polylines: per-row (N, 2) local metre polylines (``benches.row_polylines``).
        out_path: where to write ``Stage.obj``.
        base_elevation_m: Z=0 elevation, the terrain minimum
            (``benches.terrain_base_elevation``).
        ground: terrain height function the deck must clear (e.g.
            ``terrain_mesh.TerrainSurface``); optional.
    """
    parts = stage_parts(rows, polylines, base_elevation_m, ground)

    all_v: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    faces: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    v_off = uv_off = n_tris = 0
    comments: list[str] = []
    for part in parts:
        verts, tris, uvs, tuvs = prism(part.plan, part.z0, part.z1)
        all_v.append(verts)
        all_uv.append(uvs)
        start = n_tris
        comments.append((len(faces), f"# part {part.name}"))
        for t, tu in zip(tris, tuvs):
            faces.append("f " + " ".join(f"{v_off + v + 1}/{uv_off + u + 1}" for v, u in zip(t, tu)))
        n_tris += len(tris)
        ranges[part.name] = (start, n_tris)
        v_off += len(verts)
        uv_off += len(uvs)

    verts_cm = np.vstack(all_v) * CM_PER_M
    lines = [
        "# Red Rocks stage blockout: deck, roof, columns, shell, retaining walls, walkway",
        f"# {len(parts)} parts, {n_tris} triangles, centimetres, base {base_elevation_m:.3f} m",
        "# dimensions and sources: see redrocks.stage.DIMENSIONS",
        "o Stage",
    ]
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in verts_cm]
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in np.vstack(all_uv)]
    # Part markers go between faces as comments so they never disturb parsers.
    out_faces: list[str] = []
    marks = {idx: text for idx, text in comments}
    for i, f in enumerate(faces):
        if i in marks:
            out_faces.append(marks[i])
        out_faces.append(f)
    lines += out_faces

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    a = stage_axis(rows)
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - a * STAGE_FRONT_SETBACK_M
    return StageMesh(
        path=out_path,
        parts=ranges,
        vertices=len(verts_cm),
        triangles=n_tris,
        base_elevation_m=base_elevation_m,
        deck_top_elevation_m=deck_top_elevation(rows, ground),
        walkway_elevation_m=rows[-1].elevation_m,
        front_edge_east=float(origin[0]),
        front_edge_north=float(origin[1]),
        axis=(float(a[0]), float(a[1])),
    )


def dimensions_table() -> str:
    """Markdown table of every dimension with its kind and source."""
    out = ["| Dimension | Value | Kind | Source |", "|---|---|---|---|"]
    for d in DIMENSIONS:
        out.append(f"| {d.name} | {d.value_m:g} | {d.kind} | {d.source} |")
    return "\n".join(out)
