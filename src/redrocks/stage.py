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
* ``measured``: read off the ml0w photogrammetry scan (``tools/measure_stage_scan.py``).
  The scan has no units, so its scale was found from published figures (about 0.52 m
  per scan unit, within a few percent); each measured source says how and how
  confident it is;
* ``estimate``: an honest rough estimate, not measured on site or on the scan.

Sources:

* ml0w, "Red Rocks, Colorado" photogrammetry (Sketchfab, CC BY 4.0, drone data from
  DroneMapper), ``assets/scan``: stage deck, roof and the wall behind it.
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

from shapely.geometry import LineString

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
    kind: str  # "published" | "derived" | "measured" | "estimate"
    source: str


# SkyDeck I-beam grid at 40'2" above the stage floor (IA Stage). The roof slab sits on it.
ROOF_GRID_ABOVE_DECK_M = (40 + 2 / 12.0) * FT
# Y-columns fork at 36'10" above the stage floor (IA Stage) before meeting the 40'2" grid.
Y_COLUMN_FORK_ABOVE_DECK_M = (36 + 10 / 12.0) * FT

# Roof truss stations: 12 ft behind the downstage edge, then 25 ft apart, three trusses.
_TRUSS_2_FROM_FRONT_M = (12 + 25) * FT
_TRUSS_3_FROM_FRONT_M = (12 + 25 + 25) * FT

# Two steps per seating row at the row ends keep each rise around 22 cm (< UE MaxStepHeight).
STEPS_PER_ROW = 2
# With the terraced seating's tread levels (redrocks.seating): the walkway sits
# this far under the top tread, so the tread's curved back edge shows over it.
WALKWAY_UNDER_TREAD_M = 0.02
# Steel handrail on the walkway parapet (the photo from the side shows a pipe
# rail along the top of the bowl): top rail at the usual guard height.
HANDRAIL_HEIGHT_M = 1.07
HANDRAIL_POST_SPACING_M = 2.0
HANDRAIL_POST_M = 0.06
HANDRAIL_RAIL_M = 0.05
SKYDECK_BEAM_COUNT = 3

_SCAN = "ml0w scan (CC BY 4.0), "

DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("stage_width", 26.3, "measured",
              _SCAN + "roof plan width: z extent of the connected up-facing roof surface, 51 units x 0.516 m. "
              "Medium-low confidence (+-1.5 m): the roof is only partly reconstructed and the deck under it is "
              "hidden; the flat deck seen from above is at least 20.9 m wide. Was an 80 ft estimate."),
    Dimension("stage_depth", 27.9, "measured",
              _SCAN + "deck front lip to where the flat deck ends at the wall behind, along the axis on a clean "
              "strip: 54 units x 0.516 m. Medium confidence (+-1.5 m): the lip is where the ramp's gutter meets the "
              "deck, the rear is where the profile leaves the deck by 1.5 units. Was 21.3 m (roof span plus 8 ft)."),
    Dimension("stage_front_setback", 3.0, "estimate",
              "Not changed: the scan puts row 1 about 3.5 m (+-1.5) uphill of the deck lip, consistent with 3 m, "
              "but row 1's own position is not resolved in the scan."),
    Dimension("stage_drop", 0.2, "measured",
              _SCAN + "deck height against the row-1 tread on the along-axis profile: level to within 0.4 m. "
              "Low confidence (the ramp ends in a gutter 1 m lower than the deck). Was a 4 ft estimate."),
    Dimension("deck_thickness", 1.5, "estimate",
              "Blockout slab thickness; footers are 6 ft below the deck (IA Stage), the rest is buried."),
    Dimension("roof_grid_above_deck", ROOF_GRID_ABOVE_DECK_M, "published",
              "SkyDeck I-beam grid at 40'2\" above the stage floor (IA Stage). The scan agrees: the roof's "
              "underside surface is 12.33 m above the deck (+-0.6 m with the scale)."),
    Dimension("roof_thickness", 2.1, "measured",
              _SCAN + "roof top surface 14.46 m above the deck minus the underside surface 12.33 m (which matches "
              "the published grid). Medium-low confidence (+-0.7 m): blobby photogrammetry. Was a 1.5 m estimate. "
              "The roof now sits on the published grid height."),
    Dimension("roof_depth", 21.9, "measured",
              _SCAN + "x extent of the up-facing roof surface, 42.5 units x 0.516 m, starting within 0.5 m of the "
              "deck front lip. Medium confidence (+-1.5 m). The deck runs about 6 m further back than the roof."),
    Dimension("column_size", 0.9, "estimate",
              "Rough estimate, not measured: round/splayed column base about 3 ft across, modelled square."),
    Dimension("column_front_station", 12 * FT, "published",
              "Truss 1 is ~12 ft behind the downstage roof edge (IA Stage); front columns sit there."),
    Dimension("truss_mid_station", _TRUSS_2_FROM_FRONT_M, "derived",
              "Truss 2: 12 + 25 ft behind the downstage roof edge (IA Stage: three trusses 25 ft apart)."),
    Dimension("column_rear_station", _TRUSS_3_FROM_FRONT_M, "derived",
              "Truss 3: 12 + 25 + 25 ft behind the downstage roof edge (IA Stage); rear columns sit there."),
    Dimension("y_column_fork_above_deck", Y_COLUMN_FORK_ABOVE_DECK_M, "published",
              "Y-columns meet at 36'10\" above the stage floor (IA Stage) before splaying to the 40'2\" grid."),
    Dimension("y_arm_spread", 2.2, "estimate",
              "Estimate: fore-and-aft spread of the Y-column arms from the 36'10\" fork up to the roof truss."),
    Dimension("truss_depth", 0.9, "estimate",
              "Estimate: structural depth of the three primary roof trusses below the SkyDeck grid."),
    Dimension("truss_width", 0.45, "estimate",
              "Estimate: chord width of the primary roof trusses and SkyDeck purlin beams."),
    Dimension("shell_thickness", 0.6, "estimate", "Estimate: back wall of the shell. The scan shows one rear face, not a separate shell."),
    Dimension("shell_wing_length", 8.0, "estimate", "Rough estimate, not measured: splayed side walls (not resolved in the scan)."),
    Dimension("shell_wing_height", 6.0, "estimate", "Rough estimate, not measured."),
    Dimension("shell_wing_splay_degrees", 15.0, "estimate", "Estimate: outward splay of the side walls."),
    Dimension("shell_canopy_depth", 2.4, "estimate",
              "Estimate: acoustic hood projecting forward from the top of the rear stage shell."),
    Dimension("shell_canopy_thickness", 0.5, "estimate",
              "Estimate: slab thickness of the rear stage shell acoustic hood."),
    Dimension("retaining_wall_gap", 2.1, "measured",
              _SCAN + "rear face of the wall minus where the flat deck ends: 4 units x 0.516 m. Low confidence "
              "(+-1 m); close to the old 2 m estimate."),
    Dimension("retaining_wall_thickness", 0.9, "estimate", "Estimate: typical 3 ft concrete retaining wall, not measured."),
    Dimension("retaining_wall_height", 12.9, "measured",
              _SCAN + "near-vertical rear face from the deck to its crest, 25 units x 0.516 m, on a clean strip "
              "behind the stage. Low confidence (+-2 m): the scan cannot separate a built wall from the rock face "
              "it stands against, so this is the whole face. Was a 6 m estimate."),
    Dimension("retaining_wall_buried", 2.0, "estimate", "Estimate: depth below the deck so the wall meets terrain."),
    Dimension("retaining_wall_overhang", 3.0, "estimate", "Estimate: wall extends this far past each side of the stage."),
    Dimension("retaining_gusset_length", 6.0, "estimate", "Estimate: triangular return at each wall end."),
    Dimension("foh_width", 6.0, "derived",
              "FOH sound/lighting booth width in the lower bowl, from OSM way 229359709 (~6.0 x 6.1 m)."),
    Dimension("foh_depth", 3.2, "estimate",
              "Estimate: front-to-back depth of the FOH mix booth platform in the lower seating bowl."),
    Dimension("foh_wall_height", 1.05, "estimate",
              "Estimate: parapet wall height enclosing the FOH mix booth."),
    Dimension("foh_wall_thickness", 0.25, "estimate",
              "Estimate: concrete parapet thickness around the FOH mix booth."),
    Dimension("stair_width", 1.8, "estimate",
              "Estimate: width of the stone/concrete side aisle stairs at the two ends of the bench rows."),
    Dimension("stair_tread_depth", 0.45, "estimate",
              "Estimate: front-to-back tread depth of each half-row step on the bowl side stairs."),
    Dimension("stair_wall_thickness", 0.35, "estimate",
              "Estimate: thickness of the stepped stone cheek walls flanking the outer edges of the bowl stairs."),
    Dimension("stair_wall_height", 0.40, "estimate",
              "Estimate: curb height of the outer cheek wall above each row's upper stair tread."),
    Dimension("walkway_width", 4.0, "estimate",
              "Rough estimate, not measured: top-of-bowl promenade. The scan shows a flat plateau behind the top "
              "row with no edge that fixes a walkway width."),
    Dimension("walkway_thickness", 0.4, "estimate", "Blockout slab thickness."),
    Dimension("walkway_gap", 0.5, "estimate", "Estimate: clearance between the last bench and the walkway."),
    Dimension("walkway_parapet_height", 0.85, "estimate",
              "Estimate: low perimeter retaining/parapet wall along the rear edge of the top promenade."),
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
    handrail_path: Path | None = None
    handrail_triangles: int = 0


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


DECK_CLEARANCE_M = 0.05  # deck top above the ground under it
DECK_SAMPLE_M = 1.0  # ground sample spacing over the deck footprint
# Only the front of the deck has to clear the ground: the rear is cut into the
# hillside, which is what the retaining wall behind the shell holds back.
DECK_CLEAR_FRACTION = 0.5
# The deck clears this fraction of the ground samples under its front, not all
# of them: on the ml0w scan the real stage's wing walls and column bases stand
# a few metres proud along the deck's sides (the blockout's own walls and
# columns are there instead), and lifting the deck onto them would float it a
# metre or more over the scanned deck everywhere else. On a smooth DEM the
# highest few per cent are within centimetres of the highest.
DECK_CLEAR_QUANTILE = 0.75
WALKWAY_CLEARANCE_M = 0.05  # walkway top above the ground under it


def deck_top_elevation(
    rows: Sequence[Row], ground: Callable[[float, float], float] | None = None
) -> float:
    """Deck top elevation (m above sea level).

    Without ``ground``: row 1's elevation minus ``STAGE_DROP_M``. With it, row
    1's tread is the ground under row 1's centre (where its bench stands), the
    deck is ``STAGE_DROP_M`` below that, raised if needed so it clears
    ``DECK_CLEAR_QUANTILE`` of the ground under the front
    ``DECK_CLEAR_FRACTION`` of the deck: a deck below the ground is invisible
    and buries the player's spawn.
    """
    if ground is None:
        return rows[0].elevation_m - STAGE_DROP_M
    top = ground(rows[0].centre_east, rows[0].centre_north) - STAGE_DROP_M
    a = stage_axis(rows)
    l = np.array([a[1], -a[0]])
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - a * STAGE_FRONT_SETBACK_M
    hw = STAGE_WIDTH_M / 2.0
    under = [
        ground(*(origin + x * l + y * a))
        for x in np.arange(-hw, hw + 1e-9, DECK_SAMPLE_M)
        for y in np.arange(-STAGE_DEPTH_M * DECK_CLEAR_FRACTION, 1e-9, DECK_SAMPLE_M)
    ]
    return max(top, float(np.quantile(under, DECK_CLEAR_QUANTILE)) + DECK_CLEARANCE_M)


def _walkway_heights(plan: np.ndarray, top_m: float, ground) -> tuple[float, float]:
    """(bottom, top) of the walkway slab, absolute metres.

    Without ``ground`` it is ``walkway_thickness`` thick with its top at
    ``top_m`` (the last row's elevation). With it, the top clears
    ``DECK_CLEAR_QUANTILE`` of the ground under the slab and the bottom
    reaches the lowest ground there, so the slab neither floats nor buries.
    """
    if ground is None:
        return top_m - dim("walkway_thickness"), top_m
    p0, p1, _, p3 = plan
    under = [
        ground(*(p0 + s * (p1 - p0) + t * (p3 - p0)))
        for s in np.linspace(0.0, 1.0, max(2, int(np.hypot(*(p1 - p0)) / DECK_SAMPLE_M) + 1))
        for t in np.linspace(0.0, 1.0, max(2, int(np.hypot(*(p3 - p0)) / DECK_SAMPLE_M) + 1))
    ]
    top = float(np.quantile(under, DECK_CLEAR_QUANTILE)) + WALKWAY_CLEARANCE_M
    return min(top - dim("walkway_thickness"), min(under)), top


def stage_parts(
    rows: Sequence[Row],
    polylines: Sequence[np.ndarray],
    base_elevation_m: float,
    ground: Callable[[float, float], float] | None = None,
    levels: Sequence[float] | None = None,
) -> list[Part]:
    """Lay out every blockout part from the seating-bowl rows.

    ``ground(east, north)`` (metres) is the ground the benches stand on (the
    ml0w scan's, or the terrain's); the deck and the walkway sit on it. Without
    it the deck sits ``STAGE_DROP_M`` below row 1's elevation and the walkway
    at the last row's, regardless of the terrain.

    ``levels`` are the terraced seating's tread elevations, one per row
    (``seating.tread_levels``). With them the side stairs climb from each tread
    to the next across the row's band, the stage steps reach row 1's tread and
    the walkway sits just under the top tread.
    """
    if len(rows) != len(polylines):
        raise ValueError(f"{len(rows)} rows but {len(polylines)} polylines")
    if levels is not None and len(levels) != len(rows):
        raise ValueError(f"{len(rows)} rows but {len(levels)} tread levels")
    a = stage_axis(rows)
    l = np.array([a[1], -a[0]])
    first, last = rows[0], rows[-1]
    origin = np.array([first.centre_east, first.centre_north]) - a * STAGE_FRONT_SETBACK_M

    W, D = STAGE_WIDTH_M, STAGE_DEPTH_M
    deck_top = deck_top_elevation(rows, ground) - base_elevation_m
    deck_bot = deck_top - dim("deck_thickness")
    roof_bot = deck_top + ROOF_GRID_ABOVE_DECK_M  # the slab sits on the SkyDeck grid
    roof_top = roof_bot + dim("roof_thickness")
    RD = dim("roof_depth")
    hw = W / 2.0

    parts: list[Part] = []
    parts.append(Part("deck", _rect(origin, a, l, -hw, hw, -D, 0.0), deck_bot, deck_top))
    parts.append(Part("roof", _rect(origin, a, l, -hw, hw, -RD, 0.0), roof_bot, roof_top))

    # Four columns, in from each side, under trusses 1 and 3, plus their 36'10" Y-column forks.
    cs = dim("column_size")
    fork_z = deck_top + dim("y_column_fork_above_deck")
    arm_spread = dim("y_arm_spread")
    for sx in (-1, 1):
        side = "e" if sx > 0 else "w"
        for station in (dim("column_front_station"), dim("column_rear_station")):
            pos = "front" if station == dim("column_front_station") else "rear"
            cx = sx * (hw - cs)
            y = -station
            parts.append(Part(
                f"column_{side}_{pos}",
                _rect(origin, a, l, cx - cs / 2, cx + cs / 2, y - cs / 2, y + cs / 2),
                deck_top, roof_bot))
            # Y-column splayed arms above the 36'10" fork meeting the 40'2" SkyDeck grid.
            parts.append(Part(
                f"y_arm_{side}_{pos}_fore",
                _rect(origin, a, l, cx - cs * 0.4, cx + cs * 0.4, y + cs / 2 + 0.05, y + cs / 2 + arm_spread * 0.5),
                fork_z, roof_bot))
            parts.append(Part(
                f"y_arm_{side}_{pos}_aft",
                _rect(origin, a, l, cx - cs * 0.4, cx + cs * 0.4, y - cs / 2 - arm_spread * 0.5, y - cs / 2 - 0.05),
                fork_z, roof_bot))

    # Three primary roof trusses (12 ft, 37 ft, 62 ft behind downstage edge) and SkyDeck purlin beams.
    td = dim("truss_depth")
    tw = dim("truss_width")
    truss_span = hw - 1.6 * cs
    truss_stations = (
        ("truss_1", dim("column_front_station")),
        ("truss_2", dim("truss_mid_station")),
        ("truss_3", dim("column_rear_station")),
    )
    for tname, tstat in truss_stations:
        parts.append(Part(
            tname,
            _rect(origin, a, l, -truss_span, truss_span, -tstat - tw / 2, -tstat + tw / 2),
            roof_bot - td, roof_bot))

    # Longitudinal SkyDeck I-beam purlins running between the trusses.
    s1, s2, s3 = (st for _, st in truss_stations)
    for bi, bx in enumerate((-truss_span * 0.6, 0.0, truss_span * 0.6), start=1):
        parts.append(Part(
            f"skydeck_beam_{bi}_front",
            _rect(origin, a, l, bx - tw / 2, bx + tw / 2, -s2 + tw / 2 + 0.05, -s1 - tw / 2 - 0.05),
            roof_bot - td * 0.55, roof_bot))
        parts.append(Part(
            f"skydeck_beam_{bi}_rear",
            _rect(origin, a, l, bx - tw / 2, bx + tw / 2, -s3 + tw / 2 + 0.05, -s2 - tw / 2 - 0.05),
            roof_bot - td * 0.55, roof_bot))

    # Shell: back wall, overhead acoustic reflector canopy, splayed outer wing walls, and inner baffles.
    t = dim("shell_thickness")
    parts.append(Part("shell_back", _rect(origin, a, l, -hw, hw, -D, -D + t), deck_top, roof_bot))
    canopy_d = dim("shell_canopy_depth")
    canopy_t = dim("shell_canopy_thickness")
    parts.append(Part(
        "shell_canopy",
        _rect(origin, a, l, -hw * 0.85, hw * 0.85, -D + t, -D + t + canopy_d),
        roof_bot - td - canopy_t - 0.2, roof_bot - td - 0.2))
    wl = dim("shell_wing_length")
    off = wl * math.tan(math.radians(dim("shell_wing_splay_degrees")))
    wh = dim("shell_wing_height")
    for sx, nm in ((1, "e"), (-1, "w")):
        pts = [(sx * hw, -D + t), (sx * (hw + t), -D + t),
               (sx * (hw + t + off), -D + t + wl), (sx * (hw + off), -D + t + wl)]
        parts.append(Part(f"shell_wing_{nm}", _poly(origin, a, l, pts), deck_top, deck_top + wh))
        # Inner stepped acoustic baffle wall inside the shell wing.
        b_wl = wl * 0.65
        b_off = b_wl * math.tan(math.radians(dim("shell_wing_splay_degrees") * 0.75))
        bx0 = hw * 0.78
        b_pts = [(sx * bx0, -D + t + 0.4), (sx * (bx0 + t), -D + t + 0.4),
                 (sx * (bx0 + t + b_off), -D + t + 0.4 + b_wl), (sx * (bx0 + b_off), -D + t + 0.4 + b_wl)]
        parts.append(Part(f"shell_baffle_{nm}", _poly(origin, a, l, b_pts), deck_top, deck_top + wh * 0.8))

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

    # Downstage side access stairs bridging the stage deck corners toward Row 1's side aisles.
    stair_w = dim("stair_width")
    tread_d = dim("stair_tread_depth")
    row1_z = first.elevation_m - base_elevation_m
    row1_target = levels[0] - base_elevation_m if levels is not None else row1_z + 0.45
    for sx, nm in ((1, "e"), (-1, "w")):
        for step_idx in (1, 2):
            frac = step_idx / 2.0
            sz1 = deck_top + frac * max(0.35, row1_target - deck_top)
            sz0 = min(deck_bot, sz1 - 0.6)
            sy0 = (step_idx - 1) * tread_d
            sy1 = step_idx * tread_d
            parts.append(Part(
                f"stage_step_{nm}_{step_idx}",
                _rect(origin, a, l, sx * (hw - stair_w), sx * hw, sy0, sy1),
                sz0, sz1))

    # Front-of-House (FOH) sound & lighting booth in the lower bowl (around Row 8, per OSM way 229359709).
    if len(rows) >= 8:
        foh_row = rows[min(7, len(rows) - 1)]
        foh_c = np.array([foh_row.centre_east, foh_row.centre_north])
        foh_hw = dim("foh_width") / 2.0
        foh_d = dim("foh_depth")
        foh_wt = dim("foh_wall_thickness")
        foh_wh = dim("foh_wall_height")
        foh_ground = (
            float(ground(foh_row.centre_east, foh_row.centre_north))
            if ground is not None else foh_row.elevation_m
        ) - base_elevation_m
        foh_top = foh_ground + 0.50
        foh_bot = foh_ground - 0.80
        parts.append(Part("foh_platform", _rect(foh_c, a, l, -foh_hw, foh_hw, -foh_d / 2, foh_d / 2), foh_bot, foh_top))
        parts.append(Part(
            "foh_wall_front",
            _rect(foh_c, a, l, -foh_hw, foh_hw, -foh_d / 2, -foh_d / 2 + foh_wt),
            foh_top, foh_top + foh_wh))
        for sx, nm in ((1, "e"), (-1, "w")):
            x0 = sx * (foh_hw - foh_wt) if sx > 0 else -foh_hw
            x1 = sx * foh_hw if sx > 0 else -foh_hw + foh_wt
            parts.append(Part(
                f"foh_wall_{nm}",
                _rect(foh_c, a, l, x0, x1, -foh_d / 2 + foh_wt + 0.02, foh_d / 2),
                foh_top, foh_top + foh_wh))

    # End-of-row side aisle stairs & stepped outer cheek walls on both ends of every seating row.
    bench_half = 0.45  # half of benches.SEAT_DEPTH_M
    bench_h = 0.45     # benches.SEAT_HEIGHT_M
    wall_t = dim("stair_wall_thickness")
    wall_h = dim("stair_wall_height")
    stair_gap = 0.12
    shapes = [LineString(np.asarray(p, dtype=float)[:, :2]) for p in polylines]
    for idx, (row, line) in enumerate(zip(rows, polylines)):
        pts = np.asarray(line, dtype=float)[:, :2]
        if len(pts) < 2:
            continue
        # Two ends of the bench row: start (pts[0], outward along pts[0]-pts[1]) and end (pts[-1], outward along pts[-1]-pts[-2]).
        ends = (
            ("start", pts[0], pts[0] - pts[1]),
            ("end", pts[-1], pts[-1] - pts[-2]),
        )
        for side_tag, p_end, out_vec in ends:
            olen = float(np.hypot(*out_vec))
            u_out = out_vec / olen if olen > 1e-6 else (-l if side_tag == "start" else l)
            n_up = np.array([-u_out[1], u_out[0]])
            if n_up @ a < 0:
                n_up = -n_up
            end_ground = (
                float(ground(float(p_end[0]), float(p_end[1])))
                if ground is not None else row.elevation_m
            ) - base_elevation_m
            if levels is None:
                # Two steps per row (each ~22.5 cm rise) so a character walks up smoothly without jumping.
                lo, hi = end_ground, end_ground + bench_h
                band0, band = -bench_half, STEPS_PER_ROW * tread_d
            else:
                # Terraces: from the tread below (row 1: the ground at its end)
                # to this row's, across the band from its riser to the next row.
                hi = levels[idx] - base_elevation_m
                lo = levels[idx - 1] - base_elevation_m if idx > 0 else min(end_ground, hi - bench_h)
                near = shapes[idx + 1] if idx + 1 < len(shapes) else shapes[idx - 1] if idx else None
                band0, band = 0.0, float(shapes[idx].distance(near)) if near is not None else 2 * tread_d
            z_base = min(end_ground, lo) - 0.60
            for step_k in range(1, STEPS_PER_ROW + 1):
                y0 = band0 + (step_k - 1) * band / STEPS_PER_ROW
                y1 = band0 + step_k * band / STEPS_PER_ROW
                z_step_top = lo + (step_k / STEPS_PER_ROW) * (hi - lo)
                parts.append(Part(
                    f"stair_{side_tag}_r{row.index}_s{step_k}",
                    _rect(p_end, n_up, u_out, stair_gap, stair_gap + stair_w, y0, y1),
                    z_base, z_step_top))
            # Stepped stone cheek/planter wall flanking the outer edge of this row's stair treads.
            parts.append(Part(
                f"stair_wall_{side_tag}_r{row.index}",
                _rect(
                    p_end, n_up, u_out,
                    stair_gap + stair_w + 0.02, stair_gap + stair_w + 0.02 + wall_t,
                    band0, band0 + band,
                ),
                z_base, hi + wall_h))

    # Top-of-bowl walkway behind the last row, along its chord, plus a rear parapet wall.
    chord = np.asarray(polylines[-1], dtype=float)[:, :2]
    p0, p1 = chord[0], chord[-1]
    c = p1 - p0
    c = c / float(np.hypot(*c))
    n = np.array([-c[1], c[0]])
    if n @ a < 0:
        n = -n
    n0 = bench_half + dim("walkway_gap")
    n1 = n0 + dim("walkway_width")
    walk = np.array([p0 + n * n0, p1 + n * n0, p1 + n * n1, p0 + n * n1])
    wz0, wz1 = _walkway_heights(walk, last.elevation_m, ground)
    if levels is not None:
        wz1 = levels[-1] - WALKWAY_UNDER_TREAD_M
        wz0 = min(wz0, wz1 - dim("walkway_thickness"))
    parts.append(Part("walkway", walk, wz0 - base_elevation_m, wz1 - base_elevation_m))
    parapet_t = dim("stair_wall_thickness")
    parapet_h = dim("walkway_parapet_height")
    parapet = np.array([
        p0 + n * (n1 - parapet_t), p1 + n * (n1 - parapet_t),
        p1 + n * n1, p0 + n * n1,
    ])
    parts.append(Part("walkway_parapet", parapet, wz1 - base_elevation_m, wz1 - base_elevation_m + parapet_h))
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
    levels: Sequence[float] | None = None,
) -> StageMesh:
    """Write the blockout as one OBJ object, ``Stage``.

    Args:
        rows: venue rows (``venue.rows``), row 1 = stage end, last = top.
        polylines: per-row (N, 2) local metre polylines (``benches.row_polylines``).
        out_path: where to write ``Stage.obj``.
        base_elevation_m: Z=0 elevation, the terrain minimum
            (``benches.terrain_base_elevation``).
        ground: the ground the benches stand on, which the deck and walkway
            sit on (``scan_blend.ScanGround`` or ``terrain_mesh.TerrainSurface``);
            optional, see ``stage_parts``.
    """
    parts = stage_parts(rows, polylines, base_elevation_m, ground, levels)

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

    handrail_path = out_path.parent / "Handrail.obj"
    handrail_tris = _write_handrail(parts, handrail_path, base_elevation_m)

    a = stage_axis(rows)
    origin = np.array([rows[0].centre_east, rows[0].centre_north]) - a * STAGE_FRONT_SETBACK_M
    return StageMesh(
        path=out_path,
        parts=ranges,
        vertices=len(verts_cm),
        triangles=n_tris,
        base_elevation_m=base_elevation_m,
        deck_top_elevation_m=deck_top_elevation(rows, ground),
        walkway_elevation_m=next(p.z1 for p in parts if p.name == "walkway") + base_elevation_m,
        front_edge_east=float(origin[0]),
        front_edge_north=float(origin[1]),
        axis=(float(a[0]), float(a[1])),
        handrail_path=handrail_path,
        handrail_triangles=handrail_tris,
    )


def handrail_parts(parts: Sequence[Part]) -> list[Part]:
    """Steel posts every ``HANDRAIL_POST_SPACING_M`` on the walkway parapet, and a top rail.

    The parapet's plan runs (p0, p1) along its bowl-side face and (p1, p0) back
    along its rear face; the rail follows its middle, ``HANDRAIL_HEIGHT_M`` over
    the walkway.
    """
    walkway = next(p for p in parts if p.name == "walkway")
    parapet = next(p for p in parts if p.name == "walkway_parapet")
    a0, a1, b1, b0 = (np.asarray(v, dtype=float) for v in parapet.plan)
    m0, m1 = (a0 + b0) / 2.0, (a1 + b1) / 2.0
    length = float(np.hypot(*(m1 - m0)))
    along = (m1 - m0) / length
    across = np.array([-along[1], along[0]])
    top = walkway.z1 + HANDRAIL_HEIGHT_M

    def box(c0, c1, w):
        h = across * (w / 2.0)
        return np.array([c0 - h, c1 - h, c1 + h, c0 + h])

    out = []
    for k in range(int(length // HANDRAIL_POST_SPACING_M) + 1):
        # The last post stops at the end of the parapet rather than past it.
        c0 = m0 + along * min(k * HANDRAIL_POST_SPACING_M, length - HANDRAIL_POST_M)
        c1 = c0 + along * HANDRAIL_POST_M
        out.append(Part(f"post_{k + 1}", box(c0, c1, HANDRAIL_POST_M), parapet.z1, top - HANDRAIL_RAIL_M))
    out.append(Part("rail", box(m0, m1, HANDRAIL_RAIL_M), top - HANDRAIL_RAIL_M, top))
    return out


def _write_handrail(parts: Sequence[Part], path: Path, base_elevation_m: float) -> int:
    rail = handrail_parts(parts)
    lines = [
        "# Red Rocks top-of-bowl steel handrail on the walkway parapet",
        f"# {len(rail) - 1} posts and a rail, centimetres, base {base_elevation_m:.3f} m",
        "o Handrail",
    ]
    vs, uvs, faces = [], [], []
    v_off = uv_off = 0
    for part in rail:
        verts, tris, puv, tuvs = prism(part.plan, part.z0, part.z1)
        vs.append(verts)
        uvs.append(puv)
        faces.append(f"# {part.name.replace('_', ' ')}")
        for t, tu in zip(tris, tuvs):
            faces.append("f " + " ".join(f"{v_off + v + 1}/{uv_off + u + 1}" for v, u in zip(t, tu)))
        v_off += len(verts)
        uv_off += len(puv)
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in np.vstack(vs) * CM_PER_M]
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in np.vstack(uvs)]
    lines += faces
    path.write_text("\n".join(lines) + "\n")
    return sum(1 for f in faces if f.startswith("f "))


def dimensions_table() -> str:
    """Markdown table of every dimension with its kind and source."""
    out = ["| Dimension | Value | Kind | Source |", "|---|---|---|---|"]
    for d in DIMENSIONS:
        out.append(f"| {d.name} | {d.value_m:g} | {d.kind} | {d.source} |")
    return "\n".join(out)
