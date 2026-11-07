"""Blockout of the Red Rocks stage, its roof, steel frame and stone towers, the
shell, retaining walls and top walkway.

None of this exists in free geodata, so it is built from a handful of boxes and
wedges (prisms with a convex plan) and positioned from the data the rest of the
package already has: row 1 of the generated seating bowl (the stage end) and
the last row (the top of the bowl).

The OBJs share the terrain and bench frame, so they need no transform in Unreal:
centimetres, X east, Y north, Z up, centred on the same origin, Z measured from
the terrain's lowest elevation (pass the same ``base_elevation_m`` the benches
used; see ``benches.terrain_base_elevation``). Faces wind counter-clockwise seen
from outside, every face carries a UV index (Unreal's OBJ importer decimates
meshes without ``vt``) and every part is a closed prism. The parts are split by
material, one OBJ each (``stage_mesh_for``): ``Stage`` (concrete: deck, shell,
walls, walkway, the towers' bases and caps), ``Stage_Steel`` (columns, trusses,
the side roofs' frames), ``Stage_Roof`` (the roof slab and side roofs),
``Stage_Fascia`` (the roof's dark front band) and ``Stage_Towers`` (the towers'
stacked flagstone and the walls that join them to the stage).

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
* The photo board (``docs/superpowers/specs/2026-11-05-look-like-red-rocks-design.md``):
  Commons photos from the top row ("Red Rocks Amphitheater.JPG", Scapler,
  CC BY-SA 3.0) and the front rows ("Red Rock Amphitheatre.jpg", Mitul0520,
  CC BY-SA 3.0), for what the scan cannot size: the towers' concrete bases and
  rounded corners, the arch, the side roofs' frames and the roof's dark band.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from shapely.geometry import LineString, Point, Polygon

import numpy as np

from .benches import _face_normal
from .scan_blend import ScanHeights
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
# The aisle stairs butt up to the row ends (a hair's gap so faces don't fight).
STAIR_GAP_M = 0.02
# With the terraced seating's tread levels (redrocks.seating): the walkway sits
# this far under the top tread, so the tread's curved back edge shows over it.
WALKWAY_UNDER_TREAD_M = 0.02
# Steel handrail on the walkway parapet (the photo from the side shows a pipe
# rail along the top of the bowl): top rail at the usual guard height.
HANDRAIL_HEIGHT_M = 1.07
HANDRAIL_POST_SPACING_M = 2.0
HANDRAIL_POST_M = 0.06
HANDRAIL_RAIL_M = 0.05
# The sound (FOH) booth, when its OSM roof footprint is known: a deck this far
# over the highest tread under it (a step up from the row behind), and a maroon
# roof on four posts (the photo from the top row shows a flat red roof with open
# sides, like the stage's).
BOOTH_RAISE_M = 0.30
BOOTH_ROOF_M = 2.6  # roof underside over the deck
BOOTH_ROOF_THICK_M = 0.18
BOOTH_ROOF_OVERHANG_M = 0.3
BOOTH_POST_M = 0.12
SKYDECK_BEAM_COUNT = 3

_SCAN = "ml0w scan (CC BY 4.0), "
_TOWER = _SCAN + (
    "tools/measure_stage_towers.py: the tower's flat top on the scan's top surface (scan_top.npz, 1 m cells), "
    "its 5-95 % extent across and along the scan's own roof (from that roof's front-edge centre: the scan's stage "
    "is turned 11 degrees and sits 1.3 m off the blockout's), scaled from the scan as placed (0.4748 m per unit) "
    "to 0.516 like the stage's other measured sizes. Medium-low confidence (+-1 m). ")

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
    Dimension("stair_wall_height", 0.45, "estimate",
              "Estimate: height of the stone wall along the aisle's outer side, above each row's "
              "upper stair tread. The aisle is one straight-ish flight (redrocks.seating "
              "AISLE_SLOPE_M), so the wall runs along it rather than across it."),
    Dimension("walkway_width", 4.0, "estimate",
              "Rough estimate, not measured: top-of-bowl promenade. The scan shows a flat plateau behind the top "
              "row with no edge that fixes a walkway width."),
    Dimension("walkway_thickness", 0.4, "estimate", "Blockout slab thickness."),
    Dimension("walkway_gap", 0.5, "estimate", "Estimate: clearance between the last bench and the walkway."),
    Dimension("walkway_parapet_height", 0.85, "estimate",
              "Estimate: low perimeter retaining/parapet wall along the rear edge of the top promenade."),
    # The two stone towers (WPA, 1941) either side of the stage. "Left" and
    # "right" are as seen from the seats: left is north at Red Rocks.
    Dimension("tower_left_inner", 20.9, "measured",
              _TOWER + "Left tower: inner face from the stage centre line, 7.8 m out from the roof's side "
              "(the photo from the top row shows about 8.8 m)."),
    Dimension("tower_left_outer", 27.7, "measured",
              _TOWER + "Left tower: outer face from the stage centre line, so 6.7 m wide (the top-row photo "
              "shows it about 7.9 m across, its front and a little of its inner side)."),
    Dimension("tower_left_front", 2.4, "measured",
              _TOWER + "Left tower: front face, in front of the deck's front edge."),
    Dimension("tower_left_back", 6.8, "measured",
              _TOWER + "Left tower: back face, behind the deck's front edge (9.2 m deep)."),
    Dimension("tower_left_height", 15.7, "measured",
              _TOWER + "Left tower: top of its cap over the deck, the flat top's median; 1.4 m over the roof's "
              "top, as the top-row photo shows it, just over the roof's front edge."),
    Dimension("tower_right_inner", 23.3, "measured",
              _TOWER + "Right tower: inner face from the stage centre line, 10.2 m out from the roof's side "
              "(the top-row photo shows about 11 m)."),
    Dimension("tower_right_outer", 28.8, "measured",
              _TOWER + "Right tower: outer face from the stage centre line, so 5.5 m wide (about 6.4 m across "
              "in the top-row photo)."),
    Dimension("tower_right_front", 1.6, "measured",
              _TOWER + "Right tower: front face, in front of the deck's front edge."),
    Dimension("tower_right_back", 4.6, "measured",
              _TOWER + "Right tower: back face, behind the deck's front edge (6.2 m deep)."),
    Dimension("tower_right_height", 15.0, "measured",
              _TOWER + "Right tower: top of its cap over the deck; 0.6 m over the roof's top (the top-row "
              "photo shows both tops at about the roof's front edge)."),
    Dimension("tower_base_height", 6.4, "estimate",
              "Photos (top row, front rows): each tower is stacked flagstone on a pale concrete base, the stone "
              "about 52 % of what shows above the ground; this is the base's top over the deck."),
    Dimension("tower_corner_radius", 2.0, "estimate",
              "Top-row photo: each tower's inner front corner is rounded, the curve about a third of its width."),
    Dimension("tower_cap_thickness", 0.3, "estimate", "Photos: a thin pale concrete cap on each tower."),
    Dimension("tower_cap_overhang", 0.15, "estimate", "Photos: the cap stands a little proud of the stone."),
    Dimension("tower_buried", 2.0, "estimate",
              "Without a ground to stand on, the towers and their walls reach this far under the deck."),
    Dimension("tower_arch_width", 2.0, "estimate",
              "Top-row photo: the right tower's base has a tall round-arched opening through it, about a "
              "third of its front, near its outer side."),
    Dimension("tower_arch_crown", 5.9, "estimate",
              "Top-row photo: the arch's crown sits just under the stone, 0.5 m under the base's top."),
    Dimension("tower_arch_outer_pier", 0.6, "estimate", "Top-row photo: the narrow pier outside the arch."),
    Dimension("tower_link_thickness", 0.8, "estimate",
              "Estimate: the stone wall that joins each tower to the stage, between its side roof and the tower."),
    Dimension("tower_link_low", 10.0, "estimate",
              "Front-row photo: that wall's top slopes up from about 10 m over the deck by the side roof..."),
    Dimension("tower_link_below_top", 1.5, "estimate", "...to about 1.5 m under the tower's top."),
    # The lower side roofs either side of the main roof, and its dark front band.
    Dimension("roof_wing_width", 4.1, "measured",
              _SCAN + "tools/measure_stage_towers.py: the right side roof (the only one the scan reconstructs) "
              "reaches 4.1 m past the main roof's side. OSM's roof outline (way 91483344) has 4.0-4.2 m wide "
              "notches there; the top-row photo shows about 4.5 m."),
    Dimension("roof_wing_depth", 5.1, "measured",
              _SCAN + "tools/measure_stage_towers.py: the right side roof runs 5.1 m back from the front edge "
              "(5-95 %; OSM's notches are about 4 m deep)."),
    Dimension("roof_wing_top", 13.0, "measured",
              _SCAN + "tools/measure_stage_towers.py: the right side roof's top over the deck, 1.3 m under the "
              "main roof's; the top-row photo puts it about 1.5 m lower."),
    Dimension("roof_wing_thickness", 0.6, "estimate", "Top-row photo: the side roofs' edge is a thin band."),
    Dimension("roof_chamfer", 1.0, "estimate",
              "Top-row photo: the main roof's front corners are cut at 45 degrees, its dark band turning with them."),
    Dimension("fascia_thickness", 0.12, "estimate",
              "The near-black band across the roof's front and its cut corners (photos), the roof's full depth."),
    Dimension("wing_column_size", 0.5, "estimate",
              "Front-row photo: square rust-red columns carry the side roofs at their outer corners."),
    Dimension("wing_header_depth", 1.0, "estimate", "Front-row photo: deep box-beam headers under the side roofs."),
    Dimension("wing_header_width", 0.6, "estimate", "Front-row photo: the headers' width."),
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
    """A vertical prism: convex plan polygon (world E, N metres) from ``z0`` to ``z1`` (metres above base).

    With ``axes`` (rows e1, e2, e3 of a right-handed orthonormal frame) the
    polygon is in the (e1, e2) plane instead and the prism runs from ``z0`` to
    ``z1`` along e3: a profile in a vertical plane, for a wall's sloped top or
    the curve of an arch.
    """

    name: str
    plan: np.ndarray
    z0: float
    z1: float
    # Row-end stair steps: (uphill unit, outward unit, the row's band depth), for the aisle path.
    frame: tuple | None = None
    axes: tuple | None = None


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
    booth_path: Path | None = None
    booth_deck_elevation_m: float | None = None
    # Heading the booth's front (its parapet) faces, degrees counter-clockwise
    # from east: straight at the stage front centre. And its (width, depth), m.
    booth_facing_deg: float | None = None
    booth_size_m: tuple[float, float] | None = None
    walkway_centre: tuple[float, float] | None = None
    # Every row-end stair step's centre (east, north, top elevation in absolute
    # metres) up each side of the bowl, bottom first: the aisles people walk up.
    aisles: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)
    stairs_path: Path | None = None
    stairs_parts: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Plan polygons (local metres) of the stairs and the booth deck: the scan is cut out under them too.
    cut_plans: list = field(default_factory=list)
    stairs_triangles: int = 0
    handrail_triangles: int = 0
    # The other stage OBJs, one per material (Stage_Steel, Stage_Roof, ...): name -> PartsMesh.
    meshes: dict = field(default_factory=dict)
    # Each tower ("left", "right" as seen from the seats): its centre (east, north, m),
    # the top of its cap (absolute m) and its plan's corners (local metres).
    towers: dict = field(default_factory=dict)


@dataclass
class PartsMesh:
    """One OBJ object written from parts: where, which triangles are which part, how many."""

    path: Path
    parts: dict[str, tuple[int, int]]  # name -> (first triangle, end triangle)
    triangles: int
    vertices: int


# The OBJ (and so the material) each part goes in, by its name; the rest is concrete Stage.obj.
STAGE_MESHES = ("Stage", "Stage_Steel", "Stage_Roof", "Stage_Fascia", "Stage_Towers")
STEEL_PREFIXES = ("column_", "y_arm_", "truss_", "skydeck_beam_", "wing_column_", "wing_header_")


def stage_mesh_for(name: str) -> str:
    """Which OBJ a stage part is written to: ``Stairs`` or one of ``STAGE_MESHES``.

    Steel (columns, Y-arms, trusses, purlins, the side roofs' columns and
    headers) is painted oxide red; the roof slab and side roofs are maroon
    metal; the fascia is the roof's dark front band; the towers' stone (their
    masonry and the walls joining them to the stage) is stacked flagstone. The
    towers' concrete bases and caps stay with the concrete.
    """
    if name.startswith("stair_"):
        return "Stairs"
    if name.startswith(STEEL_PREFIXES):
        return "Stage_Steel"
    if name.startswith("roof"):
        return "Stage_Roof"
    if name.startswith("fascia_"):
        return "Stage_Fascia"
    if name.startswith("tower_") and name.endswith(("_masonry", "_link")):
        return "Stage_Towers"
    if name.startswith("foh_"):
        # The sound booth's base and parapet are the risers' red flagstone at the
        # real venue; in the stage concrete they read as a white box mid-bowl.
        return "Stage_Towers"
    return "Stage"


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
    booth: np.ndarray | None = None,
) -> list[Part]:
    """Lay out every blockout part from the seating-bowl rows.

    ``ground(east, north)`` (metres) is the ground the benches stand on (the
    ml0w scan's, or the terrain's); the deck and the walkway sit on it. Without
    it the deck sits ``STAGE_DROP_M`` below row 1's elevation and the walkway
    at the last row's, regardless of the terrain. The towers, the walls joining
    them to the stage and the side roofs' columns reach half a metre under the
    lowest ground below them (without it, ``tower_buried`` under the deck).

    ``levels`` are the terraced seating's tread elevations, one per row
    (``seating.tread_levels``). With them the side stairs climb from each tread
    to the next across the row's band, the stage steps reach row 1's tread and
    the walkway sits just under the top tread.

    ``booth`` is the sound booth's roof footprint (4 corners, local metres; the
    CLI takes OSM's ``building=roof`` in the bowl). With it and ``levels`` the
    booth deck keeps that footprint's centre and size, ``BOOTH_RAISE_M`` over
    the highest tread under it, squared to the stage: its parapet faces the
    stage front centre (``_booth_frame``). ``booth_roof_parts`` adds the roof.
    Without, it is the old blockout at row 8.
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
    # The roof slab on the SkyDeck grid, its front corners cut at 45 degrees (the photo from the top row).
    c = dim("roof_chamfer")
    parts.append(Part("roof", _poly(origin, a, l, [(-hw, -RD), (hw, -RD), (hw, -c), (hw - c, 0.0),
                                                   (c - hw, 0.0), (-hw, -c)]), roof_bot, roof_top))

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

    # The roof's dark front band, and the lower side roofs either side on their own steel frames.
    parts += _roof_trim(origin, a, l, hw, deck_bot, roof_bot, roof_top, deck_top, ground, base_elevation_m)

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

    # The two stone towers either side of the downstage end, and the walls joining them to the stage.
    parts += tower_parts(origin, a, l, hw, deck_top, ground, base_elevation_m)

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

    # Front-of-House (FOH) sound booth: on its OSM roof footprint (way 229359709, row ~13) when the
    # CLI found it, turned to face the stage front centre; otherwise the old blockout at row 8.
    if booth is not None and levels is not None:
        parts += _booth_parts(np.asarray(booth, dtype=float), a, polylines, levels, base_elevation_m, ground,
                              toward=origin)
    elif len(rows) >= 8:
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
    stair_gap = STAIR_GAP_M
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
                    z_base, z_step_top, frame=(tuple(n_up), tuple(u_out), band)))
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


# Below each tower, wall or column that stands on the ground, by this much.
GROUND_EMBED_M = 0.5
ARCH_SEGMENTS = 6  # pieces of the arch over the right tower's opening
CORNER_SEGMENTS = 4  # sides on each tower's rounded corner


def _ground_min(plan: np.ndarray, ground, base_elevation_m: float) -> float | None:
    """Lowest ground (metres over base) under a plan polygon: its corners and a
    ``DECK_SAMPLE_M`` grid inside it. None without a ground."""
    if ground is None:
        return None
    poly = Polygon(np.asarray(plan, dtype=float)[:, :2])
    x0, y0, x1, y1 = poly.bounds
    pts = [tuple(p) for p in np.asarray(plan, dtype=float)[:, :2]]
    pts += [(x, y) for x in np.arange(x0, x1, DECK_SAMPLE_M) for y in np.arange(y0, y1, DECK_SAMPLE_M)
            if poly.contains(Point(x, y))]
    return min(float(ground(x, y)) for x, y in pts) - base_elevation_m


def _bottom(plan: np.ndarray, deck_top: float, ground, base_elevation_m: float) -> float:
    """Bottom (metres over base) of something standing beside the deck: under the ground there."""
    low = deck_top - dim("tower_buried")
    g = _ground_min(plan, ground, base_elevation_m)
    return low if g is None else min(low, g - GROUND_EMBED_M)


def _vertical_axes(l: np.ndarray) -> tuple:
    """Rows (e1, e2, e3) for a profile across the stage: e1 = ``l``, e2 up, e3 = e1 x e2 (toward the back)."""
    return ((float(l[0]), float(l[1]), 0.0), (0.0, 0.0, 1.0), (float(l[1]), -float(l[0]), 0.0))


def _profile_part(name: str, origin, a, l, xz, y0: float, y1: float) -> Part:
    """A prism with a convex profile ``xz`` ((x across the stage, z) pairs, z
    metres over base) in a vertical plane, running from y0 to y1 along the axis."""
    axes = _vertical_axes(l)
    e3 = np.array(axes[2][:2])
    ol, oe = float(origin @ l), float(origin @ e3)
    plan = np.array([(ol + x, z) for x, z in xz], dtype=float)
    # e3 is -a: a point y along the axis is at oe - y along e3.
    t0, t1 = sorted((oe - y0, oe - y1))
    return Part(name, plan, t0, t1, axes=axes)


def _roof_trim(origin, a, l, hw, deck_bot, roof_bot, roof_top, deck_top, ground, base_elevation_m) -> list[Part]:
    """The roof's dark front band, the lower side roofs and their columns and box-beam headers.

    ``left`` and ``right`` are as seen from the seats (``+l`` is the left).
    The fascia covers the roof slab's front face and its cut corners, a
    ``fascia_thickness`` plate standing proud of them. Each side roof runs
    ``roof_wing_width`` out from the main roof's side (tucked in under its cut
    corner) and ``roof_wing_depth`` back from the front, its top
    ``roof_wing_top`` over the deck, on two columns at its outer corners with a
    header along its front and one along its outer side.
    """
    c, t = dim("roof_chamfer"), dim("fascia_thickness")
    parts = [Part("fascia_front", _rect(origin, a, l, c - hw - t / 2, hw - c + t / 2, 0.0, t), roof_bot, roof_top)]
    for sx, side in ((1, "left"), (-1, "right")):
        p0, p1 = np.array([sx * (hw - c), 0.0]), np.array([sx * hw, -c])
        n = np.array([sx, 1.0]) / math.sqrt(2.0)
        parts.append(Part(f"fascia_chamfer_{side}", _poly(origin, a, l, [p0, p1, p1 + n * t, p0 + n * t]),
                          roof_bot, roof_top))

    ww, wd = dim("roof_wing_width"), dim("roof_wing_depth")
    wtop = deck_top + dim("roof_wing_top")
    wbot = wtop - dim("roof_wing_thickness")
    cs, hd, hb = dim("wing_column_size"), dim("wing_header_depth"), dim("wing_header_width")
    for sx, side in ((1, "left"), (-1, "right")):
        out = hw + ww
        parts.append(Part(f"roof_wing_{side}", _rect(origin, a, l, *sorted((sx * (hw - c), sx * out)), -wd, 0.0),
                          wbot, wtop))
        xc = sx * (out - cs / 2)
        for pos, yc in (("front", -cs / 2), ("back", -(wd - cs / 2))):
            plan = _rect(origin, a, l, xc - cs / 2, xc + cs / 2, yc - cs / 2, yc + cs / 2)
            low = deck_bot if ground is None else min(deck_bot, _ground_min(plan, ground, base_elevation_m) - 0.3)
            parts.append(Part(f"wing_column_{side}_{pos}", plan, low, wbot - hd))
        parts.append(Part(f"wing_header_{side}_front",
                          _rect(origin, a, l, *sorted((sx * (hw - 0.5), sx * out)), -hb, 0.0), wbot - hd, wbot))
        parts.append(Part(f"wing_header_{side}_side",
                          _rect(origin, a, l, *sorted((sx * (out - hb), sx * out)), -wd, 0.0), wbot - hd, wbot))
    return parts


def _tower_plan(sx: int, inner: float, outer: float, front: float, back: float, r: float) -> np.ndarray:
    """Stage (x across, y along) corners of a tower: a rectangle from ``inner``
    to ``outer`` across (on the ``sx`` side) and from ``back`` behind the front
    edge to ``front`` in front of it, its inner front corner rounded to ``r``."""
    xi, xo = sx * inner, sx * outer
    pts = [(xi, -back), (xo, -back), (xo, front)]
    cx, cy = xi + sx * r, front - r
    for k in range(CORNER_SEGMENTS + 1):
        t = 0.5 * math.pi * k / CORNER_SEGMENTS
        pts.append((cx - sx * r * math.sin(t), cy + r * math.cos(t)))
    return np.array(pts)


def _clip_across(plan_xy: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """The part of a convex stage-coordinate plan between x0 and x1 across (still convex)."""
    from shapely.geometry import box

    lo, hi = sorted((x0, x1))
    clipped = Polygon(plan_xy).intersection(box(lo, -1e4, hi, 1e4)).simplify(1e-6)
    return np.array(clipped.exterior.coords)[:-1]


def tower_parts(origin, a, l, hw: float, deck_top: float, ground=None, base_elevation_m: float = 0.0
                ) -> list[Part]:
    """The two stone towers (WPA, 1941) beside the downstage end, and the walls joining them to the stage.

    Each tower (``tower_left_*``, ``tower_right_*`` as seen from the seats;
    ``+l`` is the left) stands on the plan its ``tower_<side>_*`` dimensions
    give, measured on the scan: a pale concrete base up to ``tower_base_height``
    over the deck, stacked flagstone above it (``*_masonry``) and a concrete cap
    ``tower_cap_overhang`` proud of the stone whose top is the tower's measured
    height. The inner front corner is rounded. The right tower's base has a
    tall round-arched opening running through it front to back, near its outer
    side: two piers and the arch's pieces over the opening, which is a real
    passage. A sloped stone wall (``*_link``) joins each tower's back to the
    stage, from by its side roof (``tower_link_low`` over the deck) up to
    ``tower_link_below_top`` under the tower's top. Bottoms reach under the
    ground (``_bottom``). Everything here is convex, like every other part.
    """
    r, base_h = dim("tower_corner_radius"), dim("tower_base_height")
    cap_t, cap_o = dim("tower_cap_thickness"), dim("tower_cap_overhang")
    parts: list[Part] = []
    for sx, side in ((1, "left"), (-1, "right")):
        inner, outer = dim(f"tower_{side}_inner"), dim(f"tower_{side}_outer")
        front, back = dim(f"tower_{side}_front"), dim(f"tower_{side}_back")
        top = deck_top + dim(f"tower_{side}_height")
        if not (hw < inner < outer and r < min(outer - inner, front + back)):
            raise ValueError(f"tower_{side}: plan does not fit beside the deck")
        xy = _tower_plan(sx, inner, outer, front, back, r)
        plan = _poly(origin, a, l, xy)
        z_bot, z_base = _bottom(plan, deck_top, ground, base_elevation_m), deck_top + base_h
        z_stone = top - cap_t
        if side == "right":
            # Piers either side of the arched opening, near the outer side.
            pier, w = dim("tower_arch_outer_pier"), dim("tower_arch_width")
            if pier + w > (outer - inner) - r:
                raise ValueError("tower_right: the arch runs into the rounded corner")
            xo, xa, xb = sx * outer, sx * (outer - pier), sx * (outer - pier - w)
            parts.append(Part(f"tower_{side}_pier_outer", _poly(origin, a, l, _clip_across(xy, xo, xa)), z_bot, z_base))
            parts.append(Part(f"tower_{side}_pier_inner", _poly(origin, a, l, _clip_across(xy, xb, sx * inner)),
                              z_bot, z_base))
            # The arch: pieces from its curve up to the stone, through the base's depth.
            rad = w / 2
            xc, zs = (xa + xb) / 2, deck_top + dim("tower_arch_crown") - rad
            if zs + rad >= z_base:
                raise ValueError("tower_right: the arch's crown is above its base")
            ring = [(xc + rad * math.cos(math.pi * k / ARCH_SEGMENTS), zs + rad * math.sin(math.pi * k / ARCH_SEGMENTS))
                    for k in range(ARCH_SEGMENTS + 1)]
            for k, (p, q) in enumerate(zip(ring, ring[1:]), start=1):
                parts.append(_profile_part(f"tower_{side}_arch_{k}", origin, a, l,
                                           [p, q, (q[0], z_base), (p[0], z_base)], -back, front))
        else:
            parts.append(Part(f"tower_{side}_base", plan, z_bot, z_base))
        parts.append(Part(f"tower_{side}_masonry", plan, z_base, z_stone))
        cap = np.array(Polygon(xy).buffer(cap_o, join_style=2).simplify(1e-6).exterior.coords)[:-1]
        parts.append(Part(f"tower_{side}_cap", _poly(origin, a, l, cap), z_stone, top))

        # The sloped stone wall from by the side roof to the tower's inner face, along its back.
        x_end = sx * (hw + dim("roof_wing_width") + 0.3)
        if abs(x_end) < inner - 0.5:
            t = dim("tower_link_thickness")
            foot = _poly(origin, a, l, [(x_end, -back), (sx * inner, -back), (sx * inner, t - back), (x_end, t - back)])
            zb = _bottom(foot, deck_top, ground, base_elevation_m)
            z_lo, z_hi = deck_top + dim("tower_link_low"), top - dim("tower_link_below_top")
            parts.append(_profile_part(f"tower_{side}_link", origin, a, l,
                                       [(x_end, zb), (sx * inner, zb), (sx * inner, z_hi), (x_end, z_lo)],
                                       -back, t - back))
    return parts


def part_mesh(part: Part):
    """(verts[N,3] world metres over base, tris, uvs, tri_uvs) of a part, its ``axes`` applied."""
    verts, tris, uvs, tuvs = prism(part.plan, part.z0, part.z1)
    if part.axes is not None:
        verts = verts @ np.asarray(part.axes, dtype=float)
    return verts, tris, uvs, tuvs


def part_footprint(part: Part) -> Polygon:
    """A part's outline seen from above (local metres)."""
    if part.axes is None:
        return Polygon(np.asarray(part.plan, dtype=float)[:, :2])
    from shapely.geometry import MultiPoint

    return MultiPoint([tuple(v) for v in part_mesh(part)[0][:, :2]]).convex_hull


# Where the scan stands this far over the deck near the new stage it is its own
# roof, side roofs or towers: the scan's stage is turned about 11 degrees and
# sits a few metres off the blockout's (tools/measure_stage_towers.py), so they
# would show beside the blockout's. Linking walls and balconies stay (under 10 m).
SCAN_STAGE_TALL_M = 11.0
SCAN_STAGE_ZONE_M = 4.0  # around the roof, side roofs and towers
SCAN_STAGE_EDGE_M = 1.0  # each tall cell's cut reaches this far past it, so its walls go too


def scan_stage_cut(top: ScanHeights, parts: Sequence[Part], origin, a, deck_top_m: float) -> list[Polygon]:
    """Plans (local metres) to cut the ml0w scan out of: its own stage's tall structures.

    The zone is the outline of the roof, the side roofs and the towers grown by
    ``SCAN_STAGE_ZONE_M``, in front of the rock behind the stage (no further
    back than a metre behind the roof). Every ``top`` cell in it standing
    ``SCAN_STAGE_TALL_M`` over the deck top (``deck_top_m``, absolute) is cut,
    ``SCAN_STAGE_EDGE_M`` around.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    keep = [part_footprint(p) for p in parts if p.name.startswith(("roof", "tower_"))]
    if not keep:
        return []
    a = np.asarray(a, dtype=float)
    l = np.array([a[1], -a[0]])
    back = dim("roof_depth") + 1.0
    hull = unary_union(keep).convex_hull.buffer(SCAN_STAGE_ZONE_M)
    front_half = Polygon([origin - a * back + l * 1e4, origin - a * back - l * 1e4,
                          origin + a * 1e4 - l * 1e4, origin + a * 1e4 + l * 1e4])
    zone = hull.intersection(front_half)
    east, north = top.centres()
    with np.errstate(invalid="ignore"):
        tall = np.nan_to_num(top.heights - deck_top_m, nan=-1e9) > SCAN_STAGE_TALL_M
    x0, y0, x1, y1 = zone.bounds
    tall &= (east > x0 - 1) & (east < x1 + 1) & (north > y0 - 1) & (north < y1 + 1)
    half = top.cell / 2
    cells = [box(e - half, n - half, e + half, n + half) for e, n in zip(east[tall], north[tall])
             if zone.contains(Point(e, n))]
    if not cells:
        return []
    cut = unary_union(cells).buffer(SCAN_STAGE_EDGE_M, join_style=2)
    return [g for g in getattr(cut, "geoms", [cut]) if not g.is_empty]


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
    booth: np.ndarray | None = None,
    scan_top: ScanHeights | None = None,
) -> StageMesh:
    """Write the blockout: ``Stage.obj`` (concrete) and one OBJ per other material.

    Every part goes in the OBJ ``stage_mesh_for`` names: the concrete in
    ``out_path`` (object ``Stage``), the rest in ``Stage_Steel.obj``,
    ``Stage_Roof.obj``, ``Stage_Fascia.obj`` and ``Stage_Towers.obj`` beside it
    (objects named like the files, with flat ``vn`` normals so a material can
    tell tops from sides), the row-end stairs in ``Stairs.obj``.

    Args:
        rows: venue rows (``venue.rows``), row 1 = stage end, last = top.
        polylines: per-row (N, 2) local metre polylines (``benches.row_polylines``).
        out_path: where to write ``Stage.obj``.
        base_elevation_m: Z=0 elevation, the terrain minimum
            (``benches.terrain_base_elevation``).
        ground: the ground the benches stand on, which the deck and walkway
            sit on (``scan_blend.ScanGround`` or ``terrain_mesh.TerrainSurface``);
            optional, see ``stage_parts``.
        scan_top: the ml0w scan's top raster; with it ``cut_plans`` also
            covers the scan's own tall stage structures (``scan_stage_cut``).
    """
    parts = stage_parts(rows, polylines, base_elevation_m, ground, levels, booth)
    # The row-end aisle stairs and cheek walls go in Stairs.obj, which takes the
    # seating's flagstone material; everything else is the concrete Stage.obj.
    stairs = [p for p in parts if p.name.startswith("stair_")]
    stairs_path = Path(out_path).parent / "Stairs.obj"
    stairs_tris, stairs_ranges = _write_parts_obj(stairs, stairs_path, "Stairs", [
        "Red Rocks row-end aisle stairs and stone cheek walls, two steps per row",
        f"centimetres, base {base_elevation_m:.3f} m",
    ]) if stairs else (0, {})

    # Steel, roof, fascia and the towers' stone: one OBJ (so one material) each.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, PartsMesh] = {}
    for name in STAGE_MESHES[1:]:
        group = [p for p in parts if stage_mesh_for(p.name) == name]
        path = Path(out_path).parent / f"{name}.obj"
        if group:
            meshes[name] = _write_group(group, path, name, base_elevation_m)
        else:
            path.unlink(missing_ok=True)

    all_v: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    faces: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    v_off = uv_off = n_tris = 0
    comments: list[str] = []
    concrete = [p for p in parts if stage_mesh_for(p.name) == "Stage"]
    for part in concrete:
        verts, tris, uvs, tuvs = part_mesh(part)
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
        "# Red Rocks stage blockout, the concrete: deck, steps, shell, retaining walls, walkway, "
        "the towers' bases and caps",
        f"# {len(concrete)} parts, {n_tris} triangles, centimetres, base {base_elevation_m:.3f} m",
        "# steel, roof, fascia and the towers' stone are in Stage_*.obj beside this file",
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
    booth_path, booth_deck, booth_facing, booth_size = None, None, None, None
    if booth is not None and levels is not None:
        booth_path = out_path.parent / "Booth.obj"
        roof = booth_roof_parts(parts, booth, a, toward=origin)
        _write_parts_obj(roof, booth_path, "Booth", [
            "Red Rocks sound booth: posts and roof over the FOH deck (deck and parapet are in Stage.obj)",
            f"centimetres, base {base_elevation_m:.3f} m",
        ])
        booth_deck = next(p.z1 for p in parts if p.name == "foh_platform") + base_elevation_m
        _, _, up, hw, depth = _booth_frame(booth, a, toward=origin)
        booth_facing = float(np.degrees(np.arctan2(-up[1], -up[0])) % 360.0)
        booth_size = (2.0 * hw, depth)
    else:
        (out_path.parent / "Booth.obj").unlink(missing_ok=True)

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
        booth_path=booth_path,
        booth_deck_elevation_m=booth_deck,
        booth_facing_deg=booth_facing,
        booth_size_m=booth_size,
        walkway_centre=tuple(float(v) for v in next(p.plan for p in parts if p.name == "walkway").mean(axis=0)),
        aisles=_aisles(stairs, base_elevation_m),
        stairs_path=stairs_path if stairs else None,
        stairs_parts=stairs_ranges,
        cut_plans=[Polygon(np.asarray(p.plan, dtype=float)) for p in parts
                   if p.name.startswith(("stair_", "foh_", "stage_step_"))]
        + [part_footprint(p) for p in parts if p.name.startswith("tower_")]
        + (scan_stage_cut(scan_top, parts, origin, a, deck_top_elevation(rows, ground))
           if scan_top is not None else []),
        stairs_triangles=stairs_tris,
        meshes=meshes,
        towers=_tower_summary(parts, base_elevation_m),
    )


def _tower_summary(parts: Sequence[Part], base_elevation_m: float) -> dict:
    """``left`` / ``right``: each tower's centre (east, north m), cap top (absolute m) and plan corners."""
    out = {}
    for side in ("left", "right"):
        stone = next((p for p in parts if p.name == f"tower_{side}_masonry"), None)
        cap = next((p for p in parts if p.name == f"tower_{side}_cap"), None)
        if stone is None or cap is None:
            continue
        plan = Polygon(np.asarray(stone.plan, dtype=float))
        out[side] = {
            "east_m": float(plan.centroid.x),
            "north_m": float(plan.centroid.y),
            "top_elevation_m": float(cap.z1 + base_elevation_m),
            "plan_m": [tuple(float(v) for v in p) for p in np.asarray(stone.plan, dtype=float)],
        }
    return out


STAGE_MESH_ABOUT = {
    "Stage_Steel": "Red Rocks stage steel, painted oxide red: Y-columns, trusses and SkyDeck purlins, "
                   "the side roofs' columns and box-beam headers",
    "Stage_Roof": "Red Rocks stage roof, maroon painted metal: the slab on the SkyDeck grid and the lower side roofs",
    "Stage_Fascia": "Red Rocks stage roof's near-black front band and its cut corners",
    "Stage_Towers": "Red Rocks stage towers' stacked flagstone, the sloped stone walls joining them to the stage, "
                    "and the sound booth's flagstone base and parapet",
}


def _write_group(parts: Sequence[Part], path: Path, name: str, base_elevation_m: float) -> PartsMesh:
    tris, ranges = _write_parts_obj(parts, path, name, [
        STAGE_MESH_ABOUT[name],
        f"{len(parts)} parts, centimetres, base {base_elevation_m:.3f} m; dimensions: redrocks.stage.DIMENSIONS",
    ], normals=True)
    return PartsMesh(path=path, parts=ranges, triangles=tris, vertices=sum(2 * len(p.plan) for p in parts))


def _row_at(point: np.ndarray, shapes: Sequence[LineString]) -> int:
    """Index of the row whose band (its line up to the next row's) holds ``point``."""
    d = np.array([s.distance(Point(*point)) for s in shapes])
    i, j = sorted(np.argsort(d)[:2])
    # Between two neighbouring lines the band is the lower one's; past the last line, the last row.
    return int(i) if j == i + 1 and d[i] + d[j] <= float(shapes[i].distance(shapes[j])) + 1e-6 else int(np.argmin(d))


def _booth_frame(plan: np.ndarray, a: np.ndarray, toward: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(centre, along-the-rows unit, uphill unit, half width, depth) of a 4-corner footprint.

    The booth keeps the footprint's centre and its two side lengths. With
    ``toward`` (the stage front centre) it is squared to the stage: ``up`` runs
    from ``toward`` through the centre, so the front (the parapet, on the
    ``-up`` side) looks straight at it. OSM's roof outline is turned about 21
    degrees off that line at the real venue. The depth is the footprint side
    most nearly along ``up``, the width the other. Without ``toward``, ``up`` is
    the footprint side most nearly along the stage axis ``a``.
    """
    plan = np.asarray(plan, dtype=float)[:4]
    centre = plan.mean(axis=0)
    edges = [plan[(k + 1) % 4] - plan[k] for k in range(4)]
    ref = np.asarray(a, dtype=float)
    if toward is not None:
        d = centre - np.asarray(toward, dtype=float)[:2]
        if float(np.hypot(*d)) > 1e-6:
            ref = d / float(np.hypot(*d))
    # The depth is whichever side pair is most nearly along the uphill direction.
    depth_edge = max(edges, key=lambda e: abs(float(e @ ref)) / float(np.hypot(*e)))
    if toward is not None:
        up = ref
    else:
        up = depth_edge / float(np.hypot(*depth_edge))
        if up @ a < 0:
            up = -up
    along = np.array([up[1], -up[0]])
    depth = float(np.hypot(*depth_edge))
    k = next(i for i, e in enumerate(edges) if e is depth_edge)
    width = float(np.hypot(*edges[(k + 1) % 4]))
    return centre, along, up, width / 2.0, depth


def _booth_parts(plan, a, polylines, levels, base, ground, toward=None) -> list[Part]:
    centre, along, up, hw, depth = _booth_frame(plan, a, toward)
    shapes = [LineString(np.asarray(p, dtype=float)[:, :2]) for p in polylines]
    samples = [centre + along * x + up * y
               for x in np.linspace(-hw, hw, 5) for y in np.linspace(-depth / 2, depth / 2, 5)]
    top = max(levels[_row_at(p, shapes)] for p in samples) + BOOTH_RAISE_M
    low = min(levels[_row_at(p, shapes)] for p in samples)
    if ground is not None:
        low = min(low, min(float(ground(*p)) for p in samples))
    z_top, z_bot = top - base, low - base - 0.6
    wt, wh = dim("foh_wall_thickness"), dim("foh_wall_height")
    y0, y1 = -depth / 2, depth / 2
    parts = [
        Part("foh_platform", _rect(centre, up, along, -hw, hw, y0, y1), z_bot, z_top),
        Part("foh_wall_front", _rect(centre, up, along, -hw, hw, y0, y0 + wt), z_top, z_top + wh),
    ]
    for sx, nm in ((1, "e"), (-1, "w")):
        x0, x1 = (hw - wt, hw) if sx > 0 else (-hw, -hw + wt)
        # Side walls stop short of the back so the row behind walks straight in.
        parts.append(Part(f"foh_wall_{nm}", _rect(centre, up, along, x0, x1, y0 + wt + 0.02, y0 + depth * 0.6),
                          z_top, z_top + wh))
    return parts


def booth_roof_parts(parts: Sequence[Part], booth: np.ndarray, a: np.ndarray,
                     toward: np.ndarray | None = None) -> list[Part]:
    """Four corner posts on the booth deck and the flat roof over them.

    Pass the same ``toward`` (stage front centre) the deck was built with, so
    the roof is squared to the stage like the deck under it.
    """
    centre, along, up, hw, depth = _booth_frame(booth, a, toward)
    deck = next(p for p in parts if p.name == "foh_platform")
    z0, z1 = deck.z1, deck.z1 + BOOTH_ROOF_M
    out = []
    pi = BOOTH_POST_M
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        x = sx * (hw - 0.15 - pi / 2)
        y = sy * (depth / 2 - 0.15 - pi / 2)
        out.append(Part(f"post_{k + 1}", _rect(centre, up, along, x - pi / 2, x + pi / 2, y - pi / 2, y + pi / 2), z0, z1))
    o = BOOTH_ROOF_OVERHANG_M
    out.append(Part("roof", _rect(centre, up, along, -hw - o, hw + o, -depth / 2 - o, depth / 2 + o),
                    z1, z1 + BOOTH_ROOF_THICK_M))
    return out


# An aisle path turns a corner when the next row's stair is this much further out
# (only low in the bowl now: above AISLE_FROM_ROW the aisle line jogs 1 m at most).
AISLE_CORNER_M = 1.2
# The corner is this far up onto the next row's tread.
AISLE_CORNER_IN_M = 0.6
# The path keeps this far in from each stair's centre line, toward the rows, so
# a walker running wide on a turn stays on the stairs, not over the outer curb.
AISLE_INSET_M = 0.4


def _aisles(stairs: Sequence[Part], base_elevation_m: float) -> dict[str, list[tuple[float, float, float]]]:
    """A walkable path up each side of the bowl: every row's two stair steps,
    bottom row first (east, north, top elevation in absolute metres).

    Where the next row reaches further out (the rows widen low in the bowl),
    heading straight for its stair would cut across this row's band and its
    stair's cheek wall. There the path turns a corner instead: up onto the next
    row's tread beside this stair, then out along that tread to the next stair's
    upper step (going past its lower one, which sits half a step down).
    """
    steps: dict[str, dict[int, dict[int, Part]]] = {}
    for part in stairs:
        m = re.fullmatch(r"stair_(start|end)_r(\d+)_s(\d+)", part.name)
        if m and part.frame is not None:
            steps.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[int(m.group(3))] = part

    def centre(part):
        u_out = np.asarray(part.frame[1], dtype=float)
        return np.asarray(part.plan, dtype=float).mean(axis=0) - u_out * AISLE_INSET_M

    out: dict[str, list[tuple[float, float, float]]] = {}
    for side, rows in steps.items():
        path: list[tuple[float, float, float]] = []
        order = sorted(rows)
        for k, idx in enumerate(order):
            row = rows[idx]
            for step_k in sorted(row):
                if step_k == 1 and path and path[-1][3] == "corner":
                    continue  # the corner leads straight to this row's upper step
                c = centre(row[step_k])
                path.append((float(c[0]), float(c[1]), row[step_k].z1 + base_elevation_m, "step"))
            if k + 1 < len(order):
                nxt = rows[order[k + 1]]
                top = row[max(row)]
                n_up, u_out, band = (np.asarray(v, dtype=float) if i < 2 else v for i, v in enumerate(top.frame))
                here, there = centre(top), centre(nxt[min(nxt)])
                if float((there - here) @ u_out) > AISLE_CORNER_M:
                    c = here + n_up * (0.25 * band + AISLE_CORNER_IN_M)
                    path.append((float(c[0]), float(c[1]), nxt[max(nxt)].z1 + base_elevation_m, "corner"))
        out[side] = [(round(e, 3), round(n, 3), round(z, 3)) for e, n, z, _ in path]
    return out


def _write_parts_obj(parts: Sequence[Part], path: Path, name: str, header: Sequence[str],
                     normals: bool = False) -> tuple[int, dict[str, tuple[int, int]]]:
    """One OBJ object of prisms; returns (triangles, part name -> (first, end) triangle).

    With ``normals`` every face also carries its flat normal (``vn``), so
    Unreal keeps the prisms' edges hard and a material can tell tops from sides.
    """
    lines = [f"# {h}" for h in header] + [f"o {name}"]
    vs, uvs, faces = [], [], []
    vns: dict[tuple[float, float, float], int] = {}
    ranges: dict[str, tuple[int, int]] = {}
    v_off = uv_off = n_tris = 0
    for part in parts:
        verts, tris, puv, tuvs = part_mesh(part)
        vs.append(verts)
        uvs.append(puv)
        faces.append(f"# {part.name.replace('_', ' ')}")
        for t, tu in zip(tris, tuvs):
            if normals:
                nid = vns.setdefault(_face_normal(verts, t), len(vns) + 1)
                faces.append("f " + " ".join(f"{v_off + v + 1}/{uv_off + u + 1}/{nid}" for v, u in zip(t, tu)))
            else:
                faces.append("f " + " ".join(f"{v_off + v + 1}/{uv_off + u + 1}" for v, u in zip(t, tu)))
        ranges[part.name] = (n_tris, n_tris + len(tris))
        n_tris += len(tris)
        v_off += len(verts)
        uv_off += len(puv)
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in np.vstack(vs) * CM_PER_M]
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in np.vstack(uvs)]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in vns]
    lines += faces
    path.write_text("\n".join(lines) + "\n")
    return n_tris, ranges


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
    return _write_parts_obj(rail, path, "Handrail", [
        "Red Rocks top-of-bowl steel handrail on the walkway parapet",
        f"{len(rail) - 1} posts and a rail, centimetres, base {base_elevation_m:.3f} m",
    ])[0]


def dimensions_table() -> str:
    """Markdown table of every dimension with its kind and source."""
    out = ["| Dimension | Value | Kind | Source |", "|---|---|---|---|"]
    for d in DIMENSIONS:
        out.append(f"| {d.name} | {d.value_m:g} | {d.kind} | {d.source} |")
    return "\n".join(out)
