"""Where the player starts: on the stage, facing up the bowl, above any ground.

Unreal discards a PlayerStart whose capsule overlaps blocking geometry (and
finds no free spot nearby), then spawns the pawn at the world origin, which for
this level is under the terrain. The stage deck is a blockout whose top can sit
below the terrain mesh where the DEM bulges, so the spawn height is taken from
whichever is higher under the capsule's footprint: the deck top or the terrain
mesh as exported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

SPAWN_BACK_M = 5.0  # behind the downstage edge, onto the deck
CAPSULE_RADIUS_M = 0.55  # the FirstPerson character's capsule, plus margin
CLEARANCE_M = 0.25  # gap between the ground and the bottom of the capsule


@dataclass(frozen=True)
class Spawn:
    east_m: float
    north_m: float
    ground_m: float  # elevation the pawn stands on (metres above sea level)
    yaw_deg: float  # terrain heading: 0 = east, 90 = north (Unreal yaw: unreal/level_frame.level_yaw)
    deck_top_m: float
    terrain_m: float  # highest terrain sample under the capsule
    clearance_m: float = CLEARANCE_M


def player_spawn(
    front_east_m: float,
    front_north_m: float,
    axis_east_north: tuple[float, float],
    deck_top_m: float,
    surface: Callable[[float, float], float],
    back_m: float = SPAWN_BACK_M,
) -> Spawn:
    """Spawn on the deck, ``back_m`` behind the downstage edge, facing the seats.

    ``axis_east_north`` runs from row 1 up to the top row; the deck lies against
    it. ``surface(east, north)`` is the terrain mesh height in metres.
    """
    ax, ay = axis_east_north
    norm = math.hypot(ax, ay) or 1.0
    ax, ay = ax / norm, ay / norm
    east = front_east_m - back_m * ax
    north = front_north_m - back_m * ay
    samples = [surface(east, north)]
    for k in range(8):
        a = k * math.pi / 4
        samples.append(surface(east + CAPSULE_RADIUS_M * math.cos(a), north + CAPSULE_RADIUS_M * math.sin(a)))
    terrain = max(samples)
    return Spawn(
        east_m=east,
        north_m=north,
        ground_m=max(deck_top_m, terrain),
        yaw_deg=math.degrees(math.atan2(ay, ax)),
        deck_top_m=deck_top_m,
        terrain_m=terrain,
    )
