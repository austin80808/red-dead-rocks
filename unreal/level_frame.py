"""Where the pipeline's terrain frame meets Unreal's level frame.

The pipeline (``src/redrocks``, ``tools/``, ``out/manifest.json``, every OBJ)
works in a right-handed terrain frame: X east, Y north, Z up, like a map.
Unreal is left-handed: X forward, Y right, Z up, and yaw turns +X toward +Y,
which seen from above is clockwise. Putting north on +Y therefore builds a
mirror image of Red Rocks: facing east, north lands on your right. (It did,
until 2026-11-04: Creation Rock and Ship Rock had swapped sides.)

The level frame is the terrain frame with north negated:

    level X = east,  level Y = -north (south),  level Z = up.

That is the one proper way to put a right-handed map into Unreal, and it is
exactly what Unreal's OBJ importer already does to every vertex (it negates Y),
so OBJ actors take scale (1, 1, 1). Yaw follows: a terrain heading of h degrees
counter-clockwise from east is the Unreal yaw -h, and a compass bearing b
(clockwise from north) is the Unreal yaw b - 90.

Every Unreal script converts through this module, so the rule lives in one
place. Plain Python, no ``unreal``: tests/test_level_frame.py pins it.
"""

import math

# The OBJ importer negates Y, which is the terrain -> level map: no actor flip.
OBJ_ACTOR_SCALE = (1.0, 1.0, 1.0)


def to_level(east, north):
    """Level (x, y) of a terrain-frame point, same units."""
    return east, -north


def to_level_cm(east_m, north_m):
    """Level (x, y) in cm of a terrain-frame point in metres."""
    return east_m * 100.0, -north_m * 100.0


def from_level(x, y):
    """Terrain-frame (east, north) of a level point, same units."""
    return x, -y


def level_yaw(heading_deg):
    """Unreal yaw for a terrain-frame heading (degrees counter-clockwise from east).

    The manifest's ``yaw_deg`` values are such headings (0 east, 90 north).
    """
    return -heading_deg


def heading(level_yaw_deg):
    """Terrain-frame heading (degrees counter-clockwise from east) of an Unreal yaw."""
    return -level_yaw_deg


def compass_yaw(bearing_deg):
    """Unreal yaw for a compass bearing (degrees clockwise from north), in [-180, 180)."""
    return (bearing_deg - 90.0 + 180.0) % 360.0 - 180.0


def forward(yaw_deg):
    """Level (x, y) unit vector a yaw faces."""
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def right_of(yaw_deg):
    """Level (x, y) unit vector to the right of a yaw (Unreal: yaw + 90)."""
    return forward(yaw_deg + 90.0)


def left_of(yaw_deg):
    """Level (x, y) unit vector to the left of a yaw (Unreal: yaw - 90)."""
    return forward(yaw_deg - 90.0)


def level_rotator(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """(yaw, pitch, roll) in the level for a rotation given in the terrain frame.

    Mirroring Y conjugates a rotation: turns about Z (yaw) and X (roll) change
    sign, turns about Y (pitch) do not.
    """
    return -yaw_deg, pitch_deg, -roll_deg


def raster_uv_transform(x0_cm, y1_cm, width_cm, height_cm):
    """(origin, inverse size) so that a material's uv = (level XY - origin) * inverse size.

    For a north-up raster in the terrain frame (an image: column 0 at x0_cm,
    the west edge; row 0 at y1_cm, the north edge; rows running south), such
    as the scan's fade mask. Level Y is -north, so v grows with level Y.
    """
    ox, oy = to_level(float(x0_cm), float(y1_cm))
    return (ox, oy), (1.0 / float(width_cm), 1.0 / float(height_cm))


def scan_actor_transform(scale, yaw_deg, mirror, centre, offset):
    """Level transform of the ml0w scan's chunk actors from tools/fit_scan.py's fit.

    The fit maps a scan vertex p (the imported chunk's local x, y) into the
    terrain frame: ``(east, north) = R(yaw) . diag(mirror * s, s) . (p - centre) + offset``.
    In the level that is ``R(-yaw) . diag(mirror * s, -s) . (p - centre) + to_level(offset)``.
    With ``mirror == -1`` (the scan imports correctly handed) the scale's two
    signs cancel into a half turn, so the actor needs no mirror at all.

    Returns ((x, y) location of the actor, yaw deg, (sx, sy) scale): the actor
    transform is ``R_unreal(yaw) . diag(sx, sy) . p + location``.
    """
    sx, sy, yaw = mirror * scale, -scale, -yaw_deg
    if sx < 0.0 and sy < 0.0:
        sx, sy, yaw = -sx, -sy, yaw + 180.0
    yaw = (yaw + 180.0) % 360.0 - 180.0
    r = math.radians(yaw)
    c, s = math.cos(r), math.sin(r)
    px, py = sx * centre[0], sy * centre[1]
    ox, oy = to_level(offset[0], offset[1])
    return (ox - (c * px - s * py), oy - (s * px + c * py)), yaw, (sx, sy)
