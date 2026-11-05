"""Create the Red Rocks materials and apply them to the imported meshes.

Run headless (close the editor first):
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_materials.py -unattended -nosplash -stdout
or inside a running editor console:
    py "/Users/austin/dev/redrocks-ue/unreal/setup_materials.py"

Creates under /Game/RedRocks/Materials, all procedural (world-position noise,
no textures to import):

    M_Sandstone         master for rock and ground. Fountain Formation red
                        sandstone in strata dipping 30 deg west: wide pale and
                        iron bands, warped by noise so they are not ruled lines,
                        fading to their average colour with distance so they
                        never shimmer. Steep ground turns to that rock; flat
                        ground is red soil with patches of olive and dry grass.
    MI_Sandstone_Rock   instance for Creation_Rock and Ship_Rock: all rock
    MI_Sandstone_Ground instance for the terrain: soil and grass, rock on slopes
    M_Benches           the terraced rows (redrocks.seating): dusty sandstone
                        treads on top, stacked red flagstone on the risers
    M_BenchWood         the bench planks: pale weathered boards with seams
    M_Steel             bench brackets and the top handrail: dark, rust-tinted
    M_Roof              the sound booth's posts and roof: maroon painted metal
    M_Concrete          stage: warm grey with stains and speckle
    M_Asphalt           roads: oxidised asphalt, worn patches, dusty edges
    M_Lot               parking: the same asphalt without road edges
    M_Gravel            paths: red decomposed-granite gravel, dusty edges

Then sets the StaticMesh assets' materials and saves them. setup_gameplay.py
assigns the stage, road, path and parking materials after it imports those
meshes; it reuses these assets when they exist. Safe to re-run: every graph is
rebuilt, existing instances are reused. Log: /tmp/redrocks_materials.log.
"""

import math
import os

import unreal

GAME_DIR = "/Game/RedRocks"
MAT_DIR = GAME_DIR + "/Materials"
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_materials.log")

log = []

lib = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary


def _step(label, fn, *args, **kwargs):
    """Run one step, log success or failure, never raise."""
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def srgb_to_linear(hex_color):
    """'#B5532E' -> unreal.LinearColor (vector parameters are linear)."""
    h = hex_color.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return unreal.LinearColor(r=out[0], g=out[1], b=out[2], a=1.0)


# Bedding dips 30 deg to the west (X east, Y north, Z up): the planes descend
# toward -X, i.e. z = x * tan30, whose upward unit normal is (-sin30, 0, cos30).
DIP = math.radians(30.0)
BED_NORMAL = (-math.sin(DIP), 0.0, math.cos(DIP))

# Noise feature sizes, cm.
MACRO_CM = 30000.0  # hillside-scale patches of grass and soil
MID_CM = 2500.0     # weathering stains, band warp
FINE_CM = 180.0     # clumps and grain
MICRO_CM = 25.0     # grit, faded out beyond a few metres

# ----------------------------------------------------------- sandstone presets
# Fountain Formation: deep red-orange arkose with pale buff and iron-dark beds.
SANDSTONE_DEFAULTS = {
    "scalars": {
        "BandThickness": 600.0,   # cm, coarse beds; fine beds are 1/3.7 of it
        "BandContrast": 0.75,
        "BandWarp": 350.0,        # cm the beds wander
        "BandFadeNear": 4000.0,   # cm: full bands nearer than this
        "BandFadeFar": 30000.0,   # cm: average colour beyond this
        "DustAmount": 0.15,
        "SlopeRed": 1.0,          # how hard steep ground turns to rock
        "RockStart": 0.42,        # 1 - normal.z where rock begins (~55 deg)
        "RockAmount": 0.0,        # 1 = rock everywhere
        "GrassAmount": 0.0,
        "Variation": 0.25,
    },
    "vectors": {
        "BaseColor": "#A2482A",
        "BuffColor": "#C98A5E",
        "IronColor": "#5C2416",
        "DustColor": "#B27D58",
        "SoilColor": "#7E4530",
        "GrassColor": "#5F6236",
        "DryGrassColor": "#8E8058",
    },
}
ROCK_INSTANCE = {"scalars": {"RockAmount": 1.0}, "vectors": {}}
GROUND_INSTANCE = {
    "scalars": {"GrassAmount": 0.6, "DustAmount": 0.1, "SlopeRed": 2.2,
                "BandThickness": 700.0, "BandContrast": 0.6},
    "vectors": {},
}


class Graph:
    """Small helper for wiring a material graph from Python.

    Inputs may be expression nodes or plain numbers (which become Constant
    nodes). Node positions are only for a readable graph in the editor.
    """

    def __init__(self, mat):
        self.mat = mat
        self.n = 0

    def node(self, cls_name, **props):
        self.n += 1
        x = -2400 + 220 * (self.n % 12)
        y = -1200 + 160 * (self.n // 12)
        node = lib.create_material_expression(self.mat, getattr(unreal, cls_name), x, y)
        for key, value in props.items():
            node.set_editor_property(key, value)
        return node

    def _in(self, value):
        if isinstance(value, (int, float)):
            return self.node("MaterialExpressionConstant", r=float(value))
        return value

    @staticmethod
    def link(src, dst, pin, src_pin=""):
        if not lib.connect_material_expressions(src, src_pin, dst, pin):
            raise RuntimeError("connect failed: %s -> %s.%s" % (src, dst, pin))

    def link1(self, src, dst):
        """Single-input nodes name their pin '' or 'Input' depending on class."""
        for pin in ("", "Input"):
            try:
                if lib.connect_material_expressions(src, "", dst, pin):
                    return
            except Exception:
                pass
        raise RuntimeError("could not connect %s -> %s" % (src, dst))

    def binop(self, cls_name, a, b):
        n = self.node(cls_name)
        self.link(self._in(a), n, "A")
        self.link(self._in(b), n, "B")
        return n

    def mul(self, a, b):
        return self.binop("MaterialExpressionMultiply", a, b)

    def add(self, a, b):
        return self.binop("MaterialExpressionAdd", a, b)

    def sub(self, a, b):
        return self.binop("MaterialExpressionSubtract", a, b)

    def div(self, a, b):
        return self.binop("MaterialExpressionDivide", a, b)

    def dot(self, a, b):
        return self.binop("MaterialExpressionDotProduct", a, b)

    def append(self, a, b):
        return self.binop("MaterialExpressionAppendVector", a, b)

    def lerp(self, a, b, t):
        n = self.node("MaterialExpressionLinearInterpolate")
        self.link(self._in(a), n, "A")
        self.link(self._in(b), n, "B")
        self.link(self._in(t), n, "Alpha")
        return n

    def unary(self, cls_name, src, **props):
        n = self.node(cls_name, **props)
        self.link1(self._in(src), n)
        return n

    def sat(self, a):
        return self.unary("MaterialExpressionSaturate", a)

    def frac(self, a):
        return self.unary("MaterialExpressionFrac", a)

    def floor(self, a):
        return self.unary("MaterialExpressionFloor", a)

    def absv(self, a):
        return self.unary("MaterialExpressionAbs", a)

    def one_minus(self, a):
        return self.unary("MaterialExpressionOneMinus", a)

    def mask(self, a, channel):
        flags = {c: (c == channel) for c in "rgba"}
        return self.unary("MaterialExpressionComponentMask", a, **flags)

    def smooth(self, lo, hi, x):
        """saturate((x - lo) / (hi - lo)), a linear ramp (cheap smoothstep)."""
        return self.sat(self.mul(self.sub(x, lo), 1.0 / (hi - lo)))

    def scalar(self, name, value):
        return self.node("MaterialExpressionScalarParameter",
                         parameter_name=name, default_value=float(value))

    def vector(self, name, hex_color):
        return self.node("MaterialExpressionVectorParameter",
                         parameter_name=name, default_value=srgb_to_linear(hex_color))

    def color(self, hex_color):
        return self.node("MaterialExpressionConstant3Vector",
                         constant=srgb_to_linear(hex_color))

    def vec3(self, x, y, z):
        return self.node("MaterialExpressionConstant3Vector",
                         constant=unreal.LinearColor(r=x, g=y, b=z, a=1.0))

    def world_pos(self):
        return self.node("MaterialExpressionWorldPosition")

    def depth(self):
        return self.node("MaterialExpressionPixelDepth")

    def normal_z(self):
        return self.mask(self.node("MaterialExpressionVertexNormalWS"), "b")

    def uv(self):
        return self.node("MaterialExpressionTextureCoordinate", coordinate_index=0)

    def noise(self, position, size_cm, levels=2, func="SIMPLEX_TEX"):
        """0..1 noise with features about size_cm across."""
        n = self.node(
            "MaterialExpressionNoise",
            scale=1.0 / size_cm, levels=levels, quality=1,
            noise_function=getattr(unreal.NoiseFunction, "NOISEFUNCTION_" + func),
            output_min=0.0, output_max=1.0, turbulence=False,
        )
        self.link(position, n, "World Position")
        return n

    def near_fade(self, start_cm, end_cm):
        """1 near the camera, 0 beyond end_cm: for detail that would alias."""
        return self.one_minus(self.smooth(start_cm, end_cm, self.depth()))

    def output(self, base_color, roughness, metallic=0.0, normal=None):
        mp = unreal.MaterialProperty
        lib.connect_material_property(base_color, "", mp.MP_BASE_COLOR)
        lib.connect_material_property(self._in(roughness), "", mp.MP_ROUGHNESS)
        lib.connect_material_property(self._in(metallic), "", mp.MP_METALLIC)
        if normal is not None:
            lib.connect_material_property(normal, "", mp.MP_NORMAL)


def _band(g, coord, thickness, contrast):
    """0..1 sine band mask: sin(2*pi*coord/thickness) * contrast * 0.5 + 0.5."""
    sine = g.unary("MaterialExpressionSine", g.div(coord, thickness), period=1.0)
    return g.sat(g.add(g.mul(g.mul(sine, contrast), 0.5), 0.5))


# --------------------------------------------------------------------- graphs

def build_sandstone(mat):
    """Rock and ground in one graph; the instances move the balance."""
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    p = {k: g.scalar(k, v) for k, v in SANDSTONE_DEFAULTS["scalars"].items()}
    c = {k: g.vector(k, v) for k, v in SANDSTONE_DEFAULTS["vectors"].items()}

    wpos = g.world_pos()
    n_macro = g.noise(wpos, MACRO_CM, levels=3)
    n_mid = g.noise(wpos, MID_CM, levels=3)
    n_fine = g.noise(wpos, FINE_CM, levels=2)
    n_micro = g.noise(wpos, MICRO_CM, levels=1)
    depth = g.depth()

    # Strata coordinate: cm along the bedding normal, wandered by mid noise.
    coord = g.dot(wpos, g.vec3(*BED_NORMAL))
    coord = g.add(coord, g.mul(g.sub(n_mid, 0.5), p["BandWarp"]))
    # Distance fade: lerp each band mask toward its mean (0.5) so far bands
    # resolve to their average colour instead of aliasing into stripes. The
    # fine beds fade three times sooner.
    fade = g.sat(g.div(g.sub(depth, p["BandFadeNear"]),
                       g.sub(p["BandFadeFar"], p["BandFadeNear"])))
    fade_fine = g.sat(g.mul(fade, 3.0))
    coarse = g.lerp(_band(g, coord, p["BandThickness"], p["BandContrast"]), 0.5, fade)
    fine = g.lerp(_band(g, coord, g.mul(p["BandThickness"], 1.0 / 3.7), p["BandContrast"]),
                  0.5, fade_fine)

    rock = g.lerp(c["BaseColor"], c["BuffColor"], g.mul(coarse, 0.8))
    rock = g.lerp(rock, c["IronColor"], g.mul(fine, 0.45))
    # Weathering: broad darker/lighter stains, then grain.
    stain = g.lerp(g.sub(1.0, p["Variation"]), g.add(1.0, g.mul(p["Variation"], 0.5)), n_mid)
    rock = g.mul(rock, stain)
    rock = g.mul(rock, g.lerp(0.88, 1.08, n_fine))
    rock = g.lerp(rock, c["DustColor"], g.mul(p["DustAmount"], n_macro))

    # Ground: red soil, with grass where the noise clears a threshold on flat
    # ground. Coverage mixes three scales so patches have ragged edges.
    soil = g.lerp(c["SoilColor"], c["DustColor"], g.mul(n_fine, 0.5))
    grass = g.lerp(c["GrassColor"], c["DryGrassColor"], n_mid)
    grass = g.mul(grass, g.lerp(0.8, 1.1, n_fine))
    cover = g.add(g.add(g.mul(n_macro, 0.5), g.mul(n_mid, 0.38)), g.mul(n_fine, 0.12))
    nz = g.sat(g.normal_z())
    flat = g.smooth(0.65, 0.9, nz)
    grass_mask = g.mul(g.sat(g.add(g.mul(g.sub(g.add(cover, p["GrassAmount"]), 1.0), 3.5), 0.5)), flat)
    grass_mask = g.mul(grass_mask, g.sat(g.mul(p["GrassAmount"], 4.0)))
    ground = g.lerp(soil, grass, grass_mask)

    # Rock where the ground is steep, with a ragged noisy edge.
    steep = g.one_minus(nz)
    rock_mask = g.add(g.mul(g.sub(steep, p["RockStart"]), g.mul(p["SlopeRed"], 4.0)),
                      g.mul(g.sub(n_mid, 0.5), 0.8))
    rock_mask = g.sat(g.add(rock_mask, p["RockAmount"]))
    col = g.lerp(ground, rock, rock_mask)

    # Grit close to the camera only.
    grit = g.lerp(1.0, g.lerp(0.86, 1.1, n_micro), g.near_fade(800.0, 3000.0))
    col = g.mul(col, grit)

    rough = g.lerp(0.95, 0.86, g.mul(rock_mask, n_fine))
    g.output(col, rough)


def build_benches(mat):
    """The terraces: dusty sandstone treads, stacked red flagstone risers.

    Terrace UVs are metres (redrocks.seating): U along the row; V across the
    tread on top. The risers' flagstone courses use world height, so they run
    level whatever depth the buried bottom reaches: thin courses (about 8 cm,
    as in the photos) of stones 22-72 cm long with wandering joints, each its
    own red-brown. Stairs.obj (the row-end aisle stairs) takes it too.
    """
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    uv = g.uv()
    u = g.mask(uv, "r")
    wpos = g.world_pos()
    wz = g.mask(wpos, "b")
    near = g.near_fade(1500.0, 7000.0)

    # Courses: world Z over ~8 cm, warped so they wander like hand-laid flagstone.
    warp = g.mul(g.sub(g.noise(wpos, 90.0, levels=2), 0.5), 4.0)
    c = g.div(g.add(wz, warp), 8.0)
    course = g.floor(c)
    cf = g.frac(c)
    # Each course has its own stone length and offset (a noise lookup on its
    # index), and the joints wander along it so the stones are not bricks.
    pick = g.noise(g.append(g.append(g.mul(course, 1931.0), 0.0), 0.0), 400.0, levels=1)
    length = g.add(0.22, g.mul(pick, 0.5))  # metres
    jitter = g.mul(g.sub(g.noise(g.append(g.append(g.mul(u, 100.0), g.mul(course, 1931.0)), 0.0), 45.0, levels=1), 0.5), 0.9)
    sc = g.add(g.div(g.add(u, g.mul(pick, 7.0)), length), jitter)
    stone = g.floor(sc)
    sf = g.frac(sc)
    bed = g.add(g.smooth(0.88, 0.97, cf), g.one_minus(g.smooth(0.0, 0.06, cf)))
    head = g.add(g.smooth(0.95, 0.99, sf), g.one_minus(g.smooth(0.0, 0.02, sf)))
    joint = g.mul(g.sat(g.add(bed, g.mul(head, 0.8))), near)
    tint = g.noise(g.append(g.append(g.mul(stone, 713.0), g.mul(course, 1931.0)), 0.0), 1000.0, levels=1)
    # Lyons/Fountain sandstone: dark red-brown to rust, a few paler buff stones.
    rock = g.lerp(g.color("#5E2618"), g.color("#8E4227"), tint)
    rock = g.lerp(rock, g.color("#A0694C"), g.mul(g.smooth(0.8, 0.97, tint), 0.7))
    blot = g.noise(wpos, 25.0, levels=2)
    rock = g.mul(rock, g.lerp(0.82, 1.1, blot))
    grain = g.noise(wpos, 4.0, levels=1, func="VALUE_ALU")
    rock = g.mul(rock, g.lerp(1.0, g.lerp(0.9, 1.08, grain), near))
    # Red dust settles on the courses' top edges.
    rock = g.lerp(rock, g.color("#9A6A50"), g.mul(g.smooth(0.6, 0.88, cf), 0.25))
    riser = g.lerp(rock, g.color("#2A140D"), g.mul(joint, 0.75))

    # Treads: packed red-tan sandstone and dust, worn paler where people walk.
    blotch = g.noise(wpos, 350.0, levels=3)
    tread = g.lerp(g.color("#A2735A"), g.color("#86573F"), g.mul(g.smooth(0.3, 0.8, blotch), 0.7))
    worn = g.noise(wpos, 2500.0, levels=2)
    tread = g.lerp(tread, g.color("#B58C70"), g.mul(g.smooth(0.55, 0.85, worn), 0.35))
    grit = g.noise(wpos, 3.0, levels=1, func="VALUE_ALU")
    tread = g.mul(tread, g.lerp(1.0, g.lerp(0.88, 1.1, grit), g.near_fade(300.0, 1500.0)))

    top = g.smooth(0.55, 0.75, g.sat(g.normal_z()))
    col = g.lerp(riser, tread, top)
    rough = g.lerp(g.lerp(0.88, 0.97, joint), 0.95, top)
    g.output(col, rough)


def build_bench_wood(mat):
    """The bench planks: pale, sun-greyed boards with dark seams and grain.

    Plank UVs are metres: U along the row, V across one 14 cm board on its top.
    """
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    uv = g.uv()
    u = g.mask(uv, "r")
    v = g.mask(uv, "g")
    # Boards are 3.6 m long; each gets its own tint and a dark butt joint.
    board = g.floor(g.div(u, 3.6))
    board_tint = g.noise(g.append(g.append(g.mul(board, 1700.0), 0.0), 0.0), 1000.0, levels=1)
    butt = g.smooth(0.995, 0.9995, g.frac(g.div(u, 3.6)))
    # Edges of the board's top: rounded, darker.
    edge = g.smooth(0.85, 0.98, g.mul(g.absv(g.sub(g.div(v, 0.14), 0.5)), 2.0))
    grain_pos = g.append(g.append(g.mul(u, 60.0), g.mul(v, 2500.0)), g.mul(board, 977.0))
    grain = g.noise(grain_pos, 100.0, levels=3)
    wpos = g.world_pos()
    weather = g.noise(wpos, 1500.0, levels=2)

    wood = g.lerp(g.color("#B49A7C"), g.color("#7E6650"), g.mul(grain, 0.6))
    wood = g.lerp(wood, g.color("#A79F92"), g.mul(g.add(g.mul(weather, 0.6), g.mul(board_tint, 0.4)), 0.55))
    wood = g.mul(wood, g.lerp(0.88, 1.1, board_tint))
    near = g.near_fade(1500.0, 6000.0)
    wood = g.lerp(wood, g.color("#2A1E16"), g.mul(g.sat(g.add(g.mul(edge, 0.6), butt)), near))
    g.output(wood, 0.8)


def build_roof(mat):
    """The sound booth's roof and posts: maroon painted metal, sun-faded on top."""
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    wpos = g.world_pos()
    fade = g.smooth(0.4, 0.9, g.noise(wpos, 300.0, levels=2))
    top = g.smooth(0.55, 0.75, g.sat(g.normal_z()))
    col = g.lerp(g.color("#6E2620"), g.color("#8E3A2E"), g.mul(fade, g.add(0.3, g.mul(top, 0.5))))
    g.output(col, g.lerp(0.5, 0.7, fade), metallic=0.3)


def build_steel(mat):
    """Bench brackets and the top handrail: dark painted steel, rust coming through."""
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    wpos = g.world_pos()
    rust = g.smooth(0.55, 0.85, g.noise(wpos, 40.0, levels=3))
    col = g.lerp(g.color("#262524"), g.color("#5A3420"), g.mul(rust, 0.7))
    # Painted, not bare: from the top row a metallic strap mirrored the sky and
    # the bracket lines read white; in the photos they are dark.
    g.output(col, g.lerp(0.65, 0.9, rust), metallic=0.1)


def _concrete_color(g, wpos, light_hex, dark_hex):
    blotch = g.noise(wpos, 400.0, levels=3)
    speck = g.noise(wpos, 6.0, levels=1, func="VALUE_ALU")
    col = g.lerp(g.color(light_hex), g.color(dark_hex), g.mul(g.smooth(0.3, 0.8, blotch), 0.6))
    return g.mul(col, g.lerp(1.0, g.lerp(0.9, 1.08, speck), g.near_fade(300.0, 1500.0)))


def build_concrete(mat):
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    wpos = g.world_pos()
    col = _concrete_color(g, wpos, "#ADA597", "#8E8577")
    # A red-dust tint from the hillside, in broad patches.
    dust = g.noise(wpos, 2500.0, levels=2)
    col = g.lerp(col, g.color("#9C6F55"), g.mul(g.smooth(0.55, 0.85, dust), 0.35))
    g.output(col, 0.85)


def _asphalt_color(g, wpos, fresh_hex, worn_hex):
    worn = g.noise(wpos, 1800.0, levels=3)
    stones = g.noise(wpos, 3.0, levels=1, func="VALUE_ALU")
    col = g.lerp(g.color(fresh_hex), g.color(worn_hex), g.mul(g.smooth(0.3, 0.85, worn), 0.7))
    return g.mul(col, g.lerp(1.0, g.lerp(0.8, 1.25, stones), g.near_fade(200.0, 1200.0)))


def _dusty_edges(g, col, wpos, soil_hex, width):
    """Blend toward red soil near the ribbon edges (ribbon V runs 0..1 across)."""
    v = g.mask(g.uv(), "g")
    centre = g.absv(g.sub(g.frac(v), 0.5))  # 0 mid-ribbon, 0.5 at the edges
    ragged = g.mul(g.noise(wpos, 120.0, levels=2), 0.12)
    edge = g.smooth(0.5 - width, 0.5, g.add(centre, ragged))
    return g.lerp(col, g.color(soil_hex), g.mul(edge, 0.85))


def build_asphalt(mat):
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    wpos = g.world_pos()
    col = _asphalt_color(g, wpos, "#4A4845", "#67635D")
    col = _dusty_edges(g, col, wpos, "#8A5A40", 0.14)
    g.output(col, 0.82)


def build_lot(mat):
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    wpos = g.world_pos()
    col = _asphalt_color(g, wpos, "#57544F", "#706B64")
    g.output(col, 0.85)


def build_gravel(mat):
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    wpos = g.world_pos()
    patches = g.noise(wpos, 600.0, levels=2)
    pebbles = g.noise(wpos, 4.0, levels=1, func="VORONOI_ALU")
    col = g.lerp(g.color("#9B6A4B"), g.color("#B98F6C"), patches)
    col = g.mul(col, g.lerp(1.0, g.lerp(0.75, 1.2, pebbles), g.near_fade(300.0, 1500.0)))
    col = _dusty_edges(g, col, wpos, "#8C4F33", 0.2)
    g.output(col, 0.95)


def build_buildings(mat):
    """Pueblo Revival sandstone/adobe walls with dark weathered roofs on top faces."""
    lib.delete_all_material_expressions(mat)
    g = Graph(mat)
    wpos = g.world_pos()
    wall = _concrete_color(g, wpos, "#B67856", "#8E553A")
    roof = _concrete_color(g, wpos, "#5A534D", "#423D39")
    top = g.smooth(0.55, 0.75, g.sat(g.normal_z()))
    col = g.lerp(wall, roof, top)
    rough = g.lerp(0.88, 0.82, top)
    g.output(col, rough)


# --------------------------------------------------------------- asset helpers

def _get_or_create(name, asset_class, factory):
    path = "%s/%s" % (MAT_DIR, name)
    if eal.does_asset_exist(path):
        log.append("reuse  %s" % path)
        return eal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset = tools.create_asset(name, MAT_DIR, asset_class, factory)
    log.append("create %s" % path)
    return asset


def _save_asset(asset):
    if not eal.save_loaded_asset(asset, False):
        raise RuntimeError("save failed for %s (read-only file?)" % asset.get_path_name())


def make_material(name, builder):
    mat = _get_or_create(name, unreal.Material, unreal.MaterialFactoryNew())
    if mat is None:
        raise RuntimeError("could not create %s" % name)
    _step("build graph %s" % name, builder, mat)
    _step("recompile %s" % name, lib.recompile_material, mat)
    _step("save %s" % name, _save_asset, mat)
    return mat


def make_instance(name, parent, preset):
    inst = _get_or_create(
        name, unreal.MaterialInstanceConstant,
        unreal.MaterialInstanceConstantFactoryNew(),
    )
    if inst is None:
        raise RuntimeError("could not create %s" % name)
    _step("parent %s" % name, inst.set_editor_property, "parent", parent)
    # Clear old overrides so a parameter dropped from the preset falls back to
    # the master's default instead of keeping a stale value.
    if hasattr(lib, "clear_all_material_instance_parameters"):
        _step("clear %s" % name, lib.clear_all_material_instance_parameters, inst)
    for key, value in preset["scalars"].items():
        _step("%s.%s" % (name, key),
              lib.set_material_instance_scalar_parameter_value, inst, key, value)
    for key, hex_color in preset["vectors"].items():
        _step("%s.%s" % (name, key),
              lib.set_material_instance_vector_parameter_value,
              inst, key, srgb_to_linear(hex_color))
    _step("update %s" % name, lib.update_material_instance, inst)
    _step("save %s" % name, _save_asset, inst)
    return inst


def apply_to_mesh(mesh_name, material):
    path = "%s/%s" % (GAME_DIR, mesh_name)
    if not eal.does_asset_exist(path):
        log.append("skip  %s (no such asset)" % path)
        return False
    mesh = eal.load_asset(path)
    if mesh is None or material is None:
        raise RuntimeError("could not load mesh or material for %s" % path)
    for i in range(max(1, len(mesh.get_editor_property("static_materials")))):
        mesh.set_material(i, material)
    _step("save mesh %s" % mesh_name, _save_asset, mesh)
    log.append("applied %s -> %s" % (material.get_name(), mesh_name))
    return True


# Built here; setup_gameplay.py assigns the ones for meshes it imports.
SURFACES = [
    ("M_Benches", build_benches, "Benches"),
    ("M_BenchWood", build_bench_wood, "Bench_Planks"),
    ("M_Steel", build_steel, "Bench_Brackets"),
    ("M_Roof", build_roof, "Booth"),
    ("M_Concrete", build_concrete, "Stage"),
    ("M_Asphalt", build_asphalt, "Roads"),
    ("M_Lot", build_lot, "Parking"),
    ("M_Gravel", build_gravel, "Paths"),
    ("M_Buildings", build_buildings, "Buildings"),
]


def main():
    if not eal.does_directory_exist(MAT_DIR):
        eal.make_directory(MAT_DIR)

    sandstone = _step("M_Sandstone", make_material, "M_Sandstone", build_sandstone)

    rock = ground = None
    if sandstone is not None:
        rock = _step("MI_Sandstone_Rock", make_instance,
                     "MI_Sandstone_Rock", sandstone, ROCK_INSTANCE)
        ground = _step("MI_Sandstone_Ground", make_instance,
                       "MI_Sandstone_Ground", sandstone, GROUND_INSTANCE)

    for name in ("Creation_Rock", "Ship_Rock", "Stage_Rock", "Rocks"):
        _step("apply %s" % name, apply_to_mesh, name, rock or sandstone)
    _step("apply RedRocks_Terrain", apply_to_mesh, "RedRocks_Terrain",
          ground or sandstone)

    for mat_name, builder, mesh_name in SURFACES:
        mat = _step(mat_name, make_material, mat_name, builder)
        if mat is not None and mesh_name:
            _step("apply %s" % mesh_name, apply_to_mesh, mesh_name, mat)
        if mat_name == "M_Concrete" and mat is not None:
            _step("apply Walls", apply_to_mesh, "Walls", mat)
        if mat_name == "M_Steel" and mat is not None:
            _step("apply Handrail", apply_to_mesh, "Handrail", mat)

    log.append("done")


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
