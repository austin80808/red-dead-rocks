"""Build the sasquatch from the UE5 mannequin: `redrocks sasquatch`.

Reads the mannequin glTF that unreal/export_mannequin.py writes, and writes
    sasquatch.glb      the skinned mesh: body and eyes, the mannequin's bones in his proportions
    fur_body.rrgroom   body fur, strands in the mesh's space
    fur_head.rrgroom   head and neck fur (hidden from your own view in first person)
    manifest.json      measured proportions, strand counts and the gait settings
                       setup_sasquatch.py puts in the retargeter

Steps (all in the body frame: cm, Z up, facing +Y, his left on +X):
  1. weld the mannequin into one surface with dense skin weights
  2. re-proportion the bones (rig.reproportion) and scale to his height
  3. add mass where an ape carries it (trapezius, belly, buttocks, calves)
  4. refine the head, hands and feet, sculpt the head onto sdf.HEAD
  5. flat soles and splayed toes; eyes
  6. fur fields and strands; masks for the skin material
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gltf, groom, mesh_ops, rig, sdf

TARGET_HEIGHT_CM = 255.0
# sdf.HEAD is drawn for a 40 cm head (jaw to crest); this brings it to about 35 cm.
HEAD_SCALE = 0.88
SEGMENTS = {
    "pelvis": rig.Segment(1.0, 1.18),
    "spine_01": rig.Segment(1.0, 1.24),
    "spine_02": rig.Segment(1.0, 1.28),
    "spine_03": rig.Segment(1.0, 1.30),
    "spine_04": rig.Segment(0.96, 1.30),
    "spine_05": rig.Segment(0.86, 1.30),
    "neck_01": rig.Segment(0.45, 1.55),
    "neck_02": rig.Segment(0.45, 1.45),
    "clavicle": rig.Segment(1.22, 1.30),
    "upperarm": rig.Segment(1.16, 1.32),
    "lowerarm": rig.Segment(1.20, 1.30),
    "hand": rig.Segment(1.20, 1.24),
    "thigh": rig.Segment(0.94, 1.30),
    "calf": rig.Segment(0.94, 1.24),
    "foot": rig.Segment(1.24, 1.30),
    "ball": rig.Segment(1.12, 1.32),
}
for _f in ("thumb", "index", "middle", "ring", "pinky"):
    SEGMENTS[_f + "_metacarpal"] = rig.Segment(1.12, 1.28)
    for _i in ("01", "02", "03"):
        SEGMENTS["%s_%s" % (_f, _i)] = rig.Segment(1.10, 1.38)

# His retarget pose: knees bent (pelvis lowered, as a fraction of hip height)
# and the trunk pitched forward, head and neck lifted back so he looks ahead.
GAIT = {"pelvis_drop": 0.07, "pitch_deg": {"spine_01": 7.0, "spine_03": 5.0, "neck_01": -7.0, "head": -5.0}}


@dataclass
class Body:
    pos: np.ndarray        # (n, 3)
    faces: np.ndarray      # (m, 3)
    weights: np.ndarray    # (n, J) dense skin weights
    names: list[str]
    parents: np.ndarray
    heads: np.ndarray      # (J, 3) joint positions
    globals_: np.ndarray   # (J, 4, 4) joint rest pose, body frame

    def bone(self, name: str) -> int:
        return self.names.index(name)

    def w(self, *names: str) -> np.ndarray:
        """Summed weight of the named bones (both sides when a name has no _l/_r)."""
        idx = [i for i, n in enumerate(self.names)
               if n in names or (n[:-2] in names and n.endswith(("_l", "_r")))]
        return self.weights[:, idx].sum(axis=1) if idx else np.zeros(len(self.pos))

    def normals(self) -> np.ndarray:
        return mesh_ops.vertex_normals(self.pos, self.faces)


# ---------------------------------------------------------------- frames

_K = np.eye(4)
_K[:3, :3] = gltf.body_frame() * gltf.GLTF_TO_CM
_K_INV = np.linalg.inv(_K)


def to_body(m: np.ndarray) -> np.ndarray:
    return _K @ m @ _K_INV


def from_body(m: np.ndarray) -> np.ndarray:
    return _K_INV @ m @ _K


# ---------------------------------------------------------------- 1. weld

def load_body(model: gltf.Model) -> Body:
    skin = model.skin
    g_glb = skin.global_matrices()
    bind = np.einsum("jab,jbc->jac", g_glb, skin.inverse_bind)       # rest x inverse bind (identity when they agree)
    pos_all, faces_all, w_all = [], [], []
    offset = 0
    nj = len(skin.names)
    for prim in model.primitives:
        p = np.zeros_like(prim.positions)
        hom = np.concatenate([prim.positions, np.ones((len(p), 1))], axis=1)
        dense = np.zeros((len(p), nj))
        for k in range(prim.joints.shape[1]):
            j = prim.joints[:, k]
            w = prim.weights[:, k]
            p += w[:, None] * np.einsum("nab,nb->na", bind[j], hom)[:, :3]
            np.add.at(dense, (np.arange(len(p)), j), w)
        pos_all.append(p)
        faces_all.append(prim.indices + offset)
        w_all.append(dense)
        offset += len(p)
    pos = np.concatenate(pos_all) @ gltf.body_frame().T * gltf.GLTF_TO_CM
    faces = np.concatenate(faces_all)[:, ::-1]                          # reflection flips winding back
    weights = np.concatenate(w_all)
    keep, remap = mesh_ops.weld(pos, 0.01)
    faces = mesh_ops.drop_degenerate(remap[faces])
    g_body = np.array([to_body(m) for m in g_glb])
    return Body(pos[keep], faces, weights[keep], list(skin.names), skin.parents.copy(),
                g_body[:, :3, 3].copy(), g_body)


# ---------------------------------------------------------- 2. proportions

def reproportion(body: Body, target_height: float = TARGET_HEIGHT_CM, crest_cm: float = 8.0) -> float:
    """Re-proportion in place; returns the overall scale used."""
    rp = rig.reproportion(body.names, body.parents, body.heads, SEGMENTS, scale=1.0)
    nj = len(body.names)
    joints = np.tile(np.arange(nj), (len(body.pos), 1))
    body.pos = rig.skin_points(rp, body.pos, joints, body.weights)
    body.heads = rp.heads
    height = body.pos[:, 2].max() - body.pos[:, 2].min()
    scale = (target_height - crest_cm) / height
    body.pos *= scale
    body.heads *= scale
    body.globals_[:, :3, 3] = body.heads
    return scale


# --------------------------------------------------------------- 3. mass

def add_mass(body: Body) -> None:
    """Push the skin out to fill an ape's bulk: smooth fields around the joints."""
    h = {n: body.heads[body.bone(n)] for n in body.names}
    n = body.normals()
    p = body.pos
    side = np.array([1.0, 0, 0])
    fwd = np.array([0, 1.0, 0])
    up = np.array([0, 0, 1.0])
    target = None
    shapes = []
    for s in (1.0, -1.0):
        sh = h["upperarm_l"] * [s, 1, 1] if s > 0 else h["upperarm_r"]
        neck_back = h["neck_01"] - fwd * 6 + up * 6
        shapes.append(sdf.tapered_capsule(p, neck_back, sh + up * 4 - fwd * 4, 13.0, 10.0))      # trapezius
        hip = h["thigh_l"] if s > 0 else h["thigh_r"]
        shapes.append(sdf.ellipsoid(p, hip - fwd * 10 + up * 2 + side * s * 2, (12, 11, 13)))   # buttocks
        knee = h["calf_l"] if s > 0 else h["calf_r"]
        ankle = h["foot_l"] if s > 0 else h["foot_r"]
        shapes.append(sdf.ellipsoid(p, knee + (ankle - knee) * 0.3 - fwd * 3, (8.5, 9.0, 14.0)))  # calves
        elbow = h["lowerarm_l"] if s > 0 else h["lowerarm_r"]
        wrist = h["hand_l"] if s > 0 else h["hand_r"]
        shapes.append(sdf.tapered_capsule(p, elbow + (wrist - elbow) * 0.15, elbow + (wrist - elbow) * 0.6, 8.5, 6.5))
    chest = h["spine_04"]
    belly = h["spine_02"]
    shapes.append(sdf.ellipsoid(p, belly + fwd * 9 - up * 4, (23, 17, 20)))                    # pot belly
    shapes.append(sdf.ellipsoid(p, chest + fwd * 4, (27, 19, 20)))                             # barrel chest
    shapes.append(sdf.ellipsoid(p, h["neck_01"] - fwd * 2 - up * 3, (16, 14, 13)))             # no neck: fill it
    target = shapes[0]
    for s in shapes[1:]:
        target = sdf.smin(target, s, 6.0)
    # Only flesh on the trunk, arms and legs moves (not hands, feet or head).
    movable = 1.0 - np.clip(body.w("hand", "head", "foot", "ball") * 1.5, 0, 1)
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        movable *= 1.0 - np.clip(sum(body.w("%s_%s" % (f, i)) for i in ("metacarpal", "01", "02", "03")) * 2, 0, 1)
    push = np.clip(-target, 0, 12.0) * movable
    body.pos = p + n * push[:, None]
    body.pos = mesh_ops.taubin_smooth(body.pos, body.faces, weight=np.clip(push / 3.0, 0, 1) * 0.8, iterations=6)


# ------------------------------------------------------------- 4. refine

def refine(body: Body, mask: np.ndarray) -> np.ndarray:
    """Subdivide faces touching `mask` vertices once; returns the new mask."""
    faces_in = mask[body.faces].any(axis=1)
    pos, faces, attrs, _ = mesh_ops.subdivide(body.pos, body.faces, faces_in,
                                              {"w": body.weights, "m": mask.astype(float)})
    body.pos, body.faces, body.weights = pos, faces, attrs["w"]
    return attrs["m"] > 0.25


def head_frame(body: Body) -> tuple[np.ndarray, np.ndarray]:
    """Origin (head joint) and 3x3 head->body axes (x forward, y his left, z up)."""
    left = np.array([1.0, 0, 0]) if body.heads[body.bone("hand_l")][0] > 0 else np.array([-1.0, 0, 0])
    axes = np.stack([np.array([0, 1.0, 0]), left, np.array([0, 0, 1.0])], axis=1)
    return body.heads[body.bone("head")].copy(), axes


def sculpt_head(body: Body, head_scale: float) -> dict:
    origin, axes = head_frame(body)
    hw = np.clip(body.w("head") + 0.6 * body.w("neck_02"), 0, 1)
    region = hw > 0.02
    local = ((body.pos[region] - origin) @ axes) / head_scale
    centre = np.array([3.0, 0.0, 3.0])
    moved = sdf.project_radially(local, centre, sdf.head_sdf)
    blend = np.clip(hw[region] * 1.25, 0, 1) ** 1.5
    new_local = local + (moved - local) * blend[:, None]
    body.pos[region] = origin + (new_local * head_scale) @ axes.T
    body.pos = mesh_ops.taubin_smooth(body.pos, body.faces, weight=(region & (hw < 0.9)) * 0.7, iterations=4)
    return {"origin": origin, "axes": axes, "scale": head_scale}


# ------------------------------------------------------------------- 5.

def shape_feet(body: Body) -> None:
    """Flat soles on the ground and wider, splayed fronts."""
    for s in ("l", "r"):
        fw = body.w("foot_" + s, "ball_" + s)
        sel = fw > 0.3
        if not sel.any():
            continue
        ground = body.pos[:, 2].min()
        z = body.pos[sel, 2]
        low = z < ground + 3.0
        body.pos[np.flatnonzero(sel)[low], 2] = ground + (z[low] - ground) * 0.25
        ball = body.heads[body.bone("ball_" + s)]
        front = body.pos[sel, 1] > ball[1] - 2
        widen = np.where(front, 1.18, 1.06)
        cx = ball[0]
        body.pos[sel, 0] = cx + (body.pos[sel, 0] - cx) * widen


def uv_sphere(centre: np.ndarray, radius: float, rings: int = 16, sides: int = 24) -> tuple[np.ndarray, np.ndarray]:
    pts = [centre + radius * np.array([0, 0, 1.0])]
    for i in range(1, rings):
        th = np.pi * i / rings
        for j in range(sides):
            ph = 2 * np.pi * j / sides
            pts.append(centre + radius * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    pts.append(centre - radius * np.array([0, 0, 1.0]))
    faces = []
    for j in range(sides):
        faces.append((0, 1 + j, 1 + (j + 1) % sides))
    for i in range(rings - 2):
        for j in range(sides):
            a = 1 + i * sides + j
            b = 1 + i * sides + (j + 1) % sides
            faces += [(a, a + sides, b + sides), (a, b + sides, b)]
    last = len(pts) - 1
    base = 1 + (rings - 2) * sides
    for j in range(sides):
        faces.append((base + j, last, base + (j + 1) % sides))
    return np.array(pts), np.array(faces)


# ------------------------------------------------------------------- 6.

def fur_fields(body: Body, head: dict) -> tuple[groom.FurField, np.ndarray, np.ndarray]:
    """Density, length, comb and the rest per vertex; also the bare-skin mask
    and the head-fur mask."""
    p = body.pos
    n = body.normals()
    nv = len(p)
    w = body.w
    origin, axes, hs = head["origin"], head["axes"], head["scale"]
    local = ((p - origin) @ axes) / hs                                    # head frame

    length = (
        15.0 * w("clavicle", "upperarm", "spine_05")
        + 14.0 * w("lowerarm")
        + 10.0 * w("spine_01", "spine_02", "spine_03", "spine_04", "pelvis")
        + 10.5 * w("thigh")
        + 7.5 * w("calf")
        + 3.0 * w("foot") + 1.0 * w("ball")
        + 3.0 * w("hand")
        + 11.0 * w("neck_01", "neck_02")
        + 8.0 * w("head")
    )
    fingers = sum(w("%s_%s" % (f, i)) for f in ("thumb", "index", "middle", "ring", "pinky")
                  for i in ("metacarpal", "01", "02", "03"))
    length = length + 1.3 * fingers
    density = np.full(nv, 4.2)

    # Face: bare in the centre (eyes, nose, mouth, between the brows), short hair round it.
    hw = w("head")
    face_d = np.sqrt(((local[:, 0] - 14.0) / 1e9) ** 2 + (local[:, 1] / 7.5) ** 2 + ((local[:, 2] + 2.5) / 9.0) ** 2)
    front = np.clip((local[:, 0] - 6.0) / 4.0, 0, 1) * (hw > 0.3)
    face_core = front * np.clip((1.25 - face_d) / 0.35, 0, 1)
    face_ring = front * np.clip((1.7 - face_d) / 0.45, 0, 1)
    density *= 1.0 - face_core
    length = length * (1 - face_ring) + face_ring * np.clip(1.5 + 2.5 * (face_d - 0.9), 1.0, 4.0)
    crest = (hw > 0.3) * np.clip((local[:, 2] - 12.0) / 6.0, 0, 1)
    length += 5.0 * crest                                                 # longer on the crest: the cone

    # Palms, finger pads and soles are bare.
    bare_hand = np.zeros(nv)
    for s in ("l", "r"):
        hand = w("hand_" + s) + sum(w("%s_%s_%s" % (f, i, s)) for f in ("thumb", "index", "middle", "ring", "pinky")
                                    for i in ("metacarpal", "01", "02", "03"))
        hb = body.globals_[body.bone("hand_" + s)][:3, :3]
        palm = _palm_direction(body, s, hb)
        bare_hand = np.maximum(bare_hand, np.clip(hand * 1.5, 0, 1) * np.clip((n @ palm + 0.1) / 0.3, 0, 1))
    sole = np.clip(w("foot", "ball") * 1.5, 0, 1) * np.clip((-n[:, 2] - 0.3) / 0.3, 0, 1)
    toes = np.clip(w("ball") * 1.5, 0, 1)
    bare = np.clip(np.maximum.reduce([face_core, bare_hand, sole, toes * 0.9]), 0, 1)
    density *= 1.0 - bare
    density *= 1.0 - 0.6 * np.clip(fingers, 0, 1)                         # sparse on the fingers

    # Comb: down the trunk and legs, toward the elbow on both arm bones, toward
    # the fingers on the hand, back over the crown, down the face.
    comb = np.zeros((nv, 3))
    down = np.array([0, 0, -1.0])
    for j, name in enumerate(body.names):
        wj = body.weights[:, j]
        if wj.max() < 1e-4:
            continue
        base = name[:-2] if name.endswith(("_l", "_r")) else name
        if base.startswith(("upperarm", "clavicle")):
            c = _toward(body, name, "lowerarm" + name[-2:])
        elif base.startswith("lowerarm"):
            c = _toward(body, "hand" + name[-2:], "lowerarm" + name[-2:])
        elif base == "hand" or base.startswith(("thumb", "index", "middle", "ring", "pinky")):
            c = _toward(body, "hand" + name[-2:], "middle_01" + name[-2:])
        elif base.startswith(("thigh", "calf")):
            c = _toward(body, "thigh" + name[-2:], "foot" + name[-2:])
        elif base.startswith(("foot", "ball")):
            c = _toward(body, "foot" + name[-2:], "ball" + name[-2:])
        else:
            c = down
        comb += wj[:, None] * c
    back = -axes[:, 0]
    head_comb = np.where((local[:, 0] > 5.0)[:, None] & (local[:, 2] < 4.0)[:, None], down + 0.3 * axes[:, 0],
                         back + 0.4 * down)
    hm = np.clip(hw * 1.2, 0, 1)[:, None]
    comb = comb * (1 - hm) + head_comb * hm

    lift = 0.32 + 0.25 * crest
    droop = 0.18 * np.clip(w("spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "pelvis", "thigh"), 0, 1)
    clump = np.full(nv, 0.55) - 0.3 * face_ring
    tip_light = np.clip(w("spine_05", "clavicle", "upperarm", "neck_01") + 0.4 * w("head"), 0, 1)
    field = groom.FurField(density, length, comb, lift, droop, clump, tip_light)
    head_mask = np.clip(w("head", "neck_02", "neck_01"), 0, 1) > 0.4
    return field, bare, head_mask


def _toward(body: Body, a: str, b: str) -> np.ndarray:
    d = body.heads[body.bone(b)] - body.heads[body.bone(a)]
    return d / max(np.linalg.norm(d), 1e-9)


def _palm_direction(body: Body, side: str, hand_rot: np.ndarray) -> np.ndarray:
    """The way his palm faces: perpendicular to the hand, away from the thumb's side of the knuckles' plane."""
    wrist = body.heads[body.bone("hand_" + side)]
    index = body.heads[body.bone("index_01_" + side)]
    pinky = body.heads[body.bone("pinky_01_" + side)]
    middle = body.heads[body.bone("middle_01_" + side)]
    nrm = np.cross(index - wrist, pinky - wrist)
    nrm /= max(np.linalg.norm(nrm), 1e-9)
    thumb = body.heads[body.bone("thumb_02_" + side)]
    # the thumb sits on the palm side of the knuckle plane
    if (thumb - middle) @ nrm < 0:
        nrm = -nrm
    return nrm


# ------------------------------------------------------------------ main

def measure(body: Body) -> dict:
    def hz(name):
        return float(body.heads[body.bone(name)][2])

    ground = float(body.pos[:, 2].min())
    arm = sum(np.linalg.norm(body.heads[body.bone(b)] - body.heads[body.bone(a)])
              for a, b in (("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l"), ("hand_l", "middle_01_l"),
                           ("middle_01_l", "middle_03_l")))
    fingertip = hz("upperarm_l") - arm - 3.0
    return {
        "height_cm": float(body.pos[:, 2].max() - ground),
        "shoulder_width_cm": float(abs(body.heads[body.bone("upperarm_l")][0] - body.heads[body.bone("upperarm_r")][0])),
        "hip_height_cm": hz("thigh_l") - ground,
        "knee_height_cm": hz("calf_l") - ground,
        "hanging_fingertip_cm": float(fingertip - ground),
        "foot_length_cm": float(np.ptp(body.pos[body.w("foot_l", "ball_l") > 0.3][:, 1])),
    }


def build(manny_glb: str | Path, out_dir: str | Path, seed: int = 7, fur_density: float = 1.0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = gltf.read_glb(manny_glb)
    body = load_body(model)
    manny = {"height_cm": float(np.ptp(body.pos[:, 2])), "hip_height_cm": float(body.heads[body.bone("thigh_l")][2])}
    scale = reproportion(body)
    add_mass(body)

    detail = body.w("head", "neck_02") > 0.02
    detail = refine(body, detail)
    detail = refine(body, detail & (body.w("head") > 0.05))
    extremities = sum(body.w("%s_%s" % (f, i)) for f in ("thumb", "index", "middle", "ring", "pinky")
                      for i in ("metacarpal", "01", "02", "03")) + body.w("hand", "foot", "ball")
    refine(body, extremities > 0.05)

    head_scale = HEAD_SCALE
    head = sculpt_head(body, head_scale)
    shape_feet(body)
    normals = body.normals()

    field, bare, head_mask = fur_fields(body, head)
    field.density *= fur_density
    uvs = np.stack([np.arctan2(body.pos[:, 0], body.pos[:, 1]) / (2 * np.pi) + 0.5,
                    body.pos[:, 2] / max(body.pos[:, 2].max(), 1.0)], axis=1)
    rng = np.random.default_rng(seed)
    strands = groom.grow(body.pos, normals, body.faces, uvs, field, rng)
    on_head = head_mask[body.faces[strands.root_face]].mean(axis=1) > 0.5
    head_fur = strands.subset(on_head)
    body_fur = strands.subset(~on_head)
    groom.write_rrgroom(body_fur, out / "fur_body.rrgroom")
    groom.write_rrgroom(head_fur, out / "fur_head.rrgroom")

    # Eyes, skinned to the head.
    eye_pos, eye_faces = [], []
    origin, axes = head["origin"], head["axes"]
    for side in (1.0, -1.0):
        c = np.array(sdf.EYE_CENTRE) * [1, side, 1] * head_scale
        sp, sf = uv_sphere(origin + axes @ c, sdf.EYE_RADIUS * head_scale)
        eye_faces.append(sf + sum(len(e) for e in eye_pos))
        eye_pos.append(sp)
    eye_pos = np.concatenate(eye_pos)
    eye_faces = np.concatenate(eye_faces)

    write_mesh(model, body, normals, bare, uvs, eye_pos, eye_faces, out / "sasquatch.glb")

    m = measure(body)
    hip_ratio = m["hip_height_cm"] / manny["hip_height_cm"]
    eye_height = float((origin + axes @ (np.array(sdf.EYE_CENTRE) * head_scale))[2])
    pitch = {}
    for bone, deg in GAIT["pitch_deg"].items():
        pitch[bone] = _pitch_offset(body.globals_[body.bone(bone)][:3, :3], deg)
    manifest = {
        **m,
        "eye_height_cm": eye_height,
        "scale_vs_mannequin": hip_ratio,
        "mannequin": manny,
        "overall_scale": scale,
        "vertices": int(len(body.pos)),
        "triangles": int(len(body.faces) + len(eye_faces)),
        "strands": {"body": int(len(body_fur.counts)), "head": int(len(head_fur.counts)),
                    "points": int(len(strands.points))},
        "gait": {"pelvis_drop_cm": GAIT["pelvis_drop"] * m["hip_height_cm"], "rotation_offsets": pitch,
                 "pitch_deg": GAIT["pitch_deg"]},
        "seed": seed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _pitch_offset(g_rot: np.ndarray, deg: float) -> list[float]:
    """Local rotation offset (x, y, z, w) that pitches a bone forward by `deg` in
    the body frame: the retargeter applies it as global' = global * offset."""
    t = np.radians(deg)
    r = np.array([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]])
    g = g_rot / np.linalg.norm(g_rot, axis=0)
    d = g.T @ r @ g
    m = np.eye(4)
    m[:3, :3] = d
    _, q, _ = gltf.matrix_to_trs(m)
    return [float(v) for v in q]


def write_mesh(model: gltf.Model, body: Body, normals: np.ndarray, bare: np.ndarray, uvs: np.ndarray,
               eye_pos: np.ndarray, eye_faces: np.ndarray, path: Path) -> None:
    sw = gltf.body_frame()
    to_g = lambda v: (v / gltf.GLTF_TO_CM) @ sw.T  # noqa: E731  (the swap is its own inverse)
    k = 4  # Interchange's glTF importer reads JOINTS_0/WEIGHTS_0 only
    order = np.argsort(-body.weights, axis=1)[:, :k]
    w = np.take_along_axis(body.weights, order, axis=1)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    colours = np.stack([bare, np.zeros_like(bare), np.zeros_like(bare), np.ones_like(bare)], axis=1)
    skin_prim = gltf.Primitive("M_SasquatchSkin", to_g(body.pos), normals @ sw.T, uvs, order, w,
                               body.faces[:, ::-1].copy(), colours)
    head_j = body.bone("head")
    eye_n = mesh_ops.vertex_normals(eye_pos, eye_faces)
    eye_prim = gltf.Primitive("M_SasquatchEye", to_g(eye_pos), eye_n @ sw.T, np.zeros((len(eye_pos), 2)),
                              np.full((len(eye_pos), 1), head_j), np.ones((len(eye_pos), 1)),
                              eye_faces[:, ::-1].copy(), None)
    g_new = np.array([from_body(m) for m in body.globals_])
    locals_ = np.zeros_like(g_new)
    for j, p in enumerate(body.parents):
        locals_[j] = g_new[j] if p < 0 else np.linalg.inv(g_new[p]) @ g_new[j]
    skin = gltf.Skin(body.names, body.parents, locals_, np.linalg.inv(g_new))
    gltf.write_glb(gltf.Model(skin, [skin_prim, eye_prim], np.eye(4), "Sasquatch"), path)
