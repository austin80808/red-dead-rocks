"""The sasquatch generator's parts, on small synthetic meshes (the real base,
the UE mannequin, is Epic content and stays out of the repo)."""

import numpy as np
import pytest

from redrocks.sasquatch import gltf, groom, mesh_ops, rig, sdf


def cylinder(radius=10.0, height=100.0, rings=21, sides=16):
    """A capped-free tube along +Z, faces wound outward (cross product out)."""
    z = np.linspace(0, height, rings)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    pos = np.array([[radius * np.cos(t), radius * np.sin(t), h] for h in z for t in a])
    faces = []
    for r in range(rings - 1):
        for s in range(sides):
            i0, i1 = r * sides + s, r * sides + (s + 1) % sides
            j0, j1 = i0 + sides, i1 + sides
            faces += [(i0, i1, j1), (i0, j1, j0)]
    return pos, np.array(faces)


def two_bone_model():
    """A tube skinned to root -> lower (0..50 cm) -> upper (50..100 cm), in glTF metres."""
    pos_cm, faces = cylinder()
    w_upper = np.clip((pos_cm[:, 2] - 40) / 20, 0, 1)
    joints = np.tile([[1, 2, 0, 0]], (len(pos_cm), 1))
    weights = np.stack([1 - w_upper, w_upper, np.zeros_like(w_upper), np.zeros_like(w_upper)], axis=1)
    # body frame (cm, Z up) -> glTF (m, Y up): swap Y/Z, reverse winding
    sw = gltf.body_frame()
    pos_g = (pos_cm @ sw.T) / 100.0
    locals_ = np.array([gltf.trs_matrix(), gltf.trs_matrix(), gltf.trs_matrix((0, 0.5, 0))])
    skin = gltf.Skin(["root", "lower", "upper"], np.array([-1, 0, 1]), locals_, np.zeros((3, 4, 4)))
    skin.inverse_bind = np.linalg.inv(skin.global_matrices())
    prim = gltf.Primitive("Skin", pos_g, np.zeros_like(pos_g), np.zeros((len(pos_g), 2)), joints, weights,
                          faces[:, ::-1].copy())
    return gltf.Model(skin, [prim])


def test_glb_round_trip(tmp_path):
    model = two_bone_model()
    path = tmp_path / "t.glb"
    gltf.write_glb(model, path)
    back = gltf.read_glb(path)
    assert back.skin.names == ["root", "lower", "upper"]
    assert list(back.skin.parents) == [-1, 0, 1]
    np.testing.assert_allclose(back.skin.global_matrices(), model.skin.global_matrices(), atol=1e-6)
    np.testing.assert_allclose(np.linalg.inv(back.skin.inverse_bind), back.skin.global_matrices(), atol=1e-5)
    p = back.primitives[0]
    np.testing.assert_allclose(p.positions, model.primitives[0].positions, atol=1e-6)
    np.testing.assert_allclose(p.weights.sum(axis=1), 1.0, atol=1e-6)
    assert p.indices.shape == model.primitives[0].indices.shape


def test_trs_round_trip():
    q0 = np.array([0.2, -0.3, 0.1, 0.927])
    m = gltf.trs_matrix((1, 2, 3), q0 / np.linalg.norm(q0), (1, 1, 1))
    t, q, s = gltf.matrix_to_trs(m)
    np.testing.assert_allclose(gltf.trs_matrix(t, q, s), m, atol=1e-9)


def test_lengthening_a_bone_stretches_its_flesh_evenly():
    names = ["root", "lower", "upper"]
    parents = np.array([-1, 0, 1])
    heads = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 50.0]])
    rp = rig.reproportion(names, parents, heads, {"lower": rig.Segment(length=1.5, girth=1.2)})
    np.testing.assert_allclose(rp.heads[2], [0, 0, 75.0])
    pts = np.array([[10.0, 0, 25.0]])
    moved = rig.skin_points(rp, pts, np.array([[1]]), np.array([[1.0]]))
    np.testing.assert_allclose(moved, [[12.0, 0, 37.5]])           # girth 1.2 across, 1.5 along
    # the upper bone keeps its shape and rides out on the longer lower one
    moved_up = rig.skin_points(rp, np.array([[10.0, 0, 60.0]]), np.array([[2]]), np.array([[1.0]]))
    np.testing.assert_allclose(moved_up, [[10.0, 0, 85.0]])


def test_overall_scale_is_linear():
    names = ["root", "a"]
    parents = np.array([-1, 0])
    heads = np.array([[0, 0, 0], [0, 0, 10.0]])
    rp = rig.reproportion(names, parents, heads, {}, scale=1.4)
    np.testing.assert_allclose(rp.heads[1], [0, 0, 14.0])
    out = rig.skin_points(rp, np.array([[1.0, 2.0, 3.0]]), np.array([[0]]), np.array([[1.0]]))
    np.testing.assert_allclose(out, [[1.4, 2.8, 4.2]])


def test_twist_bones_follow_their_parent_segment():
    names = ["root", "upperarm_l", "upperarm_twist_01_l", "lowerarm_l"]
    parents = np.array([-1, 0, 1, 1])
    seg = rig.segment_for("upperarm_twist_01_l", names, parents, {"upperarm": rig.Segment(1.2, 1.3)})
    assert seg == rig.Segment(1.2, 1.3)


def test_red_green_subdivision_leaves_no_t_junctions():
    pos, faces = cylinder(rings=5, sides=8)
    refine = np.zeros(len(faces), bool)
    refine[:6] = True
    p2, f2, attrs, origin = mesh_ops.subdivide(pos, faces, refine, {"w": np.arange(len(pos), dtype=float)})
    # every interior edge is shared by exactly two faces (the tube's open ends have one)
    e = np.sort(np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert counts.max() == 2
    assert np.isclose(mesh_ops.face_areas(p2, f2).sum(), mesh_ops.face_areas(pos, faces).sum())
    assert len(attrs["w"]) == len(p2)
    assert (np.bincount(origin, minlength=len(faces))[refine] == 4).all()


def test_vertex_normals_point_out_of_the_tube():
    pos, faces = cylinder()
    n = mesh_ops.vertex_normals(pos, faces)
    radial = pos[:, :2] / np.linalg.norm(pos[:, :2], axis=1, keepdims=True)
    assert (np.einsum("ij,ij->i", n[:, :2], radial) > 0.9).all()


def test_project_radially_lands_on_a_sphere():
    sphere = lambda p: np.linalg.norm(p, axis=-1) - 7.0  # noqa: E731
    pts = np.random.default_rng(1).normal(size=(50, 3)) * 3
    out = sdf.project_radially(pts, np.zeros(3), sphere)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 7.0, atol=1e-3)


def test_head_is_conical_with_a_face():
    # the crest is the highest point and sits behind the face
    xs, zs = np.meshgrid(np.linspace(-20, 20, 81), np.linspace(-20, 30, 101))
    p = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
    inside = sdf.head_sdf(p) < 0
    top = p[inside][np.argmax(p[inside][:, 2])]
    assert top[0] < 0                                    # crest peaks toward the back
    front = p[inside][:, 0].max()
    assert front > 15                                    # nose and lips stand out front


@pytest.fixture
def furry_tube():
    pos, faces = cylinder()
    n = mesh_ops.vertex_normals(pos, faces)
    count = len(pos)
    field = groom.FurField(
        density=np.where(pos[:, 2] > 50, 2.0, 0.0),           # bare below 50 cm
        length=np.full(count, 10.0),
        comb=np.tile([0, 0, -1.0], (count, 1)),
        lift=np.full(count, 0.3),
        droop=np.full(count, 0.2),
        clump=np.full(count, 0.5),
        tip_light=np.full(count, 0.5),
    )
    return pos, n, faces, field


def test_fur_grows_only_where_dense_and_lies_down(furry_tube):
    pos, n, faces, field = furry_tube
    strands = groom.grow(pos, n, faces, np.zeros((len(pos), 2)), field, np.random.default_rng(0))
    roots = strands.roots
    assert len(roots) > 100
    assert roots[:, 2].min() > 35                         # nothing grows on the bare half (faces straddle 50)
    starts = np.concatenate([[0], np.cumsum(strands.counts)[:-1]])
    tips = strands.points[starts + strands.counts - 1]
    assert np.median(tips[:, 2] - roots[:, 2]) < -5       # combed downward
    r_root = np.linalg.norm(roots[:, :2], axis=1)
    r_tip = np.linalg.norm(tips[:, :2], axis=1)
    assert np.median(r_tip - r_root) > 0                  # and standing off the skin, not into it
    lum = strands.colours @ [0.2126, 0.7152, 0.0722]
    assert lum[starts].mean() < lum[starts + strands.counts - 1].mean()  # dark roots, lighter tips


def test_rrgroom_round_trip(tmp_path, furry_tube):
    pos, n, faces, field = furry_tube
    strands = groom.grow(pos, n, faces, np.zeros((len(pos), 2)), field, np.random.default_rng(0))
    path = tmp_path / "fur.rrgroom"
    groom.write_rrgroom(strands, path)
    back = groom.read_rrgroom(path)
    np.testing.assert_array_equal(back.counts, strands.counts)
    np.testing.assert_allclose(back.points, strands.points, atol=1e-4)
    np.testing.assert_allclose(back.colours, strands.colours, atol=1e-6)
    half = strands.subset(strands.roots[:, 0] > 0)
    assert 0 < len(half.counts) < len(strands.counts)
    assert len(half.points) == half.counts.sum()
