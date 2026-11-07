"""redrocks.vegetation_metrics: the numbers the vegetation spec is judged by."""

import numpy as np
import pytest

from redrocks import vegetation_metrics as vm


def test_delta_e_of_identical_colours_is_zero():
    assert vm.delta_e("#505B46", "#505B46") == pytest.approx(0.0, abs=1e-6)


def test_delta_e_black_to_white_is_100():
    assert vm.delta_e("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.5)


def test_delta_e_accepts_lab_and_hex_alike():
    lab = vm.hex_to_lab("#68744B")
    assert vm.delta_e(lab, "#68744B") == pytest.approx(0.0, abs=1e-6)
    assert vm.delta_e("#68744B", "#5A5236") > 5.0


def test_rgb_to_lab_known_value():
    # sRGB mid grey 119 is L* ~50, neutral.
    lab = vm.rgb_to_lab(np.array([119, 119, 119]))
    assert lab[0] == pytest.approx(50.0, abs=0.5)
    assert abs(lab[1]) < 0.5 and abs(lab[2]) < 0.5


def test_median_hex_ignores_outliers():
    img = np.full((50, 50, 3), (0x50, 0x5B, 0x46), np.uint8)
    img[25, 25] = 255
    assert vm.median_hex(img, (25, 25), r=5) == "#505B46"


def test_median_hex_clips_at_the_image_edge():
    img = np.full((20, 20, 3), (10, 20, 30), np.uint8)
    assert vm.median_hex(img, (0, 19), r=6) == "#0A141E"


# ------------------------------------------------------------------ shapes

from redrocks import vegetation as veg  # noqa: E402


def _model_with(crown=(), cards=(), trunk=None, tips=(), height=2.0):
    """A Model from blobs ((centre, r) pairs), cards ((base, width) pairs) and a trunk tube."""
    parts = {p: veg.Mesh() for p in veg.PARTS}
    rng = np.random.default_rng(0)
    for centre, r in crown:
        veg.add_blob(parts["Crown"], centre, np.array([r, r, r]), rng, subdiv=2, lump=0.0)
    for base, width in cards:
        veg.add_card(parts["Leaves"], base, (0, 0, 1), (1, 0, 0), width, width, 0, lambda p: (0, 0, 1))
    if trunk is not None:
        (z0, z1), r = trunk
        veg.add_tube(parts["Trunk"], [(0, 0, z0), (0, 0, (z0 + z1) / 2), (0, 0, z1)], [r, r, r], 32)
    return veg.Model("Test_0", "test", "Oak", height, 0.1, parts, tips=[np.asarray(t, float) for t in tips])


def test_side_fill_of_a_solid_blob_is_near_one():
    assert vm.side_fill(_model_with(crown=[((0, 0, 1), 1.0)])) > 0.9


def test_side_fill_of_two_far_blobs_is_low():
    model = _model_with(crown=[((-3, 0, 1), 0.5), ((3, 0, 1), 0.5)])
    assert vm.side_fill(model) < 0.3


def test_side_fill_counts_only_opaque_leaf_texels():
    card = _model_with(cards=[((0, 0, 0), 1.0)])
    assert vm.side_fill(card) > 0.95
    # Tile 0 is the bottom-left quarter of the image (rows 256.. are V 0..0.5, columns ..256 U 0..0.5).
    # Clear its middle half: the two opaque edge strips keep the hull at the full card, so fill ~0.5.
    alpha = np.ones((512, 512))
    alpha[256:, 64:192] = 0.0
    assert vm.side_fill(card, alpha) == pytest.approx(0.5, abs=0.12)


def test_card_widths_and_tip_fraction():
    model = _model_with(cards=[((0, 0, 1), 0.4)], tips=[(0, 0, 1.5)])
    assert vm.card_widths(model) == pytest.approx([0.4])
    assert vm.tip_fraction(model, 1.0) == 1.0 and vm.tip_fraction(model, 0.2) == 0.0


def test_trunk_diameter_of_a_cylinder():
    model = _model_with(trunk=((-0.5, 5.0), 0.3))
    assert vm.trunk_diameter_at(model, 1.4) == pytest.approx(0.6, rel=0.05)


def test_crown_base_fraction_is_the_lowest_leaf_over_height():
    model = _model_with(cards=[((0, 0, 0.8), 0.4)], height=2.0)
    assert vm.crown_base_fraction(model) == pytest.approx(0.4)


def test_canopy_cover_of_one_shrub():
    by_name = {"Oak_0": veg.build_model("Oak", 0)}
    r = vm.footprint_radius_m(by_name["Oak_0"]) * 100
    assert 30 < r < 200
    pts = np.array([[0, 0], [r * 0.5, 0], [r * 2, 0], [0, r * 3]])
    assert vm.canopy_cover({"Oak_0": [[0, 0, 0, 0, 0, 0, 1, 1]]}, by_name, pts, ("Oak",)) == 0.5
    assert vm.canopy_cover({"Oak_0": [[0, 0, 0, 0, 0, 0, 1, 1]]}, by_name, pts, ("Mahogany",)) == 0.0


def test_ponderosa_records_its_branch_tips():
    p = veg.build_model("Ponderosa", 0)
    tips = np.asarray(p.tips)
    assert len(tips) > 20
    assert tips[:, 2].max() <= p.height_m + 1.0 and tips[:, 2].min() > 0.3 * p.height_m
