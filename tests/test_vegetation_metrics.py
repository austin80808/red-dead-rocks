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
