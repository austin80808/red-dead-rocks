import numpy as np
import pytest
from PIL import Image

from redrocks.dem import BBox, Dem
from redrocks.landscape import export_landscape, nearest_valid_size, resample

BBOX = BBox(west=-105.21, south=39.66, east=-105.20, north=39.67)


def make_dem(values: np.ndarray) -> Dem:
    return Dem(data=values.astype(np.float32), bbox=BBOX)


def test_nearest_valid_size_snaps_to_unreal_sizes():
    assert nearest_valid_size(2000) == 2017
    assert nearest_valid_size(1000) == 1009
    assert nearest_valid_size(500) == 505
    assert nearest_valid_size(130) == 127
    # 200 is 73 from 127 but only 53 from 253, so 253 wins.
    assert nearest_valid_size(200) == 253


def test_resample_preserves_corners():
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    out = resample(data, 5)
    assert out.shape == (5, 5)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, -1] == pytest.approx(1.0)
    assert out[-1, 0] == pytest.approx(2.0)
    assert out[-1, -1] == pytest.approx(3.0)


def test_resample_is_identity_at_same_size():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert np.array_equal(resample(data, 3), data)


def test_export_writes_16bit_png(tmp_path):
    ramp = np.linspace(1900.0, 2000.0, 127 * 127).reshape(127, 127)
    result = export_landscape(make_dem(ramp), tmp_path / "h.png", size=127)

    image = Image.open(result.png_path)
    assert image.mode == "I;16"
    assert image.size == (127, 127)

    pixels = np.array(image)
    assert pixels.dtype == np.uint16
    assert pixels.min() == 0
    assert pixels.max() == 65535


def test_z_scale_matches_unreal_formula(tmp_path):
    """Z scale must be relief / 512 * 100 so terrain is 1:1 with reality."""
    ramp = np.linspace(1900.0, 2000.0, 127 * 127).reshape(127, 127)
    result = export_landscape(make_dem(ramp), tmp_path / "h.png", size=127)

    assert result.min_elevation == pytest.approx(1900.0)
    assert result.max_elevation == pytest.approx(2000.0)
    assert result.z_scale == pytest.approx(100.0 / 512.0 * 100.0)


def test_export_rejects_flat_dem(tmp_path):
    flat = np.full((127, 127), 1900.0, dtype=np.float32)
    with pytest.raises(ValueError, match="no relief"):
        export_landscape(make_dem(flat), tmp_path / "h.png", size=127)


def test_export_snaps_invalid_size(tmp_path):
    ramp = np.linspace(1900.0, 2000.0, 250 * 250).reshape(250, 250)
    result = export_landscape(make_dem(ramp), tmp_path / "h.png", size=250)
    assert result.size == 253
