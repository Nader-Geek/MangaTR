import numpy as np
import pytest

from mangatr.inpaint import FlatTier, Router
from mangatr.model_tiers import (
    ModelOutputError,
    OnnxInpaintTier,
    OpenCvTier,
    pad_to_multiple,
)


class Item:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Stands in for an ONNX session: records the feeds, answers with `answer`."""

    def __init__(self, answer):
        self.answer = answer
        self.feeds = None

    def get_inputs(self):
        return [Item("image"), Item("mask")]

    def run(self, _outputs, feeds):
        self.feeds = feeds
        return [self.answer(feeds)]


def constant(value):
    def answer(feeds):
        _, _, h, w = feeds["image"].shape
        return np.full((1, 3, h, w), value, np.float32)

    return answer


def scene(h=40, w=50):
    crop = np.full((h, w, 3), 200, np.uint8)
    mask = np.zeros((h, w), bool)
    mask[10:20, 10:30] = True
    crop[mask] = 0
    return crop, mask


def test_pad_to_multiple_rounds_up_and_respects_the_minimum():
    assert pad_to_multiple(np.zeros((5, 9, 3)), 8).shape == (8, 16, 3)
    assert pad_to_multiple(np.zeros((5, 9)), 8, min_size=128).shape == (128, 128)
    assert pad_to_multiple(np.zeros((16, 8)), 8).shape == (16, 8)


def test_lama_scaling_and_only_masked_pixels_change():
    crop, mask = scene()
    session = FakeSession(constant(0.5))
    out = OnnxInpaintTier("lama", session).fill(crop, mask, mask)
    assert (out[mask] == 128).all()  # 0.5 * 255 rounded
    assert (out[~mask] == crop[~mask]).all()
    image = session.feeds["image"]
    assert image.shape == (1, 3, 40, 56) and image.dtype == np.float32
    assert session.feeds["mask"].shape == (1, 1, 40, 56)
    assert session.feeds["mask"].sum() == mask.sum()
    assert image.max() <= 1.0


def test_aot_scaling_pads_to_the_minimum_and_zeroes_the_masked_pixels():
    crop, mask = scene()
    session = FakeSession(constant(1.0))
    out = OnnxInpaintTier("aot", session).fill(crop, mask, mask)
    assert (out[mask] == 255).all()
    image = session.feeds["image"]
    assert image.shape == (1, 3, 128, 128)
    assert (image[0, :, :40, :50][:, mask] == 0).all()
    assert abs(image[0, 0, 0, 0] - (200 / 127.5 - 1)) < 1e-6


def test_large_crops_are_scaled_down_for_aot_and_back():
    crop = np.full((2048, 1024, 3), 90, np.uint8)
    mask = np.zeros((2048, 1024), bool)
    mask[500:700, 300:600] = True
    session = FakeSession(constant(0.0))
    out = OnnxInpaintTier("aot", session).fill(crop, mask, mask)
    assert session.feeds["image"].shape[2:] == (1024, 512)
    assert out.shape == crop.shape
    assert (out[mask] == 127).all() or (out[mask] == 128).all()
    assert (out[~mask] == 90).all()


@pytest.mark.parametrize(
    "bad",
    [
        lambda feeds: np.zeros((1, 3, 8, 8), np.float32),
        lambda feeds: np.full((1, 3, 40, 56), np.nan, np.float32),
    ],
)
def test_a_bad_output_is_refused(bad):
    crop, mask = scene()
    with pytest.raises(ModelOutputError):
        OnnxInpaintTier("lama", FakeSession(bad)).fill(crop, mask, mask)


def test_unknown_model_or_wrong_inputs_are_refused():
    with pytest.raises(ValueError):
        OnnxInpaintTier("nope", FakeSession(constant(0)))

    class One(FakeSession):
        def get_inputs(self):
            return [Item("image")]

    with pytest.raises(ValueError):
        OnnxInpaintTier("lama", One(constant(0)))


def test_opencv_tier_fills_and_keeps_the_rest():
    crop, mask = scene()
    out = OpenCvTier().fill(crop, mask, mask)
    assert (out[~mask] == crop[~mask]).all()
    assert out[mask].mean() > 150


def test_router_falls_through_to_a_model_tier():
    rng = np.random.default_rng(1)
    crop = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    mask = np.zeros((60, 60), bool)
    mask[20:40, 20:40] = True
    session = FakeSession(constant(0.5))
    router = Router(
        [__import__("mangatr.inpaint", fromlist=["x"]).FlatTier(), OnnxInpaintTier("lama", session)]
    )
    result, tier = router.clean(crop, mask, None)
    assert tier == "lama"
    assert (result[~mask] == crop[~mask]).all()
