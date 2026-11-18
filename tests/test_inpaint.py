import numpy as np
import pytest

from mangatr.inpaint import CHEAP_TIERS, FlatTier, Router, SmoothTier, ring_of

H, W = 120, 200


def blob(x0=70, x1=130, y0=45, y1=75):
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def flat_image(value=255):
    return np.full((H, W, 3), value, dtype=np.uint8)


def gradient_image():
    ramp = np.linspace(60, 220, W, dtype=np.float32)
    return np.repeat(np.repeat(ramp[None, :, None], H, axis=0), 3, axis=2).astype(np.uint8)


class Terminal:
    """A stand-in for the model tier: always succeeds, paints a marker colour."""

    name = "terminal"

    def __init__(self):
        self.calls = 0

    def fill(self, crop, mask, ring):
        self.calls += 1
        out = crop.copy()
        out[mask] = (1, 2, 3)
        return out


def router(terminal=None):
    return Router([*CHEAP_TIERS, terminal or Terminal()])


def test_flat_background_is_filled_with_its_colour():
    image = flat_image(240)
    image[blob()] = 0  # black text
    out, tier = router().clean(image, blob())
    assert tier == "flat"
    assert (out == 240).all()


def test_gradient_goes_to_the_smooth_tier_and_is_rebuilt():
    clean = gradient_image()
    image = clean.copy()
    mask = blob(x0=50, x1=150)
    image[mask] = 0
    out, tier = router().clean(image, mask)
    assert tier == "smooth"
    error = np.abs(out[mask].astype(int) - clean[mask].astype(int))
    assert error.mean() < 6


def test_flat_tier_refuses_the_gradient():
    image = gradient_image()
    mask = blob(x0=50, x1=150)
    assert FlatTier().fill(image, mask, ring_of(mask)) is None


def test_textured_background_reaches_the_model_tier():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    terminal = Terminal()
    _, tier = router(terminal).clean(image, blob())
    assert tier == "terminal"
    assert terminal.calls == 1


def test_structured_background_is_not_smeared_by_the_cheap_tiers():
    """Vertical stripes (speed lines): neither cheap tier may claim them."""
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[:, ::4] = 255
    mask = blob()
    ring = ring_of(mask)
    assert FlatTier().fill(image, mask, ring) is None
    assert SmoothTier().fill(image, mask, ring) is None


def test_ring_too_small_goes_to_the_model_tier():
    image = flat_image()
    mask = np.ones((H, W), dtype=bool)  # nothing left around the mask
    _, tier = router().clean(image, mask)
    assert tier == "terminal"


def test_only_masked_pixels_change():
    image = gradient_image()
    mask = blob(x0=50, x1=150)
    before = image.copy()
    out, _ = router().clean(image, mask)
    assert (out[~mask] == before[~mask]).all()
    assert (image == before).all()  # the input is not modified either


def test_valid_region_keeps_the_ring_inside_the_balloon():
    """Left part white (the balloon), right part black art. The mask sits at
    the balloon's edge, so its ring reaches the art. Read whole, the ring is
    not flat; limited to the balloon it is, and the text is filled white."""
    image = flat_image(255)
    image[:, 100:] = 0
    mask = blob(x0=60, x1=98)
    balloon = np.zeros((H, W), dtype=bool)
    balloon[:, :100] = True

    whole = ring_of(mask)
    assert whole[:, 100:].sum() > 0  # the ring does reach the black art
    assert FlatTier().fill(image, mask, whole) is None

    inside = ring_of(mask, valid=balloon)
    assert inside[:, 100:].sum() == 0
    out = FlatTier().fill(image, mask, inside)
    assert out is not None
    assert (out[mask] == 255).all()


def test_empty_mask_changes_nothing():
    image = flat_image(7)
    out, tier = router().clean(image, np.zeros((H, W), dtype=bool))
    assert tier == "none"
    assert (out == image).all()


def test_router_needs_a_tier():
    with pytest.raises(ValueError):
        Router([])


def test_router_without_a_succeeding_last_tier_fails_loudly():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        Router(list(CHEAP_TIERS)).clean(image, blob())
