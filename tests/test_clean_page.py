import numpy as np

from mangatr.clean_page import clean_page
from mangatr.model_tiers import OpenCvTier
from mangatr.scene import Block, BlockKind, Rect


def block(block_id, x, y, w, h):
    return Block(id=block_id, kind=BlockKind.DIALOGUE, frame=Rect(x, y, w, h), order=0)


def test_flat_blocks_use_the_cheap_tier_and_the_rest_the_model():
    rng = np.random.default_rng(0)
    page = np.full((300, 600, 3), 240, np.uint8)
    page[:, 300:] = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)  # noisy right half
    for x in (40, 400):
        page[100:140, x : x + 60] = 0  # "text"
    masks = {
        "flat": (40, 100, np.ones((40, 60), bool)),
        "noisy": (400, 100, np.ones((40, 60), bool)),
    }
    out, steps = clean_page(
        page,
        [block("flat", 40, 100, 60, 40), block("noisy", 400, 100, 60, 40)],
        masks,
        model=OpenCvTier(),
    )
    tiers = {s.block_ids: s.tier for s in steps}
    assert tiers[("flat",)] == "flat"
    assert tiers[("noisy",)] == "opencv"
    assert (out[100:140, 40:100] == 240).all()
    untouched = np.ones(page.shape[:2], bool)
    untouched[100:140, 40:100] = False
    untouched[100:140, 400:460] = False
    assert (out[untouched] == page[untouched]).all()


def test_blocks_without_a_mask_are_skipped():
    page = np.full((50, 50, 3), 9, np.uint8)
    out, steps = clean_page(page, [block("a", 5, 5, 10, 10)], {}, model=OpenCvTier())
    assert steps == [] and (out == page).all()
