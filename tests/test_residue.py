import numpy as np

from mangatr.inpaint import ring_of
from mangatr.residue import scrub_residue


def scene(paper=240):
    crop = np.full((80, 120, 3), paper, np.uint8)
    mask = np.zeros((80, 120), bool)
    mask[25:55, 30:90] = True
    return crop, mask


def test_a_dark_ghost_on_plain_paper_is_rebuilt():
    crop, mask = scene()
    crop[30:36, 40:70] = 150  # a dark smear the model left
    out, changed = scrub_residue(crop, mask, ring_of(mask))
    assert changed >= 30
    assert (np.abs(out[mask].astype(int) - 240) <= 2).all()
    assert (out[~mask] == crop[~mask]).all()


def test_a_bright_ghost_on_grey_paper_is_rebuilt():
    crop, mask = scene(paper=120)
    crop[40:46, 40:70] = 170
    out, changed = scrub_residue(crop, mask, ring_of(mask))
    assert changed > 0 and (np.abs(out[mask].astype(int) - 120) <= 2).all()


def test_a_textured_surround_is_left_alone():
    rng = np.random.default_rng(0)
    crop = rng.integers(40, 220, (80, 120, 3), dtype=np.uint8)
    mask = np.zeros((80, 120), bool)
    mask[25:55, 30:90] = True
    out, changed = scrub_residue(crop, mask, ring_of(mask))
    assert changed == 0 and (out == crop).all()


def test_a_clean_fill_is_not_touched():
    crop, mask = scene()
    out, changed = scrub_residue(crop, mask, ring_of(mask))
    assert changed == 0 and (out == crop).all()
