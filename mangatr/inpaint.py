"""Stage: inpaint. Phase 1.

Reads:  the page image, each block's mask, its balloon and its panel.
Writes: clean, the patch under the block's mask.
How:    a router picks the cheapest tier that rebuilds the block's surroundings,
        per block and not per page; the model is last. Tiers: one colour, a
        smooth background (Telea), OpenCV, AOT, LaMa. The order of OpenCV and
        AOT is decided by measuring on the target GPU (scripts/), not guessed.
        Models run on windows (block plus 256 px of context), neighbouring
        windows merged while the merge stays under a megapixel and sent to the
        GPU in batches. Never copy the whole page per step.
See docs/DESIGN.md, "Cleaning: a router by cost".
"""
