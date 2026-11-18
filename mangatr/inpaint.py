"""Stage: inpaint. Phase 1.

Reads:  the page image, each block's mask, its balloon and its panel.
Writes: clean, the patch under the block's mask.
Tiers:  one colour (ring within 10 levels of its median) -> a smooth
        background (Telea) -> the model (LaMa). The first tier that can
        rebuild the ring fills the mask; the model is last and runs in windows.
"""
