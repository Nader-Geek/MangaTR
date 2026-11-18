"""Stage: detect. Phase 1.

Reads:  the page image.
Writes: Blocks with id, order, kind, frame, mask, balloon, panel, angle, vertical.
Model:  the old engine's RT-DETR first; Koharu's layout model is evaluated in
        phase 1 (format and license unverified, see docs/DESIGN.md "Risks").
Order:  panel by panel, then top to bottom and right to left, as Koharu does.
Locked blocks are never replaced.
"""
