"""Stage: ocr. Phase 1.

Reads:  the image inside each block's frame, with other blocks' glyphs blanked.
Writes: source_text, source_lang.
Model:  routed by source language (Japanese, Korean, Chinese, English).
Skips locked blocks, and blocks that already have source_text.
"""
