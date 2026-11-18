"""Stage: typeset. Phase 1.

Reads:  translation, balloon, clean, the language spec.
Writes: style, and the final page image.
How:    HarfBuzz shaping per run (one script, one direction, one font), bidi
        reordering, Unicode line breaking (UAX 14). One font size for the page
        (the median of the sources'), fitted `best`: the largest size that
        fits, shrunk down to half if it must. Whole words shrink first; a
        hyphen only when a word is wider than its line at half size.
"""
