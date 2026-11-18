"""Stage: translate. Phases 1 and 2.

Reads:  source_text of the page's blocks in reading order, the glossary.
Writes: translation, target_lang.
How:    one conversation per chapter, one turn per page. A page is a JSON
        object keyed by reading order and the reply must keep the keys; a
        missing key or a blank translation fails the page and never enters the
        history. The system prompt is fixed, in English, and names the target
        language by its English name. Phase 2 adds the glossary, never-translate
        masking and the QA detectors.
"""
