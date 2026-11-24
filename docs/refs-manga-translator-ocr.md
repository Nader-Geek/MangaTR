# 06 - Manga-Translator-OCR (kaushikpaul) reference summary

Repo: `.../scratchpad/ref/Manga-Translator-OCR`. Python 3.12, ~13k LOC (renderer.py 5477, pipeline.py 3569, detector.py 1035, layout.py 756, translator.py 547).
Tags: **[R]** = read code directly, **[I]** = inferred / not fully read.

## 0. Licence [R]
- `LICENSE` = MIT, "Copyright (c) 2026 Kaushik Paul" (grep "Permission is hereby" hits LICENSE + README). README says MIT. pyproject has no licence field.
- MIT: reusable with copyright notice retained. Model weights (manga-ocr-base, comic-text-detector-onnx) and bundled fonts (animeace2, CCWildWords, Noto CJK, WenQuanYi, Droid) carry their OWN licences - check separately before shipping (fonts are checked into `main/fonts/` despite README saying gitignored).

## 1. README / AGNETS.md (pipeline, conventions) [R]
- Pipeline: detect (comic-text-detector ONNX -> text mask + YOLO blocks) -> manga-ocr per region -> LLM batch translate + repair -> cv2 Telea inpaint -> PIL render ALL CAPS.
- Constraints stated: source = Japanese only; output ASCII-only + basic punctuation; ALL CAPS; concise.
- Conventions: BGR internally, RGB only for PIL; model singletons via `__new__` + lazy load; factory `get_ocr_engine/get_text_detector`; `RenderTextUnit` dataclass carries per-region state; page-level `ThreadPoolExecutor` (2 pages default local); LLM concurrency `BoundedSemaphore` (default 3); no tests policy; output WebP q82 / JPEG q85.
- Pitfalls: AGNETS.md filename is a typo (not AGENTS.md); README claims fonts gitignored but they're committed; README "Python 3.11+" vs 3.12 elsewhere; `fc-query` (fontconfig CLI) needed for charset (see 4c).

## 2. layout.py - bubble grouping (all [R])
Entry: `group_text_by_bubbles(image_bgr, text_mask, blocks: list[DetectedBlock]) -> (list[TextGroup], residual_mask)`.
Dataclasses: `BubbleMask(x,y,mask)`, `TextGroup(x1,y1,x2,y2,text_mask,bubble|None,component_ids)`.

### Algorithm
```
binary = text_mask>0; CC(8-conn) -> labels, stats
min_cc = max(6, int(min(H,W)^2 * 4e-6)); drop smaller comps
# 1. candidate blocks
for blk in blocks sorted by area ascending:        # small first so nested boxes win
    m = max(2, int(min(w,h)*0.08)); expand box by m
    take comps with (intersection/comp_bbox_area >= 0.5) and unassigned
# comps left over -> extract_regions_from_mask(residual) -> mask-only candidates (is_model=False)
# 2. per candidate: _flood_fills_for_components(..., require_paper = not is_model)
# 3. union-find fills with IoU>=0.5 (valid only) ; assign comps to bubble roots
# 4. post-passes: demote_surfaces_containing_balloons, absorb_text_inside_balloons, merge_split_columns
# 5. sort manga order; residual = valid comps not claimed
```
### _flood_fills_for_components (core)
```
pad = int(max(bw,bh)*1.25)+16 ; ROI = bbox +/- pad (clamped)
text_d = dilate(text,5x5);  ring = dilate(own,11x11) & ~text_d ; need >=12 px
paper = _dominant_color(ring px)   # 16-level colour bins, mode bin's median
paper_gray, paper_sat(HSV S)
light = gray>=190 (>=200 if require_paper); dark = gray<=60 and not require_paper
reject if sat>50 or not(light or dark)          # coloured surfaces = SFX/free text
fill_img = roi; fill_img[text_d]=paper; medianBlur(3)    # glyphs treated as paper
edges = Canny(GaussianBlur(gray,3x3), 30, 90)
barrier = _strip_glyph_edges(edges, text) ; dilate 2x2   # keeps balloon outline
    strip: zero edges inside dilate(text,3); CC edge chains; drop chain if >=95% of its px within dilate(text,9)
for comp in comps sorted by area desc (skip if already inside an earlier fill):
    seed = a glyph pixel (median index of its pixels)
    floodFill(4-conn, FIXED_RANGE, MASK_ONLY, loDiff=upDiff=(20,20,20), mask=barrier)
    touches = region touches a ROI side that is NOT a page border   -> leak -> invalid
    filled = fill_holes(region)
    if holes(non-glyph) > 15% of filled: invalid   # art surface enclosing objects
    if valid: filled = _regularize_balloon_shape(filled)
validate each fill:
    median sat on surface (fill minus glyph) >30 or p90>40 -> invalid
    covered(text_box in fill) < 0.7  -> invalid
    fill_area < max(text_box*1.08, glyph_footprint*1.25) -> invalid
    fill_area > text_box*(10 if require_paper else 14) or > 8% of page -> invalid   # wall/sky
```
- `_regularize_balloon_shape`: if solidity (fill/convex-hull area) in [0.45,0.85) replace by hull (translucent balloons); else keep exact.
- `_union_fills`: OR the masks, MORPH_OPEN ellipse k=odd(max(3, 3% of min dim)) (kept only if >=60% area survives), fill holes, crop.
- `_include_text_footprint`: adds MORPH_CLOSE(rect k=odd(min(41,max(4,0.25*sqrt(n_px))))) of source glyphs into bubble - "ink was printed there so it is safe".
- Unit decision per block: if >=50% of block glyph pixels sit in valid fills -> bubble unit(s) (block spanning two balloons is SPLIT; loose comps join nearest balloon by bbox distance); elif model block -> free-text "block" unit (no bubble); mask-only non-bubble -> residual (legacy).
- `_demote_surfaces_containing_balloons`: outer.bubble.area > 1.5*inner and overlaps >=50% of inner -> outer.bubble=None.
- `_absorb_text_inside_balloons`: bubble-less unit merged into host if >=50% of its glyph px inside host mask.
- `_merge_split_columns` / `_continues_line`: merge if boxes overlap >=25% of smaller area; or same-line continuation: thickness ratio <=1.6, overlap>=0.7*thick, gap in [-2, 0.5*thick], len_b<=2.5*len_a, a_len>=1.5*thickness; for "line" case with two bubbles require bubble overlap >=15% of smaller.
- `_sort_groups_manga_order`: 3 vertical column bands (page_w/3), right-to-left, then top-to-bottom by centre y. Crude (no panel detection).

### SFX path [R]
layout.py has no SFX logic of its own: SFX = mask components not in any valid balloon and not boxed by a model block (`require_paper=True` there makes the test stricter: gray>=200, no dark, 10x area cap) -> returned as `residual_mask`. pipeline.py (L95-111) runs legacy `extract_regions_from_mask` + `_merge_overlapping_/_same_surface_detected_regions` on it. Style then from `_infer_unit_style` (pipeline L2581): kana/kanji/punct profile; `_is_effect_like_source` (compact repeated kana) -> "sfx", else "dialogue"; <=2 chars or no script -> sfx; layout units with a bubble are always "dialogue". `_is_short_colored_lettering` can flip dialogue->sfx. [R for flow; internals of helpers I]

### Pitfalls
- Heavy tuning constants, no tests; many thresholds tied to scan resolution (ring 11x11, dilations 5/9 px are absolute pixels, not relative to page size - breaks on very large/small scans) [I].
- Canny 30/90 + tol 20 fail on screentone-filled balloons; hull fallback only partly helps.
- O(n^2) pairwise fill overlap and `np.isin` per comp: fine per page, slow on dense pages [I].
- Reading order is 3-band heuristic; wrong for single-column/wide panels.
- Requires `DetectedBlock` (x1,y1,x2,y2,w,h) and `extract_regions_from_mask` from detector.py (not read).
- Colour/black balloons: dark paper allowed only for model blocks.

### Reusable as-is
Yes, with small shim: layout.py is self-contained (cv2/numpy) except the 2 detector imports. Best reusable ideas: fill-validity checks, `_strip_glyph_edges`, `_dominant_color`, union-find merge, hole-ratio / area-ratio / saturation rejection of non-balloon surfaces, `_include_text_footprint`.

## 3. translator.py [R]
`translate_texts(texts, source_lang, model, max_retries=3, constraints)`:
```
if len>10: recurse in chunks of 10 (_MAX_TRANSLATION_BATCH_SIZE)   # keeps 1:1 contract
indexed = non-empty texts only (positions restored at end)
msg = "Translate the following Japanese manga text to English. ALL CAPS... Return ONLY ... [N]\n" + "[i] (style=dialogue, max_words=8, max_chars=34) <src>" lines
attempt loop (3): call -> _parse_numbered_response ; backoff sleep 2**attempt; on final failure return ORIGINAL texts
repair candidates = [i for i if _needs_repair_translation(src, out, constraint)]
repair pass (batch, REPAIR_SYSTEM_PROMPT, 3 tries) -> accept only fixes that no longer need repair
then per-line singleton repair (3 tries each, prompt "Return only the translation, with no number")
final: _postprocess_translation(raw or original src)
```
- Constraint hint is serialised by `_constraint_tag` as `(style=sfx|dialogue, max_words=N, max_chars=N)`; prompt says these are SOFT. `_violates_constraint` exists but is NOT a repair trigger (renderer shortens).
- Constraints computed in pipeline.py ~L1010-1040: SFX: max_words 3, max_chars 12 (16 if area>=12000, +2 up to 18 if box w>=160 or h>=220); dialogue by box width: w<60 -> 5w/22c; <80 -> 7/28; <100 -> 8/34; default 14w/60c, tightened for tall aspect (>=1.75 -> 8-10w/32-44c; >=1.35 -> 9-12w/38-50c) [R partial].
- System prompt (L21-57): professional localizer, no omission/safety commentary, contractions, ALL CAPS, SFX -> punchy 1-3 word English never romaji, drop honorifics, OCR-noise -> "...", never "..." for real reactions, ASCII only, straight quotes, no markdown, "answer immediately" (no reasoning). Worth copying wholesale.
- `_needs_repair_translation` returns True when: empty; (source not pure punctuation) and: tgt==src; tgt in {"[","]","[8","[9",".."}; "..." with >=3 content chars in src; refusal markers ("i can't provide", "i cannot assist", "i'm unable to", "sorry, i can't"...); any CJK (kana U+3040-30FF, ext-A 3400-4DBF, 4E00-9FFF, F900-FAFF); romaji SFX; box-drawing/geometric chars (U+2500-25FF, □■▪▫▲△●○); src>=8 chars with tgt<=1 char. The final "ASCII ratio" block is dead code (returns False either way).
- Pure punctuation regex for source: `[．。、・\s\.\!\?♡♪～〜＊＾「」『』（）ッ♥]`.
- `_parse_numbered_response`: regex `^\s*(?:\[(\d+)\]|(\d+)[.)])\s*[:.)-]?\s*(.*)$`; 1-based -> 0-based; indices outside range ignored; repeated marker overwrites; unnumbered continuation lines appended (space-joined) to current index; if no numbers at all, positional fallback by line. Missing indices left absent -> repair.
- `_is_romaji_sfx`: <=2 words, in a ~30-word set (biku, doki, gata, goku...) or doubled halves (len>=6, half in set or half ends with vowel & len>=3).
- `_postprocess_translation`: strip honorific regex `(?<=[A-Za-z])[-\s](san|kun|chan|sama|senpai|sensei|dono)\b`; strip wrapping `*`; uppercase if <=4 words ending in `!`; romaji-SFX -> "...".
- model_client.py: httpx POST `{BASE_URL}/chat/completions`, temperature 0.3, max_tokens 4096, `"reasoning": {"effort":"none","enabled":False}` (provider-specific field), timeouts connect15/read120/write30/pool15, process-wide BoundedSemaphore (default 3) with 2s-poll acquire [R for constants].
### Pitfalls
- No context across bubbles/pages; no glossary; chunks of 10 lose continuity.
- Returning untranslated source on total failure means Japanese may be re-rendered (the singleton pass exists to avoid that, but the fallback `translated_map.get(idx, texts[i])` still can).
- Backoff is plain 2**n, no jitter, no 429/Retry-After handling [I from retry loop; model_client status handling only skimmed].
- `_looks_like_romaji_noise` is unused [I]. Mojibake "[8"/"[9" magic strings are hacks.
### Reusable as-is
Yes: parser, repair predicate, repair->singleton escalation, constraint tag, post-process. Swap `call_translation_model` for your client.

## 4. renderer.py (only the relevant functions read; 5.4k lines, rest [I])

### 4a. Bubble-band line fitting - `render_text_in_bubble` + `_layout_in_shape` (L5224-5477) [R]
```
clean = " ".join(_sanitize_for_font(text, font).split())
inside = bubble_mask>0 (& ~avoid_mask) ; need >=64 px
dist = distanceTransform(pad(inside,1), L2, 5)       # pad so crop edge counts as boundary
floor = max(12, round(page_h*0.0085)); rescue = max(10, round(page_h*0.0072))
max_size = max(floor, min(max_font_from_source_glyphs, page_h*0.06))
base_margin = clip(min(bw,bh)*0.07, 3, 18)
for margin in (base_margin, base_margin*0.5, 1.0):
    inner = dist>=margin ; centre=(centroid of inner, else argmax(dist))
    left,right = _row_spans(inner, cx)               # per row: run of interior containing cx (avoids straddling gaps)
    for size from max_size down to floor:
        stroke=max(1,size//12); word_w = getlength(w)+2*stroke
        line_h = ascent+descent+2*stroke ; gap=max(1,round(0.08*size)); pitch=line_h+gap
        for n in 1..min(nwords, rows//pitch):
            for shift in (0,-pitch/2,+pitch/2,-pitch,+pitch):   # try centred, then nudged
                top = cy - total/2 + shift ; per-line cap = min(right)-max(left) over that line's rows
                lines = greedy pack words into caps (None if any word > cap or leftovers; reject empty lines)
                return first success      # => biggest font, fewest lines
if fail: try narrow/condensed face with margins (0.5*base, 1.0); then rescue sizes floor-1..rescue
draw each line centred at band centre (anchor "ms")
```
- Words never broken here (no hyphenation in bubble path). Fewest lines at the largest font first = max size wins.
- Pitfalls: font load per size (`ImageFont.truetype` in the loop - slow, cache it); `_row_spans` is a Python per-row loop; fit never balances line lengths (greedy -> ragged last line); caps use the tightest row in a line band so ovals give narrow lines; 5 vertical shifts only.
- Reusable as-is: yes (needs `_sanitize_for_font`, `_outline_color_for_text`, constants). Best single piece of the repo.

### 4b. Rect/box fitting, wrap, hyphenation [R for _wrap_text/_hyphenate_word/_should_prefer_hyphenated_layout; `_fit_text_to_box` body I]
- `_wrap_text(draw,text,font,max_width,allow_char_wrap,allow_hyphenation,stroke_width)`: greedy word wrap via `textbbox` (with stroke); if both flags False and any word too wide returns `[]` = "shrink font" signal; otherwise hyphenate then hard char split (`_split_long_token_to_fit`).
- `_hyphenate_word`: words < 8 letters never split; preferred break before common suffix (tion sion ment ness able ible ful less ous ive ing ting ally ely ily ly er ed en est ize ise ated ious eous tial cial) if suffix within last 1 char of the end window (`idx>1 and idx+len(pat) >= len-1`, picks largest idx); fallback: consonant->vowel boundary searching outward from the middle; parts get "-" appended. English-only, not a real hyphenation dictionary (use `pyphen` instead).
- Policy `_should_prefer_hyphenated_layout`: hyphen layout used only if it yields a larger font AND contains zero renderer-inserted breaks (i.e. effectively "never hyphenate when a whole-word layout fits").
- `_line_spacing_for_size`: dialogue max(4, 0.25*size); sfx max(2, 0.12*size). `_stroke_width_for_font`: max(1, size//12).

### 4c. Font charset sanitising [R] (`_get_font_charset`, `_sanitize_for_font`, L79-167)
```
charset = parse `fc-query --format=%{charset}\n <font>`  (hex ranges "20-7e a0-ff"), cached; on failure -> ASCII 0x20-0x7E only
if all chars in charset -> return
map: curly quotes->' or ", en/em/figure dash->-, … -> "...", ★☆->*, •->-, ·->., →->"->", ♪♫->~, ♥❤->"<3", 〜～->~
replacement chars kept only if the font supports them; other non-ASCII dropped; collapse spaces
if result empty -> return ORIGINAL text (can still tofu)
```
Pitfalls: depends on external `fc-query` binary (not on Windows/slim containers) - better use fontTools `cmap` or `PIL ImageFont.getmask`/`font.getbbox` compare against `.notdef`; ASCII chars absent from font are dropped silently; empty-result fallback returns unsanitised text. Reusable: the replacement table + flow; replace charset source with fontTools.

### 4d. Stroke / colour [R]
- Stroke width `max(1, size//12)`; stroke colour `_outline_color_for_text`: luma=0.299R+0.587G+0.114B of FILL; `<128 -> white outline else black`.
- Fill: mean gray of balloon paper pixels (`inside`) >140 -> black text else white; override with source colour if chromatic and `|luma_text - paper_luma| >= 70`.
- `detect_text_color`: median of ML-mask text pixels (fallback: border-ring bg median + Otsu on colour distance, threshold >= 25); `_snap_extreme_neutrals` returns None when chroma (max-min RGB) < 60, i.e. only clearly coloured text (chroma>=60) is returned; else None -> brightness logic. (Docstring/name mismatch: it no longer snaps.)
- Note stroke used inside bubbles even though text is black on white: stroke white around black text (cosmetic halo, helps over screentone).

### 4e. SFX box [R `_resolve_sfx_box`]
Largest contour (area>=12) + neighbours whose centre within `max(24, 0.28*min(w,h))`; expand by 0.45*text_w / 0.60*text_h (min 5); min box 28 px / 28% of w, 22 px / 24% of h. `_compact_sfx_text`: <=3 alpha tokens, collapse 3+ repeated letters to 2, keep 2 tokens, <=14 chars, capitalize. SFX uses a separate font (`_SFX_FONT_PATH`) [R paths, I usage].

## 5. Summary: what to lift for a Python core
1. layout.py flood-fill balloon segmentation (MIT, near drop-in).
2. `_layout_in_shape` + distance-transform margin + `_row_spans` band packing (cache fonts; consider balancing lines).
3. translator.py: batch(10) -> parse -> repair predicate -> batch repair -> singleton repair; the prompts; constraint tags.
4. Font sanitising table (swap fc-query for fontTools).
Avoid/skip: renderer.py's 5k-line rect-fallback maze (`_resolve_dialogue_box`, mask-anchored art dialogue heuristics), pipeline's Modal/GCP/Gradio glue, dead helpers (`_looks_like_romaji_noise`, `_violates_constraint`).
