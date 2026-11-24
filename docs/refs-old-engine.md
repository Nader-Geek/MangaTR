# Old engine study: manga-autotranslate--arabic (branch new-engine)

Repo: /home/claude/nader-geek/manga-autotranslate--arabic. Tags: [read] = read in code, [inf] = inferred.
Paths below are relative to `manga_sections/` unless noted. Line numbers are as of this checkout.

## 0. Orientation (read)
- `manga.py` (357 lines) is a loader that marshal-caches the `manga_sections/*.py` bodies (concatenated into one `MangaTranslator` class via `_MangaTranslatorBody_NN` mixins).
- The engine's translation target is **Persian (fa)**, not Arabic. All the comments, log lines, prompts and fonts (Vazirmatn) are Persian; the `MANGA_TARGET_LANG=ar` env is set by the bot (`bot/runner.py`), and Arabic translation actually runs on `legacy/manga_ar.py` (10,410 lines, NOT read) via Ollama. [read: runner.py build_plan; inf: that manga_ar.py is a separate older engine]
- Source language is mostly English. Latin guards are everywhere (`_latin_lang`, `_latin_junk_line`, `_is_non_english_script`). JA/KO/ZH are an opt-in via `ocr_langs`.
- Comments and docstrings are stripped; the logic is only in code and in the Persian log strings. `PROGRESS.md` holds the history of per-bug fixes.
- Pipeline phases: `extract_regions_phase` (70:4074) -> `translate_regions` (80:1243) -> `finish_page_phase` (70:4498) -> `clean_image` (85:8902) -> `render_translations` (90:286 -> 95:2669 `_render_one_region`).
- Page kinds: `dialogue | promo | sfx | junk`. Det classes: `bubble | text_bubble | text_free` (RT-DETR v2 `ogkalu/comic-text-and-bubble-detector`, conf 0.35, iou 0.40, 640 input, max 120 dets, tiled at 1680px with 520 overlap, only when h > 2100) (20_models.py:1867-2149).
- 16 tone labels (`BalloonToneClassifier.TONES`, 40:659): normal, shout, comedy_shout, whisper, sun_thought, thought, free_text, system, monster, cry, fear, broadcast, letter, narrator, square_thought, black. The classifier is pixel heuristics (Otsu, contour counts, e.g. 10+ small dashed contours -> whisper), not an ML model. [read]

## 1. OCR (30_ocr.py)
Backends [read]:
| Class | Line | Role |
|---|---|---|
| `RapidOCRBackend` | 692 | Main desktop path. `rapidocr` (new API) on ONNX. PP-OCRv6 det small + PP-OCRv6 medium rec (en); `PP-OCRv5_korean_rec` (ko); `japan_PP-OCRv4_rec_mobile` (ja). Cls OFF. Falls back to old `rapidocr_onnxruntime` v3. |
| `PaddleAndroidOCR` | 47 | Android only, via a Java bridge (`recognizeAuto`); pipe-string protocol `META|conf|x1,y1..x4,y4|text`. Drops conf < 0.25. |
| `MtLadderOCR` | 271 | Android wrapper. Crop mode boosts crops: pad 12px white; scale x3 if min side < 96, x2 if max side < 480. Maps boxes back. |
| `PaddleOCRWrapper` | 1319 | Thin shim around a Paddle engine. |
Details worth porting:
- Det resize policy (20_models:2358): for non-CJK, `Det.limit_side_len=960`, `limit_type=max` (env `MANGA_OCR_DET_LIMIT/TYPE`). CJK uses the library default (736/min).
- Model pair choice (20_models:2327): default det is `PP-OCRv6_det_small`; medium det only with `MANGA_OCR_DET_MEDIUM`; small/tiny sets via env/emulator. Models are downloaded into `<MANGA_FILES_DIR>/rapidocr_models` from HF/ModelScope mirrors; explicit `Det/Rec/Cls.model_path` are set only if the file is > 100 KB. Korean/v6 need an extra `rec_keys_path` file.
- Threads: intra = 2 (desktop default), 1-2 on Android, overridable (`MANGA_OCR_THREADS`, `MANGA_ORT_THREADS`); inter = 1. (ORT option parity is already ported.)
- GPU safety net (already partly covered by your ORT parity work): CUDA is probed by creating a real `InferenceSession` on the det model; DML via `_dml_probe_model`; DML runs are serialized with `_DML_RUN_LOCK`; `_gpu_warmup` runs two synthetic images (110x520, 320x900 with "MANGA 12345"); both failing -> `_gpu_recover(force)` rebuilds the engine on CPU. Runtime errors containing 887a0006/80070057/dml/cuda/cudnn/oom/"device hung" trigger a one-time CPU rebuild and a retry (`ocr()`, 30:1174).
- Text normalization: `_deaccent` (NFKD + strip combining marks) on every OCR line. CJK guard (`_cjk_dominant`: CJK count >= max(2, latin count)) drops Chinese-hallucinated lines when the source is Latin.
- Auxiliary engines: `det_only` (1006) and `rec_only` (1060) build separate engines from `_eng_params` with GPU keys stripped (CPU-only, so they can run in parallel with the GPU engine). `rec_only` is used for the fast single-line path and the missed-quad rescue.
- Multi-language Android (`_ocr_multi`, 144): try "multi" first; if mean conf < 0.72 try "ko" and switch to it if conf is better by > 0.02; sticky `_multi_pref`.
- Speed-lab (speed_test_*, 423-690): a benchmarking harness; not needed.

## 2. Extraction, detection, SFX, watermark (70_extract.py)
Per-crop OCR, `_ocr_crop_run` (70:402) [read]:
1. Crop = rect + 8px pad. Skew estimate (`_estimate_skew_angle`, 70:5: connected components of ink, min-area-rect long side, median; components need area >= 150, max side >= 40, aspect >= 2). If |skew| >= 8 deg the crop is widened by `min(120, sin(skew)*maxside + 16)`.
2. Upscale `base_scale`: 2.4 if max side < 200; 1.8 if < 360; 1.35 if < 600; else 1.0. Core inset 6% when min side >= 160.
3. **Fast single-line path** (`_rec_fast_single_line`, 70:295): skips det, only rec. Conditions: not tilted; background median >= 120 (dark ink) or <= 108 (light ink); exactly one row band (rows with ink density > 3%, gap < 3 rows, band >= 5 px); that band holds >= 88% of the ink; touches no edge unless edge density < 55% of core; rec conf >= 0.85; >= 2 Latin letters and Latin >= half of the chars. Off with `MANGA_REC_FAST_OFF`.
4. Variants run in order: `raw` (scale), `clahe` (LAB L channel, clip 2.5, 8x8; skipped if p90-p10 >= 120), `otsu` (scale+0.25, max 2.2), `inv` (only if median < 110; scale+0.4, max 2.6). Lines are merged across variants by bbox overlap > 0.35 (higher conf wins). Variant score = `latin*2 + conf*3 + min(len,24)*0.15`.
5. Early-stop on variant 0: conf >= 0.80 and >= 3 Latin letters (or >= 2 CJK) and line y-span covers >= 60% of the ink rows; or (conf >= 0.86 and latin >= 8) or (conf >= 0.78 and latin >= 20).
6. Per-line filters: `min_confidence`; `_is_non_english_script`; `_line_is_foreign` (a line whose center lies in a neighbor block and which overlaps it >= 45% belongs to that neighbor, dropped); `_latin_junk_line` (drops <=2-char vowel-less tokens, digit-only etc., may repair to "I").
7. Gap rescue ("شکاف"): if a tall crop (max side >= 260, aspect >= 1.5) has a large gap between lines, re-read the gap; a +-90 deg rotation rescue ("r90", 70:~1180) for vertical text.
- OCR crop cache (<= 512 entries per page); a thread pool for the crops (`_ocr_pool_workers`: min(4, physical cores // 2), env `MANGA_OCR_WORKERS`) only for `RapidOCRBackend`.

Detector-driven flow `_extract_regions_from_bubbles` (70:2383) [read]:
- Drop contained boxes (`_drop_contained_boxes`, containment 0.72); split glued balloons (`_split_adjacent_balloons`, 1316) but cancel the split if it cuts a text line (`_split_cuts_text`, 1400).
- Drop boxes < 16 px, or area < 0.08% of the page and max side < 60.
- Empty OCR + text-like ink -> region with `ocr_empty=True` (cleaned only). "Dots/tiny symbols" ink (<= 10 comps, max comp height < 18% of box) -> untouched.
- Angle: poly long-side angle (kept if >= 3 deg), else engine angle (median, |a| <= 45), else ink slant (`_ink_slant_angle`, threshold 5 deg horizontal / 7 deg vertical). `_verify_angle_signs` (3562) re-checks (> 45 deg -> 0; flat-veto).
- Post: `_merge_overlapping_regions`, `_dedup_duplicate_text_regions`, `_split_mixed_column_regions`, `_rescue_orphan_lines` (3227, env `MANGA_ORPHAN_RESCUE`; band reach 2.2x line height, ink density 0.4-55%), `_rec_rescue_missed_quads` (3935: rec on detector quads OCR missed).
- Reading order sort: row bucket `y // 80`, then x (rtl: by -(x1+x2)).
- `_classify_text` (85:71): dialogue / promo / junk using `DOMAIN_RE`, `PROMO_RE`, `WATERMARK_PATTERNS` (40:781, ~30 scan-group names), a big list of short dialogue words, vowel checks.
- No-touch rule: region text without any letter (`source_has_letters`, 40:330) -> `no_touch=True` (neither translated nor cleaned).

System windows (game UI boxes) [read]:
- `_SYSTEM_KEYWORDS` (70:50, EN/JA/KO/ZH, ~120 words: quest, status, level, reward...).
- `_system_window_score` (73) is additive: bracketed `[..]`/`【..】` +3; keyword >= 1 +2, >= 3 +1 more; >= 2 colons +2; >= 3 lines with similar length +2; line gaps >= 1.35x height (>= 3 lines) +2; >= 4 lines each <= 42 chars +1; >= 3 lines with no sentence-final punctuation +1; aspect >= 2.2 with >= 3 lines +1.
- `_system_window_strong` (179): bracket OR keyword OR >= 2 colons. `_is_label_block` (152): 2-12 lines, max 46 chars, avg <= 30, no `?!...` end, >= 2 colon lines or bracket.
- Accept (`_sys_ok`): strong AND (score >= 5 OR single line <= 28 chars OR label block). Split into one region per line (`_sys_can_split`: >= 2 polys, label block or score >= 6) using `_cluster_ocr_lines` (y-tol 0.65 x median height); each gets `bubble_style="system"`, `style_locked`, `sys_split`, `cover_rect`. The system style uses a monospace/"code/mono/system/sahel" font looked up in `fonts/` (50_translator:635). Cleanup fills the box interior (`_system_box_interior`, 85:6037; `_fill_system_box_interiors`, 8744).

SFX [read]:
- `clean_sfx` flag (default off). SFX are `kind="sfx"`, untouched unless `clean_sfx`.
- `_discover_sfx_regions` (70:3686) calls `self._discover_free_text_regions(...)`, **which does not exist anywhere in this checkout** (grep: only the call site). It is wrapped in try/except returning `[]`, so SFX discovery is dead code here. [read; caveat inf: may live in an older/other branch]
- `_merge_sfx_fragments` (3695): merges rects within 70px; re-expands to the ink mask (|gray-median31| > 52) +-80px, 3 iterations. Reusable idea.
- `text_free` detector class is only cleaned when `MANGA_CLEAN_TEXTFREE` (Android gate).

Watermarks / promos [read]:
- `_add_watermark_regions` (70:2142): a **second** OCR pass over the whole page (fresh `rapidocr_onnxruntime.RapidOCR()` each call - slow) matching `www.|https?://|domain.(com|net|org|io|xyz)|patreon|chapter N ... page N`, conf >= 0.60, rect >= 12x6, not overlapping existing regions -> `kind=promo`.
- `_scan_watermark_banners` (4192): tiles of 2200 px with 120 overlap; stroke map |gray - median21| > 42, close 25x5 and dilate 11x5; candidate blobs need w 60..1000, h 12..170, w >= 2.2h, fill 0.06..0.95, stroke ratio 0.025..0.60, not in existing regions; edge-near candidates (<= 40 px from left/right) first; OCR budget `min(40, 24 + 6*(tiles-2))` (`MANGA_BANNER_OCR_MAX`), round-robin across tiles, early stop after 8 empty OCRs (`MANGA_BANNER_EARLY_STOP`), time limit `max(8s, 3s*tiles)`. Hits -> `kind=promo`.
- Promo regions are left untouched in the output ("ازش رد شه") but protected/flattened in cleanup (`_promo_band_flatten`, `_expand_promo_rect` pads 22px, colored-banner HSV test H 95..178, S >= 45, V >= 25).
- Text-level: `_is_watermark_text` (80:106), `_strip_watermark_tokens` (85:51), `_dialogue_despite_watermark` (>= 3 vowel words after removing domain).

Page-level helpers (70:4788-5445) [read titles only]: `_is_mostly_blank` (std < 12, unique < 24), `_cluster_widths`, `_normalize_page_width`, `_find_safe_cut_y_smart`, `_repair_page_seams`, `_stitch_pages_for_efficiency` (webtoon strips; the bot disables them via `--stitch-max-height 0 --no-seam-repair`).

## 3. Translation, rate limiting, key rotation, fallback (50_translator, 80_translate, 40_api_text)
Defaults [read]: `max_retries=8`, `api_timeout=30`, `request_delay=0`, `translation_temperature=0.7`, `top_p=0.9`, `gemini_rpm_budget=5` per key.
Providers (80:1076): `gemini` (google-genai, streaming JSON), `openai` (chat completions, also custom endpoints), `openai-responses`, `anthropic`, `google-endpoint`, `ollama` via OpenAI-compatible `/v1`. [read]
Gemini call (80:600): `response_mime_type=application/json`, `thinking_level="high"` if `deep_thinking`, safety BLOCK_NONE on 4 categories (disabled for the session via `_sbn_ok=False` if the API rejects it), temp/top_p special-cased to 0.4/0.7 on `gemini-3.6-flash`. Streaming parses finished pages incrementally (`_salvage_stream_pages`, `on_item`). Non-structured calls: temp 0.2, max_output_tokens 2048. Per-call timeout `min(1800, max(api_timeout, 60 + 1.5*n_regions + chars/40))`; executed in a 1-worker ThreadPool with a timeout that recreates the client. [read]

Pacing (50:1838) [read]:
- Sliding window of fire times over 60s; `budget = 60 / (rpm_per_key * n_keys)`; `max_burst = round(60/budget)`; if the window is full sleep until the oldest ages out (cap 30s). One global lock (`_pace_lock`); calls inside `_api_lock`.
- Key cooldown: parse `retryDelay`/`Retry-After`/"N seconds until"; if >= 15s the key sleeps `min(d+2, 300)s` in `_key_cooldown_until`.
- Random key per page (`_pick_random_api_key`): random among non-cooldown keys, falls back to all.

Error classification [read]:
| Class | Function (line) | Match |
|---|---|---|
| Bad/banned key | `_is_banned_or_invalid_key_error` 50:1277 | "api key not valid", API_KEY_INVALID, consumer_suspended, permission_denied, 401+key/auth... -> remove key |
| Model unavailable | 1299 | 503, 404, UNAVAILABLE, "high demand", "model not found", INVALID_ARGUMENT... |
| Permanently gone | 1319 | 404/NOT_FOUND, "no longer available", "developer instruction is not enabled", ... -> drop model |
| Rate / per-minute quota | `_is_rate_or_model_quota_error` 80:5 | RESOURCE_EXHAUSTED, 429, PerMinute/PerModel/PerHour (timeouts/SSL excluded first; daily excluded) |
| Daily quota | `_is_daily_quota_error` 50:1878 | "PerDay", "GenerateRequestsPerDay", "requests per day", daily+quota |
| Timeout / dead conn | in 80:1451 loop | timeout, 504, deadline, "bad file descriptor", "ssl:", reset, broken pipe |
| Truncation / filter | `_last_finish_reason` | MAX_TOKENS/LENGTH -> stop; SAFETY/PROHIBITED/RECITATION/BLOCKLIST/SPII -> stop after 2 |
| Censored/refusal output | `_is_censored_translation` 80:341 | cores like "censored"/"redacted"/"سانسور"; "i'm sorry..."; "cannot translate" -> retried with `_CENSOR_RETRY_NOTE` |

Model cascade [read]:
- Models from `gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-4-hydrogen, gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, gemini-3-flash-preview` (offline list, 40:340). Live discovery via API/REST (`_discover_models_from_api`), filtered by `_is_bad_translate_model` (bans computer-use, veo, image, tts, live, audio, embedding, gemma, ..., non-gemini, major version <= 2), then optional live style probe (`gemini_style_probe_many`) that drops models giving "bookish" Persian.
- Sort key (`_model_sort_key`): (age penalty for <2.0, type rank lite < flash < preview < pro, newer version first). Costly "pro" models go last. If the user gave a model list, ONLY that list is used.
- Per-page "local cascade" = 6 models starting at last-good (or first "3."/flash-latest/lite). `_note_model_fail` bans a model after 3 fails without ever having succeeded; `_note_model_ok` clears.
Retry ladder (80:1451 `_translate_regions_batch`) [read]:
1. Timeout/conn-dead: recreate the client; same-model retry up to `MANGA_TIMEOUT_RETRIES` (3, max 10) with wait `min(MANGA_TIMEOUT_WAIT=2 * 2^n, 30)`; then next model; whole cascade reset once; then next key (cycle) + reset; else sleep 0.25 and retry.
2. Invalid key -> delete key, apply the next; raise `GeminiQuotaExhausted` if none left.
3. Daily quota: same model twice in a row (streak >= 2) -> drop model; else next key; else next model (reset key index to 0); else raise `GeminiQuotaExhausted`.
4. Rate/minute: if daily (`_daily_dead[(model,key)]`) try keys not yet dead for the model, else drop the model; if cascade has > 1 distinct models switch model immediately; else cooldown the key, sleep `min(max(retryDelay, 2+attempt), 60)` (or `min(2+attempt, 6)`), then next key, next model, cascade reset once (sleep 1.0), next key + reset (1.5s), final sleep 2.0.
5. Permanently gone model: use `_extract_suggested_model` ("use models/xxx"), drop, switch.
6. Partial output: untranslated regions are retried in the next attempt (only the missing ones); after success `_force_translate_missing` does a last per-region rescue.
Other translation features [read]: duplicate dialogues fanned out (`_fanout_duplicate_translations`) and a cross-page memory (`_mt_dupes_prefill/record`) to save tokens; `_fix_ocr_text` (small regex fixes, trailing-number strip unless preceded by lv/ch/hp...); `uncensor_swears`; style anchoring (`_note_style_samples`, `_fa_style_verdict`, `_style_retry_bookish` re-asks if the Persian is formal/bookish, adding `_STYLE_RETRY_NOTE`); prompts are `PROMPT_PRESETS`/`mt_user_prompt` (page-batched JSON: `{filename, dialogues:[[idx,text]]}`).
Bot-side translator (bot/translate_ollama.py, 119 lines) [read]: batches of 8 numbered lines to Ollama `/api/chat` format=json (temp 0.2, num_ctx 8192) or OpenAI-compatible `/chat/completions` (retries without `response_format` on HTTP 400). Reply must be `{"translations":[...]}` of identical length; a bad reply is retried once and then goes line-by-line; failures stay "". Dialects: msa / egyptian. Provider table (`bot/providers.py`): gemini, openai, deepseek, groq, openrouter, xai, together (keys only from env; `auto` = first with a key, else ollama).

## 4. Rendering (90_render.py, 95_run.py)
Where it really lives [read]: `_render_one_region` (95:2669), `_render_curved_lines` (95:3075), `_render_arc_text` (95:~3155); 90_render.py has the helpers (fit, font, colors), exports (PDF/ZIP/PSD/HTML) and the debug overlay.
Arabic/Persian shaping [read]: `arabic_reshaper.reshape(text)` then `bidi.get_display(...)` (80:2082 `_shape_farsi`) before every measure/draw call. Fonts are loaded with `ImageFont.Layout.BASIC` (no raqm), because shaping is done manually. Wrapping measures the *shaped* string word by word.
Font coverage guard [read]: `_font_covers` (90:24) draws a probe string made of all Persian letters in isolated/initial/medial/final forms plus digits, `ﻵ-ﻼ` lam-alef forms, and compares glyph bitmaps to three Private-Use "tofu" references; a font that fails is replaced (`_cover_fallback_font`) with a warning. Per-style fonts via `font_by_style` with the same guard.
Fit algorithm (`_wrap_and_fit`, 90:145): font size loops from `max_size` (<= 48; from OCR line height x1.2 clipped 12..48, and capped at 0.80 x original line height if >= 9) down to min 11 (14 if <= 2 words and <= 12 chars); outer loop over line gaps (4, 2, 1, 0); then continues to size 6 with gap 0. Stroke width `_stroke_width_for`: 1 if size <= 14, 2 if <= 22, else max(2, size // 16). Line height = bbox of "آیگچ" + gap.
Text color (`_pick_text_and_stroke`, 90:225): median color of original ink pixels inside the OCR polygon (threshold +-20 around the cleaned background median); if saturation < 25, use near-black (18) with white stroke when background >= 140, else near-white (245) with black stroke; saturated ink keeps its color with a contrast stroke (lum >= 140 -> dark stroke).
Layout [read]:
- Box: region rect; if the OCR poly center is > 35% off the rect center use the poly box; for vertical text or dense polys use `_text_block_box`. Short text (<= 2 words) in a small rect (w < 90 or h < 50) expands by max(4, 12%). Padding 5% (short) / 8%.
- Horizontal (|angle| < 3): if the number of wrapped lines equals the number of original OCR lines (`_group_polys_to_lines`, median line height >= 8), each line is vertically aligned to the original line center ("ALIGN" mode); otherwise a centered block. Lines are centered horizontally in the box.
- Rotated (|angle| >= 3): the text is drawn onto an RGBA tile, rotated by `-angle` with BICUBIC, `expand=True`, rescaled with LANCZOS to fit <= 104% of the rect, pasted by center. The fit box is derived from the rotated rect via cos/sin of the angle (x0.94). Angle sources: polys long-side, ink slant (`_ink_slant_*`, `_ink_bitmap_polar`), engine angle; flat ink vetoes a poly angle (`angle_flat_veto`); |angle| > 45 -> 0.
- Curved text: `_curve_path_from_ink` (70:1926) accepts only a single, non-vertical line; fits a quadratic y = a x^2 + b x + c on the rotated ink column means (nb = clip(span/3, 10, 64) buckets), requires span >= 64, sag >= max(3.5, 0.12 x row height), total turn 12..75 deg, residual std <= 0.55 x line height, >= 82% of ink within 0.70 x line height, the poly tangent agrees within 16 deg, no multi-band gap (>= 20% of band height), uniform column height (std <= 28%). Returns a 24-point path + `lh`, `sag`, `d_ang`, `arc_len`. `_render_arc_text` extends the path 18% beyond both ends for single-source lines, lays each wrapped row along an offset of the path (normal offset `((K-1)/2 - i) x line_h`), draws each glyph tile rotated to its local tangent, and returns False on failure so the straight path is used. Max rows 4 (single-line source) or 5. `_render_curved_lines` handles multi-line text where each OCR line has its own angle.
- Failure ladder: `_render_one_region` -> retry after gc -> `_render_one_region_fallback` -> `_restore_region_pixels` (put back the original pixels, never leave a black bubble). JSON-looking translations are refused.
- Not found in 90/95: vertical (tategaki) Arabic layout, per-bubble-shape text boxes (ellipse inscribed rect). The text box is the detector/OCR rect, not the bubble interior. [read; absence inferred from the greps]
bot/typeset.py (151 lines) [read]: a PIL-only renderer for `chapter.json`: auto-fit loop from `fontSize*1.25` down to 8, line height 1.25, anchor "ma", effects: drop shadow (offset + blur), glow (stroke r/2 + gaussian r, x1.8 gain), stroke, vertical gradient, fill; 24 px padded crop so effects can spill.

## 5. Cleanup special-case strategies (85_cleanup.py, 14,284 lines)
Already ported by you: bubble interior (`_bubble_interior_mask` 536, `_bubble_interior_sweep` 13117), halo absorb (`_absorb_text_halo` 8815), residue scrub (`_residue_sweep` 11663, `_scrub_dark/bright_residuals` 14160/14178). Per-page driver: `_clean_image_page` (8910-11216, ~2300 lines) in this order [read via log strings]: no-touch masks -> halo absorb -> structure/light-ink protect (`_structural_protect_mask`, `_rect_structure_lock`) -> uncovered ink bands/strips -> system-box interiors -> enclosure/line instant fill -> dark-container forced LaMa -> cluster fills (flat/paper/color/ring-plane) -> LaMa (clustered, batched, parallel) -> per-crop verify/repair -> two-image compose (donor vs original) -> residue sweep -> bubble sweep -> ghost polish -> MT polygon clean -> final verify.
Cleaning method choice (`_decide_lama`, 50:310): `--clean-method opencv|lama-manga (lama.onnx)|lama-big (torch big-lama.pt)`; auto: CUDA with VRAM >= `_LAMA_MIN_VRAM_GB` -> big-lama; desktop with ORT -> lama.onnx + flat fills; low free RAM (< 2.2 GB) -> OpenCV. `MANGA_CLEAN_LEVEL` quality|fast. LaMa grouping: `_lama_group_prep` (4094) pad = clip(15% of group size, 12..48), min crop 96, promo overlap pad clip(0.55*max side, 48..200), downscale when max side > `MANGA_LAMA_MAX_SIDE` (2048), sizes rounded to multiples of 8; `_lama_workers_auto` (8/5/2 workers by cores and free RAM, cap 8); `_lama_batch_cap` (`MANGA_LAMA_BATCH`; 2 on Android or < 3 GB RAM, else 8); 600 s per-patch timeout.

Strategy table (name, line, case, key constants) [read heads of each function; constants are those visible in the code head only - bodies up to thousands of lines may hold more]:
| Function | Line | Case solved | Constants seen |
|---|---|---|---|
| `_protect_bubble_wall` | 450 | Keep the speech bubble outline from being erased | border = clip(min(h,w)//7, 6, 20); Canny 50/120 |
| `_ink_mask_inside_bubble` / `_text_zone_in_crop` | 362 / 411 | Ink restricted to the bubble interior / text zone | - |
| `_drop_non_text_components` / `_fill_holes_mask` | 505 / 527 | Remove art-like blobs from the text mask | - |
| `_box_bg_stats` / `_box_bg_textured` | 583 / 615 | Is the box background flat or textured (chooses flat fill vs LaMa) | - |
| `_uncovered_ink_bands` | 623 | Text ink OCR polys missed (bands above/below lines) | box >= 24x24; dark = g < med-32; bright = g > med+42; reject if core > 30%; known dilated 7x7 |
| `_expand_donor_text_strips` | 835 | Grow the mask for donor strips (halo/anti-alias) | dilate 9x9; remove `_page_protect` |
| `_bubble_domain_local` / `_fast_bubble_guard` | 1040 / 1108 | Decide whether a mask leaks outside the bubble; shrink it | margin M=34 |
| `_limit_mask_to_bubble`, `_pad_dilate_wall_safe` | 1263 / 1322 | Clip dilation at walls | - |
| `_carve_structural_walls` | 1378 | Carve walls/lines (panel borders, bubble outline) out of the mask; skipped for light ink | M=24 |
| `_free_text_tight_ok`, `_ink_tight_ok` | 1487 / 1558 | Tight mask for free text on art (avoid LaMa smearing art) | M=26 |
| `_enclosure_tight_mask` | 1599 | Text inside a closed enclosure (box/caption) | polygon fill ratio check |
| `_white_bubble_tighten` | 1774 | White bubbles: shrink to ink only | M=30; interior area >= 400; median interior >= 185 |
| `_group_polys_to_lines`, `_line_strips_mask`, `_line_strip_ink` | 1879-1996 | Per-line strip masks | - |
| `_panel_leftover_ink` | 2211 | Leftover ink in a panel after strip cleaning | max h = max(64, 1.15*med_h+4); max area = max(2600, 0.6*med_h^2); max w = max(180, 2.6*med_h) |
| `_banner_like_free_text` | 2294 | Colored banner text (blue/purple bar) treated as promo-like | rect pad (10,8,12,10); w<=0.7W, h<=0.25H; HSV H 95..178, S>=45, V>=25; non-paper >= 35% |
| `_build_text_mask_fast` / `_build_text_mask` | 2354 / 3558 | Mask builder (OCR polys + ink + absorb) | ~570 lines + 140 lines |
| `_ink_absorb_mask`, `_stray_ink_reclaim`, `_absorb_block_neighbors` | 2927 / 3090 / 3267 | Pull in stray ink and neighbor-block ink; skips promo/sfx | - |
| `_line_instant_fill` | 3343 | Flat-background lines: fill directly (no LaMa), with Sobel gradient check | - |
| `_tilted_ink_mask`, `_glyph_mask_in_zone` | 3697 / 3741 | Rotated text masks | - |
| `_flat_fill_uniform` | 3769 | Uniform ring -> flat color fill | ring 5x5 dilation, >= 40 px; 96% of ring within 20 of median |
| `_color_fill_cluster`, `_paper_fill_cluster`, `_flat_fill_cluster` | 3793 / 3838 / 3875 | Colored panel / paper white / generic flat fills for a cluster | paper: ring dilate 15x15, >= 200 px, lum >= 190; flat: ring 9x9, >= 60 px, fit_tol 6 |
| `_mask_clusters`, `_wall_lines`, `_group_nearby_patches` | 3947-4021 | Group masks to LaMa patches; keep wall lines | - |
| `_lama_run_dual`, `_lama_run_groups_batched`, `_lama_group_finish` | 5489 / 5571 / 4332 | LaMa execution paths (GPU+CPU twin, batched, finish/compose) | see above |
| `_lama_rescue_boxes`, `_telea_fix_boxes` | 5723 / 5818 | Re-run failing boxes via LaMa, then OpenCV Telea | Telea ring dilate 13x13 |
| `_enclosure_instant_fill` | 5861 | Dark container (black box) filled with its own bg; skip promo | - |
| `_system_box_interior` | 6037 | Game system-window interior detection | box >= 24x16; bright = g>=120, bright ratio >= 0.30 |
| `_rotated_region_strips` | 6095 | Strip masks for rotated text (rotate page crop, strip, rotate back) | pad 30 |
| `_compose_region_strips`, `_compose_strip_mask` | 6175 / 6278 | Strip-based compose for regions | ~300 lines |
| `_strip_ghost_polish` | 6583 | Faint shadow/ghost left in a strip | median blur 41; dark/bright delta 22..60 |
| `_compose_polish` | 6746 | Polish connected components of the compose mask | ellipse 15, 25; dilate 9x9 |
| `_texture_refill_donor` | 7228 | Re-add halftone/texture lost by flat fills | Gaussian sigma 3.5; comp w >= 24, 6 <= h <= 380 |
| `_fill_harmony_qc` | 7391 | Reject donor fills whose color/saturation differs from the original | diff > 12 |
| `_promo_band_flatten` | 7692 | Flatten the colored promo band without erasing it | HSV H 95..178 |
| `_compose_two_image` | 7812 | Master: take donor only where mask, else original (+ refill, QC) | ~540 lines |
| `_reject_dark_strip_fills` | 8350 | Dark smear from LaMa on strips: revert | - |
| `_compose_corner_feather` | 8522 | Feather corners at compose boundary; forbid wall band / protect | halo dilate 25x25; skip if < 40 px |
| `_flat_white_rescue` | 8584 | LaMa turned white area gray/colored: restore flat white | per component |
| `_fill_system_box_interiors` | 8744 | Fill system-window interiors by group (`cover_rect`) | - |
| `_faint_ghost_polish`, `_compose_dot_sweep` | 11216 / 11305 | Ghost lines / dots left after fill | - |
| `_leftover_text_sweep` | 11507 | OCR the cleaned page in chunks (`_ocr_cleaned_chunked` 11470) and remove text still readable | - |
| `_mt_polygon_clean` / `_mt_ring_samples` | 12025 / 12005 | Polygon-based fill from OCR quads using ring samples; heal | - |
| `_opencv_fill_components`, `_inpaint_big_component_banded` | 12188 / 12238 | Big components inpainted in horizontal bands | - |
| `_bg_is_textured`, `_glyph_refine_mask` | 12284 / 12301 | Textured background test; refine mask to glyph pixels | - |
| `_ring_plane_fill` | 12336 | Fit a plane to the ring color for gradients | pad 10; ellipse 9; comps >= 8 px |
| `_opencv_inpaint_hq` | 12393 | Telea with radius `inpaint_radius` (3), second downscaled pass to 640 px for big masks | mask > 0.4% of page and max side > 640 |
| `_smooth_bg_fill` | 12449 | Large-kernel blurred background fill | k1=51, k2=21, feather 3 |
| `_rect_structure_lock`, `_structural_protect_mask` | 12489 / 12559 | Lock panel frames / light structure (env `MANGA_NO_STRUCT_LOCK`) | ~500 lines |
| `_expand_promo_rect` | 13067 | Expand promo box to the whole colored banner | pad 22 |
| `_final_verify_repair`, `_verify_repair_clean` | 13293 / 13554 | Per-crop check after LaMa/OpenCV and a union-level repair | crop pad 26 |
| `_line_tail_sweep`, `_dark_remnant_sweep` | 13387 / 13478 | Line ends / dark remnants after fill | near dilate 19x19; zone 35x35 |
Port priority (inferred from the amount of per-bug history in PROGRESS.md): two-image compose (donor vs original with QC), uncovered ink bands, structural/wall protection, instant flat fills + ring-plane fill, system-box interior, promo/banner protection, leftover OCR sweep. LaMa is `lama.onnx`/`big-lama.pt` (already covered by your model layer).

## 6. Discord layer (bot/, 4,078 lines) [read]
- Stack: discord.py client with `app_commands` tree (`GuardedTree` checks allowed guilds/admins), asyncpg Postgres job queue, subprocess runner. Config from env/`bot/.env` (`bot/config.py`, frozen dataclass).
- Commands (`bot.py`): `/clean`, `/extract`, `/typer` (cleaned pages + TypeR bubble coordinates), `/chapter` (chapter.zip = clean images + `chapter.json`, schema `manhuarmtl-v1`, optional Arabic translation + `render`), `/translate` (Arabic via Ollama, legacy engine), `/styles upload|show|map|clear` (per-user TypeR style packs), `/status`, `/history`, `/cancel`, `/queue`, `/help`, `/ping`. UI strings are Arabic; options: OCR (en, ja+en, ko+en, zh+en), reading order rtl/ltr, quality fast(OpenCV)/lama, format jpg/png/webp, dialect msa/egyptian, zip/pdf, provider choices.
- Submit flow (`MangaBot.submit`): defer -> per-user active cap (2) -> extension and size checks (`max_upload_mb` 100) or link fetch (Drive/Drive folder/Gofile, `fetch.py`, `max_link_mb` 500, zip-bomb guard `max_unzipped_mb` 600, path-traversal check) -> `inspect_upload` (page count; `max_pages` 60 / 30 translate) -> `create_job` -> queue embed -> status message id saved.
- Queue (`db.py`, `schema.sql`): `jobs` table (kind in clean/extract/typer/translate/chapter; status pending/running/done/failed/cancelled; options JSONB, stats JSONB, result_files, attempts, last_activity_at, expires_at). `claim_job` = `UPDATE ... WHERE id=(SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1)`; `requeue_stale` after a crash (max 2 attempts).
- Worker (`worker.py`): runs `manga.py` as a subprocess (never a shell string; secrets stripped from the env), pumps stdout to `run.log`, parses progress with regexes on the (Persian/Arabic) log lines (`stats.LogProgress`: "[*] استخراج متن: n/m", "[TIME] page#N clean+render="), edits one status message every `progress_interval` 8s, heartbeat 30s, monitor line (CPU/RAM/GPU) 30s, job timeout 120 min, kill-tree on cancel, cleanup of workdirs after `file_ttl_minutes` 20.
- Delivery: results zipped; `shrink_oversized_images` (JPEG quality ladder 95,90,85,80,72,65, then vertical slicing at quiet rows `_quiet_cut_rows`), `split_zip`, `batch_by_size` (<= 10 files and <= upload limit per message), fallback to Gofile (`hosting.py`) when the file does not fit. Typer export uses `--export-typer` and `typer_job.json` (pages -> regions: kind, tone, source, text, rect...).
- Engine contract worth keeping: bot invokes `manga.py -i -o --gpu/--cpu --provider ollama --reading-order --ocr-lang --img-format --export-typer DIR --workers N --stitch-max-height 0 --no-seam-repair [--clean-only --no-resume] [--lama]`, with env `MANGA_TARGET_LANG`, `MANGA_TYPER_FORMAT`, `MANGA_CLEAN_PAGES`. [read]

## 7. Gaps / uncertainties
- `legacy/manga_ar.py` (the actual Arabic translate/render engine, 10,410 lines) was not read; Arabic specifics (dialect prompts, Arabic font choice) there are unknown. Only its bot-side contract is known.
- `20_models.py`, `60_input.py`, `95_run.py` orchestration (3,355 lines), `99_cli.py` were read only in the parts above.
- The 85_cleanup constants column is limited to what the function heads show; thresholds deeper in the bodies (e.g. `_build_text_mask_fast` 570 lines, `_compose_two_image` 540 lines) were not extracted.
