# Koharu (koharu-rs/koharu v0.84.0) - reference summary

Path: /home/claude/koharu-rs/koharu (crates under `crates/`). Rust, edition 2024, Tauri desktop app.
Tags: **[R]** = read code; **[I]** = inferred. All paths below are relative to `crates/`.

## 0. Headline findings for a Python port
- **[R] Licences.** Repo code: `MIT OR Apache-2.0` (LICENSE-MIT, LICENSE-APACHE). koharu-pipeline is `MIT OR Apache-2.0 AND BSD-3-Clause` (Telea port keeps OpenCV notice); koharu-leptonica adds BSD-2-Clause. Index-Translate prompt template is from bilibili/Index-Translate (Apache-2.0).
- **[R] "Koharu Model License": NOT FOUND.** `grep -ri "model license|koharu model"` over the whole repo (md/mdx/rs/toml/json, excl. node_modules) returns nothing. The only model licences stated: manga-ocr-ar-experimental = **CC BY-NC-SA 4.0** (koharu-ml/src/manga_ocr_ar/mod.rs:4, docs/models/vision.mdx:23, pipeline README:179); Qwen-Image 2.1 = Qwen Research License (research/eval only). Others say "check each model card". Weights (mayocream/koharu-layout-highres, mayocream/lama-manga, mayocream/fonts) are fetched from HF at pinned commit hashes; their HF-card licences are not in this repo (unverified; check cards on HF if provenance matters).
- **[R] "manga_ocr_ar" is NOT Arabic OCR.** "ar" is an author tag; it reads *Japanese* manga (`language()` returns "ja", pipeline/stages/recognize/model.rs). Baberu = Japanese. Hayai = ja/zh/ko/en. **There is no Arabic OCR or Arabic-source support.** Arabic is only a possible *target* of translation/typesetting.
- **[R] No explicit Arabic code path** anywhere (grep "arabic" = 0 hits). RTL works only via generic `unicode-bidi` + HarfRust shaping + base-direction check. Arabic quality is **[I] untested by the project**.
- **[R] Typeset is not a pipeline stage.** `Stage` = Detect, Recognize, Inpaint, Translate only (pipeline/src/request.rs:63). Placement (frame/room/size) is written at translate time; rendering is koharu-renderer on demand; export is koharu-psd.

## 1. koharu-scene (6.5k LOC) - data model
- **Purpose [R]:** Photoshop-style layer tree + manga kinds, stored in one SQLite file (store.rs; app id 0x4B485255 "KHRU"), undo via `Edit{label, ops}`/`Session::apply`.
- Kinds (kinds.rs): Group, Image, Raster, Shape, Mask, **Panel{outline}**, **Balloon{outline}**, **Block{role: Text|Sfx, source: Text, translation: Option<Text>, mask: Layer<Mask>, cleanup: Layer<Raster>}**.
- `Text` (text.rs:224): content (RichText spans), style, `frame` (Point | Box{bounds,fit,anchor} | Path{path,fit,anchor,inset}), `direction` (Horizontal/Vertical), `language` (BCP-47), `transform` (2x3 affine; carries tilt). `CharStyle`: family, weight, italic, size, fill, stroke, stroke_width. `Text::bounds()` = frame bbox through transform.
- Pixels: 64x64 tiles, zlib level 3 each (tiles.rs:17; "3x faster than 6 for +4% bytes"). Attributes namespace `c2pa` records model provenance per stage.
- **Port advice [I]:** Port only the *shape*: page -> panels -> balloons -> blocks{source, translation, mask, cleanup}. Use a plain dataclass/JSON + PNG masks; skip the SQLite/undo machinery.
- **Weakness [I]:** heavy design for editing; the `translation: Option<Text>` is single target (one translation per block, no multi-language).

## 2. koharu-pipeline (5.8k LOC) - orchestration
- Scheduler (scheduler/mod.rs): `AHEAD=3` pages in flight (read / GPU / write), `RECOGNIZE_PAGES=8` per OCR call, model `KEEP=5min` idle. One GPU thread owns all models; priority = interactive > batch. Stage never calls another stage; a run replaces earlier output (locked layers are kept).
- **Detect** (stages/detect/mod.rs): scan -> `KoharuLayoutHighres.inference` (GPU) -> `Detected::layout(page)` + `Lettering::typography` (CPU, rayon) -> `layers::additions`. Removes all unlocked Panel/Balloon/Block layers then adds new. `GROW=3` px: block mask = glyph mask clipped to bbox (+1 glyph past line ends, not sides) and dilated 3 px (detect/layers.rs:122).
- **Recognize** (stages/recognize/mod.rs): crop = scan inside block source bbox; **pixels of *other* blocks' masks that aren't in this block's mask are painted with the crop's per-channel median colour** (`blank()`), preventing neighbour text leakage. Crops batched across <=8 pages. [R]
- **Inpaint** (stages/inpaint/): tiered, cheap first. Per block: ring of unmasked pixels `RING=4` beyond grown mask; if >=`INSIDE=0.9` of the mask is inside its balloon and ring has >= `RING_PIXELS=16` px: (a) flat colour (median) if mean abs deviation <= `FLAT=10`; (b) per-channel least-squares plane (gradient) fill; (c) Telea fast marching (`RADIUS=5`, OpenCV port, telea.rs); a ring pixel >= `LINE_CONTRAST=64` off the fit = line/art -> goes to model (mask dilated 1px, LInf). `EDGE=3.0` px inside balloon outline when sampling paper. Else -> LaMa. [R: flat.rs consts]
  - Windows for the model (windows.rs): `CONTEXT=256` px around each lettering, clipped to its **panel**, merged while same panel, overlapping, area <= `MOST=1<<20` px and side <= pass size (LaMa 800). Only masked pixels are written back. [R]
- **Translate** - see section 6.
- **Weaknesses [I]:** deeply tied to the scene + GPU thread; the tier thresholds were tuned on Blue Archive / Manga109 (colour pages, Japanese), so they may not transfer. The balloon-aware logic needs balloon outlines from the detector.

## 3. koharu-ml - models
### 3a. koharu_layout_highres (detector) [R: mod.rs, processor.rs, suppress.rs, order.rs, layout.rs, rooms.rs, measure.rs]
- RF-DETR Seg 2XL (roboflow rf-detr @4ab7c18), classes `text, onomatopoeia, bubble, panel` (+ background logit slot 5th, `logits [1,300,5]`, `num_select` from config, masks at res/4). HF `mayocream/koharu-layout-highres@53fef6f`. Needs torch (libtorch port), CUDA recommended.
- **Pre-process (must match training):** `LongestMaxSize(resolution)` (cv2 INTER_AREA down / INTER_LINEAR up, round-half-even), pad to square top-left with **white 255**, ImageNet mean/std. `resolution % 24 == 0`.
- **Post-process:** sigmoid, flatten, top-k over (query x class), **per-class thresholds** (`config.recommended_thresholds`, not in repo). Koharu overrides: **onomatopoeia 0.4 (checkpoint says 0.3), bubble 0.4 (checkpoint 0.5)**; text/panel = checkpoint's. **One label per query:** keep the text label if any else best score (classes are exclusive). Boxes cxcywh * long side of page. Masks: bilinear up to square, `>0`, crop to page.
- **Dedup (suppress.rs):** per-class NMS `IOU=0.5`; nested text (>=`NESTED=0.8` share of smaller inside bigger) is dropped (text only); onomatopoeia **merged** (box union, glyph union, area-weighted score) if IoU>=0.5 or nested>=0.8 AND `SHARED_GLYPHS=0.5` of smaller mask's pixels shared; repeat to fixpoint. `disjoint()` gives shared glyph pixels to the smaller box.
- **Glyph mask refinement (measure.rs `glyph_mask`):** Otsu split of the crop; "ink" side picked by paper = border colour; mask closed with radius 6 px per 1024 of long side; widened by 2 px per 1024; holes filled. Typography estimate: vertical?, tilt (+/-45 deg, 0 if < a stroke), font size (median-ink run of the cross-line profile), stroke width, ink/paper colours, outline band (`OUTLINE_TOLERANCE=48`, `PAPER_TOLERANCE=32`, `PLAIN_PAPER=0.99`).
- **Reading order (order.rs):** panels by recursive XY-cut: split at gutters into rows (top-to-bottom) first, else columns **right-to-left**, else pairwise `before()` with progressive box erosion `[0,.1,.2,.3,.4,.5]`. Lettering assigned to panel with >= `IN_PANEL=0.25` of its box (smallest among ties); panels nested if >=`NESTED=0.9`; "beside" panel needs >=`BESIDE=0.5` height overlap. Order is **hard-coded Japanese RTL**; no LTR (western) option [I].
- **Bubble assignment (layout.rs):** text belongs to bubble holding >= `IN_BUBBLE=0.5` of its *glyph pixels* (not box). Balloon outline = largest contour, simplified.
- **Rooms (rooms.rs):** per-text region inside a balloon: conjoined balloons split along chord between convexity-defect notches (`NOTCH=0.04` of sqrt(area)), chord must leave each text >= `SIDE=0.9` on one side; remaining sharing texts split by seeded watershed on distance transform; texts straddling the edge (< `STRADDLE=0.75` inside) get their box as room. Valuable; portable with OpenCV/skimage.
- **Weakness [I]:** only 4 classes; no text-direction/furigana class; ordering assumes manga RTL; thresholds tuned for Blue Archive.
### 3b. lama [R: lama/mod.rs, processor.rs]
- Big-LaMa FFC-ResNet fine-tuned on manga (`mayocream/lama-manga@f91c85b`), run **as IOPaint**: `WHOLE=800` (long side <= 800 -> whole image), else per-connected-mask-component crop with **margin 128**, shifted (not clipped) to stay inside image. Pad to multiple of 8 (symmetric pad), mask `>127`->binary, clamp output, **blend only inside mask** (`out*a + img*(1-a)`). Port = trivial with `simple-lama-inpainting` / IOPaint.
### 3c. manga_ocr_ar [R] (Japanese; CC BY-NC-SA 4.0; `muscgab/manga-ocr-ar-experimental@7873d84`)
- MonkeyOCRv2-B vision tower (12 layers, all-depth states fused) + 1-block AR decoder; char vocabulary (`vocab.json`), greedy decode with KV cache, `max_length` from config, `BATCH_SIZE=64`.
- Preprocess: patch 14; binary search largest uniform scale s.t. `ceil(h*s/14)*ceil(w*s/14) <= patch_budget` (default tests use 324 -> e.g. 53x280 -> 8x40 grid); resize Catmull-Rom (Pillow bicubic); canvas padded with **border median colour**; CLIP mean/std (0.481,0.458,0.408 / 0.269,0.261,0.276). Includes row/col index tensors (2D pos).
### 3d. baberu_ocr [R] (`genshiai-daichi/baberu-ocr@v1.2`, licence not in repo)
- DINOv2-base encoder + projector + small decoder; Catmull-Rom resize to `vision_image_size`; generation `max_new_tokens`, `repetition_penalty` from generation_config; **degenerate-run guards `max_content_run=12`, `max_symbol_run=16`** (mod.rs:100). Japanese only.
### 3e. hayai_ocr [R] (`JustANormalTinkerer/hayai-ocr-v2.7-supernova@5289b2c`, licence not in repo)
- SigLIP2 NaFlex encoder, 2x2 patch merge, GQA decoder, 2D M-RoPE, greedy. ja/zh/ko/en. **NaFlex budget chooser** (processor.rs:88-143): budgets {128,192,256,384}; `MAX_SCALE=4.0`, `MIN_PATCHES=64`, patch size from siglip2; thresholds: square = aspect 0.77..1.30; 128 if s128>=1.65&aspect<3.8 (vertical) / s128>=2.35&aspect<3.2 (horizontal); 384 if n384>n256 and (s256<1.28 | square&s256<2.20 | horiz&s256<1.45 | aspect>=3.8&s256<1.75 | aspect>=5&s256<2.20); 192 if s192>=1.55&aspect<3.6 (vert) / >=1.85&<3.0 (horiz); else 256. Only multilingual one but **no Arabic** [I; README lists ja/zh/ko/en].
- **Port advice [I]:** For Arabic source you need a different OCR; none here. For Japanese, the crop prep (blank neighbours, median pad) is the reusable idea.

## 4. koharu-renderer (7.7k LOC) - typesetting
Stack [R: Cargo.toml]: **harfrust** (HarfBuzz port), `unicode-bidi`, `icu_segmenter` (UAX14), `hypher` (hyphenation), `jieba-rs` (Chinese words), fontique/skrifa, vello_gpu. Pipeline (text/mod.rs `layout`): Styled -> itemize -> shape (once) -> breaks -> fit -> arrange.
- **Itemize (itemize.rs):** runs cut at span, bidi level, script, vertical turns, font fallback. Font fallback chosen per grapheme (first font in chain covering it; common/inherited chars take the previous char's font). Missing chars reported in `uncovered`. Vertical: UAX#50 orientation; `!?` upright full-width; 2-3 `!?` and 1-2 digit numbers set as tate-chu-yoko (itemize.rs:263).
- **Shape (shape.rs):** `BidiInfo::new(text, None)` per paragraph; HarfRust direction from run level (RTL levels -> RTL buffer). Reshaping at line ends where HarfBuzz marks "unsafe to break" (arrange.rs:168) - matters for Arabic joining.
- **Break opportunities (breaks.rs):** UAX#14, jieba for zh, Korean between words, hyphenation points (`HYPHENATION_KEEPS=2` letters each side; only languages `hypher::Lang` supports - **no Arabic**).
- **Line breaking (lines.rs):** DP over (line, end, hyphen-run) minimising cost: line cost = `100*slack^2` (slack=(room-used)/room), +2 if break not after punctuation, +1000 for hyphen, +1e6*(1+overflow/room) if overflow. `RUNS=3` -> max 2 consecutive hyphenated lines. `Balance::Even` (centred / path text: last line also fills) vs `Ragged`. Path rooms try every line count in `fewest..=most`, cheapest wins.
- **Size fitting (fit.rs):** try size scale 1.0; if fits and frame "fills", `grow` by doubling to `CEILING=256` then 8 halvings; if overflow: shrink to floor 0.5 by probing in 1/8 steps then 8 halvings (whole words first); then halve balloon inset; then hyphenate; else overflow whole words. Default `line_height=1.2` (style.rs:73). Placement (translate/placement/mod.rs): initial size = **median source size of page text blocks**; centre-align for non-CJK; stroke width scaled by new/old size; SFX stroke = size/16, black if fill luminance>0.5 else white.
- **Alignment/RTL:** `rtl` from `unicode_bidi::get_base_direction` (fit.rs:393) for path edge choice; Start/End swap for RTL; justify supports only spacing (arrange.rs:330) - **no kashida** [R: grep "kashida" 0]. Frame vertical only if source vertical AND target ja/zh.
- Fonts: catalog from HF dataset `mayocream/fonts@304901a` (downloaded lazily); CSS-style nearest face matching.
- **Port advice [I]:** In Python use `uharfbuzz` + `python-bidi`/`fribidi` + PIL/`skia-python`/cairo; reimplement: (1) shaping per bidi/script run with fallback, (2) the DP line-breaker above, (3) binary-search size fit inside a balloon polygon (scanline spans per line -> centred on one common axis, see room.rs), (4) stroke outside glyphs. Arabic needs a font with Arabic coverage (catalog is Google-like; **[I]** coverage unverified).
- **Weaknesses [I]:** GPU (vello) dependency for draw; no Arabic-specific tuning (diacritics line height, tatweel justification, digits Eastern/Western); hyphenation absent for Arabic (fine); line height fixed at 1.2.

## 5. koharu-llm - translation backend (5.7k LOC)
- Local GGUF via llama.cpp binding (koharu-llama) + hosted (OpenAI-compatible, ChatGPT, Grok, LM Studio). Default model `koharu/index-translate-9b` (default 2B/9B/35B-A3B variants), target default `en`. Catalog in local/catalog.rs; also Gemma4, Qwen3.5, LFM3.
- Structured output through a JSON schema (`koharu_llm::answers(model, schema)`), llguidance grammar for local models [R: Cargo `llguidance`].
- Conversation context (context.rs): history kept per project; when over the window, summarised into 3 sections (Characters / Terms / Story); translation-only models cannot summarise so oldest messages are dropped. Reserve capped at 1/4 of model window.

## 6. Translation prompt + JSON format (port verbatim) [R: pipeline/stages/translate/mod.rs, turn.rs]
System prompt (target name from CLDR in English, e.g. "Arabic"; zh by script):
```
Translate the manga text in each message into {target}. Each message is a page's texts as a JSON object, numbered in reading order: translate only the values, and never alter the structure or the keys. Output only the translated JSON, without any explanation.
```
User instructions are appended after a blank line (cannot replace the template).
- **User message** = `{\n  "01": "text",\n  "02": "text"\n}` one line per text, keys `1..N` zero-padded to the width of N (so lexical sort = reading order). Blank source texts are skipped.
- **Schema**: `{"type":"object","properties":{k:{"type":"string"}},"required":keys,"additionalProperties":false}`.
- **Validation (`answers`)**: must be a JSON object with exactly those keys, each a non-blank string (trimmed); otherwise error (page fails, history not updated).
- **Echo guard**: if >=2 and >= half of the answers equal the previous page's answers while the source texts differ, **re-ask with empty history**.
- History keeps text turns only (images + reasoning removed). One conversation per project; changing model/target/prompt resets.
- Optional vision: page thumbnail longest side `SIGHT=1280`, JPEG base64, attached beside the JSON (only if model reports image input).
- Unchanged answer (text == source) is treated as already target-script (e.g. "!?"): translation=null, cleanup hidden, source lettering kept.
- **Weaknesses [I]:** no glossary field except free-form instructions; no speaker/context info per bubble; no retry loop beyond echo (failure = error); Index-Translate prompt is not Arabic-specific; per-page only (no cross-page batching but history carries it).

## 7. koharu-psd (1.4k LOC)
- `export(session, renderer, page, labels)` -> PSD bytes via `photocraft-psd` [R]. `MAX_SIDE=30000` (else PSB needed). Renders page once (merged image = renderer composite); each layer as pixel layer with mask; panels/balloons/blocks -> folders; balloon outline = hidden shape (`vmsk`); text -> **editable type layer** `TySh` + EngineData (ag-psd encoder, UTF-16BE strings) + Outside stroke via `lfx2/FrFX`; kinsoku sets for Japanese (engine_data.rs:342). `rtl` flag drives justification (engine_data.rs:226) and `WritingDirection` int (0 horizontal /2 vertical) - **Arabic type layers get justification flag only; no Arabic-specific EngineData [I]**.
- Port advice: in Python use `psd-tools` (read) / `pytoshop` / `ag-psd` via node; editable text layers are hard, so a safe fallback is rasterised text layers + hidden text-metadata layers [I].

## 8. Porting priorities (stage by stage) [I]
1. **detect**: swap in any detector with {text, sfx, bubble, panel} masks; port suppress/merge/disjoint, glyph_mask, XY-cut order, bubble holder (0.5 glyph share), rooms. Provide LTR order option.
2. **ocr**: crop with neighbour blanking; Japanese model of choice; **choose an Arabic-capable OCR separately** if sources are Arabic.
3. **translate**: prompt + numbered-JSON + schema + validation + echo guard + summary-on-overflow.
4. **inpaint**: tier ladder (flat -> plane -> Telea -> LaMa) with windows <=1MP clipped to panel, LaMa 800/128 rule.
5. **typeset**: HarfBuzz+bidi, DP breaker, balloon-room fit with 0.5 shrink floor.
6. **export**: PNG/PSD; for Arabic ensure RTL base direction per block (`get_base_direction`) and bidi reordering.

## 9. Not read / uncertain
- Model weight HF licences (not in repo); `Thresholds` recommended defaults live in remote config.json (not read); koharu-agent, canvas, desktop, runtime, config, diffusion/flux/qwen-image paths not studied; hosted LLM adapters only skimmed.
