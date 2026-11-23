# OverTranslate reference summary (for Python manga-translation port)

Root: `.../scratchpad/ref/OverTranslate` (C#/WPF/ONNX Runtime, Windows only). `S` = `src/OverTranslate/Services`.
All claims below are **[read]** (read the code) unless marked **[inferred]**. Pixel numbers are page pixels.

## 0. Licence / provenance (verified by grep)
- Repo code: MIT, "Copyright (c) 2026 Hon-Lu" (`LICENSE`). Algorithms may be re-implemented; keep attribution as provenance.
- `Imaging/Inpaint.cs`: port of OpenCV 4.13 `icvInpaint` (Apache-2.0 + Intel BSD notice). In Python just call `cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)`.
- CRAFT (`craft_mlt_25k`, clovaai CRAFT-pytorch, MIT) -> `tools/MangaModelExport/export_craft.py`.
- manga-ocr-base-CPT-flex weights (muscgab, rev 9cc1574...): **CC BY-NC-SA 4.0** (THIRD-PARTY-NOTICES.txt l.1434-1442; NOTICE: weights NC-SA, inference code Apache-2.0, training data animetext-ocr CC BY-NC-SA). ONNX conversions by Hon-Lu are redistributed under the same licence (hf `hon-lu/manga-ocr-base-CPT-flex-onnx-fp16`). Not bundled in repo: downloaded on request. => Private use only; record provenance, don't ship.
- Detector `hon-lu/comic-text-and-bubble-detector-onnx-fp16` (RT-DETR, 3 classes) - licence not checked here (check THIRD-PARTY-NOTICES before use). The column-pipeline OCR is PP-OCR (RapidOcr, det/cls/rec ONNX) - not covered in depth.

## 1. Model set `ocrmodels/manga-vertical.json` (version "2") [read]
Files (fp16 internally, fp32 I/O; sha256 in manifest): `detector.fp16.onnx` (85MB), `ocr-encoder.fp16.onnx` (172MB), `ocr-decoder-cross.fp16.onnx` (4.7MB), `ocr-decoder-step.fp16.onnx` (56MB), `vocab.txt` (27KB), `craft.fp16.onnx` (41MB). Runs under DirectML; in Python use onnxruntime CPU/CUDA - graphs themselves are generic.
Exports: `tools/MangaModelExport/{export_flex,export_kv,export_craft,export_fp16}.py`.

## 2. Page pipeline (orchestration) - `S/Ocr/Manga/MangaVerticalReader.ReadAsync`, `MangaOcrEngine.DetectUnderGate/ReadUnderGate`
```
rgb = page RGB (Pillow-like)
boxes = Detector(rgb)                       # section 3
luma  = L-image (Pillow weights 19595/38470/7471 >>16)
for box in text boxes:
    if IsLong(box, luma): -> column pipeline (PP-OCR vertical), not manga-ocr
    else: text.append(box)
for box in text:
    pieces[box] = CRAFT-cut(box) if Asks(box) else None         # section 5
readers[box] = ReaderFor(box, line-direction)   # Manga | ColumnsAcross | ColumnsDown
   ColumnsDown (width >= 2*height, lines run down) -> back to Manga if pieces!=null (cut already handles it)
manga-ocr reads: all Manga/ColumnsAcross boxes (or their pieces), batches of 4  # section 4
column pipeline reads: Long boxes + non-Manga readers (crop padded 6px)
ColumnsAcross: column pipeline result wins only if latin letters>=3 and latin*2>=non-space chars; else manga reading kept
Assemble(read, bubbles, passedOn, luma)  -> text blocks grouped per bubble
MangaColumnAngles.Apply -> tilt (section 7)
```
- IsLong: `w >= 7*h` or (`h >= 7*w` and luma.WritingSideBySide(box)) (`MangaPageLayout.cs:60`). RunsAcross: `w > 2.5*h`. LongBlockCrop pad=6.
- Bubble ownership (`MangaPageLayout.Assemble`): text box belongs to smallest bubble covering >=60% of its area (`BubbleCover=0.6`); columns of a group sharing >50% of the narrower's width = one band (`BandOverlap`). Union-find over shared bubble; Lines via `MangaTextRows.Find`.
- ReadAhead trick: when a wide block exists, read settled blocks (multiple of 4) concurrently with column pipeline. Pure perf.
- Pitfall: any GPU failure -> whole page falls to column pipeline until models change.

## 3. Text/bubble detector `MangaTextDetector.cs`
- Model: RT-DETR-style. Input `images` [1,3,640,640] float = RGB/255 (no mean/std) of page **squashed** (no aspect keep) to 640x640 via Pillow bilinear (section 9). Second input `orig_target_sizes` int64 [1,2] = (width, height) of the original page. Outputs `labels`(int64), `boxes`(x0,y0,x1,y1 in original px, float), `scores`.
- Filter: score >= 0.3; clip to page; drop side < 4px.
- Classes: 0 Bubble, 1 text_in_bubble, 2 text_outside_bubble. Text = 1&2, bubbles = 0.
- De-dupe text boxes: sort by score desc; drop a box if intersection > 0.7 * min(area_a, area_b) with a kept box.
- Warm-up: first run is slow (DirectML kernel build); irrelevant for ORT CPU.

## 4. OCR recogniser (CPT-flex, 3 ONNX graphs + KV cache) `MangaTextRecognizer.cs`, `MangaPatchInput.cs`
Architecture: ViT-style encoder (flex: variable aspect grid inside a fixed 196-patch budget) + 2-layer BERT decoder with cross-attention (L=2 layers, 12 heads, D=64, hidden 768; from export_kv.py [read]; head count inferred from bert-base config).

### 4a. Encoder input (per block) - `MangaPatchInput.Fill`
```
crop = RecognitionCrop(block)   # round() edges, clamp to page, >=1px
grey = Luma(crop)               # Pillow convert("L") integer weights
Grid(h,w): bisect scale in [1e-4, 1000] 60 iterations: largest s with ceil(h*s/16)*ceil(w*s/16) <= 196
  rows=ceil(h*s/16) cols=ceil(w*s/16)
sw=max(1,min(cols*16, round(w*s))); sh=max(1,min(rows*16, round(h*s)))   # python round (banker's)
small = PillowBilinearResize(grey, sw, sh)            # exact Pillow integer algorithm
fill  = median of border pixels (top row, bottom row, left col, right col; corners twice) as uint8 (mean of 2 middle values floor)
patch grid rows x cols, each 16x16, pixels from `small` at top-left, rest = fill
value = (px/255 - 0.5)/0.5 ; 3 identical channels, layout per patch [C,16,16] flattened (768)
patches[196,768] row-major patch order, unused rows zero
patch_mask[196] = 1 for first rows*cols
position_weights[196,196]: row r (patch i,j) = outer(Wdown[i,:], Wacross[j,:]) flattened over the 14x14 learned table;
  if rows==cols==14 -> identity
  W = torch bicubic (A=-0.75, align_corners=False, clamp edges) 14->size: x = (14/size)*(i+0.5)-0.5; taps floor-1..floor+2
```
Pitfall: author reports bit-exact match to Python preprocessing only because Pillow resize and torch bicubic are reproduced exactly; in Python you can use PIL `Image.resize(BILINEAR)` directly and `F.interpolate(mode="bicubic")` weights (get weights by interpolating identity matrix).

### 4b. Graphs
- `ocr-encoder`: inputs `patches` [B,196,768], `patch_mask` [B,196], `position_weights` [B,196,196] -> `last_hidden_state` [B,197,768] (class token + 196).
- `ocr-decoder-cross`: `encoder_hidden_states` -> `cross_k`, `cross_v` [L,B,H,197,D] (run once).
- `ocr-decoder-step`: `input_ids`[B,1] int64, `position_ids`[B,1] int64, `past_k`,`past_v` [L,B,H,T,D], `cross_k`,`cross_v`, `memory_mask` [B,197] -> `logits`[B,V], `present_k`,`present_v` [L,B,H,T+1,D].
- `memory_mask` = [1 (class token), patch_mask(196)]. Batch is lock-step (no self-attention mask), so finished rows keep feeding id 0.
- Batch cap = 4 (`BatchCap`); pad batch with all-zero patches (masked) and mark rows done.

### 4c. Decode loop (greedy)
```
ids = [START=2]*B ; positions = step ; past = empty tensors with T=0 (shape [L,B,H,0,D])
for step in range(MaxTokens=160):
    logits, present_k, present_v = step_graph(ids, step, past_k, past_v, cross_k, cross_v, memory_mask)
    past = present
    for each row not done:
        best = argmax(logits); certainty += 1/sum(exp(logit - max))   # = softmax prob of best
        if best == END(3): done
        else tokens.append(best); if EndsInLoop(tokens): done
    ids = best (0 for done rows); stop when all done
```
- MaxTokens=160 (longest non-runaway reading measured 134).
- `EndsInLoop` (`MangaDecodeGuard.cs`): for unit in 1..10: copies = max(3, ceil(30/unit)); span=unit*copies; if the last `span` tokens are periodic with period `unit` -> stop (ie ~30 tokens of repetition). Note the repeated tokens stay in the output; there is no trim [read].
- Confidence = sum(per-step top prob)/(n_tokens + 1 if ended by END). Multi-piece blocks: confidence = length-weighted mean.
- Decode: vocab.txt one token per line (BERT specials first: id0 [PAD], 2 [CLS]=start, 3 [SEP]=end; 6817 chars after specials [read from notice]); stop at id 0; skip tokens like `[...]`; strip `##`.
- PostProcess: remove all whitespace; "…"->"..."; runs of `[・.]{2,}` -> same number of "."; NFKC normalise.

## 5. CRAFT column cuts `MangaColumnDetector.cs`, `MangaColumnCuts.cs`
Purpose: CPT-flex misses whole columns past ~40 chars in multi-column blocks (measured 29% char error at 45-59 chars, 57% at >=60; cut pieces <=30 chars: 3.6%).
- Asked when `w >= 1.3*h` or `h >= 4*w`, among blocks that are not Long.
- CRAFT input: crop block (RecognitionCrop), scale = min(4, 384/max(cw,ch)); resize bilinear (Pillow) to (w,h)<=384; canvas 384x384 filled with white (255 normalised); ImageNet mean/std on 0-255 (123.675,116.28,103.53 / 58.395,57.12,57.375); input name `image` [1,3,384,384]; output [1,192,192,2] = (region, affinity). Map cell = `2/scale` crop px; used region = ceil(w/2) x ceil(h/2).
- `Pieces(maps, block, luma)` returns null (not this class's) or list of pieces (right-to-left, page coords; single [block] means read whole):
```
Vote: blobs(4-connected) of (region>0.35 or affinity>0.3), keep cells>=4 and >=10% of largest;
      v = area-weighted mean log(blob_h/blob_w); if v < -0.8 -> null (lines run across)
chars = blobs of region>0.35 with cells>=3  (boxes in block px)
columns: sort chars by centre x; join to last column if (cx - colMid) < 0.5*max(colW,charW) and cx < colRight+0.2*charW; col width = median char width
Leans: over columns with >=4 chars, slope=|cov(x,y)/var(y)| of char centres; median>0.07 (4 deg) -> null (leaning)
bases: ruby (col width < 0.6*percentile75(widths)) is folded into the base column to its left (RubyRight)
if len(bases)<2 or median(height/width of bases) < 2 -> null
est(col)=max(1,(bottom-top)/max(1,width)); if sum(est)<=40 -> [block]
runs: iterate columns right-to-left, start new run when count+est > 30 (a single huge column stays alone); 1 run -> [block]
cut between runs i,i+1: x in [runs[i].last.Left, runs[i+1].first.RubyRight], y range of both runs;
    choose x minimising dark-pixel count along the vertical line (Otsu-based ink; ties nearest the midpoint) [LeastInkLine]
piece y-range = column tops/bottoms +-4px; pieces read right to left, text concatenated in that order
```
- Measured: cutting at gap midpoint split a glyph; least-ink line gives 3.6% vs 8.2% for "refuse if ink".
- `LumaPage.Ink(block)` (threshold + dark-vs-light polarity by Otsu) is in `MangaPageLayout.cs` (not fully read).

## 6. Column (PP-OCR) pipeline helpers `S/Ocr/Vertical*`
Context: PP-OCR (RapidOcr, DB detector `det.onnx`, PP-OCRv6-det-tiny thresholds in `OnnxOcrEngine.ExportedThresholds`: box_thresh(binarise) 0.2, box_score_thresh 0.4, unclip 1.4) reading vertical columns in realtime.
### 6a. VerticalSecondLook (`VerticalSecondLook.cs`)
- Detector output varies non-monotonically with input size in 32px steps; reading twice at different sizes and keeping the better per balloon: 98/132 balloons whole -> 111-119.
- `OtherSize(w,h,primary)`: native=max(w,h); null if native<800 (`RealtimeDetectorSize.DownscaleMinSide`); other = floor(native*0.77/32)*32; null if |other-primary|<64.
- `Merge(first, second)`: union-find blocks across both reads whose bounds overlap > 30% of the smaller area (`InTheSamePlace`). Per place: if only one side has blocks keep them; else pick side by (1) longest block text (whitespace-stripped), (2) total chars, (3) fewer blocks.
- Not applied when manga models read the page (their 640x640 squash makes size irrelevant; measured identical).
### 6b. VerticalRepeatedColumns.Drop
For ordered pairs (i,j) not tilted: if >=80% of i's LayoutBounds area lies inside j's AND `SameWriting.SaysTheSame` (LCS length >= 0.6*shorter, whitespace stripped), drop the one with less text (fewer chars; tie -> lower confidence).
### 6c. VerticalRubyColumns.Separate (furigana)
- Reading of column W if: reading is kana-only (ぁ-ヿ, ー, ・), W holds kanji, glyph size(reading) <= 0.62*glyph size(W) [glyph = min(width, height/nchars)], reading's left is within +-0.5*W.width of W's right, vertical overlap >= 0.85*reading height. Nearest (rightmost) candidate wins.
- Certain readings are dropped and their bounds merged into the base; "might be" readings (same geometry, not kana-only) are kept out of the sentence and re-added later.

## 7. Tilted columns `MangaColumnAngles.cs`
1. For non-across blocks: ink angle by Otsu-selected ink point cloud (~500 jittered points, hash-jitter), project onto angles -20..20 deg (2.5 -> 0.5 -> 0.25 step coarse-to-fine) maximising sum of squared 1-D histogram counts; NaN if at edge of range. Candidate if |angle|>=3 deg.
2. Candidate crops (pad max(6,10% of short side)) packed into a mosaic (gap 12, grey 128 bg), area capped 22000 at scale 0.35 (down to 0.27, then drop least-leaning).
3. Run the PP-OCR line detector on the mosaic; line angle by `TiltedColumns.Angle(lines, trusted length 2.5)`. Accept only if |angle - ink angle| <= 2 deg. Group tilt = length-weighted mean; reject if spread > `TiltedColumns.MaxSpread`.
(Constants in `TiltedColumns/TiltedText` not read in full.)

## 8. Inpainting / text erase
### 8a. `Imaging/Inpaint.cs` [read]
Navier-Stokes (Telea-like FMM) exact port of OpenCV. Used only by `CpuAdaptiveBackground.cs` (lines ~1025, ~1319 with `FillRadius`, not read). **Python: `cv2.inpaint(..., INPAINT_NS)`.** Radius clamped to 1..100.
### 8b. `RealtimeNaturalBackground.cs` (rect-based erase; no mask needed)
```
ErasedArea(rect): hb=clamp(rect.h,12,72); padX=max(5,ceil(.16*hb)); padTop=max(8,ceil(.34*hb)); padBot=max(4,ceil(.16*hb))
  (top largest: dakuten/dots sit above glyph body)
GlyphBounds(line,glyphH): if 0<glyphH<line.h: top=y+(h-glyphH)/2; bottom=min(line.bottom, top+glyphH*1.30)
excluded mask = union of all ErasedArea of all source lines (so neighbouring text never donates colour)
for each erase rect (sorted by top,left): sample from the ORIGINAL pixels only
  band(side) = per ray, 3 nearest non-excluded pixels outward, channel-wise MEDIAN of 3; missing rays interpolated/extended; positions = mean donor coordinate
  if above and below bands exist: vertical lerp between them per column (t from actual donor y)
  elif left and right: horizontal lerp per row
  elif one of above/below -> extend; elif left/right -> extend
  else: mean of non-excluded border pixels (black if none)
```
Good for flat bubbles/gradients; does not reconstruct texture (use cv2.inpaint for screentone) [inferred].
- `SampleText`: text colour accepted only if |dR|+|dG|+|dB| >= 90 vs background.

## 9. Pillow-exact bilinear resize `PillowImage.cs`
Integer fixed-point (PRECISION_BITS=22), triangle filter with support = max(scale,1), horizontal pass first (rounded to uint8) then vertical, coefficients normalised then rounded half-away. In Python: `PIL.Image.resize(size, Image.BILINEAR)` (default reducing_gap=None) is identical. L conversion: `(R*19595+G*38470+B*7471+0x8000)>>16`. Pitfall: do NOT use cv2.resize (different antialiasing).

## 10. LLM translation `OverTranslate.Translation/OpenAi/*`
### 10a. Prompt (`OpenAiPrompt.cs`)
Default template (single user message, no system message; roles = only `user`; `BuildMessages`): 
```
{context}\n\n<Instruction>\n{text}            # batch adds "\n\n(Output each translation by its number, for example <1>translation</1>, one per line.)"
Single instr: "Translate the following text into {target_name}. Output only the translation, with no tags and no explanation:"
Batch  instr: "Translate the following text into {target_name}, English words and phrases included. Output only the translation, with no tags and no explanation:"
Style (<=200 chars, braces stripped, trailing punctuation trimmed) inserted before ". Output only": ". {style}. Output only"
Batch body: "<1>text1</1>\n<2>text2</2>..." (each text collapsed to one line)
```
(zh-Hant / zh-Hans variants exist; templates user-editable with `{context}`, `{text}`, `{target_name}` placeholders; placeholder newlines collapsed when context empty.)
- Context (history): `"Earlier translations, for reference while translating:\n" + join("\n\n", ["Source: <one-line src>\n<TargetLabel>: <one-line translation>"])`.
- Glossary section: `"Use these translations for the following names and terms:\n" + pairs "Source: x\n<Target>: y"` appended after history (inside {context}; if template lacks {context}, section is prepended).
### 10b. Glossary matcher (`OpenAiGlossary.cs`)
- Terms trimmed, empty dropped, later duplicate source wins, original order kept. Match longest-source-first on raw text, occurrences may not overlap an already taken span (so longer term suppresses its substrings). Terms containing Latin letters need whole-word boundaries (Latin/digit neighbours). Results ordered by first appearance (text index, position). Caps: 20 terms, 600 chars (source+target) per request. Only terms appearing in the batch are sent. Fingerprint = count:hash for cache keys.
### 10c. History (`S/Realtime/RealtimeTranslationHistory.cs`)
Pairs list; MaxPairs=16 -> when exceeded keep last 8; while token estimate > 2000 drop the older half. Token estimate: CJK/Hangul/fullwidth chars count 1 each, others 1/4. Add only if `stillCurrent()` (stale-session guard).
### 10d. Batching / answer parsing (`OpenAiChatTranslator.cs`, `OpenAiAnswer.cs`)
- Batch: as many texts as fit (limits: `MaxBatchTexts=40`, char budget `BatchSize` ~ message length + 32 + 2*chars of texts, with carried context); answers parsed by number `<(\d+)>(.*?)(?=</\1>|<\d+>|\z)` (first non-empty wins, out-of-range ignored). Only missing numbers are re-sent, each alone as a batch of one (with prior translations as context, <=MaxConcurrentRequests). A batch failing for text-related reasons is re-sent as single batches. One retry on empty/unthinkable answers.
- `StripThinking`: remove `<think...>...</think>`, leading text up to a lone `</think>`, and an unclosed `<think>` tail.
- Single-answer clean-up (default template only): strip ``` fences; unwrap `<t>..</t>`; take last `<t id=1>` block; strip matching outer quotes (" “” 「」 『』) unless the source contained them.
- Optional `temperature`, `max_tokens` in payload; OpenAI-compatible `/chat/completions`.

## 11. Not read / caveats
- `MangaLineGeometry.Directions` (line direction via PP-OCR line detector), `MangaTextRows.Find`, `LumaPage.*` (Ink, WritingSideBySide, SamePaper, rule detection), `VerticalColumnGrouping` (779 lines), `OcrTextBlockGrouper`, `CpuAdaptiveBackground`, `TiltedText/TiltedColumns` constants, SettingsService, realtime scroll logic: not analysed.
- The whole codebase assumes Windows + DirectML; ONNX graphs are portable but fp16 internals with fp32 IO - on CPU ORT they run (fp16 kernels are slow on CPU; consider converting to fp32 if licensing/private use allows) [inferred].
- Porting order of value: (1) PatchInput + 3-graph decode, (2) detector + dedupe, (3) CRAFT cuts, (4) prompt/glossary/history, (5) background rect fill.
