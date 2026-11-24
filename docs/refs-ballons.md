# BallonsTranslator reference summary (for a Python manga-translation core)

Source: `.../scratchpad/ref/BallonsTranslator` @ b59f0031df85e5144bea13e7a6879a0bdbbc7699. Licence GPL-3.0 (record as provenance; private non-commercial reuse).
Package root: `ballontranslator/` (abbrev `BT/`). Tags: **[R]** = read code, **[I]** = inferred / not verified by running.
Nothing was executed; no weights were downloaded. All pseudo-code below is paraphrased from the code read.

## 0. Pipeline shape [R]
detect (mask + TextBlocks with `lines` quads) -> OCR per block (crop per line, warp) -> translate (per page, list of strings) -> inpaint (mask from detector, per-block or full) -> text layout (fit translated words into balloon mask) -> render (Qt only; not portable).
Core domain object: `BT/utils/textblock.py::TextBlock` (xyxy, lines [N,4,2] quads, vertical, src_is_vertical, angle, font_size, text, translation, vec/norm/distance, _detected_font_size, fg/bg colors). Qt is only needed for rendering/`bubble_shape.py` (imports qtpy) and the editing UI; detector/OCR/inpaint/translators contract modules are Qt-free [R].

## 1. Comic Text Detector (CTD)
Files: `BT/modules/textdetector/detector_ctd.py` (wrapper), `ctd/inference.py` (TextDetector class, `det_rearrange_forward`, pre/post), `ctd/basemodel.py` (nets), `ctd/textmask.py` (mask refinement), `db_utils.py::SegDetectorRepresenter` (DBNet decode), `utils/textblock.py::group_output`.

### 1.1 Weights [R]
- Torch: `https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt` sha256 `1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb` -> `data/models/comictextdetector.pt`. Dict ckpt with keys `blk_det` (YOLOv5 ckpt), `text_seg` (UnetHead state), `text_det` (DBHead state); activation `leaky`.
- ONNX: `.../comictextdetector.pt.onnx` sha256 `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`. Used when device == cpu; loaded with `cv2.dnn.readNetFromONNX` (no onnxruntime). **Fixed 1024x1024 input** (code forces detect_size=1024 for the opencv backend).
- Wrapper defaults: detect_size 1280 (options 896/1024/1152/1280), rearrange max batches 4, mask dilate size 3, device auto.

### 1.2 Network I/O [R]
- Input: float32 NCHW, scale 1/255, no mean/std, N=1 (square `S x S`, S = detect_size).
- 3 outputs from `net.forward(uoln)`: `blks` (YOLO decoded rows `[cx,cy,w,h,obj,cls...]`, shape [1, M, 7] [I]), `mask` [1,1,S,S] (sigmoid 0..1 text-pixel prob), `lines_map` [1,2,S,S] (DBNet: ch0 = shrink map prob, ch1 = threshold map). With some OpenCV versions the output order of `mask` and `lines_map` is swapped; code detects it by `mask.shape[1] == 2` and swaps. Keep that check.
- Torch path: `TextDetBase.forward` = YOLO backbone(detect=True) -> feature list -> `UnetHead(forward_mode=INFERENCE)` gives mask + feats -> `DBHead(step_eval=False)` returns cat(shrink, thresh) [1,2,S,S].

### 1.3 Preprocess (single-shot path) [R]
```
letterbox(img, new_shape=(S,S), color=0, auto=False, stride=64):
    r = min(S/h, S/w)                      # scaleup allowed
    new_unpad = (round(w*r), round(h*r))
    dw, dh = S-new_unpad[0], S-new_unpad[1]   # padding ONLY bottom/right (top=left=0)
    resize(INTER_LINEAR); copyMakeBorder(0,dh,0,dw, constant 0)
torch: chw = hwc.transpose(2,0,1)[::-1] (channel flip!), /255
opencv: blobFromImage(letterboxed, 1/255, (1024,1024)) with swapRB=False (no flip)
```
Pitfall [R/I]: torch path flips channels, opencv path does not; BT feeds RGB images (PIL imread). The original CTD repo fed BGR to both. Feed BGR (cv2.imread order) to the ONNX to match the training convention [I]; the difference is probably small but untested.

### 1.4 Postprocess [R]
```
resize_ratio = (im_w/(S-dw), im_h/(S-dh))
blks = NMS(yolo, conf 0.4, iou 0.35) -> boxes*ratio, cls, conf     # NOTE: computed but NOT USED, see below
mask  = mask[:S-dh, :S-dw]  (crop padding);  lines_map = lines_map[..., :S-dh, :S-dw]
mask  = uint8(mask*255); then cv2.resize to (im_w, im_h) INTER_LINEAR
lines, scores = SegDetectorRepresenter(thresh=0.3, unclip=1.5)(lines_map, height=im_h, width=im_w)
keep lines with score > 0.6                      # box_thresh in inference.py
blk_list = group_output([], lines, im_w, im_h, mask, canvas=img)   # YOLO blocks passed as [] => ignored
mask_refined = refine_mask(img, mask, blk_list, REFINEMASK_INPAINT)
```
DBNet decode `boxes_from_bitmap` [R]: `bitmap = shrink>0.3`; `findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)`, up to 1000 candidates; per contour: `minAreaRect` -> 4 pts (sorted tl,tr,br,bl via `get_mini_boxes`), skip if short side < 2; score = mean of shrink map inside the contour bbox/mask (`box_score_fast`); `unclip`: `distance = area*1.5/perimeter` (shapely Polygon), pyclipper round offset; re-`minAreaRect`; scale coords to (dest_w/map_w, dest_h/map_h), clip. Needs `shapely` + `pyclipper`. Note zero-size contours leave rows of zeros in the array (filtered later by score).
Pitfall [R]: `blks` (YOLO) is dropped in this version; blocks come only from DB text lines grouped by `group_output`. Lines scoring below 0.6 never appear; undetected-but-masked text can be recovered only via `refine_undetected_mask` (not used by the default `_detect`).

### 1.5 Rearrange for tall/wide pages (`det_rearrange_forward`) [R]
```
h,w = img shape; if h < w: transpose (work with tall layout)
asp = h/w ; down = h/tgt_size
if not (down > 2.5 and asp > 3): return None  -> use single-shot path
pw_num = max(floor(2*tgt/w), 2); patch = pw_num*w           # square patches w x (pw_num*w)
ph_num = ceil(h/patch); step = (h-patch)/(ph_num-1)         # overlapping vertical crops
pad patch list to multiple of pw_num with zeros
group pw_num patches side-by-side -> (patch_h x pw_num*w) images -> square_pad_resize(tgt) -> batches of <=max_batch
run dbnet+mask on batch (torch: NHWC/255 -> NCHW; opencv: one image at a time)
un-rearrange: split columns back, place at rel_step*H offsets, average overlap (/2 for interleave rows)
```
Output is (db [1,2,h',w'], mask [1,1,h',w']) at a reduced resolution, then resized exactly as above. Needs `einops`. Pitfall: overlaps averaged only once (`/2`), >2 overlapping patches would be wrong [I]. For webtoon-style strips use this; for normal pages it is skipped.

### 1.6 `group_output` (lines -> TextBlocks) [R] (`utils/textblock.py:947`, helper `utils/textlines_merge.py::sort_pnts`)
```
for each DB quad: pts, is_vertical = sort_pnts(quad)    # orientation: long-side vectors; vertical if |vx|*1.2 <= |vy|; square => horizontal
   (YOLO blocks would be matched with >0.4 overlap; unused here)
   if mask[box].mean()/255 < 0.1: drop (needs "enough" text mask)  # mask_score_thresh
   blk = TextBlock(bbox,[quad]); examine_textblk(blk) -> fills vec, norm, angle, font_size, distance
        font_size = round(norm_h / n_lines)  (vertical) or round(norm_v / n_lines)
        angle = atan2(primary_vec) deg; vertical: angle -= 90; |angle|<3 -> 0
        distance = perpendicular distance of line centres to origin ((imw,0) for vertical, (0,0) for horizontal) for reading order
scattered['ver'/'hor'].append(blk)
sort scattered: ver by centre.x descending (right-to-left); hor by centre.y ascending
merge_textlines(hor, fntsize_tol=2.0); merge_textlines(ver)   # greedy: try_merge_textline(blk_i, blk_j>i)
sort_regions(blocks)  # manga reading order: by centre y; ties inside same y-band by x (right-to-left if any vertical)
drop block if >90% of its area lies inside an already-kept block
blk._detected_font_size = blk.font_size
```
`try_merge_textline` (merge criteria) [R]: if the last line of A and first line of B polygons do not intersect:
- vertical: need `distance_y <= 0` (overlap in y), `distance_x <= 0.8*avg_font`, `|distance_y|/min(h) >= 0.4`;
- horizontal: need `distance_x <= 0` (overlap in x), `distance_y <= avg_font*0.5` (0.6 if avg_font<24), `|distance_x|/min(w) >= 0.3`;
- font size ratio <= fntsize_tol (1.7 default, 2.0 horizontal call), `|cos(angle between vec)| >= 0.866` (30 deg).
Merge = append lines, vec sum, avg font size weighted by line count, angle recompute.
Pitfall: `sort_regions` is O(n^2) insertion sort heuristic, bubble-order not panel-aware (panel_finder.py is the panel-aware alternative, see 1.8). `right_to_left = nv/nr > 0` i.e. true if ANY block is vertical [R].

### 1.7 Mask refinement `ctd/textmask.py::refine_mask` [R]
Converts the soft CTD mask into a tight, inpaint-ready binary mask per block.
```
for blk in blk_list:
    (x1,y1,x2,y2) = enlarge_window(blk.xyxy, W, H, ratio=2.5)   # solve quadratic so area grows 2.5x, clamped to image
    im, msk = crop(img), crop(pred_mask)
    masks  = get_topk_masklist(im, msk)       # grey; histogram of pixels under eroded(3x3) msk>127 (255 bins);
                                              # top-3 colours (distinct by >10), each: inRange(grey, c-30, c+30)
          + get_otsuthresh_masklist(im, msk)  # per channel Otsu; keep the channel with min XOR vs msk
    each candidate passes minxor_thresh: choose threshed vs inverted, whichever XORs less with msk
    merge_mask_list(masks sorted by xor):
        pred = erode(msk, ellipse 3x3); pred = (pred>60)*255
        for cand: connectedComponentsWithStats(8-conn); for each component with w*h>=3:
              OR it into merged only if (sum pred over newly covered px)*2 > 255*count   # majority of its px are predicted text
        if INPAINT mode: dilate merged with 5x5
        fill holes: components of ~merged (4-conn... 8) smaller than 2nd-largest area, accept by same predicate
    mask_refined[crop] |= merged
```
Wrapper `detector_ctd.py::_detect`: `TextDetector.__call__` returns `(mask, mask_refined, blks)` but wrapper does `_, mask, blk_list = self.model(img)` so it keeps `mask_refined`, then `cv2.dilate` with an ellipse of `(2k+1)` kernel, k=3 [R].
`refine_undetected_mask` (used only with `keep_undetected_mask=True`, annotation tooling): zero pred where refined>30, 4-conn components with area>50 whose union with any block bbox <50% -> new TextBlock -> refine again [R].

### 1.8 `panel_finder.py` [R]
Standalone (OpenCV + PIL + shapely), not used by the default pipeline (only `__main__` / manual reorder).
```
panel_process_image: gray -> GaussianBlur(7x7) -> threshold(>200 => white) -> pad 10px WHITE -> invert
contours = findContours(RETR_EXTERNAL) ; keep area > 10000; convexHull; drop contours whose bbox is inside another
order_panels: sort bboxes by y; group by any vertical overlap; within group score = -y + x, sort DESC (=> right-to-left, then top-to-bottom)
reorder_boxes_indices: assign each text box to a panel by nearest-distance (shapely) then sort by same score
```
Pitfalls [R]: (a) `closest_text_to_panel_index` never records the argmin panel: it appends the *last iterated* `p_index` unless it breaks on distance 0, so text outside every panel always goes to the last panel (bug). (b) groups are y-overlap clusters, so tall panels spanning rows merge groups (it is an approximation). (c) hard-coded RTL. (d) `draw_bboxes` needs `arial.ttf`.

### 1.9 RECIPE: run CTD standalone, get mask + boxes [R for steps, I for untested glue]
```python
# pip install numpy opencv-python shapely pyclipper  (+ torch/einops only for .pt)
import cv2, numpy as np, pyclipper
from shapely.geometry import Polygon
# weights: curl -L -o comictextdetector.pt.onnx https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt.onnx
S = 1024
net = cv2.dnn.readNetFromONNX('comictextdetector.pt.onnx'); outs = net.getUnconnectedOutLayersNames()

def run(bgr):                                     # bgr: HxWx3 uint8 (cv2.imread)
    h, w = bgr.shape[:2]
    r = min(S/h, S/w); nw, nh = round(w*r), round(h*r)
    dw, dh = S-nw, S-nh
    im = cv2.copyMakeBorder(cv2.resize(bgr, (nw, nh)), 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=0)
    net.setInput(cv2.dnn.blobFromImage(im, 1/255., (S, S)))
    blks, mask, lines = net.forward(outs)
    if mask.shape[1] == 2: mask, lines = lines, mask        # OpenCV output-order quirk
    mask = mask.squeeze()[:S-dh, :S-dw]                     # crop padding
    lines = lines[0, 0, :S-dh, :S-dw]                       # shrink map only
    mask = cv2.resize((mask*255).astype(np.uint8), (w, h))  # text-pixel prob 0..255
    # DB decode: bitmap = lines>0.3 -> contours -> minAreaRect -> score(mean shrink) -> unclip 1.5 -> rescale
    ...  # port SegDetectorRepresenter.boxes_from_bitmap (db_utils.py:127-216) with dest=(w,h), keep score>0.6
    return mask, quads
```
Then: `group_output`-style grouping (port `sort_pnts`, `examine_textblk`, `try_merge_textline`, `sort_regions`), `refine_mask` for the inpaint mask, `cv2.dilate(mask, ellipse(7x7))`.
Key checks: output tensor order (swap check), padding is bottom/right only, DB map is [0]=shrink [1]=thresh, coordinate scale uses map dims not S, mask threshold for "is text" is soft (use >60 after erode as BT does). For pages with h/w > 3 and h > 2.5*S use the rearrange path (1.5) or tile yourself. Torch alternative: `TextBlock`-free port of `TextDetBase` needs the `yolov5` package files under `modules/textdetector/yolov5/` (`load_yolov5_ckpt`) - heavier; ONNX+cv2.dnn is the simplest [I].
Pitfall: `cv2.dnn` ONNX on very new OpenCV may fail on some layers (the app falls back to torch on GPU); onnxruntime can load the same file [I, not tested].

## 2. Other detectors (brief) [R]
- `detector_koharu.py` (RF-DETR, `mayocream/koharu-layout-highres` safetensors+config.json, sha in code): bilinear resize to detect_size (default 1152), ImageNet normalize, thresholds text 0.25 / sfx 0.3, gives instance masks; `_mask_angle` picks angle in [-90,90] step 0.5 by maximising projection-histogram energy (sum h^2/N) along two axes; vertical if scores differ >2%. Uses `partition_bubble` + `trace_bubble` to build `BubbleShape`.
- `detector_ysg*.py`, `detector_stariver.py` (online API), `detector_ysg_ctd.py` (YOLO + CTD hybrid). Not studied in depth.

## 3. `utils/textblock_mask.py` (per-block text/balloon mask + colours) [R]
Used by `TextBlock` for colour estimation and by the draw panel; NOT by default inpainting.
- `canny_flood(img)`: scale 0.6 if both dims >300, 1.4 if any <120; Canny(70,140,L2) on Gaussian(3x3); border rectangle 1px; contours (RETR_CCOMP); among contours with bbox area >= 40% of crop, flood-fill from centre (loDiff=upDiff=10) to find the smallest region with area > 30% => balloon mask; dilate 3x3; iteratively `bitwise_and(edges, balloon)`, flood from two corners to separate text-edge area, stop when text/outer ratio < 0.85 (erode balloon up to 2x). Then text mask = edges inside balloon; `bground_calculator` gives mean bg colour + grey variance `sd`; `need_inpaint = sd >= 10` (`inpaint_sdthresh`); text colour from Otsu on grey (inverted if bg dark). Returns `(mask, ballon_mask, {rgb,bground_rgb,inner_rect,need_inpaint})`.
- `connected_canny_flood`: pick best of 4 channels (R,G,B,grey) by counting plausible character components (w,h < 0.5*largest, bbox area > 0.1% of crop) after Otsu (invert if mean over outer region >160); text = connected components not the largest and area < 40% crop, & balloon mask; colour = mean of unsharp-masked pixels under eroded mask.
- `extract_ballon_mask(img, text_mask)` (used by inpaint base "check_need_inpaint"): Gaussian 3x3, Canny(70,140), dilate 3x3, draw crop border, remove text mask, `findContours(RETR_CCOMP)`; the **smallest-area contour whose bbox contains the text-mask bbox and whose filled area covers all mask px** = balloon. Returns `(ballon_mask, non_text_mask)`.
- Pitfall: all are heuristic on white balloons; fail on textured/transparent/multi-balloon crops (returns None / full rect).

## 4. `utils/bubble_shape.py` [R]
Qt-dependent (QPainterPath) editable closed-cubic "bubble shape" (frame xywh + 3..64 nodes (x,y,inx,iny,outx,outy) normalised). `trace_bubble(mask, page_size, angle, text_mask)`: take contour owning the text centroid; morphological open with ellipse r=max(1,round(min(w,h)*.06)) (accept if keeps >=80% area); try minAreaRect (if area/rect >=0.9 and in bounds, overlap tests >=96%/84%), else fitEllipse with shrink 1,.97,.94, else piecewise cubic least-squares fit (tolerance max(.8, .035*min(w,h)), <=12 segs, 4 initial cuts, split worst, merge), else approxPolyDP corners (<=64); then expand to cover text mask. `partition_bubble` splits a shared mask between several text anchors by neck cells / nearest-site (Voronoi) fallback. Good idea to borrow: the shape concept; the implementation is Qt-bound so would need re-implementation with OpenCV/shapely [I].

## 5. Inpainting `BT/modules/inpaint/`
### 5.1 Driver `base.py::InpainterBase.inpaint` [R]
```
if cfg.filter_mask_by_bboxes: mask &= dilate(5x5)(union of rotated block rects)
if inpaint_by_block and blocks given:
    for blk: xyxy_e = enlarge_window(blk.xyxy, W, H, ratio=1.7); crop img/mask
        if check_need_inpaint: (ballon, nontext) = extract_ballon_mask(crop, mask_crop)
             bg = median(nontext px); std_rgb = std(nontext - bg); thresh = 7 if std(std_rgb)>1 else 10
             if max(std_rgb) < thresh: fill ballon with bg colour (skip model)   # flat white bubble shortcut
        else crop = model(crop, mask_crop)
        mask[xyxy] = 0   # so later overlapping blocks don't redo it
else: whole-image model
RGBA: inpaint RGB only; alpha: if median alpha around mask <128, set alpha in mask to that value
OOM: retry after empty_cache, then fall back to CPU
```
### 5.2 AOT (`aot.py`, registered in `inpaint_default.py::AOTInpainter`) [R]
Weights: `https://huggingface.co/dreMaz/mit_models/resolve/main/aot_inpainter.ckpt` sha256 `878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f` -> `data/models/aot_inpainter.ckpt` (`torch.load`, key `model` or plain state dict; `AOTGenerator(in_ch=4,out_ch=3,ch=32)`, 10 AOT blocks, rates default 1,2,4,8 [I]; torch only).
```
mask01 = (mask>=127); img/mask = resize_keepasp to inpaint_size (default 2048) only if max side > size
pad bottom/right with BORDER_REFLECT up to >=128 if side < 128
img_t = img/127.5-1 (NCHW RGB), mask_t = mask/255 binarised at .5; img_t *= (1-mask_t)
out = model(img_t, mask_t)  (clip -1..1) -> (out+1)*127.5 -> crop padding -> resize back
result = out*mask01 + original*(1-mask01)        # paste only masked pixels
```
### 5.3 LaMa-MPE (`lama.py::LamaFourier`, wrapper `LamaInpainterMPE`) [R]
Weights: `.../dreMaz/mit_models/resolve/main/lama_mpe.ckpt` sha256 `d625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc`; ckpt keys `gen_state_dict` (FFC-ResNet generator in=4,out=3, 9 blocks, ratio_g 0.75, sigmoid out) and `str_state_dict` (MPE module: masked positional encoding). `lama_large_512px`: `https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt` sha `11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935`, 18 blocks, no MPE, default size 1536, bf16 on CUDA.
```
resize_keepasp to inpaint_size if larger; PAD (not resample) bottom/right with BORDER_REFLECT to multiple of 64 (square: longer side)
img/255 (RGB, 0..1), mask binarised .5; img *= (1-mask)
load_masked_position_encoding(mask): at 256x256, iterate dilation (3x3 filter2D) from known px inward:
    pos[i]=ring index i; direct[...,4] = 4 directional-neighbour flags (d_filter1..4);  rel_pos = clip(pos/128*128, 0..127)
    (nearest-resize back to full-res, zero outside mask)    # CPU loop: cost ~ O(max ring count)
out = gen(cat(mask,img)) * mask + (1-mask) * img ; *255 ; crop pads; resize back; paste with mask01
```
Pitfalls: inference code needs the FFC modules in `ffc.py` (Fourier conv, `torch.fft.rfftn`; fp16 problematic, CPU ok); MPE input dtype LongTensor; masks must be >=127-thresholded or ring loop can be huge/infinite (guarded by `mask3.max()>0`) [R].
### 5.4 PatchMatch / OpenCV [R]
- `opencv-tela`: `cv2.inpaint(img, mask, 3, INPAINT_NS)` (name is misleading, it is Navier-Stokes).
- `patchmatch`: ctypes into prebuilt `data/libs/libpatchmatch.so` (Linux must be user-supplied; Win/mac prebuilt downloads from `github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/{windows,macos_arm64}_patchmatch_libs.7z`). Call `PM_inpaint(image HxWx3 uint8, mask HxWx1 uint8 non-zero=hole, patch_size=3)`; returns same-size image. Mask must be uint8 [H,W,1] contiguous. Default patch_size in lib API is 15; BT uses 3. Non-portable: C lib lives in a separate repo [R]. 
- Also `flux2-klein` (diffusers GGUF), `inpaint_llm`, `lama` variants - generative, heavy, skipped.

## 6. OCR `BT/modules/ocr/`
Base (`base.py::OCRBase.run_ocr`): RGBA->RGB; for each block `ocr_img(crop of blk.xyxy)` OR (line-only models) per-line warped regions; text post-processing (keyword substitution, letter case). Result text stored in `blk.text` as list of strings/lines [R].
- `manga_ocr` (kha-white): HF `VisionEncoderDecoderModel` from `https://huggingface.co/kha-white/manga-ocr-base/resolve/main/` (`pytorch_model.bin` sha `c63e0bb5...6c2eb`, config, preprocessor_config, vocab.txt...). Input: the block's xyxy crop (RGB ndarray) -> `AutoImageProcessor` (224x224, normalised) -> `generate`; post: strip whitespace, `…`->`...`, collapse `[・.]{2,}`, `jaconv.h2z(ascii=True,digit=True)` (full-width) [R]. Best for Japanese vertical single bubbles; one crop per block, so multi-column blocks rely on the model reading order [I].
- `mit48px_ctc` (manga-image-translator 48px CTC): weights `https://huggingface.co/dreMaz/mit_models/resolve/main/ocr-ctc.zip` (ckpt sha `8b0837a2...78ed89`, `alphabet-all-v5.txt`); also `mit48px` (autoregressive; `ocr48px.zip`, alphabet-all-v7). Procedure [R]:
```
for each line quad of each block: blk.get_transformed_region(img, idx, text_height=48, maxwidth=8100)
    horizontal: perspective-warp quad -> (h=48, w=round(48/ratio)); vertical: warp -> w=48,h=..., rotate 90 deg CCW
    (for ctd horizontal blocks the quad is first expanded by max(0.1*font,3)px because CTD boxes are tight)
batch chunk_size=16 lines: pad width to 4*((max_w+7)//4)+128 with zeros; (x-127.5)/127.5 NCHW
net.decode(images, widths, 0) -> per char (id, logprob, fg rgb, bg rgb); drop line if exp(mean logprob) < 0.3
text = join; '<SP>' -> ' '; colours averaged -> blk.update_font_colors (fg/bg estimation is a free by-product)
```
Needs `torch`, `einops`; `OCR` class in `mit48px_ctc.py` (transformer encoder 768 dim + CTC; `del sd[...pe.pe]` buffers, `strict=False`). Pitfall: `textblk_lst_indices[i+chunk_idx]` assumes all lines of a chunk decode non-empty alignment [R].
- `ocr_ppv6_onnx` (PaddleOCR PP-OCRv6 medium rec via onnxruntime): `https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx` (sha `9c09abf0...b673ba`) + `inference.yml` (character dict). Line crops via `collect_textblock_regions`; heavy CUDA-lib preload code unrelated to the model [R].
- Others: `ocr_paddleVL_manga` (HF `jzhang533/PaddleOCR-VL-For-Manga`), `ocr_llm`/`llm_vision` (send crop/page to VLM), `ocr_google_vision`, `ocr_lens_proto` (Google Lens protobuf), `ocr_screen_ai`, `ocr_macos`, `ocr_windows`, `ocr_stariver` (online), `ocr_hayai_supernova`. Not studied in depth.
- Pitfall: MIT/ppocr OCR operate on lines: if your detector returns blocks only, you must supply quads or fall back to block crops (`collect_textblock_regions(..., split_textblk=True)` splits single-line-blocks using `canny_flood`+`split_text_region`).

## 7. Text-block ordering and grouping summary
Covered in 1.6 (line->block merge, reading order) and 1.8 (panel-aware ordering). Extra [R]:
- `TextBlock.sort_lines()` orders lines by `distance` (perp. distance to origin, vertical manga origin at top-right).
- `split_textblk` (distance-gap splitting) exists but is **commented out** in `group_output`; Japanese multi-line blocks are therefore kept whole.
- Block-level dedupe: drop any block >90% inside a kept earlier block (by the sorted order).
- `examine_textblk` constants: angle snap 3 deg, `font_size = line thickness`.

## 8. Layout of translated text (`utils/text_layout.py` + caller `ui/text_engine/editing/manager.py` ~L925-1045) [R]
Qt is used only for font metrics (`QFontMetricsF`) and applying results. The algorithm is portable if you supply a width-measuring function.
```
mask = extract_ballon_region(img, blk_rect, enlarge_ratio)  # imgproc_utils.py:265, canny+flood, like canny_flood
   enlarge_ratio = min(max(w/h, h/w)*1.5, 2.5 if tgt_cjk else 3);  enlarge_window(rect, ratio, aspect=h/w)
words, delimiter = seg_text(text, target_lang)     # text_processing.py: CJK char-wise / eng word-wise (+ hyphenation helpers)
wl_list = widths(words); line_height = round(line_spacing * height('X木' if cjk else 'X')); delimiter_len
auto font shrink:
   horizontal src non-cjk: ratio = max(balloon_area/1.2/text_area, 0.7); also <= src_width/total_len if source lines valid; clamp [0.6,1]
   cjk src: only if source lines valid: ratio = max(src_width/(sum wl) *1.5, 0.5); clamp [0.6,1]
   vertical->horizontal: ratio = clip(min(area_ratio/1.7, region_w/max(wl)), 0.6, 1)
layout_text -> layout_lines_aligncenter (alignment centre) or _alignside:
   central line seeded around centroid word (cumulative-width median);
   grow left/right word-by-word while the line's two end x positions hit mask rows with mean >220 (inside bubble), check width vs max (line_is_valid: new_len < max_w or old/max_w < max_w/new_len);
   then expand rows up and down the same way (line_height steps); optional start_from_top when 2-3 source lines and first line is long;
   rotate mask by blk.angle first if rotated
returns (text with \n, xywh canvas, start_from_top, adjust_xy); post-resize for vertical case: ratio clip(max(region_w/w, .5), 0, 1)
```
Constants: border_thr 220 (mask is 255 inside balloon), balloon_area_thresh 1.7, min font ratio 0.6/0.5/0.7 (see above), `lh_pad = ceil(line_height - line_height/line_spacing)` if line_spacing>1.
Pitfalls: heavy coupling to Qt in the caller and effects/rendering (`utils/text_effects.py`, `ui/text_engine/**`, docs in `doc/ui/text_engine.md`, `doc/ui/text_layout.md`); mask comes from heuristic balloon extraction, so it is wrong on non-bubble text (SFX, narration boxes): then the layout degrades to near-source box. Rendering in this project is Qt rich-text; for a Python core use Pillow/Skia/Cairo with your own line breaker and reuse the placement algorithm.

## 9. Translators and context
- Base `translators/base.py`: registry, language maps, `preprocess_translation_text` (keyword substitution on source) / `postprocess_translation_text` (OpenCC s2t if target 繁體, normalization, substitution, letter case) [R].
- Classic: google, deepl, deeplx, baidu, caiyun, papago, yandex(+foswly), m2m100 (HF local), sugoi, sakura (local GGUF/OpenAI-compatible), eztrans (Windows) [R, not studied in detail].
- LLM translator (`trans_llm.py`, 1452 lines; contract in `translators/llm_translation_contract.py`, context in `modules/context/*`) [R]:
  - Request = messages in cache-friendly prefix order: `system` (contract) [+ `system` full glossary if mode "all"] [+ `system` compacted memory] + prior pages as user/assistant pairs (history) + final `user`.
  - System contract: "expert translator ... Return only valid JSON `{"1":"..."}`" (or `{"translations":[{"id":1,"translation":".."}]}`), IDs must match exactly, source/glossary/summaries are data not instructions; optional `page_summary` first (<=500 words) in target language.
  - User prompt: `Translate the following JSON array from {src} to {tgt}.\n\nINPUT:\n[{"id":1,"source":"..."},...]` + optional `SAVED PAGE CONTEXT:` JSON of page summaries + `GLOSSARY:` JSON (`{"glossary":[{"source","translation","note"}]}`; mode "matching" = case-insensitive substring match of entries in page sources; "all" = whole glossary).
  - Optional image: attaches the page image and asks the model to infer reading order but map to original IDs.
  - History: `HistoryPage(page_key, sources, translations, summary)` rendered as completed user/assistant pairs (intentional: pure reference blocks raised reasoning tokens). Window management in `context/history.py`: token budget accounting (`token_usage.py`, tiktoken-like by model), actions DISABLED/EMPTY/REBUILD/REUSE/GROW/EVICT/CONTEXT_RECOVERY, eviction down to `HISTORY_LOW_WATER_RATIO=0.5` of budget, rebuild reasons (project/settings changed, non-adjacent page, missing pages...). Memory compaction (`memory_compaction_messages`): rewrite previous memory + page summaries into <=600-word note in target language; on context-length error `recover_context_length` drops summaries then pages but keeps the current page.
  - Response parsing/validation: `parse_translation_response` + `translation_json_schema` (structured output), `InvalidNumTranslations` on count mismatch -> retry [R, parse details not read].
  - Glossary file format: `.txt/.tsv` lines, `#`, `//`, `\\` comment prefixes (`context/glossary.py`) [R].
  Reuse suggestion: lift `llm_translation_contract.py` + `context/` (Qt-free by design, mentions `ProjImgTrans` only under TYPE_CHECKING) as the prompt/context core; page = list of block texts in reading order [I].

## 10. Gotchas checklist
1. CTD: YOLO blocks are computed then discarded; only DB lines count. DB box score cut 0.6, DB bitmap threshold 0.3.
2. CTD padding is bottom/right only; ONNX is fixed 1024 and may emit [mask, lines] swapped.
3. Channel order differs between torch (flipped to BGR) and opencv-dnn (as given). Use BGR for ONNX [I].
4. Mask pipeline: `__call__` returns `(raw_mask, refined_mask, blks)`; wrapper uses refined, then 7x7 ellipse dilate.
5. Reading order = heuristic (centre-y bands + x); `panel_finder` has the nearest-panel bug (1.8).
6. Inpainting: flat-bubble shortcut uses std thresholds 7/10 on RGB; AOT size multiple rule (>=128 pad), LaMa pad to 64; always blend only masked pixels back (`out*mask + orig*(1-mask)`).
7. Quad corner convention: `lines` are 4x2 ordered [tl,tr,br,bl] after `sort_pnts` (horizontal) / vertical sorted so that warp rotates 90 deg CCW; keep the same convention before `get_transformed_region`.
8. GPL-3.0 provenance: yolov5 (GPL-3), comic-text-detector (GPL-3, dmMaze), manga-image-translator-derived OCR/AOT/LaMa code (GPL-3); manga-ocr MIT-ish (kha-white, Apache-2.0 [I]); model weights have their own licences (check HF cards) [I].
