# 07 - clean-bot (omar soliman) study

Repo: `/tmp/claude-0/-home-claude/3f4fe986-168e-5f93-a77f-3b358ef513a1/scratchpad/ref/clean-bot`
Labels: **read code** = verified in source; **inferred** = reasoned, not verifiable from repo (e.g. third-party lib internals).

## 0. Licence (read code)
- `LICENSE`: "MIT License / Copyright (c) 2025 omar soliman"; README badge + line 366 say MIT.
- `app/services/hi_sam_src/LICENSE` is Apache-2.0 (vendored Hi-SAM source).
- **Provenance caveat**: `app/__init__.py:1` = `__program__ = "pcleaner"`; module/class names, config schema (`.conf` profiles, `#raw.json`/`#clean.json`, `MaskerData`, `pick_best_mask`, `calculate_best_splits`) are a fork of VoxelCubes' **pcleaner/PanelCleaner**. Upstream is (inferred, from memory) GPL-3.0, and there are no upstream copyright headers here. So the "MIT" label may not be valid for the masker/image_ops/config code. **Treat the algorithms as idea-level reference only; do not copy files into MangaTR without clearing this.** Hi-SAM (Apache-2), LaMa weights (Sanster releases; LaMa code Apache-2) are separately licensed.
- Model weights fetched at runtime: CTD `zyddnys/manga-image-translator` release beta-0.3 `comictextdetector.pt(.onnx)`; YOLO11s `huggingface.co/lordtrilink/manga-text-detector-v0` `best.pt/.onnx` (Ultralytics = AGPL-3 - check for a commercial bot); SAM `sam_vit_{b,l,h}` from fbaipublicfiles; Hi-SAM decoder ckpts (`vit_*_maskdecoder.pth`, `word_detection_totaltext.pth`, `line_detection_ctw1500.pth`) from **OneDrive share links** scraped via a "Badger token" hack (`model_downloader._download_onedrive`) - fragile.

## 1. Repo map (read code)
- `main.py` FastAPI app + lifespan: starts `worker.process_queue()` and a cleanup loop as asyncio tasks; mounts `app/api/main.py` at `/api`; CORS `*`.
- `app/api/main.py`: `/health /ready /device /process /status/{id} /settings/generate-key` (Bearer = `settings.API_KEY`, compared with `!=`, not constant-time; `generate-key` just returns a random string, does not store/invalidate it despite the warning log).
- `app/workers/task_worker.py` (queue + orchestration), `app/services/clean_pipeline.py` (stage driver), `preprocessor.py` (detectors + box prep), `masker.py`+`image_ops.py` (flat-colour mask fitting), `denoiser.py`, `inpainting.py` (LaMa), `model_downloader.py`, `database.py` (Supabase), `google_drive.py` + `rclone_manager.py` (misnamed: it is a Drive-API manager, no rclone), `cleanup.py`.
- `discord-bot/index.js` (677 lines, discord.js 14, ESM) - single file.
- `New Text Document.txt` = ops runbook (start.sh/stop.sh, pip CUDA 12.8 wheels `cu128`, `API_PORT=10100`, `fuser -k 10100/tcp`, pcleaner pip extras, Docker build commands for `Dockerfile.server` which is NOT in the repo). `New Text Document (2).txt` = 211-byte subset of the same commands. Both contain only shell snippets; a placeholder `HF_TOKEN`, no real secrets (read code).
- `app/config.py` (1762 lines) is mostly the pcleaner attrs-dataclass config machinery (profile load/save, validators, Qt-ish `LongString/Pixels/ThreadLimit` annotations). Real tunables are only the dataclass fields listed below.

## 2. Pipeline as actually run (read code: `task_worker._clean_images`, `clean_pipeline.py`)
```
process_request(id,user,drive_url):
  status=processing; download Drive folder (3 retries, 10MB chunks) -> temp_chapters/<uuid>/
  pixel_count = SUM(page heights)   # NB "pixel" is really total height; page>16000 or >16383 wide -> reject
  credits = 1 if <=200k, 2 <=300k, 3 <=400k, 4 <=500k, >500k reject   (5-credit branch unreachable)
  user.credits < need -> fail
  pre_compress: every image -> RGB (alpha pasted on white) -> thumbnail(3000,3000, LANCZOS) -> JPEG q85 progressive   # LOSSY + DOWNSCALE before cleaning
  split_all_images: calculate_best_splits (below) -> crops saved JPEG q92 as stem_partNNN
  run_detection(parts)  -> run_cleaning(parts) -> merge_parts_from_disk (stitch, >65500px JPEG -> PNG, rename 001.ext..)
  create Drive subfolder Cleaned_Order_<id>, permission anyone/reader, upload files sequentially
  deduct credits AFTER success; status=completed; output_url=drive folder; rmtree temp
  on exception: status=failed(+error), rmtree, delete Drive subfolder
```
Stage detail of `run_cleaning`: `prep_json_file` (ThreadPool<=8) -> `mask_page` (multiprocessing.Pool, cores) -> optional `denoise_page` -> `inpaint_page` (ThreadPool(4) sharing ONE LaMa instance) -> export via ProcessPool. Intermediates are JSON+PNG on disk per page in `parts/_work`.

### 2a. Long-strip splitting `ops.calculate_best_splits` (read code, reusable idea)
```
if not split or h<=2000 or w/h > 0.33: no split
n = ceil(h/2000)-1; ph = h//(n+1); splits=[ph*i]
tol = min(500, ph//2)
for each split: window=[s-tol, s+tol) on luminance
   cost[row] = sum((diff along x)^2)          # horizontal edge energy per row
   bitmap = cost > percentile(cost,10)
   pick centre of the longest contiguous run where bitmap==0 (quietest band); else argmin(cost)
```
Cheap, sensible "cut where the page is blank" splitter. Constants: height 2000, tol 500, aspect 0.33.

## 3. Text detection (read code: `preprocessor.py`)
Detector chosen by `[TextDetector] text_detector_model` (shipped profile `master.conf`: `hi-sam`, `hi_sam_model_type = vit_h`; code default "bubble" is remapped to hi-sam). All three return **only axis-aligned boxes**.
| Detector | Weights | Call / thresholds |
|---|---|---|
| Hi-SAM (vendored `hi_sam_src`, Apache) | SAM ViT-{b,l,h} + Hi-SAM decoder `word_detection_totaltext.pth`, `hier_det=True, attn_layers=1, prompt_len=12` | `AutoMaskGenerator.predict_text_detection(from_low_res=False, fg_points_num=900, batch_points_num=150, score_thresh=0.5, nms_thresh=0.42, zero_shot=False, dataset='totaltext')`; mask->bbox of nonzero pixels; keep area>=50 px |
| CTD | `comictextdetector.pt` (+onnx) | 4 backends tried in order: `comic_text_detector` lib, onnxruntime, cv2.dnn, ultralytics. ONNX path: plain `cv2.resize` to 640x640 (no letterbox), expects output `[N,6]` with conf>=0.5, rescales to page. |
| YOLO11s lordtrilink v0 | `best.pt/.onnx` | ultralytics `predict(conf=0.3, iou=0.5)`; same ONNX fallback with conf 0.3, area>=50 |
Notes:
- **Mask construction (key finding)**: the Hi-SAM pixel masks are *discarded*; `generate_mask_data` draws `ImageDraw.rectangle(fill=255)` for each box into `<stem>_mask.png`. So the "precise mask" downstream is a union of rectangles, not stroke-level. Hi-SAM's main benefit (pixel text strokes) is thrown away. (read code)
- Inferred bugs: CTD/YOLO ONNX branches assume NMS-decoded `[N,6]` output and 640 square resize; raw YOLOv8/11 ONNX output is `[1,4+nc,8400]`, and CTD `.pt` is not an ultralytics file, so those fallbacks likely produce garbage or fall through to "0 boxes" with only a warning. `_load_ctd_model` just `torch.load`s and never builds a net. The CTD lib branch re-instantiates the detector per image.
- Full page (<=3000px side after pre-compress, <=~2000px tall after split) is fed whole; no tiling at detection.
- No OCR anywhere in the pipeline (cleaner only); README mentions easyocr in pip line only.
- Failure policy: any exception -> empty mask + empty `blk_list`, pipeline continues (page returned uncleaned, user still charged).

### Box preparation (`prep_json_file`, config from `master.conf`)
```
drop boxes with area < box_min_size (400)            # code default 20*20
resolve_total_overlaps(); grow BOX by 2 (+3 right)
sort by y1
EXTENDED = copy; grow by 5 (+5 right)
MERGED_EXT = merge extended boxes with overlap >= 20%
REFERENCE  = MERGED_EXT grown by 20   # ring of background used for colour sampling
```
(`cluster_boxes`, reading-order helpers exist but are unused here.)

## 4. Flat-colour mask fitting (read code: `masker.mask_page`, `ops.pick_best_mask`) - this is the cheap tier
```
box_mask = rectangles of EXTENDED boxes; cut_mask = precise ∩ box_mask
for each merged-extended box (masking_box) with its reference_box:
    crop base to reference_box (box+20px ring); crop precise mask, pad to same shape
    if precise empty -> skip (false positive)
    candidates = [dilate(precise, min_thickness=2)]  then +2px steps x (11-1)  (ellipse kernel; diameter<=5 -> rounded square)
                 + box_mask appended LAST   (mask_selection_fast=False; if True box first & stop at std==0)
    for each candidate: edge = FIND_EDGES(mask); border px = base[edge==1]
         std  = np.std(gray)  or, colour: sample std of Euclidean distances to mean (ddof=1)
         colour = mode if one colour >50% of border px else geometric median (Weiszfeld, eps 1e-5, 500 it)
         if min(colour) > off_white 240 -> (255,255,255)
    best = first candidate; replace by later one only if std <= best*(1-0.1)   # larger masks must be >=10% more uniform
    if best_std > 15.0 -> mark FAILED (leave for LaMa) else paint mask with median colour
combined RGBA layer alpha-composited on page
```
Constants: growth 2px x 11 steps, min thickness 2, `allow_colored_masks=True`, `off_white_max_threshold=240`, `mask_max_standard_deviation=15`, `mask_improvement_threshold=0.1`, `mask_selection_fast=False`. Per-box result also stores `deviation` and `thickness` which later scale the LaMa mask growth.

## 5. Denoiser (read code; disabled in shipped profile `denoising_enabled=False`)
For non-failed boxes with std > 0.25: crop box, `fastNlMeansDenoising(Colored)` h=10, templ 7, search 21, over mask grown 5px and Gaussian-faded 1px; alpha-composite back. Skips 1-bit images. Intended to hide the flat-fill vs. screentone/noise mismatch.

## 6. LaMa usage (read code: `inpainting.py`; library internals inferred)
```
InpaintingModel: ensure weights (default anime-manga-big-lama.pt from Sanster/models), set env LAMA_MODEL, SimpleLama() (pip simple-lama-inpainting), eager load once; _ModelCache singleton guarded by threading.Lock
inpaint_page:
  original (RGB, full res); mask_np (flat-filled boxes) and exact_np (failed boxes)
  per box: src = exact if failed else fitted; dilate(min_mask_thickness, 3x3 ellipse)
      growth = min(10 + int(deviation*0.2), 20)          # min_inpainting_radius 10, multiplier 0.2, max 20
      iso    = growth + 8                                # inpainting_isolation_radius
      pad box by iso, dilate by growth -> max() into combined_inpaint; dilate by iso -> combined_isolation
  LaMa mask = threshold(combined_inpaint >127) BINARY, resize NEAREST to original      # comment: soft masks cause black blocks
  result = SimpleLama(original_whole_image, mask)         # WHOLE PAGE, single pass, no tiling/ROI crop in this code
  alpha = gaussian_blur(binary, fade_radius=4) bilinear-resized, zeroed outside isolation mask
  out = (original + flat-colour layer) alpha_composited with LaMa result using alpha
```
- `full_lama_inpainting = true` in shipped profile -> masker skips prefill of cleaned.png, but `inpaint_page` still pastes the flat colour layer under the composite, so flat fills + LaMa both occur (ladder is "flat colour then LaMa blended on top" rather than routed).
- Resolution/tiling/padding: none in this repo. Inferred: `simple-lama-inpainting` pads to multiple of 8 and runs the TorchScript model at full input resolution (no 512 crop), so VRAM scales with page area. The project caps this indirectly: pages <=3000 px per side, strips split to ~2000 px tall. No per-box ROI crops (Lama-cleaner-style "crop around mask + margin"), which is the main efficiency gap vs. our cost-routed cleaner.
- The README lists 5 "LaMa" models incl. `lama_large_512px.ckpt` (dreMaz) and `.safetensors`; `SimpleLama` can only load a TorchScript `.pt`, so models 3 and 5 are likely non-functional (inferred).
- `inpainting_min_std_dev=3.0`, `inpainting_max_mask_radius=6` exist in config but are **unused** in `inpaint_page` (grep: only growth/isolation/fade fields used).
- `InpaintingModel.__call__` crops result back to input size (guards padding).

## 7. GPU memory handling and concurrency (read code)
- Models are process-global singletons (`_hisam_amg`, `_ctd_model`, `_yolo_model`, `_ModelCache`). Detector switch frees old Hi-SAM (`del`, `gc.collect`, `torch.cuda.empty_cache`).
- OOM ladder in `generate_mask_data`: catch RuntimeError containing "out of memory" -> `empty_cache`+`gc` -> retry same detector on **CPU** -> if Hi-SAM not vit_b and still OOM -> vit_b on CPU -> else raise. Also `finally: empty_cache()` after every page. `clean_pipeline.run_detection` adds a second CPU-model retry for "CUDA/cudnn/kernel image" errors.
- `start.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `main.py` lifespan only *logs* a "GPU power" recommendation (>=8GB: concurrent_models=8; >=4: 4; else 2) - not enforced. `master.conf` comment: vit_h ~6GB VRAM, "34 parts may OOM on 11GB".
- `utils/device.py`: lazy cached torch probe -> `DeviceKind {CPU,CUDA,MPS}`, `torch_device_str()`, `hardware_info()` (name, CUDA ver, GiB). Good small module; note it only queries device 0 and caches forever.
- Pitfall: 4-thread pool calling one LaMa instance + `torch` model not guarded by a lock around forward (inferred risk; works because GIL + CUDA stream serialisation but VRAM spikes 4x page size).
- No batching; no fp16/`torch.inference_mode`/`no_grad` visible in this repo's code (inferred it depends on simple-lama).

## 8. Queue / job model (read code)
- **Single sequential worker** inside the API process: `process_queue` polls Supabase `requests` every 5s; `get_next_pending_request` = select oldest `pending` then `UPDATE ... status=processing WHERE id=? AND status='pending'` (optimistic claim, safe with multiple workers).
- Work runs through `asyncio.to_thread` per stage; one request at a time. No retry of whole job, no heartbeat: **a crash mid-job leaves status `processing` forever** (no reaper). Credits deducted only on success (user can't be double-charged but can be cleaned for free if balance dropped concurrently; credit check/deduct is read-modify-write, not atomic).
- `queue_position` = count of pending/processing older rows. ETA = (position+1) x `settings.estimated_processing_time_per_chapter_seconds`.
- Redis is in settings/deps but unused (inferred: not referenced in the read code paths).

## 9. Discord bot (read code: `discord-bot/index.js`)
- Gateway intents: `Guilds` only. Slash commands registered **globally** on `ClientReady` via REST `applicationCommands` (propagation up to 1h).
- Commands: `/ping`, `/balance [user]`, `/profile [user]`, `/history [user]` (last 10 `requests`), `/clean`, admin-only `/add_points` `/deduct` (admin = single `ADMIN_USER_ID` env). Non-admin viewing others -> ephemeral "Access Denied".
- `/clean`: opens a modal `clean_modal` with a Drive-folder URL text input (images are NOT attached to Discord; user uploads to Drive first). Handler: `deferReply`, check URL contains `drive.google.com`, `ensureUser` (auto-create with 5 free credits), `credits>0` check, `POST {API}/api/process` (Bearer key, 15s `AbortSignal.timeout`), render embed with request id, queue position, ETA, delivery method (channel vs DM).
- **Polling delivery**: `setInterval(10s)` queries Supabase `requests` where status in (completed,failed) and `notified=false`; `notifyUser`: in-memory `notifiedIds` Set + DB `notified=true`; try channel (`permissionsFor(me).has(SendMessages)`), mention with `allowedMentions:{users:[id]}`, embed with fallback to plain text if Embed Links missing, else DM, else give up (still mark notified). Completion embed says output deleted after 20 minutes.
- Auto-delete: server `CleanupScheduler` every 60s (`CLEANUP_INTERVAL_SECONDS`) deletes Drive files for `requests` older than `EXPIRATION_TIME_MINUTES=20` (`db.get_expired_outputs`, `mark_output_cleaned`). Note `output_url` is a *folder* URL but the cleanup splits it by "," and calls `files().delete(fileId=<url>)` - inferred bug: IDs would be full URLs, so deletion probably 404s (and `delete_file` returns True on any 404 -> silently marked cleaned). Verify before copying.
- Per-user limits: **only credits** (tiers above; admin top-ups); no concurrent-job cap per user, no cooldown, no max pages/file size enforcement at the bot (limits enforced late in the worker after the Drive download).
- Security notes: output Drive folder set `anyone/reader` (link-sharing public); API key static; `/api` CORS `*`.

## 10. What is reusable for MangaTR
Reusable ideas (re-implement, don't copy - licence provenance):
1. **Border-uniformity test for flat fill** (section 4): std of ring pixels at multiple dilations, accept growth only if >=10% better, mode-colour-else-geometric-median, off-white snap to 255. Maps directly to our "flat colour" rung and gives a *cost-routing signal* (`deviation`) we can use to decide flat vs plane-fit vs AOT/LaMa. Cheap (pure OpenCV/numpy).
2. **Per-box LaMa mask growth** = 10 + 0.2*deviation (cap 20) with an isolation ring (+8) and Gaussian-faded alpha (r=4) clipped to isolation ring; **binary mask into LaMa, soft alpha only for compositing**. Good compositing recipe.
3. **Strip splitter** (section 2a) and merge with JPEG 65500px fallback to PNG.
4. **Detector OOM ladder** (GPU -> CPU -> lighter model), per-page `empty_cache`, singleton model cache with lock, preload at startup, `expandable_segments`.
5. **Bot architecture**: HTTP submit -> DB row -> worker claims with conditional UPDATE; bot polls `notified=false` and delivers channel->DM with permission check, mention allowlist, plain-text fallback; ETA from queue position x measured seconds; credit tiers; auto-expiry of artefacts at 20 min. Our improvements: use attachments or a presigned/expiring URL instead of Drive; replace polling with a worker->bot callback/webhook or Redis pubsub; add per-user concurrency cap + cooldown; heartbeat/reaper for stuck `processing`; atomic credit RPC (`UPDATE ... WHERE credits>=n`); global command registration only on deploy (use guild commands in dev).

## 11. Comparison vs our cost-routed cleaner (flat -> plane fit -> AOT -> LaMa -> Telea)
| Aspect | clean-bot | MangaTR |
|---|---|---|
| Routing | Fixed 2-tier: flat colour (std<=15 else "failed") then LaMa blended over *whole page* for all boxes | Multi-rung by cost; stops at cheapest passing rung |
| Fit test | Border std only (flat); no gradient/plane model, no texture/screentone rung | Plane fit + AOT cover gradients/halftone |
| LaMa cost | Full-page forward even when most boxes are flat (full_lama=true makes it always run) | Run LaMa only on residual crops with margin |
| Mask | Rectangles from detector boxes (+dilation) - strokes ignored | We should keep stroke-level masks (Hi-SAM/CTD seg) |
| Detection | Hi-SAM vit_h/vit_b, CTD, YOLO11s; boxes only; fragile ONNX fallbacks | Our detect stage should keep seg masks, pick by quality/cost |
| Memory | CPU fallback on OOM, no tiling, 4 threads on 1 model | Tile/crop, bounded concurrency |
| Fidelity | Pre-downscale to 3000px + JPEG q85 -> q92 -> final recompress | Keep lossless/original res |
| Failures | Silent empty mask, user still charged | Surface + refund |
| Typeset/translate | None (clean only) | Full chain |

## 12. Pitfalls checklist
- Pre-compress thumbnail to 3000px + JPEG q85 degrades output; users get lower-res images than they sent.
- "pixel_count" is actually sum of heights; naming/limit misleading (500k "pixels" = 500k px of height).
- Charged even if detection silently returned 0 boxes.
- Stuck `processing` rows after crash; Redis dead config; unused config fields (`inpainting_min_std_dev`, `inpainting_max_mask_radius`, `suspicious_box_min_size`, `reading_order`, `concurrent_models`).
- Cleanup deletes by wrong id (see 9); public Drive folders.
- Model links via OneDrive scraping and non-TorchScript LaMa variants likely break.
- Everything on disk as JSON+PNG per stage (simple, but I/O heavy); multiprocessing.Pool created per stage.
- Dockerfile referenced in runbook is absent; docs claim Docker support.
