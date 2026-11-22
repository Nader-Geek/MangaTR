# Provenance of copied or adapted code

| File | Source | Licence | What was taken |
|---|---|---|---|
| `mangatr/model_tiers.py` | KOMA Studio (klaus-2/koma-studio), `mini-backend/models/inpainting` | MIT | Pre/post-processing contract of the AOT and LaMa ONNX models (scaling, padding, mask threshold, output range), re-written, not copied verbatim. |
| `mangatr/translate.py`, `mangatr/inpaint.py` | Koharu (koharu-rs/koharu) | MIT / Apache-2.0 | The design: one conversation per chapter with keyed JSON pages; the flat-colour thresholds (10 and 64). Re-written. |
| `mangatr/runtime.py`, parity check in `mangatr/model_tiers.py` | Old engine (Nader-Geek/Manga-AutoTranslate--Arabic, `20_models.py`) | MIT, same author | ORT session options, provider order and CUDA options; the GPU/CPU colour parity check. Adapted, Android/XNNPACK/DirectML parts left out. |
| `mangatr/bubble.py` | Old engine (`85_cleanup.py`, `_bubble_domain_local`, `_bubble_interior_mask`) | MIT, same author | The bubble-interior idea, re-written. |
| `mangatr/halo.py` | Old engine (`85_cleanup.py`, `_absorb_text_halo`) | MIT, same author | The halo-absorption logic, improved: any colour and faintness (paper/ink mixture) instead of fixed warm hues. |
| `mangatr/residue.py` | Old engine (`85_cleanup.py`, `_scrub_bright_residuals`, `_scrub_dark_residuals`) | MIT, same author | Ghost scrub on a plain surround, merged into one pass, estimated from the ring only. |
| `mangatr/detect.py` (`nms`, `holding_bubble`) | Koharu (`koharu-ml/.../koharu_layout_highres/suppress.rs`, `layout.rs`) | MIT / Apache-2.0 | The duplicate and nested-text suppression rules and the bubble-holds-half rule, with Koharu's measured thresholds (IoU 0.5, nested 0.8, in-bubble 0.5). Re-written for boxes. |
| `mangatr/strips.py` | KOMA Studio (`services/splitter_analysis.py`) | MIT | The empty-row projection idea for cutting long strips, improved (uniform rows, gap centre, target height). |
| `mangatr/ocr_models.py` | manga-ocr (kha-white/manga-ocr-base) | Apache-2.0 (model card) | The published recipe: grayscale, the model's image processor, greedy generation, whitespace and ellipsis tidy-up. Weights are downloaded, not copied. |

Models are downloaded at run time and are not part of this repository; each has its own licence (see docs/DESIGN.md, risks).

## Licence policy

This project is for private use: not commercial, not distributed. Components under GPL-3.0 (Comic Text Detector, BallonsTranslator code) or CC BY-NC-SA 4.0 (manga-ocr-base-CPT-flex weights, manga-ocr-ar) may therefore be used. Their licences are recorded as provenance. **If the project is ever shared, published or monetised, every such component must be re-checked or removed first.** See docs/REFERENCES.md.
