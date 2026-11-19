# Provenance of copied or adapted code

| File | Source | Licence | What was taken |
|---|---|---|---|
| `mangatr/model_tiers.py` | KOMA Studio (klaus-2/koma-studio), `mini-backend/models/inpainting` | MIT | Pre/post-processing contract of the AOT and LaMa ONNX models (scaling, padding, mask threshold, output range), re-written, not copied verbatim. |
| `mangatr/translate.py`, `mangatr/inpaint.py` | Koharu (koharu-rs/koharu) | MIT / Apache-2.0 | The design: one conversation per chapter with keyed JSON pages; the flat-colour thresholds (10 and 64). Re-written. |
| `mangatr/runtime.py`, parity check in `mangatr/model_tiers.py` | Old engine (Nader-Geek/Manga-AutoTranslate--Arabic, `20_models.py`) | MIT, same author | ORT session options, provider order and CUDA options; the GPU/CPU colour parity check. Adapted, Android/XNNPACK/DirectML parts left out. |
| `mangatr/bubble.py` | Old engine (`85_cleanup.py`, `_bubble_domain_local`, `_bubble_interior_mask`) | MIT, same author | The bubble-interior idea, re-written. |

Models are downloaded at run time and are not part of this repository; each has its own licence (see docs/DESIGN.md, risks).
