# Provenance of copied or adapted code

| File | Source | Licence | What was taken |
|---|---|---|---|
| `mangatr/model_tiers.py` | KOMA Studio (klaus-2/koma-studio), `mini-backend/models/inpainting` | MIT | Pre/post-processing contract of the AOT and LaMa ONNX models (scaling, padding, mask threshold, output range), re-written, not copied verbatim. |
| `mangatr/translate.py`, `mangatr/inpaint.py` | Koharu (koharu-rs/koharu) | MIT / Apache-2.0 | The design: one conversation per chapter with keyed JSON pages; the flat-colour thresholds (10 and 64). Re-written. |

Models are downloaded at run time and are not part of this repository; each has its own licence (see docs/DESIGN.md, risks).
