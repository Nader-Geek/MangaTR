"""Recognizers for `ocr.OcrStage`.

MangaOcr: Japanese, vertical and horizontal, built for manga (kha-white/
manga-ocr-base, Apache-2.0 on its model card; the weights are downloaded on first
use). It is a Vision Encoder Decoder, so the recipe is the one its authors
publish: grayscale, the model's own image processor, greedy generation, then
tidy the text. `torch` and `transformers` are imported on first use so the rest
of the package does not need them.

Korean and Chinese are not covered by this model; a recognizer for them plugs
into the same `Recognizer` protocol.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .ocr import Reading

MODEL_ID = "kha-white/manga-ocr-base"
MAX_LENGTH = 300


def clean_reading(text: str) -> str:
    """The model puts spaces between characters and writes ellipses its own way:
    drop whitespace, spell an ellipsis as dots, collapse runs of dots and middots."""
    text = "".join(text.split()).replace("…", "...")
    return re.sub("[・.]{2,}", lambda m: "." * (m.end() - m.start()), text)


class MangaOcr:
    name = "manga-ocr"
    languages = frozenset({"ja"})

    def __init__(self, model: Any, processor: Any, tokenizer: Any, device: str, batch: int = 8):
        self.model, self.processor, self.tokenizer = model, processor, tokenizer
        self.device, self.batch = device, batch

    @classmethod
    def load(cls, device: str | None = None, model_id: str = MODEL_ID) -> MangaOcr:
        import torch
        from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = ViTImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
        return cls(model, processor, tokenizer, device)

    def read(self, crops: Sequence[np.ndarray]) -> list[Reading]:
        import cv2
        import torch
        from PIL import Image

        readings: list[Reading] = []
        for start in range(0, len(crops), self.batch):
            images = []
            for crop in crops[start : start + self.batch]:
                gray = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2GRAY)
                images.append(Image.fromarray(gray).convert("RGB"))
            pixels = self.processor(images, return_tensors="pt").pixel_values.to(self.device)
            with torch.inference_mode():
                out = self.model.generate(
                    pixels,
                    max_length=MAX_LENGTH,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            texts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
            confidences = self._confidences(out, len(texts))
            for text, confidence in zip(texts, confidences, strict=True):
                readings.append(Reading(clean_reading(text), confidence))
        return readings

    def _confidences(self, out, count: int) -> list[float]:
        """One confidence per crop: the mean token probability of the chosen reading.

        The model's own config may ask for beam search, and then the per-step scores
        are per beam (batch x beams rows) and do not line up with the sequences; the
        beam search's own length-normalised score is used instead. If anything about
        this fails the reading is kept with a neutral 0.5: a text must never be lost
        to a confidence quirk (the neutral value passes the 0.25 free-text cut).
        """
        try:
            beams = getattr(out, "sequences_scores", None)
            if beams is not None:
                return [float(v) for v in beams.exp().tolist()]
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
            generated = out.sequences[:, -scores.shape[1] :]
            real = generated != self.tokenizer.pad_token_id
            result = []
            for row in range(count):
                probabilities = scores[row][real[row]].exp()
                result.append(float(probabilities.mean()) if probabilities.numel() else 0.0)
            return result
        except Exception as exc:
            import warnings

            warnings.warn(f"OCR confidence unavailable ({exc}); using 0.5", stacklevel=2)
            return [0.5] * count
