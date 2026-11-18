"""Model tiers for the cleaning router: OpenCV, AOT and LaMa.

Each tier is a `Tier` (see inpaint.py) that always fills the mask, so any of
them can be the router's last tier. The ONNX tiers take an injected session, so
tests need no model file and the caller decides the providers and the order.

The pre- and post-processing of AOT and LaMa follows KOMA's wrappers
(klaus-2/koma-studio, MIT; see NOTICE.md): RGB uint8 in, mask binarised at 128,
height and width padded (mirrored) to a multiple of 8, output cropped back.
AOT works in -1..1 with the masked pixels zeroed, LaMa in 0..1.
Only masked pixels are taken from the model; the rest of the crop is untouched.
"""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np

PAD_MOD = 8
AOT_MIN_SIZE = 128
AOT_MAX_SIDE = 1024  # larger crops are scaled down for AOT, as in KOMA


class ModelOutputError(RuntimeError):
    """The model returned something that cannot be a picture of the right size."""


class OpenCvTier:
    """Telea inpainting: no model, always succeeds. The floor of the router."""

    name = "opencv"

    def __init__(self, radius: int = 3) -> None:
        self.radius = radius

    def fill(self, crop, mask, ring):
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(crop, grown, self.radius, cv2.INPAINT_TELEA)
        out = crop.copy()
        out[mask] = filled[mask]
        return out


def pad_to_multiple(array: np.ndarray, mod: int = PAD_MOD, min_size: int = 0) -> np.ndarray:
    """Pad bottom and right (mirrored) so both sides are a multiple of `mod`
    and at least `min_size`."""
    height, width = array.shape[:2]

    def target(size: int) -> int:
        size = max(size, min_size)
        return -(-size // mod) * mod

    out_h, out_w = target(height), target(width)
    if (out_h, out_w) == (height, width):
        return array
    # symmetric padding cannot exceed the current size, so grow in steps
    while array.shape[0] < out_h or array.shape[1] < out_w:
        h, w = array.shape[:2]
        step = ((0, min(out_h - h, h)), (0, min(out_w - w, w))) + ((0, 0),) * (array.ndim - 2)
        array = np.pad(array, step, mode="symmetric")
    return array


def _scale_to(side_limit: int | None, height: int, width: int) -> float:
    if side_limit is None or max(height, width) <= side_limit:
        return 1.0
    return side_limit / max(height, width)


class OnnxInpaintTier:
    """AOT or LaMa through an ONNX Runtime session (or anything with the same
    `get_inputs()` and `run(None, feeds)`)."""

    def __init__(self, name: Literal["aot", "lama"], session: Any) -> None:
        if name not in ("aot", "lama"):
            raise ValueError(f"unknown model kind: {name}")
        self.name = name
        self._session = session
        names = [item.name for item in session.get_inputs()]
        if len(names) != 2:
            raise ValueError(f"{name} expects an image and a mask input, got {names}")
        self._image_name, self._mask_name = names

    def fill(self, crop, mask, ring):
        height, width = crop.shape[:2]
        scale = _scale_to(AOT_MAX_SIDE if self.name == "aot" else None, height, width)
        image, binary = crop, mask.astype(np.uint8) * 255
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            binary = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
        work_h, work_w = image.shape[:2]
        min_size = AOT_MIN_SIZE if self.name == "aot" else 0
        image = pad_to_multiple(image, PAD_MOD, min_size)
        binary = pad_to_multiple(binary, PAD_MOD, min_size)

        keep = (binary >= 128).astype(np.float32)
        tensor = image.astype(np.float32)
        tensor = tensor / 127.5 - 1.0 if self.name == "aot" else tensor / 255.0
        if self.name == "aot":
            tensor = tensor * (1.0 - keep[..., None])
        feeds = {
            self._image_name: np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]),
            self._mask_name: keep[None, None],
        }
        out = np.asarray(self._session.run(None, feeds)[0])
        result = self._decode(out, image.shape[:2])[:work_h, :work_w]
        if scale < 1.0:
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC)
        filled = crop.copy()
        filled[mask] = result[mask]
        return filled

    def _decode(self, out: np.ndarray, padded_hw: tuple[int, int]) -> np.ndarray:
        if out.ndim == 4:
            out = out[0]
        if out.ndim != 3 or out.shape[0] != 3 or tuple(out.shape[1:]) != padded_hw:
            raise ModelOutputError(f"{self.name} returned {out.shape}, expected (3, {padded_hw})")
        if not np.isfinite(out).all():
            raise ModelOutputError(f"{self.name} returned NaN or infinity")
        out = out.transpose(1, 2, 0)
        out = out * 127.5 + 127.5 if self.name == "aot" else out * 255.0
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


GPU_FIRST = ("CUDAExecutionProvider", "CPUExecutionProvider")


def load_session(path: str, providers: tuple[str, ...] = GPU_FIRST):
    """Open an ONNX model. ONNX Runtime falls back to the CPU silently when CUDA
    cannot load, so callers that care check `session.get_providers()`."""
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=list(providers))
