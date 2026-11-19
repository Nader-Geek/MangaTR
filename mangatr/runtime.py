"""ONNX Runtime sessions with the settings the old engine settled on.

Adapted from the old engine (Nader-Geek/Manga-AutoTranslate--Arabic, branch
new-engine, manga_sections/20_models.py: `_ort_session_options`,
`_best_providers_for`, `_build_provider_options`, `_cudnn_algo_search`; MIT, same
author). Kept: full graph optimisation, sequential execution, spinning off, the
provider order, and the CUDA options (`cudnn_conv_algo_search=HEURISTIC` avoids the
slow exhaustive search on the first run of each new shape; max workspace; copy in
the default stream; power-of-two arena). Left out: Android, NNAPI, QNN, XNNPACK,
OpenVINO and the DirectML run lock, because MangaTR does not target them.

Environment switches (all optional): MANGATR_ORT_THREADS, MANGATR_ALLOW_TRT=1,
MANGATR_CUDNN_ALGO=exhaustive.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from typing import Any

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
GPU_ORDER = (
    "TensorrtExecutionProvider",
    CUDA,
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
)


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def best_providers(prefer_gpu: bool = True, available: Sequence[str] | None = None) -> list[str]:
    """GPU providers in order of preference, then the CPU. TensorRT only when
    MANGATR_ALLOW_TRT=1 (its first run builds engines for a long time)."""
    have = list(available_providers() if available is None else available)
    chosen: list[str] = []
    if prefer_gpu:
        for name in GPU_ORDER:
            if name not in have:
                continue
            if name == "TensorrtExecutionProvider" and os.environ.get("MANGATR_ALLOW_TRT") != "1":
                continue
            chosen.append(name)
    chosen.append(CPU)
    return chosen


def cudnn_algo_search() -> str | None:
    """HEURISTIC unless MANGATR_CUDNN_ALGO asks for the exhaustive search."""
    if os.environ.get("MANGATR_CUDNN_ALGO", "").strip().lower() in (
        "exhaustive",
        "off",
        "none",
        "0",
    ):
        return None
    return "HEURISTIC"


def provider_options(providers: Sequence[str]) -> list[dict[str, str]] | None:
    """Per-provider options aligned with `providers`, or None when none are needed."""
    if CUDA not in providers:
        return None
    options: list[dict[str, str]] = [{} for _ in providers]
    cuda = {
        "cudnn_conv_use_max_workspace": "1",
        "do_copy_in_default_stream": "1",
        "arena_extend_strategy": "kNextPowerOfTwo",
    }
    algo = cudnn_algo_search()
    if algo:
        cuda["cudnn_conv_algo_search"] = algo
    options[list(providers).index(CUDA)] = cuda
    return options


def session_options(threads: int = 4) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    with contextlib.suppress(ValueError):
        threads = int(os.environ.get("MANGATR_ORT_THREADS", "") or threads)
    so.intra_op_num_threads = max(0, threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


def open_session(path: str, *, prefer_gpu: bool = True, threads: int = 4) -> Any:
    """An InferenceSession with the engine's settings. ONNX Runtime falls back to
    the CPU silently when CUDA cannot load: check `session.get_providers()`."""
    import onnxruntime as ort

    providers = best_providers(prefer_gpu)
    return ort.InferenceSession(
        path,
        sess_options=session_options(threads),
        providers=providers,
        provider_options=provider_options(providers),
    )
