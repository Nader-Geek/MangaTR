"""Report what a machine does with the models MangaTR plans to use.

Run on the target GPU (a Colab T4 first):

    python -m mangatr.gpu_check --out report.json

It records the environment, downloads each model and checks its hash, loads it
with the GPU provider and says which provider ONNX Runtime really used (it
falls back to the CPU silently when CUDA cannot load), then times a dummy
inference on the GPU and, for comparison, on the CPU. The whole report is also
printed between two marker lines so it can be copied back as text.

The timings use random inputs of the model's own input shapes, so they measure
the model and the machine, not the quality of any result. Model hashes marked
as expected come from KOMA's registry and are not verified by us.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BEGIN = "===== MANGATR REPORT BEGIN ====="
END = "===== MANGATR REPORT END ====="

GPU_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
CPU_PROVIDERS = ("CPUExecutionProvider",)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    repo: str
    filename: str
    sha256: str | None = None  # expected hash, when we have one


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("detector", "ogkalu/comic-text-and-bubble-detector", "detector.onnx"),
    ModelSpec(
        "aot",
        "ogkalu/aot-inpainting",
        "aot.onnx",
        "ffd39ed8e2a275869d3b49180d030f0d8b8b9c2c20ed0e099ecd207201f0eada",
    ),
    ModelSpec(
        "lama_manga",
        "ogkalu/lama-manga-onnx-dynamic",
        "lama-manga-dynamic.onnx",
        "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9",
    ),
)

_DTYPES: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(milliseconds: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(milliseconds),
        "mean": round(statistics.fmean(milliseconds), 2),
        "median": round(statistics.median(milliseconds), 2),
        "min": round(min(milliseconds), 2),
        "max": round(max(milliseconds), 2),
    }


def resolve_shape(shape: Sequence[Any], *, size: int = 512) -> list[int]:
    """A concrete shape for an input: fixed dims stay, a dynamic first dim is 1
    (the batch), any other dynamic dim is `size`."""
    dims: list[int] = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        elif index == 0:
            dims.append(1)
        else:
            dims.append(size)
    return dims


def make_feeds(session: Any, *, size: int = 512) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    feeds: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        dtype = _DTYPES.get(item.type)
        if dtype is None:
            raise ValueError(f"unsupported input type {item.type} for input {item.name}")
        shape = resolve_shape(item.shape, size=size)
        if np.issubdtype(dtype, np.floating):
            feeds[item.name] = rng.random(shape).astype(dtype)
        elif dtype is np.bool_:
            feeds[item.name] = np.zeros(shape, dtype=dtype)
        else:  # integer inputs are usually image sizes
            feeds[item.name] = np.full(shape, size, dtype=dtype)
    return feeds


def time_model(
    path: str | Path,
    providers: Sequence[str],
    *,
    runs: int = 10,
    warmup: int = 3,
    size: int = 512,
    memory_probe: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    """Load the model and time `runs` inferences. Never raises: a failure is
    recorded under "error" so one bad model does not hide the others."""
    result: dict[str, Any] = {"requested_providers": list(providers)}
    try:
        import onnxruntime as ort

        start = time.perf_counter()
        session = ort.InferenceSession(str(path), providers=list(providers))
        result["load_s"] = round(time.perf_counter() - start, 3)
        result["providers_used"] = session.get_providers()
        result["inputs"] = [
            {
                "name": item.name,
                "type": item.type,
                "shape": [dim if isinstance(dim, int) else str(dim) for dim in item.shape],
            }
            for item in session.get_inputs()
        ]
        feeds = make_feeds(session, size=size)
        for _ in range(warmup):
            session.run(None, feeds)
        times: list[float] = []
        for _ in range(runs):
            begin = time.perf_counter()
            session.run(None, feeds)
            times.append((time.perf_counter() - begin) * 1000)
        result["ms"] = summarize(times)
        if memory_probe is not None:
            result["gpu_memory_used_mb"] = memory_probe()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def _run(args: Sequence[str]) -> str | None:
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else None


def gpu_memory_used_mb() -> int | None:
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return int(out.splitlines()[0]) if out else None
    except ValueError:
        return None


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "gpu": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            kib = int(handle.readline().split()[1])
        info["ram_gb"] = round(kib / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        info["ram_gb"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
    except Exception as exc:
        info["onnxruntime"] = f"unavailable: {type(exc).__name__}: {exc}"[:200]
        info["available_providers"] = []
    return info


def fetch(spec: ModelSpec, models_dir: Path) -> dict[str, Any]:
    """Download one model and check its hash. Never raises."""
    result: dict[str, Any] = {"repo": spec.repo, "file": spec.filename}
    try:
        from huggingface_hub import hf_hub_download

        start = time.perf_counter()
        path = hf_hub_download(repo_id=spec.repo, filename=spec.filename, local_dir=str(models_dir))
        result["download_s"] = round(time.perf_counter() - start, 1)
        result["path"] = str(path)
        result["size_mb"] = round(Path(path).stat().st_size / 1e6, 1)
        digest = sha256_of(path)
        result["sha256"] = digest
        result["sha256_expected"] = spec.sha256
        result["sha_ok"] = None if spec.sha256 is None else digest == spec.sha256
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def render_block(report: dict[str, Any]) -> str:
    body = json.dumps(report, indent=2, ensure_ascii=False)
    return f"{BEGIN}\n{body}\n{END}"


def build_report(
    models_dir: Path, *, runs: int, size: int, cpu_runs: int, skip_download: bool
) -> dict[str, Any]:
    env = environment()
    report: dict[str, Any] = {"environment": env, "models": {}}
    has_cuda = "CUDAExecutionProvider" in env.get("available_providers", [])
    for spec in MODELS:
        entry: dict[str, Any] = {}
        local = models_dir / spec.filename
        if skip_download and local.exists():
            entry["download"] = {"path": str(local), "skipped": True}
            path: str | None = str(local)
        else:
            entry["download"] = fetch(spec, models_dir)
            path = entry["download"].get("path")
        if path:
            if has_cuda:
                gpu = time_model(
                    path, GPU_PROVIDERS, runs=runs, size=size, memory_probe=gpu_memory_used_mb
                )
                gpu["ran_on_gpu"] = "CUDAExecutionProvider" in gpu.get("providers_used", [])
                entry["gpu"] = gpu
            else:
                entry["gpu"] = {"skipped": "CUDAExecutionProvider is not available"}
            if cpu_runs > 0:
                entry["cpu"] = time_model(path, CPU_PROVIDERS, runs=cpu_runs, warmup=1, size=size)
        report["models"][spec.key] = entry
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report what this machine does with the models.")
    parser.add_argument("--out", default="mangatr_report.json")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--runs", type=int, default=10, help="timed runs on the GPU")
    parser.add_argument("--cpu-runs", type=int, default=2, help="timed runs on the CPU; 0 skips")
    parser.add_argument("--size", type=int, default=512, help="size for dynamic input dims")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        models_dir,
        runs=args.runs,
        size=args.size,
        cpu_runs=args.cpu_runs,
        skip_download=args.skip_download,
    )
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(render_block(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
