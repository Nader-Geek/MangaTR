import json

import numpy as np
import pytest

from mangatr import gpu_check
from mangatr.gpu_check import (
    BEGIN,
    CPU_PROVIDERS,
    END,
    environment,
    make_feeds,
    render_block,
    resolve_shape,
    sha256_of,
    summarize,
    time_model,
)


def test_resolve_shape_keeps_fixed_dims_and_fills_dynamic_ones():
    assert resolve_shape([1, 3, "h", "w"], size=512) == [1, 3, 512, 512]
    assert resolve_shape([None, 3, 640, 640]) == [1, 3, 640, 640]
    assert resolve_shape(["batch", 1, -1, 256], size=384) == [1, 1, 384, 256]
    assert resolve_shape([1, 2]) == [1, 2]


def test_summarize():
    stats = summarize([10.0, 20.0, 30.0])
    assert stats == {"n": 3, "mean": 20.0, "median": 20.0, "min": 10.0, "max": 30.0}


def test_sha256_of(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"abc")
    assert sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_the_expected_hashes_look_like_hashes():
    for spec in gpu_check.MODELS:
        if spec.sha256 is not None:
            assert len(spec.sha256) == 64
            int(spec.sha256, 16)


def test_environment_never_crashes_without_a_gpu():
    info = environment()
    for key in (
        "python",
        "platform",
        "numpy",
        "gpu",
        "ram_gb",
        "onnxruntime",
        "available_providers",
    ):
        assert key in info


def test_report_block_round_trips():
    report = {"environment": {"gpu": "Tesla T4, 550, 15360 MiB"}, "models": {}}
    block = render_block(report)
    assert block.startswith(BEGIN) and block.endswith(END)
    body = block.removeprefix(BEGIN).removesuffix(END)
    assert json.loads(body) == report


def test_failure_is_reported_not_raised(tmp_path):
    result = time_model(tmp_path / "missing.onnx", CPU_PROVIDERS, runs=1, warmup=0)
    assert "error" in result
    assert "ms" not in result


@pytest.fixture
def tiny_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "h", "w"])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, "h", "w"])
    sizes = helper.make_tensor_value_info("sizes", TensorProto.INT64, [1, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, "h", "w"])
    sizes_f = helper.make_tensor_value_info("sizes_f", TensorProto.FLOAT, [1, 2])
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    nodes = [
        helper.make_node("Mul", ["image", "two"], ["scaled"]),
        helper.make_node("Add", ["scaled", "mask"], ["out"]),
        helper.make_node("Cast", ["sizes"], ["sizes_f"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(nodes, "tiny", [image, mask, sizes], [out, sizes_f], [two])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / "tiny.onnx"
    onnx.save(model, path)
    return path


def test_make_feeds_matches_each_input(tiny_model):
    import onnxruntime as ort

    session = ort.InferenceSession(str(tiny_model), providers=list(CPU_PROVIDERS))
    feeds = make_feeds(session, size=64)
    assert feeds["image"].shape == (1, 3, 64, 64) and feeds["image"].dtype == np.float32
    assert feeds["mask"].shape == (1, 1, 64, 64)
    assert feeds["sizes"].shape == (1, 2) and feeds["sizes"].dtype == np.int64
    assert (feeds["sizes"] == 64).all()


def test_time_model_on_a_real_session(tiny_model):
    result = time_model(tiny_model, CPU_PROVIDERS, runs=3, warmup=1, size=32)
    assert "error" not in result, result
    assert result["providers_used"] == ["CPUExecutionProvider"]
    assert result["ms"]["n"] == 3 and result["ms"]["mean"] > 0
    assert [item["name"] for item in result["inputs"]] == ["image", "mask", "sizes"]
    assert result["inputs"][0]["shape"] == [1, 3, "h", "w"]
    assert result["load_s"] >= 0


def test_gpu_memory_probe_is_called_when_given(tiny_model):
    result = time_model(
        tiny_model, CPU_PROVIDERS, runs=1, warmup=0, size=16, memory_probe=lambda: 123
    )
    assert result["gpu_memory_used_mb"] == 123


def test_build_report_without_cuda_skips_the_gpu_run_and_keeps_the_cpu_one(
    tiny_model, tmp_path, monkeypatch
):
    spec = gpu_check.ModelSpec("tiny", "none/none", tiny_model.name)
    monkeypatch.setattr(gpu_check, "MODELS", (spec,))
    monkeypatch.setattr(
        gpu_check,
        "environment",
        lambda: {"available_providers": ["CPUExecutionProvider"]},
    )
    report = gpu_check.build_report(
        tiny_model.parent, runs=2, size=16, cpu_runs=2, skip_download=True
    )
    entry = report["models"]["tiny"]
    assert entry["download"]["skipped"] is True
    assert "CUDAExecutionProvider" in entry["gpu"]["skipped"]
    assert entry["cpu"]["ms"]["n"] == 2
