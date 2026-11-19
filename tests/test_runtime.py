import numpy as np
import pytest

from mangatr.model_tiers import PARITY_TOLERANCE, OnnxInpaintTier, parity_difference
from mangatr.runtime import CPU, CUDA, best_providers, cudnn_algo_search, provider_options


def test_gpu_providers_come_first_and_tensorrt_needs_an_opt_in(monkeypatch):
    monkeypatch.delenv("MANGATR_ALLOW_TRT", raising=False)
    have = ["TensorrtExecutionProvider", CUDA, CPU]
    assert best_providers(True, have) == [CUDA, CPU]
    monkeypatch.setenv("MANGATR_ALLOW_TRT", "1")
    assert best_providers(True, have) == ["TensorrtExecutionProvider", CUDA, CPU]
    assert best_providers(False, have) == [CPU]
    assert best_providers(True, []) == [CPU]


def test_cuda_options_use_the_heuristic_search_unless_told_otherwise(monkeypatch):
    monkeypatch.delenv("MANGATR_CUDNN_ALGO", raising=False)
    options = provider_options([CUDA, CPU])
    assert options[0]["cudnn_conv_algo_search"] == "HEURISTIC" and options[1] == {}
    assert provider_options([CPU]) is None
    monkeypatch.setenv("MANGATR_CUDNN_ALGO", "exhaustive")
    assert cudnn_algo_search() is None
    assert "cudnn_conv_algo_search" not in provider_options([CUDA])[0]


class Item:
    def __init__(self, name):
        self.name = name


class Constant:
    def __init__(self, value, tint=0.0):
        self.value, self.tint = value, tint

    def get_inputs(self):
        return [Item("image"), Item("mask")]

    def run(self, _o, feeds):
        _, _, h, w = feeds["image"].shape
        out = np.full((1, 3, h, w), self.value, np.float32)
        out[:, 2] += self.tint
        return [out]


def test_identical_sessions_have_no_difference_and_a_tinted_one_is_caught():
    same = parity_difference("lama", Constant(0.5), Constant(0.5))
    assert same == 0.0
    tinted = parity_difference("lama", Constant(0.5, tint=0.1), Constant(0.5))
    assert tinted > PARITY_TOLERANCE


def test_the_tier_wraps_the_session():
    assert OnnxInpaintTier("aot", Constant(0.0)).name == "aot"


def test_session_options_are_sequential_and_optimised():
    ort = pytest.importorskip("onnxruntime")
    from mangatr.runtime import session_options

    so = session_options(2)
    assert so.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert so.inter_op_num_threads == 1 and so.intra_op_num_threads == 2
