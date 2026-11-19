import cv2
import numpy as np

from mangatr.compare import main, make_sheet, run_page, variants
from mangatr.detect import Detector
from mangatr.model_tiers import OpenCvTier


class FakeSession:
    class Item:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self.Item("images"), self.Item("orig_target_sizes")]

    def run(self, _o, feeds):
        return [
            np.array([[1]], np.int64),
            np.array([[[40, 40, 140, 100]]], np.float32),
            np.array([[0.9]], np.float32),
        ]


def test_sheet_has_every_column_labelled():
    page = np.zeros((100, 80, 3), np.uint8)
    sheet = make_sheet([("a", page), ("b", page), ("c", page)])
    assert sheet.shape == (128, 240, 3)


def test_a_page_goes_through_and_the_record_has_timings(tmp_path):
    page = np.full((200, 300, 3), 230, np.uint8)
    cv2.putText(page, "HI", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    path = tmp_path / "p.png"
    cv2.imwrite(str(path), page)
    record = run_page(
        path, Detector(FakeSession()), {"opencv": ((), OpenCvTier())}, tmp_path, lambda _m: None
    )
    assert record["blocks"] == 1 and record["opencv"]["seconds"] >= 0
    assert (tmp_path / "p_sheet.jpg").exists()


def test_variants_without_model_files_is_just_opencv(tmp_path):
    found, infos = variants(tmp_path, cpu=True)
    assert list(found) == ["opencv"] and infos == {}


def test_main_needs_the_detector(tmp_path):
    try:
        main(["--images", str(tmp_path), "--models-dir", str(tmp_path)])
    except SystemExit as exit_:
        assert "detector.onnx" in str(exit_)
    else:
        raise AssertionError("should exit")


def test_crops_sheet_has_a_row_per_block_and_the_record_lists_detections(tmp_path):
    page = np.full((200, 300, 3), 230, np.uint8)
    cv2.putText(page, "HI", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    path = tmp_path / "p.png"
    cv2.imwrite(str(path), page)
    record = run_page(
        path, Detector(FakeSession()), {"opencv": ((), OpenCvTier())}, tmp_path, lambda _m: None
    )
    assert (tmp_path / "p_crops.jpg").exists()
    assert record["detections"][0][0] == "text_bubble" and record["detections"][0][-1] is True
