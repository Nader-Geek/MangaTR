import numpy as np
import pytest

from mangatr.detect import (
    Detection,
    Detector,
    decode,
    merge_tiles,
    nms,
    plan_tiles,
    preprocess,
    reading_order,
    to_blocks,
)
from mangatr.scene import BlockKind, Rect


def outputs(rows):
    labels = np.array([[r[0] for r in rows]], np.int64)
    boxes = np.array([[r[2] for r in rows]], np.float32)
    scores = np.array([[r[1] for r in rows]], np.float32)
    return [labels, boxes, scores]


def test_decode_filters_and_clamps():
    out = outputs(
        [
            (1, 0.9, (10, 20, 110, 80)),
            (1, 0.2, (10, 20, 110, 80)),  # low score
            (2, 0.8, (0, 0, 5, 5)),  # too small
            (2, 0.8, (-50, 10, 500, 100)),  # clamped to the page
            (7, 0.9, (10, 20, 110, 80)),  # unknown label
        ]
    )
    dets = decode(out, 300, 200)
    assert [d.label for d in dets] == ["text_bubble", "text_free"]
    assert dets[1].rect == Rect(0, 10, 300, 90)


def test_decode_scales_normalised_boxes_and_tolerates_short_output():
    dets = decode(outputs([(1, 0.9, (0.1, 0.1, 0.5, 0.3))]), 1000, 500)
    r = dets[0].rect
    assert (r.x, r.y, r.w, r.h) == pytest.approx((100, 50, 400, 100))
    assert decode([np.zeros(1)], 10, 10) == []


def test_nms_is_per_label():
    a = Detection("text_bubble", 0.9, Rect(0, 0, 100, 100))
    b = Detection("text_bubble", 0.8, Rect(5, 5, 100, 100))
    c = Detection("bubble", 0.7, Rect(0, 0, 100, 100))
    assert nms([a, b, c]) == [a, c]


def test_preprocess_shapes():
    tensor, sizes = preprocess(np.zeros((300, 200, 3), np.uint8))
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert sizes.tolist() == [[200, 300]]


def test_tiles_cover_the_page_and_short_pages_are_one_tile():
    assert plan_tiles(1500) == [(0, 1500, 0, 1500)]
    tiles = plan_tiles(6000)
    assert tiles[0][0] == 0 and tiles[-1][1] == 6000
    for (_, end, _, _), (start, _, _, _) in zip(tiles, tiles[1:], strict=False):
        assert start <= end


def test_text_in_the_overlap_is_found_once():
    tiles = plan_tiles(5000)
    (_, c0e, k0s, _), (_, _, k1s, _) = tiles[0], tiles[1]
    page_y = c0e - 20  # centre near the end of tile 0's core
    d0 = Detection("text_bubble", 0.9, Rect(10, page_y - k0s - 30, 100, 60))
    d1 = Detection("text_bubble", 0.8, Rect(10, page_y - k1s - 30, 100, 60))
    merged = merge_tiles([[d0], [d1]] + [[]] * (len(tiles) - 2), tiles)
    assert len(merged) == 1 and merged[0].rect.y == page_y - 30


def test_reading_order_manga_and_western():
    rects = [Rect(0, 0, 50, 50), Rect(100, 5, 50, 50), Rect(40, 200, 50, 50)]
    assert reading_order(rects) == [1, 0, 2]
    assert reading_order(rects, rtl=False) == [0, 1, 2]


def test_blocks_get_order_kind_and_smallest_balloon():
    dets = [
        Detection("bubble", 0.9, Rect(0, 0, 400, 400)),
        Detection("bubble", 0.9, Rect(50, 50, 150, 150)),
        Detection("text_bubble", 0.9, Rect(80, 80, 60, 90)),
        Detection("text_free", 0.9, Rect(300, 300, 60, 40)),
    ]
    blocks = to_blocks(dets)
    assert [b.order for b in blocks] == [0, 1]
    assert blocks[0].kind == BlockKind.DIALOGUE and blocks[1].kind == BlockKind.FREE_TEXT
    assert blocks[0].balloon[0] == (50, 50)
    assert blocks[1].balloon[0] == (0, 0)
    assert blocks[0].vertical is False


class FakeSession:
    class Item:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self.Item("images"), self.Item("orig_target_sizes")]

    def run(self, _o, feeds):
        assert feeds["images"].shape == (1, 3, 640, 640)
        assert feeds["orig_target_sizes"].dtype == np.int64
        return outputs([(1, 0.9, (10, 10, 100, 60))])


def test_detector_runs_one_tile_for_a_normal_page():
    dets = Detector(FakeSession()).detect(np.zeros((400, 300, 3), np.uint8))
    assert len(dets) == 1 and dets[0].label == "text_bubble"


def test_false_positives_are_thin_strips_not_text():
    from mangatr.detect import drop_false_positives

    lace = Detection("text_free", 0.9, Rect(0, 0, 700, 30))
    border = Detection("text_free", 0.9, Rect(0, 0, 12, 300))
    column = Detection("text_bubble", 0.9, Rect(0, 0, 30, 150))  # vertical Japanese
    bubble = Detection("bubble", 0.9, Rect(0, 0, 700, 30))
    weak = Detection("text_free", 0.4, Rect(0, 0, 80, 40))
    kept = drop_false_positives([lace, border, column, bubble, weak], 720, min_text_free_score=0.5)
    assert kept == [column, bubble]
