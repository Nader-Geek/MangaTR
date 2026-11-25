import numpy as np
import pytest

from mangatr.detect import (
    Detection,
    Detector,
    decode,
    holding_bubble,
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


def test_the_same_lettering_under_both_text_labels_is_kept_once():
    free = Detection("text_free", 0.70, Rect(118, 5310, 576, 123))
    bubble = Detection("text_bubble", 0.61, Rect(118, 5310, 576, 123))
    assert nms([bubble, free]) == [free]
    # a text box inside a bubble box is still a different thing
    holder = Detection("bubble", 0.5, Rect(100, 5300, 600, 150))
    assert nms([free, holder]) == [free, holder]


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
    weak = Detection("text_free", 0.42, Rect(375, 9935, 320, 163))  # vest ornament
    strip = Detection("text_free", 0.6, Rect(0, 5357, 720, 111))  # lace across the page
    real = Detection("text_free", 0.72, Rect(326, 4442, 67, 172))
    kept = drop_false_positives([lace, border, column, bubble, weak, strip, real], 720)
    assert kept == [column, bubble, real]


def test_fragments_and_merges_of_a_text_box_are_suppressed_but_bubbles_still_nest():
    whole = Detection("text_bubble", 0.8, Rect(100, 100, 100, 300))
    twin = Detection("text_bubble", 0.5, Rect(105, 100, 100, 290))  # a duplicate
    fragment = Detection("text_bubble", 0.4, Rect(150, 110, 40, 150))  # inside the whole
    merged = Detection("text_bubble", 0.3, Rect(40, 100, 160, 320))  # whole plus an aside
    aside = Detection("text_bubble", 0.35, Rect(40, 300, 50, 120))
    bubble = Detection("bubble", 0.9, Rect(90, 90, 120, 320))
    outer_bubble = Detection("bubble", 0.8, Rect(0, 0, 400, 500))  # nests in nothing smaller
    inner_bubble = Detection("bubble", 0.7, Rect(100, 100, 100, 100))
    kept = nms([whole, twin, fragment, merged, aside, bubble, outer_bubble, inner_bubble])
    assert whole in kept and aside in kept
    assert twin not in kept and fragment not in kept and merged not in kept
    assert bubble in kept and outer_bubble in kept and inner_bubble in kept


def test_a_text_is_held_by_the_bubble_with_most_of_its_box_not_by_its_centre():
    big = Rect(0, 0, 300, 300)
    small = Rect(40, 40, 120, 120)
    text = Rect(60, 60, 60, 60)  # fully in both: the smaller one wins
    assert holding_bubble(text, [big, small]) == small
    edge_remark = Rect(280, 100, 60, 40)  # mostly outside: 20/60 inside the big bubble
    assert holding_bubble(edge_remark, [big]) is None


class SizeDependentSession(FakeSession):
    """Finds the first text at any size and a second one only in the downscaled read."""

    def run(self, _o, feeds):
        w, h = (int(v) for v in feeds["orig_target_sizes"][0])
        boxes = [(1, 0.9, (10, 10, 100, 60))]
        if w < 300:
            boxes.append((1, 0.8, (150, 100, 100, 60)))
        return outputs(boxes)


def test_second_read_at_another_size_adds_what_the_first_missed():
    page = np.zeros((400, 300, 3), np.uint8)
    one = Detector(SizeDependentSession(), second_scale=None).detect(page)
    two = Detector(SizeDependentSession(), second_scale=0.77).detect(page)
    assert len(one) == 1 and len(two) == 2
    assert all(0 <= d.rect.x < 300 and d.rect.w > 0 for d in two)  # back in page pixels
