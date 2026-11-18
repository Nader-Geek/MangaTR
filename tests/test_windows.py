from mangatr.scene import Block, BlockKind, Rect
from mangatr.windows import plan_windows


def block(block_id, x, y, w=100, h=100, panel=None):
    return Block(id=block_id, kind=BlockKind.DIALOGUE, frame=Rect(x, y, w, h), panel=panel)


def test_window_is_frame_plus_context_clamped_to_the_page():
    (window,) = plan_windows([block("a", 10, 20)], 1000, 1000, context=50)
    assert (window.rect.x, window.rect.y) == (0, 0)  # 10-50 and 20-50 clamp to 0
    assert (window.rect.right, window.rect.bottom) == (160, 170)
    assert window.block_ids == ("a",)


def test_far_apart_blocks_stay_in_separate_windows():
    windows = plan_windows([block("a", 0, 0), block("b", 5000, 5000)], 9000, 9000, context=50)
    assert [w.block_ids for w in windows] == [("a",), ("b",)]


def test_overlapping_windows_merge():
    windows = plan_windows([block("a", 100, 100), block("b", 250, 100)], 2000, 2000, context=50)
    assert len(windows) == 1
    assert set(windows[0].block_ids) == {"a", "b"}
    assert windows[0].rect.x == 50
    assert windows[0].rect.right == 400


def test_merge_is_refused_over_the_pixel_budget():
    blocks = [block("a", 100, 100), block("b", 250, 100)]
    windows = plan_windows(blocks, 2000, 2000, context=50, max_pixels=30_000)
    assert len(windows) == 2


def test_a_window_over_the_budget_is_not_split():
    (window,) = plan_windows([block("a", 0, 0, 900, 900)], 2000, 2000, context=50, max_pixels=10)
    assert window.block_ids == ("a",)


def test_windows_never_span_two_panels():
    blocks = [block("a", 100, 100, panel="p1"), block("b", 250, 100, panel="p2")]
    assert len(plan_windows(blocks, 2000, 2000, context=50)) == 2


def test_a_block_with_no_panel_merges_with_either_panel():
    blocks = [block("a", 100, 100, panel="p1"), block("b", 250, 100, panel=None)]
    (window,) = plan_windows(blocks, 2000, 2000, context=50)
    assert window.panel == "p1"


def test_chain_of_overlaps_ends_as_one_window():
    blocks = [block(c, x, 100) for c, x in (("a", 100), ("b", 250), ("c", 400))]
    (window,) = plan_windows(blocks, 2000, 2000, context=50)
    assert set(window.block_ids) == {"a", "b", "c"}


def test_plan_is_reproducible_whatever_the_input_order():
    blocks = [block("c", 5000, 10), block("a", 0, 0), block("b", 2500, 5)]
    first = plan_windows(blocks, 9000, 9000, context=50)
    second = plan_windows(list(reversed(blocks)), 9000, 9000, context=50)
    assert first == second
