import numpy as np

from mangatr.detect import Detector, merge_tiles
from mangatr.strips import choose_cuts, empty_rows, gap_centres, tiles_from_cuts


def strip_with_gaps(gaps, height=6000, width=200):
    page = np.random.default_rng(1).integers(0, 255, (height, width), dtype=np.uint8)
    for top, bottom, value in gaps:
        page[top:bottom] = value
    return page


def test_empty_rows_include_dark_gaps_not_only_white_ones():
    page = strip_with_gaps([(100, 140, 255), (300, 340, 20)], height=500)
    empty = empty_rows(page)
    assert empty[110:130].all() and empty[310:330].all()
    assert not empty[200:260].any()  # noise is not a gap


def test_gap_centre_is_the_middle_of_the_gap_and_short_runs_are_ignored():
    empty = np.zeros(100, bool)
    empty[10:30] = True
    empty[50:54] = True  # too short
    assert gap_centres(empty) == [19]


def test_cuts_go_to_the_gap_nearest_the_target_or_to_the_target_itself():
    cuts = choose_cuts(6000, [1500, 1700, 3300, 5000], target=1680, window=400)
    assert cuts == [1700, 3300, 5000]
    assert choose_cuts(6000, [], 1680, 400)[0] == 1680


def test_tiles_from_cuts_cover_the_page_with_overlap():
    tiles = tiles_from_cuts(5000, [1700, 3400], overlap=260)
    assert tiles[0] == (0, 1700, 0, 1960)
    assert tiles[1] == (1700, 3400, 1440, 3660)
    assert tiles[-1][1] == 5000 and tiles[-1][3] == 5000


class Item:
    def __init__(self, n):
        self.name = n


class Sees:
    def get_inputs(self):
        return [Item("images"), Item("orig_target_sizes")]

    def run(self, _o, feeds):
        return [
            np.zeros((1, 0), np.int64),
            np.zeros((1, 0, 4), np.float32),
            np.zeros((1, 0), np.float32),
        ]


def test_detector_cuts_a_long_strip_in_a_gap():
    gray = strip_with_gaps([(1700, 1760, 255)])
    rgb = np.stack([gray] * 3, axis=2)
    tiles = Detector._tiles(rgb)
    assert tiles[0][1] == 1729  # the middle of the gap at 1700..1760
    assert Detector(Sees()).detect(rgb) == []
    assert merge_tiles([[]] * len(tiles), tiles) == []
