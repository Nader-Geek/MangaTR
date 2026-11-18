import pytest

from mangatr.pipeline import Context, StageError, run_page
from mangatr.scene import Page


class Recorder:
    def __init__(self, name, calls, fail=False):
        self.name = name
        self.calls = calls
        self.fail = fail

    def run(self, page, ctx):
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError("model crashed")


def make():
    return Page(id="p1", image="p1.png", width=10, height=10), Context(
        workdir=None, target_lang="ar"
    )


def test_stages_run_in_the_given_order():
    page, ctx = make()
    calls = []
    run_page(page, [Recorder("detect", calls), Recorder("ocr", calls)], ctx)
    assert calls == ["detect", "ocr"]


def test_failure_names_stage_and_page_and_stops_later_stages():
    page, ctx = make()
    calls = []
    stages = [
        Recorder("detect", calls),
        Recorder("ocr", calls, fail=True),
        Recorder("translate", calls),
    ]
    with pytest.raises(StageError) as info:
        run_page(page, stages, ctx)
    assert (info.value.stage, info.value.page_id) == ("ocr", "p1")
    assert isinstance(info.value.__cause__, RuntimeError)
    assert calls == ["detect", "ocr"]
