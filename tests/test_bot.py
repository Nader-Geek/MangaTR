import asyncio
import io
import zipfile

import cv2
import numpy as np
import pytest

from bot.delivery import parts
from bot.jobs import JobQueue, Status
from bot.limits import LimitError, Limits
from bot.safety import Upload, UploadError, extract


def png(width=40, height=30):
    ok, buffer = cv2.imencode(".png", np.full((height, width, 3), 200, np.uint8))
    return buffer.tobytes()


def make_zip(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return stream.getvalue()


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_limits_cooldown_quota_and_one_job_at_a_time():
    clock = Clock()
    limits = Limits(cooldown=60, daily_pages=10, max_pages_per_job=8, clock=clock)
    limits.admit(1, 5)
    with pytest.raises(LimitError, match="already have a job"):
        limits.admit(1, 1)
    limits.finish(1, 5, 5)
    with pytest.raises(LimitError, match="wait"):
        limits.admit(1, 1)
    clock.now += 61
    with pytest.raises(LimitError, match="daily limit"):
        limits.admit(1, 6)
    limits.admit(1, 5)
    limits.finish(1, 5, 5)
    clock.now += 100000  # a new day
    limits.admit(1, 8)
    with pytest.raises(LimitError, match="at most"):
        limits.admit(2, 9)
    with pytest.raises(LimitError, match="no pages"):
        limits.admit(3, 0)


def test_unprocessed_pages_are_given_back():
    clock = Clock()
    limits = Limits(cooldown=0, daily_pages=10, clock=clock)
    limits.admit(1, 10)
    limits.finish(1, pages_done=2, pages=10)
    limits.admit(1, 8)  # 8 given back


def test_extract_flattens_names_orders_naturally_and_skips_non_images(tmp_path):
    data = make_zip(
        [
            ("ch/10.png", png()),
            ("ch/2.png", png()),
            ("../evil.png", png()),
            ("ch/notes.txt", b"x"),
            ("ch/.hidden.png", png()),
        ]
    )
    paths = extract([Upload("c.cbz", data)], tmp_path / "in")
    assert [p.name for p in paths] == ["001.png", "002.png", "003.png"]
    assert all(p.parent == tmp_path / "in" for p in paths)


@pytest.mark.parametrize(
    "upload",
    [
        Upload("a.exe", b"x"),
        Upload("a.zip", b"not a zip"),
        Upload("a.png", b"not an image"),
        Upload("a.png", png(width=50000, height=1500)),
    ],
)
def test_bad_uploads_are_refused(tmp_path, upload):
    with pytest.raises(UploadError):
        extract([upload], tmp_path / "in")


def test_too_many_images_are_refused(tmp_path):
    data = make_zip([(f"{i}.png", png()) for i in range(5)])
    with pytest.raises(UploadError, match="more than"):
        extract([Upload("a.zip", data)], tmp_path / "in", max_pages=3)


def test_delivery_splits_when_over_the_limit(tmp_path):
    images = []
    for i in range(4):
        rng = np.random.default_rng(i)
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), rng.integers(0, 255, (200, 200, 3), np.uint8))
        images.append(path)
    one = parts(images, tmp_path / "a", limit=10**9)
    assert len(one) == 1
    many = parts(images, tmp_path / "b", limit=60_000)
    assert len(many) > 1
    assert all(p.stat().st_size <= 70_000 for p in many)


def test_queue_runs_jobs_in_order_and_survives_a_failure_and_a_timeout():
    async def scenario():
        finished = []

        def work(job):
            if job.payload == "boom":
                raise RuntimeError("model crashed")
            if job.payload == "slow":
                import time

                time.sleep(0.3)
            return []

        async def done(job):
            finished.append((job.id, job.status, job.error))

        queue = JobQueue(work, done, timeout=0.1, max_waiting=3)
        queue.start()
        first = queue.submit(1, 1, "ok")
        queue.submit(1, 1, "boom")
        queue.submit(2, 1, "slow")
        assert queue.position(first) == 1
        with pytest.raises(RuntimeError, match="full"):
            queue.submit(3, 1, "x")
        for _ in range(100):
            await asyncio.sleep(0.05)
            if len(finished) == 3:
                break
        await queue.stop()
        return finished

    finished = asyncio.run(scenario())
    assert [f[1] for f in finished] == [Status.DONE, Status.FAILED, Status.FAILED]
    assert "model crashed" in finished[1][2] and "too long" in finished[2][2]
