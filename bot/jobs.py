"""A single worker running jobs one at a time (a GPU is one machine), with a reaper.

A job that runs longer than `timeout` is marked failed and its slot freed, so a
stuck job never blocks the queue (clean-bot had no reaper). `work` is a
blocking function run in a thread; it returns the result files. A thread cannot be
killed, so a timed-out job's thread runs on until it ends; the queue only stops waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Status(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class Job:
    id: int
    user_id: int
    pages: int
    payload: object
    status: Status = Status.QUEUED
    result: list[Path] = field(default_factory=list)
    error: str | None = None
    started: float | None = None


class JobQueue:
    def __init__(
        self,
        work: Callable[[Job], list[Path]],
        on_finish: Callable[[Job], Awaitable[None]],
        *,
        timeout: float = 1800.0,
        max_waiting: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._work, self._on_finish = work, on_finish
        self.timeout, self.max_waiting, self._clock = timeout, max_waiting, clock
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._next = 1
        self.jobs: dict[int, Job] = {}
        self._task: asyncio.Task | None = None

    def submit(self, user_id: int, pages: int, payload: object) -> Job:
        if self._queue.qsize() >= self.max_waiting:
            raise RuntimeError("the queue is full; try again in a few minutes")
        job = Job(self._next, user_id, pages, payload)
        self._next += 1
        self.jobs[job.id] = job
        self._queue.put_nowait(job)
        return job

    def position(self, job: Job) -> int:
        """1 when it is next (or running), counting the jobs ahead of it."""
        ahead = sum(
            1
            for j in self.jobs.values()
            if j.id < job.id and j.status in (Status.QUEUED, Status.RUNNING)
        )
        return ahead + 1

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            job = await self._queue.get()
            job.status, job.started = Status.RUNNING, self._clock()
            try:
                job.result = await asyncio.wait_for(
                    asyncio.to_thread(self._work, job), timeout=self.timeout
                )
                job.status = Status.DONE
            except TimeoutError:
                job.status, job.error = Status.FAILED, "the job took too long and was stopped"
            except Exception as exc:
                job.status, job.error = Status.FAILED, str(exc)
            with contextlib.suppress(Exception):  # delivery failed; the loop must go on
                await self._on_finish(job)
