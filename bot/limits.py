"""Per-user limits: a cooldown between jobs, a daily page quota, one job at a time.

The old engine had credits only and no cooldown or concurrency cap (clean-bot
the same); here all three apply. Time is injected so tests do not sleep.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

DAY = 86400.0


class LimitError(Exception):
    """The request is refused; the message is safe to show the user."""


@dataclass(slots=True)
class _User:
    last_job: float = float("-inf")
    window_start: float = 0.0
    pages_today: int = 0
    running: int = 0


@dataclass(slots=True)
class Limits:
    cooldown: float = 60.0
    daily_pages: int = 60
    max_pages_per_job: int = 40
    max_running_per_user: int = 1
    clock: Callable[[], float] = time.monotonic
    _users: dict[int, _User] = field(default_factory=dict)

    def _user(self, user_id: int) -> _User:
        user = self._users.setdefault(user_id, _User(window_start=self.clock()))
        if self.clock() - user.window_start >= DAY:
            user.window_start, user.pages_today = self.clock(), 0
        return user

    def admit(self, user_id: int, pages: int) -> None:
        """Reserve a job or raise LimitError. Call `finish` when the job ends."""
        user = self._user(user_id)
        if pages < 1:
            raise LimitError("no pages were found in the upload")
        if pages > self.max_pages_per_job:
            raise LimitError(f"at most {self.max_pages_per_job} pages per job (got {pages})")
        if user.running >= self.max_running_per_user:
            raise LimitError("you already have a job running; wait for it to finish")
        wait = user.last_job + self.cooldown - self.clock()
        if wait > 0:
            raise LimitError(f"wait {int(wait) + 1}s before sending another job")
        if user.pages_today + pages > self.daily_pages:
            left = max(0, self.daily_pages - user.pages_today)
            raise LimitError(f"daily limit reached: {left} pages left today")
        user.running += 1
        user.pages_today += pages
        user.last_job = self.clock()

    def finish(self, user_id: int, pages_done: int | None = None, pages: int = 0) -> None:
        """Release the slot. Pages that were not processed are given back."""
        user = self._user(user_id)
        user.running = max(0, user.running - 1)
        if pages_done is not None:
            user.pages_today = max(0, user.pages_today - max(0, pages - pages_done))
