"""Per-user token bucket in process memory (one API instance, ADR 18)."""

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """`per_minute` requests on average, at most `burst` in a row."""

    def __init__(
        self, per_minute: float, burst: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._rate = per_minute / 60.0
        self._burst = float(burst)
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = _Bucket(self._burst, now)
        bucket.tokens = min(self._burst, bucket.tokens + (now - bucket.updated) * self._rate)
        bucket.updated = now
        if bucket.tokens < 1:
            return False
        bucket.tokens -= 1
        return True
