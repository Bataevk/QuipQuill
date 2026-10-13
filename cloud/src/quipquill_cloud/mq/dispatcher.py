"""Delivers generation events to WebSocket subscribers and remembers them for `resume`.

Sessions are keyed by (user_id, session_id): a client can never subscribe to another user's
session even if it guesses the id. Delivery and replay for one session happen under one FIFO
lock, so events keep their order and a reconnecting client neither loses nor duplicates one.
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from quipquill_cloud.mq.schemas import Event

logger = logging.getLogger(__name__)

SessionKey = tuple[str, str]  # (user_id, session_id)


class Subscriber(Protocol):
    async def send(self, event: Event) -> None: ...


@dataclass
class _Generation:
    channel: str | None
    meta: dict | None
    text: str = ""
    updated: float = field(default_factory=time.monotonic)


@dataclass
class _Finished:
    event: Event
    at: float


class EventDispatcher:
    def __init__(self, generation_ttl: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = generation_ttl
        self._clock = clock
        self._subscribers: dict[SessionKey, set[Subscriber]] = defaultdict(set)
        self._locks: dict[SessionKey, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._active: dict[SessionKey, dict[str, _Generation]] = defaultdict(dict)
        self._finished: dict[SessionKey, dict[str, _Finished]] = defaultdict(dict)

    async def handle(self, event: Event) -> None:
        """An event from the queue: remember it, then deliver to the session's subscribers."""
        if not event.user_id or not event.session_id:
            logger.warning("Dropping event without routing info: %s", event.type)
            return
        key = (event.user_id, event.session_id)
        async with self._locks[key]:
            self._record(key, event)
            for subscriber in list(self._subscribers.get(key, ())):
                try:
                    await subscriber.send(event)
                except Exception:
                    logger.info("Subscriber gone; unsubscribing from %s", key[1])
                    self._subscribers[key].discard(subscriber)

    async def subscribe(self, key: SessionKey, subscriber: Subscriber, *, replay: bool) -> None:
        """Start delivering a session's events; with `replay`, first send what is in flight."""
        async with self._locks[key]:
            self._expire(key)
            if replay:
                for event in self._snapshot(key):
                    await subscriber.send(event)
            self._subscribers[key].add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        for subscribers in self._subscribers.values():
            subscribers.discard(subscriber)

    def has_active(self, key: SessionKey) -> bool:
        self._expire(key)
        return bool(self._active.get(key))

    def _record(self, key: SessionKey, event: Event) -> None:
        request_id = event.request_id
        if request_id is None:
            return
        now = self._clock()
        active = self._active[key]
        if event.type in ("status", "delta", "tool_call"):
            generation = active.setdefault(request_id, _Generation(event.channel, event.meta))
            if event.type == "delta" and event.text:
                generation.text += event.text
            if event.meta and generation.meta is None:
                generation.meta = event.meta
            generation.updated = now
        elif event.type in ("done", "error"):
            active.pop(request_id, None)
            self._finished[key][request_id] = _Finished(event, now)
        self._expire(key)

    def _snapshot(self, key: SessionKey) -> list[Event]:
        user_id, session_id = key
        events = []
        for request_id, generation in self._active.get(key, {}).items():
            common = {
                "user_id": user_id,
                "request_id": request_id,
                "session_id": session_id,
                "channel": generation.channel,
                "meta": generation.meta,
            }
            if generation.text:
                events.append(Event(type="replay", text=generation.text, **common))
            else:
                events.append(Event(type="status", status="thinking", **common))
        # A finished result the client may have missed while disconnected.
        events.extend(finished.event for finished in self._finished.get(key, {}).values())
        return events

    def _expire(self, key: SessionKey) -> None:
        """Forget generations whose worker went silent and results older than the TTL."""
        deadline = self._clock() - self._ttl
        active = self._active.get(key, {})
        for request_id in [r for r, g in active.items() if g.updated < deadline]:
            del active[request_id]
        finished = self._finished.get(key, {})
        for request_id in [r for r, f in finished.items() if f.at < deadline]:
            del finished[request_id]
