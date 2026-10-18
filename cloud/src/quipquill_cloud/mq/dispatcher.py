"""Delivers generation events to WebSocket subscribers and remembers them for `resume`.

Sessions are keyed by (user_id, session_id): a client can never subscribe to another user's
session even if it guesses the id. Delivery and replay for one session happen under one FIFO
lock, so events keep their order and a reconnecting client neither loses nor duplicates one.

A session's state is dropped once nothing references it (no lock holders or waiters, no
subscribers, nothing in flight or finished within the TTL), so memory follows live sessions.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol

from quipquill_cloud.mq.schemas import Address, Event

logger = logging.getLogger(__name__)

SessionKey = tuple[str, str]  # (user_id, session_id)


class Subscriber(Protocol):
    async def send(self, event: Event) -> None: ...


@dataclass
class _Generation:
    address: Address
    meta: dict | None
    text: str = ""
    updated: float = field(default_factory=time.monotonic)


@dataclass
class _Finished:
    event: Event
    at: float


@dataclass
class _Session:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Coroutines holding or waiting for `lock`: the session must not be dropped under them.
    users: int = 0
    subscribers: set[Subscriber] = field(default_factory=set)
    active: dict[str, _Generation] = field(default_factory=dict)
    finished: dict[str, _Finished] = field(default_factory=dict)

    def expire(self, deadline: float) -> None:
        """Forget generations whose worker went silent and results older than the TTL."""
        for request_id in [r for r, g in self.active.items() if g.updated < deadline]:
            del self.active[request_id]
        for request_id in [r for r, f in self.finished.items() if f.at < deadline]:
            del self.finished[request_id]

    def idle(self) -> bool:
        return not (self.users or self.subscribers or self.active or self.finished)


class EventDispatcher:
    def __init__(
        self,
        generation_ttl: float,
        clock: Callable[[], float] = time.monotonic,
        sweep_interval: float = 60.0,
    ) -> None:
        self._ttl = generation_ttl
        self._clock = clock
        self._sweep_interval = sweep_interval
        self._last_sweep = clock()
        self._sessions: dict[SessionKey, _Session] = {}

    async def handle(self, event: Event) -> None:
        """An event from the queue: remember it, then deliver to the session's subscribers."""
        if not event.user_id or not event.session_id:
            logger.warning("Dropping event without routing info: %s", event.type)
            return
        async with self._locked((event.user_id, event.session_id)) as session:
            self._record(session, event)
            for subscriber in list(session.subscribers):
                try:
                    await subscriber.send(event)
                except Exception:
                    logger.info("Subscriber gone; unsubscribing from %s", event.session_id)
                    session.subscribers.discard(subscriber)
        self._maybe_sweep()

    async def subscribe(self, key: SessionKey, subscriber: Subscriber, *, replay: bool) -> None:
        """Start delivering a session's events; with `replay`, first send what is in flight."""
        async with self._locked(key) as session:
            if replay:
                for event in self._snapshot(session):
                    await subscriber.send(event)
            session.subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        for key, session in list(self._sessions.items()):
            session.subscribers.discard(subscriber)
            self._drop_if_idle(key, session)

    def has_active(self, key: SessionKey) -> bool:
        session = self._sessions.get(key)
        if session is None:
            return False
        session.expire(self._deadline())
        return bool(session.active)

    def session_count(self) -> int:
        """Sessions held in memory (for tests and diagnostics)."""
        return len(self._sessions)

    @asynccontextmanager
    async def _locked(self, key: SessionKey) -> AsyncIterator[_Session]:
        session = self._sessions.setdefault(key, _Session())
        session.users += 1
        try:
            async with session.lock:
                session.expire(self._deadline())
                yield session
        finally:
            session.users -= 1
            self._drop_if_idle(key, session)

    def _drop_if_idle(self, key: SessionKey, session: _Session) -> None:
        if session.idle() and self._sessions.get(key) is session:
            del self._sessions[key]

    def _maybe_sweep(self) -> None:
        """Drop sessions nobody touched since their last results expired."""
        now = self._clock()
        if now - self._last_sweep < self._sweep_interval:
            return
        self._last_sweep = now
        deadline = now - self._ttl
        for key, session in list(self._sessions.items()):
            if session.users == 0:
                session.expire(deadline)
                self._drop_if_idle(key, session)

    def _deadline(self) -> float:
        return self._clock() - self._ttl

    def _record(self, session: _Session, event: Event) -> None:
        request_id = event.request_id
        if request_id is None:
            return
        now = self._clock()
        if event.type in ("status", "delta", "tool_call"):
            generation = session.active.setdefault(
                request_id, _Generation(event.address(), event.meta)
            )
            if event.type == "delta" and event.text:
                generation.text += event.text
            if event.meta and generation.meta is None:
                generation.meta = event.meta
            generation.updated = now
        elif event.type in ("done", "error"):
            session.active.pop(request_id, None)
            session.finished[request_id] = _Finished(event, now)

    @staticmethod
    def _snapshot(session: _Session) -> list[Event]:
        events = []
        for generation in session.active.values():
            common = {**generation.address, "meta": generation.meta}
            if generation.text:
                events.append(Event(type="replay", text=generation.text, **common))
            else:
                events.append(Event(type="status", status="thinking", **common))
        # Finished results the client may have missed while disconnected.
        events.extend(finished.event for finished in session.finished.values())
        return events
