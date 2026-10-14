import asyncio

from quipquill_cloud.mq.dispatcher import EventDispatcher
from quipquill_cloud.mq.schemas import Event

KEY = ("user-1", "save-1")


class Collector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def send(self, event: Event) -> None:
        self.events.append(event)


def event(kind: str, **fields) -> Event:
    return Event(type=kind, user_id=KEY[0], session_id=KEY[1], request_id="r1", **fields)


def test_snapshot_and_ttl_expiry() -> None:
    now = [0.0]
    dispatcher = EventDispatcher(generation_ttl=10, clock=lambda: now[0])

    async def scenario() -> list[list[str]]:
        await dispatcher.handle(event("status", status="thinking", channel="gm"))
        await dispatcher.handle(event("delta", text="Hello"))
        await dispatcher.handle(event("delta", text=" there"))

        live = Collector()
        await dispatcher.subscribe(KEY, live, replay=True)
        first = [f"{e.type}:{e.text}" for e in live.events]

        await dispatcher.handle(event("done", message={"content": "Hello there"}))
        finished = Collector()
        await dispatcher.subscribe(KEY, finished, replay=True)
        second = [e.type for e in finished.events]

        now[0] = 11  # past the TTL: nothing left to replay
        expired = Collector()
        await dispatcher.subscribe(KEY, expired, replay=True)
        return [first, second, [e.type for e in expired.events]]

    first, second, third = asyncio.run(scenario())
    assert first == ["replay:Hello there"]
    assert second == ["done"]
    assert third == []
    assert not dispatcher.has_active(KEY)


def test_failing_subscriber_is_dropped() -> None:
    dispatcher = EventDispatcher(generation_ttl=10)

    class Broken:
        async def send(self, event: Event) -> None:
            raise ConnectionError("socket closed")

    good = Collector()

    async def scenario() -> None:
        await dispatcher.subscribe(KEY, Broken(), replay=False)
        await dispatcher.subscribe(KEY, good, replay=False)
        await dispatcher.handle(event("status", status="thinking"))
        await dispatcher.handle(event("delta", text="x"))

    asyncio.run(scenario())
    assert [e.type for e in good.events] == ["status", "delta"]


def test_sessions_are_dropped_when_nothing_references_them() -> None:
    now = [0.0]
    dispatcher = EventDispatcher(generation_ttl=10, clock=lambda: now[0], sweep_interval=5)
    subscriber = Collector()

    async def scenario() -> list[int]:
        counts = []
        await dispatcher.subscribe(KEY, subscriber, replay=False)
        await dispatcher.handle(event("status", status="thinking"))
        await dispatcher.handle(event("done", message={"content": "ok"}))
        dispatcher.unsubscribe(subscriber)
        counts.append(dispatcher.session_count())  # the finished result is kept for resume

        now[0] = 20  # the result expired; any later event sweeps the forgotten session
        await dispatcher.handle(
            Event(type="status", user_id="user-2", session_id="save-2", request_id="r2")
        )
        counts.append(dispatcher.session_count())  # only user-2's in-flight generation
        return counts

    assert asyncio.run(scenario()) == [1, 1]
    assert not dispatcher.has_active(KEY)
    assert dispatcher.has_active(("user-2", "save-2"))


def test_session_with_only_a_subscriber_goes_away_on_unsubscribe() -> None:
    dispatcher = EventDispatcher(generation_ttl=10)
    subscriber = Collector()

    asyncio.run(dispatcher.subscribe(KEY, subscriber, replay=True))
    assert dispatcher.session_count() == 1
    dispatcher.unsubscribe(subscriber)
    assert dispatcher.session_count() == 0
