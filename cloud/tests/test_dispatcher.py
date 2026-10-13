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
