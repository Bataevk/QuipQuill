"""RabbitMQ topology: two durable queues for the whole app (commands in, events out)."""

import asyncio
import logging

import aio_pika
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustChannel,
    AbstractRobustConnection,
)
from pydantic import ValidationError

from quipquill_cloud.mq.dispatcher import EventDispatcher
from quipquill_cloud.mq.schemas import Command, Event
from quipquill_cloud.mq.worker import CommandWorker

logger = logging.getLogger(__name__)

EVENTS_PREFETCH = 64


class Broker:
    def __init__(
        self,
        connection: AbstractRobustConnection,
        publish_channel: AbstractRobustChannel,
        prefix: str,
    ) -> None:
        self._connection = connection
        self._publish_channel = publish_channel
        self.commands_queue = f"{prefix}.commands"
        self.events_queue = f"{prefix}.events"

    @classmethod
    async def connect(cls, url: str, prefix: str, connect_timeout: float) -> "Broker":
        connection = await aio_pika.connect_robust(url, timeout=connect_timeout)
        broker = cls(connection, await connection.channel(), prefix)
        for name in (broker.commands_queue, broker.events_queue):
            await broker._publish_channel.declare_queue(name, durable=True)
        return broker

    async def ping(self, wait_seconds: float) -> bool:
        """A real round trip to the broker (a robust connection hides outages while it retries)."""
        try:
            async with asyncio.timeout(wait_seconds):
                await self._publish_channel.declare_queue(self.commands_queue, passive=True)
        except Exception:
            return False
        return True

    async def publish_command(self, command: Command) -> None:
        await self._publish(self.commands_queue, command.model_dump_json(by_alias=True).encode())

    async def publish_event(self, event: Event) -> None:
        await self._publish(self.events_queue, event.internal())

    async def _publish(self, queue: str, body: bytes) -> None:
        await self._publish_channel.default_exchange.publish(
            aio_pika.Message(
                body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue,
        )

    async def start_workers(self, worker: CommandWorker, concurrency: int) -> None:
        """Competing consumers: up to `concurrency` commands are handled at once."""
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=concurrency)
        queue = await channel.declare_queue(self.commands_queue, durable=True)
        await queue.consume(worker.on_message)

    async def start_dispatcher(self, dispatcher: EventDispatcher) -> None:
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=EVENTS_PREFETCH)
        queue = await channel.declare_queue(self.events_queue, durable=True)

        async def on_event(message: AbstractIncomingMessage) -> None:
            try:
                event = Event.from_internal(message.body)
            except (ValidationError, ValueError):
                logger.error("Rejecting malformed event: %r", message.body[:200])
                await message.reject(requeue=False)
                return
            try:
                await dispatcher.handle(event)
            finally:
                await message.ack()

        await queue.consume(on_event)

    async def delete_queues(self) -> None:
        """For tests: remove this prefix's queues."""
        channel = await self._connection.channel()
        for name in (self.commands_queue, self.events_queue):
            await channel.queue_delete(name)
        await channel.close()

    async def close(self) -> None:
        await self._connection.close()
