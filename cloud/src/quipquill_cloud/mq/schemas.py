"""Wire format between the app and the cloud. Spec: docs/architecture/protocol.md."""

import json
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PROTOCOL_VERSION = 1

Id = Annotated[str, Field(min_length=1, max_length=128)]
Purpose = Literal["gm", "guard", "npc", "summary", "character", "generator"]


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    v: Literal[1] = PROTOCOL_VERSION


# --- client → server -------------------------------------------------------------------------


class LlmRequest(_Message):
    type: Literal["llm.request"]
    request_id: Id = Field(alias="requestId")
    session_id: Id = Field(alias="sessionId")
    channel: Id
    purpose: Purpose
    # The model class the client wants (ADR 50); the server maps it to a model and providers.
    model: str = "basic"
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = Field(default=None, alias="responseFormat")
    meta: dict[str, Any] | None = None


class Auth(_Message):
    """First message on a new socket (ADR 22): the token never travels in the URL."""

    type: Literal["auth"]
    token: str = Field(min_length=1, max_length=4096)


class Resume(_Message):
    type: Literal["resume"]
    session_id: Id = Field(alias="sessionId")


class Cancel(_Message):
    type: Literal["cancel"]
    request_id: Id = Field(alias="requestId")


class Ping(_Message):
    type: Literal["ping"]


ClientMessage = Annotated[LlmRequest | Resume | Cancel | Ping, Field(discriminator="type")]
client_message = TypeAdapter(ClientMessage)


# --- inside the cloud --------------------------------------------------------------------------


class Command(BaseModel):
    """An LLM request on its way to a worker. User and tier come from the JWT, not the client."""

    command_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str
    tier: str
    request: LlmRequest


# --- server → client (also the payload of qq.events) -----------------------------------------


class Event(BaseModel):
    """One event of a generation. `user_id` routes it inside the cloud and is never sent out."""

    model_config = ConfigDict(populate_by_name=True)

    v: Literal[1] = PROTOCOL_VERSION
    type: Literal["ready", "status", "delta", "tool_call", "done", "error", "replay", "heartbeat"]
    user_id: str | None = Field(default=None, exclude=True)
    request_id: str | None = Field(default=None, alias="requestId")
    session_id: str | None = Field(default=None, alias="sessionId")
    channel: str | None = None
    seq: int | None = None
    status: Literal["queued", "thinking"] | None = None
    text: str | None = None
    call: dict[str, Any] | None = None
    message: dict[str, Any] | None = None
    finish_reason: str | None = Field(default=None, alias="finishReason")
    usage: dict[str, Any] | None = None
    code: str | None = None
    status_key: str | None = Field(default=None, alias="statusKey")
    retryable: bool | None = None
    meta: dict[str, Any] | None = None

    def wire(self) -> dict[str, Any]:
        """JSON for the client: camelCase, no internal routing fields, no empty fields."""
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")

    def internal(self) -> bytes:
        """JSON for the events queue, including the routing user id."""
        data = self.model_dump(by_alias=True, exclude_none=True, mode="json")
        data["userId"] = self.user_id
        return _json(data)

    @classmethod
    def from_internal(cls, body: bytes) -> "Event":
        data = _loads(body)
        user_id = data.pop("userId", None)
        return cls.model_validate({**data, "user_id": user_id})


def error_event(
    code: str,
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    channel: str | None = None,
    retryable: bool = False,
) -> Event:
    return Event(
        type="error",
        user_id=user_id,
        request_id=request_id,
        session_id=session_id,
        channel=channel,
        code=code,
        status_key=f"error.{code}",
        retryable=retryable,
    )


def _json(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()


def _loads(body: bytes) -> dict[str, Any]:
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("event body is not a JSON object")
    return data
