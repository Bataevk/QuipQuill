"""The story catalogue for the app (ADR 61): the list, and the file to download and import."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from quipquill_cloud.api.deps import CurrentUser
from quipquill_cloud.services import Services, get_services

router = APIRouter(prefix="/stories", tags=["stories"])

ServicesDep = Annotated[Services, Depends(get_services)]
# An SQLite file (schema/story.sql); the app does not look at the type, only at the bytes.
MEDIA_TYPE = "application/vnd.sqlite3"


class Story(BaseModel):
    id: str
    version: str
    title: str
    tagline: str
    summary: str
    heroes: list[dict[str, Any]]
    size: int


@router.get("")
def list_stories(_user: CurrentUser, services: ServicesDep) -> list[Story]:
    return [
        Story(
            id=entry.id,
            version=entry.version,
            title=entry.title,
            tagline=entry.tagline,
            summary=entry.summary,
            heroes=entry.heroes,
            size=entry.size,
        )
        for entry in services.stories.entries()
    ]


@router.get("/{story_id}/file")
def story_file(story_id: str, _user: CurrentUser, services: ServicesDep) -> FileResponse:
    entry = services.stories.get(story_id)
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="story_not_found")
    return FileResponse(
        entry.path,
        media_type=MEDIA_TYPE,
        filename=f"{entry.id}.qqstory",
        headers={"ETag": f'"{entry.version}"'},
    )
