"""
VideoSpec — the shared, render-engine-agnostic description of a Remotion video.

This is the Python half of a two-sided contract: the pydantic models here
mirror the zod schema in ``remotion/src/schema.ts``. Both halves describe the
same JSON document so either side can validate it. The Python side *produces*
specs (Adapter A turns one ``VideoTask`` into a single-clip spec; Generator B
turns an LLM response into a multi-clip spec); the Node side *renders* them.

Design
======
- One spec, two producers. A ``VideoSpec`` with exactly one ``Clip`` is a
  per-frame "Ken Burns" motion clip (flow A, slots into ``merge_videos``); a
  ``VideoSpec`` with many clips is a whole motion-graphics video (flow B).
- Layers are a discriminated union on ``type`` so the LLM-produced spec in B
  validates strictly and the renderer can switch on a single field.
- Media (``KenBurnsImage.src``, ``audio.src``, ``Clip.audio_src``) is referenced
  as a **path relative to the shared output root** (e.g.
  ``video_inputs/<id>.png``). The render service resolves these against the
  ``output/`` directory it serves statically. ``http(s)://`` and ``data:`` URLs
  are passed through untouched.

Keep this file declarative — no rendering logic, no I/O.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated


# ─────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────


class Focus(BaseModel):
    """A Ken Burns keyframe: zoom + normalized pan offset (0,0 = centered)."""

    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0


class SubtitleCue(BaseModel):
    text: str
    from_sec: float = 0.0
    to_sec: float


class TransitionIn(BaseModel):
    """How a clip enters relative to the previous one."""

    type: Literal["fade", "slide", "wipe", "none"] = "fade"
    duration_sec: float = 0.5
    direction: Optional[Literal["from-left", "from-right", "from-top", "from-bottom"]] = None


class AudioTrack(BaseModel):
    """A global audio track (e.g. background music) spanning the whole video."""

    src: str
    from_sec: float = 0.0
    volume: float = 1.0
    loop: bool = False


# ─────────────────────────────────────────────────────────────────────────
# Layers (discriminated union on ``type``)
# ─────────────────────────────────────────────────────────────────────────


class KenBurnsImageLayer(BaseModel):
    type: Literal["kenburns_image"] = "kenburns_image"
    src: str
    from_: Focus = Field(default_factory=Focus, alias="from")
    to: Focus = Field(default_factory=lambda: Focus(scale=1.12))
    easing: Literal["linear", "ease", "ease-in", "ease-out", "ease-in-out"] = "ease-in-out"
    fit: Literal["cover", "contain"] = "cover"

    model_config = {"populate_by_name": True}


class TitleCardLayer(BaseModel):
    type: Literal["title_card"] = "title_card"
    text: str
    subtitle: Optional[str] = None
    align: Literal["center", "left", "right"] = "center"
    color: str = "#ffffff"
    font_size: int = 80
    enter: Literal["fade", "rise", "none"] = "rise"


class SubtitleLayer(BaseModel):
    type: Literal["subtitle"] = "subtitle"
    cues: List[SubtitleCue] = Field(default_factory=list)
    position: Literal["bottom", "top", "center"] = "bottom"
    color: str = "#ffffff"
    font_size: int = 44
    box: bool = True  # render a translucent backing box behind the text


class BulletListLayer(BaseModel):
    type: Literal["bullet_list"] = "bullet_list"
    title: Optional[str] = None
    items: List[str] = Field(default_factory=list)
    color: str = "#ffffff"
    stagger_sec: float = 0.4  # delay between each item appearing


class BackgroundLayer(BaseModel):
    type: Literal["background"] = "background"
    color: Optional[str] = None
    gradient: Optional[List[str]] = None  # e.g. ["#0f172a", "#1e293b"]


Layer = Annotated[
    Union[
        KenBurnsImageLayer,
        TitleCardLayer,
        SubtitleLayer,
        BulletListLayer,
        BackgroundLayer,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────────────────────
# Clip + top-level spec
# ─────────────────────────────────────────────────────────────────────────


class Clip(BaseModel):
    """One timeline segment: a stack of composited layers over a duration."""

    duration_sec: float = Field(5.0, gt=0)
    layers: List[Layer] = Field(default_factory=list)
    transition_in: Optional[TransitionIn] = None
    audio_src: Optional[str] = None  # per-clip audio, e.g. dialogue (output-relative)


class VideoSpec(BaseModel):
    """Render-engine-agnostic description of a full video."""

    fps: int = 30
    width: int = 1080
    height: int = 1920
    clips: List[Clip] = Field(default_factory=list)
    audio: List[AudioTrack] = Field(default_factory=list)

    def to_payload(self) -> dict:
        """Serialize to the JSON shape the render service expects.

        ``by_alias`` emits ``from`` (not ``from_``); ``exclude_none`` keeps the
        payload small and lets the zod schema apply its own defaults.
        """
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")

    @property
    def total_duration_sec(self) -> float:
        return sum(c.duration_sec for c in self.clips)


__all__ = [
    "Focus",
    "SubtitleCue",
    "TransitionIn",
    "AudioTrack",
    "KenBurnsImageLayer",
    "TitleCardLayer",
    "SubtitleLayer",
    "BulletListLayer",
    "BackgroundLayer",
    "Layer",
    "Clip",
    "VideoSpec",
]
