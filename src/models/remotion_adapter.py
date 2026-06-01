"""
RemotionVideoAdapter — flow A: drive the Remotion render service as if it were
a video model.

Slots into the same ``VideoModelDispatcher`` registry as the cloud i2v vendors
(Wan / Kling / Vidu / ...). Instead of calling a paid image-to-video API, it
turns the frame's still image into a Ken Burns "pseudo-motion" clip (camera
push/pan + optional baked-in subtitle + dialogue audio) rendered locally for
free. The output contract is identical — ``(absolute_path, seconds)`` written
to ``ctx.output_path`` — so ``merge_videos`` stitches Remotion clips and real
i2v clips on the same timeline without knowing the difference.

The motion intent comes from the storyboard frame, which the pipeline drops
into ``ctx.extras`` (``camera_movement`` / ``shot_size`` / ``dialogue`` /
``duration_seconds`` / ``aspect_ratio`` / ``dialogue_audio``). The adapter never
reaches back into the pipeline.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Optional, Tuple

from .remotion_renderer import RemotionRenderClient, get_render_client
from .remotion_spec import (
    Clip,
    Focus,
    KenBurnsImageLayer,
    SubtitleCue,
    SubtitleLayer,
    VideoSpec,
)
from .video_dispatcher import VideoAdapter, VideoGenerationContext

logger = logging.getLogger(__name__)

_REMOTE_RE = re.compile(r"^(https?:|data:|blob:)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────
# Camera movement → Ken Burns keyframes
# ─────────────────────────────────────────────────────────────────────────

# A subtle default push so a "static" shot still breathes instead of being a
# dead freeze-frame. Keyword tables cover the Chinese terms the storyboard LLM
# emits plus common English equivalents.
_ZOOM_IN = ("推", "推近", "zoom in", "push in", "push-in", "dolly in")
_ZOOM_OUT = ("拉", "拉远", "zoom out", "pull out", "pull-out", "dolly out")
_PAN_LEFT = ("摇左", "向左", "左移", "pan left", "move left")
_PAN_RIGHT = ("摇右", "向右", "右移", "pan right", "move right")
_TILT_UP = ("上摇", "向上", "上移", "tilt up", "move up", "crane up")
_TILT_DOWN = ("下摇", "向下", "下移", "tilt down", "move down", "crane down")


def _matches(text: str, keywords: Tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def camera_to_kenburns(movement: Optional[str], shot_size: Optional[str]) -> Tuple[Focus, Focus]:
    """Map a (camera_movement, shot_size) pair to Ken Burns from/to focuses.

    ``shot_size`` only nudges the zoom amount — a close-up (特写) gets a touch
    more travel than a wide (远景) so the motion reads at the intended scale.
    """
    m = (movement or "").strip().lower()
    span = 0.14
    if shot_size:
        s = shot_size
        if any(k in s for k in ("特写", "close")):
            span = 0.18
        elif any(k in s for k in ("远景", "wide", "全景")):
            span = 0.10

    if _matches(m, _ZOOM_OUT):
        return Focus(scale=1.0 + span), Focus(scale=1.0)
    if _matches(m, _PAN_LEFT):
        return Focus(scale=1.0 + span / 2, x=span / 2), Focus(scale=1.0 + span / 2, x=-span / 2)
    if _matches(m, _PAN_RIGHT):
        return Focus(scale=1.0 + span / 2, x=-span / 2), Focus(scale=1.0 + span / 2, x=span / 2)
    if _matches(m, _TILT_UP):
        return Focus(scale=1.0 + span / 2, y=span / 2), Focus(scale=1.0 + span / 2, y=-span / 2)
    if _matches(m, _TILT_DOWN):
        return Focus(scale=1.0 + span / 2, y=-span / 2), Focus(scale=1.0 + span / 2, y=span / 2)
    # zoom-in keywords AND the default both push gently in.
    return Focus(scale=1.0), Focus(scale=1.0 + span)


# ─────────────────────────────────────────────────────────────────────────
# Resolution → frame size
# ─────────────────────────────────────────────────────────────────────────


def resolution_to_size(resolution: Optional[str], aspect_ratio: Optional[str]) -> Tuple[int, int]:
    """Turn a ``"1080p"``-style resolution + ``"9:16"`` aspect into (w, h)."""
    digits = re.findall(r"\d+", resolution or "")
    base = int(digits[0]) if digits else 1080
    ar = (aspect_ratio or "9:16").strip()
    if ar == "16:9":
        return round(base * 16 / 9), base
    if ar == "1:1":
        return base, base
    # default 9:16 (vertical micro-drama)
    return base, round(base * 16 / 9)


# ─────────────────────────────────────────────────────────────────────────
# Builder — VideoTask + frame extras → single-clip VideoSpec
# ─────────────────────────────────────────────────────────────────────────


class ClipSpecBuilder:
    """Fluent builder for the single-clip spec flow A renders per frame.

    Mirrors the storyboard ``StoryboardPromptBuilder`` style: small, explicit
    ``with_*`` steps that read the domain object and accumulate layers.
    """

    def __init__(self, width: int, height: int, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._duration = 5.0
        self._layers: list = []
        self._audio_src: Optional[str] = None

    def with_duration(self, seconds: Optional[float]) -> "ClipSpecBuilder":
        if seconds and seconds > 0:
            self._duration = float(seconds)
        return self

    def with_image(
        self, src: str, movement: Optional[str], shot_size: Optional[str]
    ) -> "ClipSpecBuilder":
        frm, to = camera_to_kenburns(movement, shot_size)
        self._layers.append(KenBurnsImageLayer(src=src, from_=frm, to=to))
        return self

    def with_subtitle(self, text: Optional[str]) -> "ClipSpecBuilder":
        text = (text or "").strip()
        if text:
            self._layers.append(
                SubtitleLayer(cues=[SubtitleCue(text=text, from_sec=0.0, to_sec=self._duration)])
            )
        return self

    def with_audio(self, src: Optional[str]) -> "ClipSpecBuilder":
        if src:
            self._audio_src = src
        return self

    def build(self) -> VideoSpec:
        clip = Clip(duration_sec=self._duration, layers=self._layers, audio_src=self._audio_src)
        return VideoSpec(
            fps=self._fps, width=self._width, height=self._height, clips=[clip]
        )


# ─────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────


class RemotionVideoAdapter(VideoAdapter):
    """Renders one storyboard frame into a Ken Burns motion clip via Remotion."""

    def __init__(self, client: Optional[RemotionRenderClient] = None):
        self._client = client

    @property
    def client(self) -> RemotionRenderClient:
        # Lazily resolved so importing this module never opens a connection and
        # tests can inject a fake client.
        if self._client is None:
            self._client = get_render_client()
        return self._client

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        task = ctx.task
        extras = ctx.extras or {}

        src = self._resolve_media(ctx.img_url)
        if not src:
            raise ValueError("RemotionVideoAdapter requires an input image (ctx.img_url)")

        duration = extras.get("duration_seconds") or task.duration or 5
        width, height = resolution_to_size(
            getattr(task, "resolution", None), extras.get("aspect_ratio")
        )
        audio = self._resolve_media(ctx.audio_url or extras.get("dialogue_audio"))

        spec = (
            ClipSpecBuilder(width=width, height=height, fps=int(extras.get("fps", 30)))
            .with_duration(duration)
            .with_image(src, extras.get("camera_movement"), extras.get("shot_size"))
            .with_subtitle(extras.get("dialogue"))
            .with_audio(audio)
            .build()
        )

        seconds = self.client.render(spec, ctx.output_path)
        return ctx.output_path, seconds

    # ── media handling ───────────────────────────────────────────────────

    def _resolve_media(self, ref: Optional[str]) -> Optional[str]:
        """Return a ref the render service can load: a path relative to the
        shared output root, or an http/data URL passed through unchanged.

        Local snapshots (the common A case, e.g. ``video_inputs/<id>.png``) are
        already output-relative. OSS keys / remote URLs / paths outside the
        output root are materialized into ``output/_remotion_cache/`` so the
        statically-served output dir can reach them.
        """
        if not ref:
            return None
        if _REMOTE_RE.match(ref):
            return ref

        output_root = self.client.output_root

        # Already a path under the output root → use as-is (output-relative).
        candidate = os.path.normpath(os.path.join(output_root, ref))
        if (
            candidate == output_root or candidate.startswith(output_root + os.sep)
        ) and os.path.isfile(candidate):
            return os.path.relpath(candidate, output_root).replace(os.sep, "/")

        # Otherwise materialize (handles OSS object keys + remote URLs) and, if
        # it didn't land under the output root, copy it in.
        try:
            from ..utils.provider_media import MediaResolver

            local = MediaResolver().to_local_file(ref)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Remotion adapter could not resolve media %s: %s", ref, e)
            return None

        local = os.path.abspath(local)
        if local.startswith(output_root + os.sep):
            return os.path.relpath(local, output_root).replace(os.sep, "/")

        cache_dir = os.path.join(output_root, "_remotion_cache")
        os.makedirs(cache_dir, exist_ok=True)
        dest = os.path.join(cache_dir, os.path.basename(local))
        if os.path.abspath(dest) != local:
            shutil.copy2(local, dest)
        return os.path.relpath(dest, output_root).replace(os.sep, "/")


def make_remotion_adapter() -> VideoAdapter:
    return RemotionVideoAdapter()


__all__ = [
    "RemotionVideoAdapter",
    "ClipSpecBuilder",
    "make_remotion_adapter",
    "camera_to_kenburns",
    "resolution_to_size",
]
