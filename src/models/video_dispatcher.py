"""
Video model dispatcher — strategy registry mapping (model family, backend) to
the adapter that knows how to drive that vendor's API.

Design notes
============
The pipeline used to grow an ``if/elif`` chain every time a new vendor was
added (Kling vendor / Vidu vendor / fall through to Wanx). This module turns
that into a registry: each adapter declares which ``(family_prefix, backend)``
pairs it serves, and the dispatcher picks the longest-matching one at request
time. Adding a new vendor is now: write an adapter + register it.

Adapters share a tiny ``VideoAdapter`` interface so the pipeline does not need
to know which provider is on the other side of the call. Provider-specific
parameters (Kling ``mode`` / Vidu ``movement_amplitude`` / etc.) are pulled
from the same ``VideoTask`` instance the pipeline already constructs — each
adapter just reads what it needs.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Adapter contract
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class VideoGenerationContext:
    """Inputs the dispatcher hands to each adapter.

    Bundling these into a dataclass means new params don't break every adapter
    signature — only the ones that care about the new field need to read it.
    """

    task: Any  # VideoTask (avoid circular import)
    output_path: str
    img_url: Optional[str] = None
    img_path: Optional[str] = None
    audio_url: Optional[str] = None
    generate_audio: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


class VideoAdapter(ABC):
    """Generates a video clip for one model family / backend pair.

    Each adapter caches its own underlying client (Wanx / KlingModel / ...).
    The pipeline holds a single dispatcher; tests can monkeypatch the client
    classes before exercising the pipeline (the lazy import inside
    :meth:`_client` picks up the patched class).
    """

    @abstractmethod
    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        """Return ``(absolute_path_to_video, generation_seconds)``."""


# ─────────────────────────────────────────────────────────────────────────
# Built-in adapters
# ─────────────────────────────────────────────────────────────────────────


class WanxDashScopeAdapter(VideoAdapter):
    """Default DashScope route — used by Wan / Kling-on-DashScope / Vidu-on-
    DashScope / Pixverse-on-DashScope. Wraps the existing ``WanxModel``."""

    def __init__(self, video_generator: Any):
        # The pipeline's existing ``video_generator.model`` is reused so OSS,
        # endpoint resolution, and credential plumbing stay in one place.
        self._video_generator = video_generator

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        task = ctx.task
        return self._video_generator.model.generate(
            prompt=task.prompt,
            output_path=ctx.output_path,
            img_path=ctx.img_path,
            img_url=ctx.img_url,
            duration=task.duration,
            seed=task.seed,
            resolution=task.resolution,
            audio_url=ctx.audio_url,
            audio=ctx.generate_audio,
            prompt_extend=task.prompt_extend,
            negative_prompt=task.negative_prompt,
            model=task.model,
            shot_type=task.shot_type,
            ref_video_urls=(
                task.reference_video_urls if task.generation_mode == "r2v" else None
            ),
            camera_motion=None,
            subject_motion=None,
        )


class KlingVendorAdapter(VideoAdapter):
    def __init__(self):
        self._client = None

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        if self._client is None:
            from .kling import KlingModel  # late import — picks up monkeypatches

            self._client = KlingModel({})
        task = ctx.task
        return self._client.generate(
            prompt=task.prompt,
            output_path=ctx.output_path,
            img_url=ctx.img_url,
            img_path=ctx.img_path,
            duration=task.duration,
            model=task.model,
            negative_prompt=task.negative_prompt,
            aspect_ratio="16:9",
            mode=task.mode or "std",
            sound=task.sound or "off",
            cfg_scale=task.cfg_scale,
        )


class ViduVendorAdapter(VideoAdapter):
    def __init__(self):
        self._client = None

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        if self._client is None:
            from .vidu import ViduModel

            self._client = ViduModel({})
        task = ctx.task
        return self._client.generate(
            prompt=task.prompt,
            output_path=ctx.output_path,
            img_url=ctx.img_url,
            img_path=ctx.img_path,
            duration=task.duration,
            model=task.model,
            resolution=task.resolution,
            aspect_ratio="16:9",
            seed=task.seed or 0,
            audio=task.vidu_audio if task.vidu_audio is not None else True,
            movement_amplitude=task.movement_amplitude or "auto",
        )


class _NotImplementedAdapter(VideoAdapter):
    """Raises a clear error when a preview vendor is invoked.

    The Settings UI marks these models ``available=False`` and disables their
    pickers — but if a stale project state references one of them, we want a
    descriptive error rather than a silent fall-through to Wanx. Once the
    actual client lands, swap the registration to point at the real adapter.
    """

    def __init__(self, family: str, hint: str):
        self._family = family
        self._hint = hint

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        raise NotImplementedError(
            f"{self._family} vendor-direct adapter is not implemented yet. "
            f"{self._hint}"
        )


def make_pixverse_vendor_adapter() -> VideoAdapter:
    return _NotImplementedAdapter(
        family="pixverse",
        hint="Pixverse vendor-direct API client is not yet wired. "
        "Switch to DashScope mode (PIXVERSE_PROVIDER_MODE=dashscope) "
        "or contribute the client at src/models/pixverse.py.",
    )


class DoubaoSeedanceVendorAdapter(VideoAdapter):
    """ByteDance Seedance vendor-direct adapter (Volcano Engine Ark).

    Bridges the dispatcher's :class:`VideoGenerationContext` to
    :func:`src.models.seedance.generate_seedance_video`.
    """

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        from .seedance import generate_seedance_video

        task = ctx.task
        return generate_seedance_video(
            prompt=task.prompt,
            output_path=ctx.output_path,
            img_url=ctx.img_url,
            img_path=ctx.img_path,
            duration=int(task.duration or 5),
            resolution=(task.resolution or "1080p"),
            model=task.model,
        )


def make_doubao_vendor_adapter() -> VideoAdapter:
    return DoubaoSeedanceVendorAdapter()


class VeoVendorAdapter(VideoAdapter):
    """Google Veo vendor-direct adapter (Gemini API)."""

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        from .veo import generate_veo_video

        task = ctx.task
        return generate_veo_video(
            prompt=task.prompt,
            output_path=ctx.output_path,
            img_url=ctx.img_url,
            img_path=ctx.img_path,
            duration=int(task.duration or 8),
            resolution=(task.resolution or "1080p"),
            model=task.model,
        )


def make_veo_vendor_adapter() -> VideoAdapter:
    return VeoVendorAdapter()


class FalVideoVendorAdapter(VideoAdapter):
    """fal.ai aggregator video adapter (Veo / Kling / Seedance / etc.)."""

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        from .fal_aggregator import generate_fal_video

        task = ctx.task
        return generate_fal_video(
            prompt=task.prompt,
            output_path=ctx.output_path,
            img_url=ctx.img_url,
            img_path=ctx.img_path,
            duration=int(task.duration or 5),
            resolution=(task.resolution or "720p"),
            model=task.model,
        )


def make_fal_vendor_adapter() -> VideoAdapter:
    return FalVideoVendorAdapter()


class HailuoVendorAdapter(VideoAdapter):
    """MiniMax Hailuo (海螺) vendor-direct adapter.

    Bridges the dispatcher's :class:`VideoGenerationContext` to the simpler
    :func:`src.models.hailuo.generate_hailuo_video` signature. Resolution
    is mapped from the pipeline's ``720p/1080p`` strings to MiniMax's
    ``768P/1080P`` casing.
    """

    _RES_MAP = {"480p": "768P", "720p": "768P", "768p": "768P", "1080p": "1080P", "2k": "1080P"}

    def generate(self, ctx: VideoGenerationContext) -> Tuple[str, float]:
        from .hailuo import generate_hailuo_video

        task = ctx.task
        target_resolution = self._RES_MAP.get((task.resolution or "").lower(), "768P")
        return generate_hailuo_video(
            prompt=task.prompt,
            output_path=ctx.output_path,
            img_url=ctx.img_url,
            img_path=ctx.img_path,
            duration=int(task.duration or 6),
            resolution=target_resolution,
            model=task.model,
        )


def make_hailuo_vendor_adapter() -> VideoAdapter:
    return HailuoVendorAdapter()


# ─────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Route:
    family_prefix: str
    backend: str  # "dashscope" | "vendor"


class VideoModelDispatcher:
    """Resolves a model name + backend to the right :class:`VideoAdapter`.

    Lookup is a longest-prefix match against the model id (e.g.
    ``viduq3-pro`` matches family ``vidu``). Backends that have no explicit
    registration fall back to ``default_adapter``.
    """

    def __init__(self, default_factory: Callable[[], VideoAdapter]):
        self._routes: Dict[_Route, Callable[[], VideoAdapter]] = {}
        self._cache: Dict[_Route, VideoAdapter] = {}
        self._default_factory = default_factory
        self._default: Optional[VideoAdapter] = None

    def register(
        self,
        family_prefix: str,
        backend: str,
        factory: Callable[[], VideoAdapter],
    ) -> None:
        family = (family_prefix or "").strip().lower()
        be = (backend or "").strip().lower()
        if not family:
            raise ValueError("family_prefix cannot be empty")
        if be not in ("dashscope", "vendor"):
            raise ValueError(f"unsupported backend: {backend}")
        self._routes[_Route(family, be)] = factory

    def resolve(self, model_name: str, backend: str) -> VideoAdapter:
        normalized = (model_name or "").strip().lower()
        be = (backend or "dashscope").strip().lower()

        best: Optional[_Route] = None
        for route in self._routes:
            if route.backend != be:
                continue
            if not normalized.startswith(route.family_prefix):
                continue
            if best is None or len(route.family_prefix) > len(best.family_prefix):
                best = route

        if best is not None:
            adapter = self._cache.get(best)
            if adapter is None:
                adapter = self._routes[best]()
                self._cache[best] = adapter
            return adapter

        if self._default is None:
            self._default = self._default_factory()
        return self._default

    def list_routes(self) -> List[Tuple[str, str]]:
        return sorted((r.family_prefix, r.backend) for r in self._routes)


def build_default_dispatcher(video_generator: Any) -> VideoModelDispatcher:
    """Compose the dispatcher used by :class:`ComicGenPipeline`.

    Adding a new vendor = one registration call. The default route
    (``video_generator.model.generate(...)``) handles every DashScope-routed
    model so families that share the DashScope transport layer don't each
    need their own adapter.
    """
    dispatcher = VideoModelDispatcher(
        default_factory=lambda: WanxDashScopeAdapter(video_generator)
    )
    # Vendor-direct adapters
    dispatcher.register("kling-", "vendor", KlingVendorAdapter)
    dispatcher.register("vidu", "vendor", ViduVendorAdapter)
    dispatcher.register("viduq2", "vendor", ViduVendorAdapter)
    dispatcher.register("viduq3", "vendor", ViduVendorAdapter)
    dispatcher.register("pixverse-", "vendor", make_pixverse_vendor_adapter)
    dispatcher.register("doubao-seedance-", "vendor", make_doubao_vendor_adapter)
    dispatcher.register("hailuo-", "vendor", make_hailuo_vendor_adapter)
    # MiniMax-Hailuo-* (canonical API model ids) shares the same adapter.
    dispatcher.register("minimax-hailuo-", "vendor", make_hailuo_vendor_adapter)
    # 2026 additions: Google Veo + fal.ai aggregator video.
    dispatcher.register("veo-", "vendor", make_veo_vendor_adapter)
    dispatcher.register("fal-", "vendor", make_fal_vendor_adapter)
    # Local Remotion render engine (flow A) — "pseudo-motion" on stills, no
    # cloud API call. Lazy import keeps the render client out of import time.
    from .remotion_adapter import make_remotion_adapter

    dispatcher.register("remotion-", "vendor", make_remotion_adapter)
    return dispatcher


__all__ = [
    "VideoAdapter",
    "VideoGenerationContext",
    "VideoModelDispatcher",
    "WanxDashScopeAdapter",
    "KlingVendorAdapter",
    "ViduVendorAdapter",
    "build_default_dispatcher",
]
