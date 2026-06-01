"""
RemotionMGGenerator — flow B: a chat-only motion-graphics / explainer video.

No image or video model is involved. An LLM turns a topic or short script into
a multi-clip ``VideoSpec`` (title cards, bullet lists, gradients, subtitles,
transitions); the shared Remotion render service turns that into an MP4. Cost
is LLM tokens only.

This is deliberately a separate, self-contained generator rather than a branch
inside the漫剧 pipeline — flow B bypasses the assets / storyboard / i2v stages
entirely and only shares the LLM stage's adapter and the Remotion renderer.

Design patterns
===============
- **Builder** (``MGCopyPromptBuilder`` / ``MGPromptBuilder``) assembles the LLM
  messages step by step, mirroring ``StoryboardPromptBuilder`` in
  ``storyboard.py``.
- **Strategy** (``MGCopyQuality`` + ``_QUALITY_PROFILE``) — one knob picks a
  target duration / section count / clip pacing that drives *both* the copy
  step and the spec step, so quality stays coherent end-to-end.
- **DTO mirror**: the LLM is constrained to the ``VideoSpec`` shape and its
  output is validated by the pydantic ``VideoSpec`` before it ever reaches the
  renderer — bad JSON fails here, not in Chrome.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ...models.remotion_renderer import RemotionRenderClient, get_render_client
from ...models.remotion_spec import VideoSpec

logger = logging.getLogger(__name__)


def _aspect_to_size(aspect_ratio: str, base: int = 1080) -> Tuple[int, int]:
    ar = (aspect_ratio or "9:16").strip()
    if ar == "16:9":
        return round(base * 16 / 9), base
    if ar == "1:1":
        return base, base
    return base, round(base * 16 / 9)  # default vertical 9:16


# ─────────────────────────────────────────────────────────────────────────
# Generation quality (Strategy) — one knob, coherent across copy + spec
# ─────────────────────────────────────────────────────────────────────────


class MGCopyQuality(str, Enum):
    """How long / dense / paced a generated MG video should be."""

    CONCISE = "concise"
    STANDARD = "standard"
    RICH = "rich"


# target_sec: rough total duration; sections: copy section count;
# clips: clip-count hint handed to the spec step so pacing matches the copy.
_QUALITY_PROFILE: Dict[MGCopyQuality, Dict[str, Any]] = {
    MGCopyQuality.CONCISE: {"target_sec": 30, "sections": 2, "clips": "4-5", "label": "精炼"},
    MGCopyQuality.STANDARD: {"target_sec": 55, "sections": 3, "clips": "6-7", "label": "标准"},
    MGCopyQuality.RICH: {"target_sec": 90, "sections": 4, "clips": "8-10", "label": "丰富"},
}


def _profile(quality: MGCopyQuality) -> Dict[str, Any]:
    return _QUALITY_PROFILE.get(quality, _QUALITY_PROFILE[MGCopyQuality.STANDARD])


# ─────────────────────────────────────────────────────────────────────────
# Copy prompt builder (title + existing content → narration copy)
# ─────────────────────────────────────────────────────────────────────────


class MGCopyPromptBuilder:
    """Assembles the chat messages that ask the LLM for Remotion-ready copy."""

    def __init__(self):
        self._title: str = ""
        self._existing: str = ""
        self._quality: MGCopyQuality = MGCopyQuality.STANDARD
        self._style_hint: Optional[str] = None

    def with_title(self, text: str) -> "MGCopyPromptBuilder":
        self._title = (text or "").strip()
        return self

    def with_existing(self, text: str) -> "MGCopyPromptBuilder":
        self._existing = (text or "").strip()
        return self

    def with_quality(self, quality: MGCopyQuality) -> "MGCopyPromptBuilder":
        self._quality = quality
        return self

    def with_style_hint(self, hint: Optional[str]) -> "MGCopyPromptBuilder":
        self._style_hint = (hint or "").strip() or None
        return self

    def build(self) -> List[Dict[str, str]]:
        p = _profile(self._quality)
        system = (
            "你是一名短视频解说编剧,擅长把一个主题写成有节奏、适合做图文/解说动效视频的口播文案。\n"
            "文案结构要求(便于后续切分成 Remotion 分镜):\n"
            "1. 开场一句钩子(将作为主标题/title_card);\n"
            f"2. {p['sections']} 个小节,每节先一句小标题,再 2~3 条要点(将作为 bullet_list);\n"
            "3. 结尾一句行动号召或总结。\n"
            "写作风格:口语化、短句、信息密度高、不要套话和 markdown 标记,直接输出可朗读的纯文本正文。\n"
            f"目标总时长约 {p['target_sec']} 秒(按正常语速估算字数)。"
        )
        if self._style_hint:
            system += f"\n语气/风格倾向:{self._style_hint}"
        user_parts = [f"视频标题:{self._title or '(未提供,请根据已有内容自拟)'}"]
        if self._existing:
            user_parts.append(f"已有内容/素材(可改写、提炼、补充):\n{self._existing}")
        user_parts.append("请直接输出最终文案正文。")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]


# ─────────────────────────────────────────────────────────────────────────
# Spec prompt builder (copy → VideoSpec JSON)
# ─────────────────────────────────────────────────────────────────────────

_SCHEMA_GUIDE = """\
你要输出一个 JSON 对象,描述一条用 Remotion 渲染的图文/解说短视频(VideoSpec)。
严格使用下面的结构(字段名用英文 snake_case,文案内容用中文):

{
  "fps": 30,
  "width": <int>, "height": <int>,
  "clips": [
    {
      "duration_sec": <number 秒>,
      "transition_in": {"type": "fade"|"slide"|"none", "duration_sec": 0.5, "direction": "from-left"|"from-right"|null},
      "layers": [ <Layer>, ... ]
    }, ...
  ]
}

Layer 是以下几种之一(用 "type" 区分):
- {"type":"background","gradient":["#0f172a","#1e293b"]}            // 或 {"type":"background","color":"#000000"}
- {"type":"title_card","text":"主标题","subtitle":"副标题(可空)","color":"#ffffff","font_size":80,"enter":"rise"|"fade"|"none"}
- {"type":"bullet_list","title":"小标题(可空)","items":["要点1","要点2"],"color":"#ffffff"}
- {"type":"subtitle","cues":[{"text":"旁白文字","from_sec":0,"to_sec":4}],"position":"bottom"|"center"}

美术与节奏要求:
- 每个 clip 通常先放一个 background,再叠加 title_card / bullet_list / subtitle。
- 整条视频用统一的配色基调:深色科技感用冷色渐变(如 #0f172a→#1e293b),明亮活泼用暖色或高饱和渐变;开场镜头适合 title_card,中段用 bullet_list 承载要点,过场用 fade/slide 转场。
- 单个 clip 时长 3~6 秒;总时长按内容自然展开。
- 只输出 JSON,不要任何额外解释或 markdown 代码块。
"""


class MGPromptBuilder:
    """Assembles the chat messages that ask the LLM for a VideoSpec."""

    def __init__(self):
        self._topic: str = ""
        self._width: int = 1080
        self._height: int = 1920
        self._style_hint: Optional[str] = None
        self._quality: MGCopyQuality = MGCopyQuality.STANDARD

    def with_topic(self, text: str) -> "MGPromptBuilder":
        self._topic = (text or "").strip()
        return self

    def with_size(self, width: int, height: int) -> "MGPromptBuilder":
        self._width, self._height = width, height
        return self

    def with_style_hint(self, hint: Optional[str]) -> "MGPromptBuilder":
        self._style_hint = (hint or "").strip() or None
        return self

    def with_quality(self, quality: MGCopyQuality) -> "MGPromptBuilder":
        self._quality = quality
        return self

    def build(self) -> List[Dict[str, str]]:
        p = _profile(self._quality)
        system = (
            "你是一名短视频动效导演,擅长把文字内容转成简洁有节奏的图文/解说视频脚本。\n"
            + _SCHEMA_GUIDE
            + f"\n本次视频画幅:width={self._width}, height={self._height}, fps=30。"
            + f"\n镜头数控制在 {p['clips']} 个,总时长约 {p['target_sec']} 秒。"
        )
        if self._style_hint:
            system += f"\n视觉风格倾向:{self._style_hint}"
        user = f"请为以下内容生成 VideoSpec JSON:\n\n{self._topic}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


# ─────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────


class RemotionMGGenerator:
    """LLM → (copy →) VideoSpec → MP4, for the Remotion motion-graphics engine."""

    def __init__(self, render_client: Optional[RemotionRenderClient] = None):
        from .llm_adapter import LLMAdapter

        self.llm = LLMAdapter()
        self._render_client = render_client

    @property
    def render_client(self) -> RemotionRenderClient:
        if self._render_client is None:
            self._render_client = get_render_client()
        return self._render_client

    def generate_copy(
        self,
        title: str,
        existing: str = "",
        *,
        quality: MGCopyQuality = MGCopyQuality.STANDARD,
        style_hint: Optional[str] = None,
    ) -> str:
        """Expand a title (+ optional existing content) into Remotion-ready copy.

        Stateless — no project required. Must be called inside a
        ``scoped_instance(..., InstanceType.LLM)`` so the bound LLM instance
        drives the model name / credentials.
        """
        messages = (
            MGCopyPromptBuilder()
            .with_title(title)
            .with_existing(existing)
            .with_quality(quality)
            .with_style_hint(style_hint)
            .build()
        )
        return (self.llm.chat(messages) or "").strip()

    def generate_spec(
        self,
        text: str,
        *,
        aspect_ratio: str = "9:16",
        style_hint: Optional[str] = None,
        quality: MGCopyQuality = MGCopyQuality.STANDARD,
    ) -> VideoSpec:
        """Ask the LLM for a VideoSpec and validate it before returning.

        Must be called inside a ``scoped_instance(..., InstanceType.LLM)`` so
        the bound LLM instance drives the model name / credentials.
        """
        width, height = _aspect_to_size(aspect_ratio)
        messages = (
            MGPromptBuilder()
            .with_topic(text)
            .with_size(width, height)
            .with_style_hint(style_hint)
            .with_quality(quality)
            .build()
        )
        content = self.llm.chat(messages, response_format={"type": "json_object"})
        data = self._parse_json(content)
        # The LLM may omit width/height/fps — backfill from the requested size.
        data.setdefault("width", width)
        data.setdefault("height", height)
        data.setdefault("fps", 30)
        return VideoSpec.model_validate(data)

    def render(self, spec: VideoSpec, output_path: str) -> float:
        """Render a validated spec to ``output_path``; return seconds."""
        return self.render_client.render(spec, output_path)

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        raw = (content or "").strip()
        if raw.startswith("```"):
            # strip ```json ... ``` fences if the model added them
            raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw.strip("`")
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid VideoSpec JSON: {e}") from e


__all__ = [
    "RemotionMGGenerator",
    "MGPromptBuilder",
    "MGCopyPromptBuilder",
    "MGCopyQuality",
]
