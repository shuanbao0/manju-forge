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
- **Builder** (``MGPromptBuilder``) assembles the LLM messages step by step,
  mirroring ``StoryboardPromptBuilder`` in ``storyboard.py``.
- **DTO mirror**: the LLM is constrained to the ``VideoSpec`` shape and its
  output is validated by the pydantic ``VideoSpec`` before it ever reaches the
  renderer — bad JSON fails here, not in Chrome.
"""
from __future__ import annotations

import json
import logging
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
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────

_SCHEMA_GUIDE = """\
你要输出一个 JSON 对象，描述一条用 Remotion 渲染的图文/解说短视频（VideoSpec）。
严格使用下面的结构（字段名用英文 snake_case，文案内容用中文）：

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

Layer 是以下几种之一（用 "type" 区分）：
- {"type":"background","gradient":["#0f172a","#1e293b"]}            // 或 {"type":"background","color":"#000000"}
- {"type":"title_card","text":"主标题","subtitle":"副标题(可空)","color":"#ffffff","font_size":80,"enter":"rise"|"fade"|"none"}
- {"type":"bullet_list","title":"小标题(可空)","items":["要点1","要点2"],"color":"#ffffff"}
- {"type":"subtitle","cues":[{"text":"旁白文字","from_sec":0,"to_sec":4}],"position":"bottom"|"center"}

要求：
- 每个 clip 通常先放一个 background，再叠加 title_card / bullet_list / subtitle。
- 单个 clip 时长 3~6 秒；总时长按内容自然展开。
- 只输出 JSON，不要任何额外解释或 markdown 代码块。
"""


class MGPromptBuilder:
    """Assembles the chat messages that ask the LLM for a VideoSpec."""

    def __init__(self):
        self._topic: str = ""
        self._width: int = 1080
        self._height: int = 1920
        self._style_hint: Optional[str] = None

    def with_topic(self, text: str) -> "MGPromptBuilder":
        self._topic = (text or "").strip()
        return self

    def with_size(self, width: int, height: int) -> "MGPromptBuilder":
        self._width, self._height = width, height
        return self

    def with_style_hint(self, hint: Optional[str]) -> "MGPromptBuilder":
        self._style_hint = (hint or "").strip() or None
        return self

    def build(self) -> List[Dict[str, str]]:
        system = (
            "你是一名短视频动效导演，擅长把文字内容转成简洁有节奏的图文/解说视频脚本。\n"
            + _SCHEMA_GUIDE
            + f"\n本次视频画幅：width={self._width}, height={self._height}, fps=30。"
        )
        if self._style_hint:
            system += f"\n视觉风格倾向：{self._style_hint}"
        user = f"请为以下内容生成 VideoSpec JSON：\n\n{self._topic}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


# ─────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────


class RemotionMGGenerator:
    """LLM → VideoSpec → MP4, for the Remotion motion-graphics engine."""

    def __init__(self, render_client: Optional[RemotionRenderClient] = None):
        from .llm_adapter import LLMAdapter

        self.llm = LLMAdapter()
        self._render_client = render_client

    @property
    def render_client(self) -> RemotionRenderClient:
        if self._render_client is None:
            self._render_client = get_render_client()
        return self._render_client

    def generate_spec(
        self,
        text: str,
        *,
        aspect_ratio: str = "9:16",
        style_hint: Optional[str] = None,
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


__all__ = ["RemotionMGGenerator", "MGPromptBuilder"]
