"""
Vendor connector registry — declarative description of every external service
the user might configure (DashScope, Kling, Vidu, Doubao, Hailuo, ...).

Why this exists
================
The Settings UI used to grow a fresh JSX section per vendor (Kling toggle,
Vidu toggle, an "advanced" disclosure for Pixverse / Doubao / Hailuo, etc.).
That worked when there were three providers; with eight or more it became
inconsistent and easy to miss.

This module describes vendors as data: id, capabilities, credential fields,
optional dual-mode (DashScope-routed vs vendor-direct), badges, docs link.
The frontend renders one ``VendorCard`` per entry, so adding a new provider
is a pure data change with no UI rewrite.

Pairs with:
- ``src/utils/model_catalog.py`` — which models exist and what they support.
- ``src/utils/provider_registry.py`` — how each model family routes between
  DashScope and vendor-direct.
- ``src/auth/credentials.py`` — which env keys are accepted for storage.

Keep this file declarative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CredentialField:
    """A single env-var-shaped form field surfaced on a vendor card.

    Fields:
        key:          credentials-store key (matches ``auth.credentials``).
        label:        human-readable label rendered above the input.
        placeholder:  hint text shown inside the empty input.
        secret:       render as password input + masked status.
        help_text:    small text under the input.
        required:     drives validation; the FE marks the field with a red ``*``.
    """

    key: str
    label: str
    placeholder: str = ""
    secret: bool = False
    help_text: str = ""
    required: bool = True


@dataclass(frozen=True)
class VendorMode:
    """A backend mode the vendor can run in (DashScope-routed or vendor-direct).

    ``id`` matches the values written to ``mode_env_key`` (e.g.
    ``KLING_PROVIDER_MODE = "vendor"``). ``fields`` are the credentials only
    needed when this mode is active — vendor-direct typically requires its
    own access keys, while DashScope-routed reuses the global ``DASHSCOPE_API_KEY``.
    """

    id: str  # "dashscope" | "vendor"
    label: str
    description: str = ""
    fields: Tuple[CredentialField, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VendorConnector:
    """One configurable third-party service.

    Capabilities tell the FE which section to render the card in; family
    prefixes link the connector back to ``ProviderRegistry`` so the dispatcher
    knows which models live behind it.
    """

    id: str
    display_name: str
    description: str
    capabilities: Tuple[str, ...]
    common_fields: Tuple[CredentialField, ...] = field(default_factory=tuple)
    modes: Tuple[VendorMode, ...] = field(default_factory=tuple)
    mode_env_key: Optional[str] = None
    family_prefixes: Tuple[str, ...] = field(default_factory=tuple)
    docs_url: str = ""
    badges: Tuple[str, ...] = field(default_factory=tuple)
    accent: str = "amber"  # tailwind palette key driving the FE card stripe


# ─────────────────────────────────────────────────────────────────────────
# Built-in connectors
# ─────────────────────────────────────────────────────────────────────────


# DashScope — the default backend for almost everything.
_DASHSCOPE = VendorConnector(
    id="dashscope",
    display_name="阿里云 DashScope",
    description="百炼平台,默认 LLM/T2I/I2I/I2V 后端,支持 Qwen / Wan / FLUX 等",
    capabilities=("llm", "t2i", "i2i", "i2v", "t2v", "r2v", "tts"),
    common_fields=(
        CredentialField(
            key="DASHSCOPE_API_KEY",
            label="API Key",
            placeholder="sk-...",
            secret=True,
        ),
        CredentialField(
            key="DASHSCOPE_BASE_URL",
            label="Base URL",
            placeholder="https://dashscope.aliyuncs.com",
            help_text="海外部署可改为 https://dashscope-intl.aliyuncs.com",
            required=False,
        ),
    ),
    family_prefixes=("wan2.7-", "wan2.6-", "wan2.5-", "wan2.2-", "qwen-image", "flux-"),
    docs_url="https://help.aliyun.com/zh/model-studio/",
    badges=("recommended",),
    accent="amber",
)


def _dual_mode_video_connector(
    *,
    id: str,
    display_name: str,
    description: str,
    family_prefixes: Tuple[str, ...],
    mode_env_key: str,
    vendor_fields: Tuple[CredentialField, ...],
    docs_url: str = "",
    accent: str = "violet",
    badges: Tuple[str, ...] = (),
) -> VendorConnector:
    """Reusable factory for Kling / Vidu / Pixverse — they share the same
    "DashScope mode by default, switch to vendor-direct with own keys" shape."""

    return VendorConnector(
        id=id,
        display_name=display_name,
        description=description,
        capabilities=("i2v", "t2v"),
        modes=(
            VendorMode(
                id="dashscope",
                label="DashScope",
                description="通过百炼路由,复用 DashScope API Key",
            ),
            VendorMode(
                id="vendor",
                label="Vendor Direct",
                description="直连厂商 API,需要独立凭证",
                fields=vendor_fields,
            ),
        ),
        mode_env_key=mode_env_key,
        family_prefixes=family_prefixes,
        docs_url=docs_url,
        badges=badges,
        accent=accent,
    )


_KLING = _dual_mode_video_connector(
    id="kling",
    display_name="Kling AI",
    description="可灵 AI,人像与运镜表现优秀",
    family_prefixes=("kling-",),
    mode_env_key="KLING_PROVIDER_MODE",
    vendor_fields=(
        CredentialField(
            key="KLING_ACCESS_KEY", label="Access Key", placeholder="Kling Access Key", secret=True,
        ),
        CredentialField(
            key="KLING_SECRET_KEY", label="Secret Key", placeholder="Kling Secret Key", secret=True,
        ),
        CredentialField(
            key="KLING_BASE_URL", label="Base URL",
            placeholder="https://api-beijing.klingai.com/v1",
            required=False,
            help_text="留空使用默认。海外部署可填 https://api.klingai.com/v1",
        ),
    ),
    docs_url="https://app.klingai.com/cn/dev/document-api",
    accent="purple",
)

_VIDU = _dual_mode_video_connector(
    id="vidu",
    display_name="Vidu",
    description="Vidu Q3 Pro / Turbo,支持音视频联动生成",
    family_prefixes=("vidu", "viduq2", "viduq3"),
    mode_env_key="VIDU_PROVIDER_MODE",
    vendor_fields=(
        CredentialField(key="VIDU_API_KEY", label="API Key", placeholder="Vidu API Key", secret=True),
        CredentialField(
            key="VIDU_BASE_URL", label="Base URL",
            placeholder="https://api.vidu.cn/ent/v2",
            required=False,
        ),
    ),
    docs_url="https://platform.vidu.com/",
    accent="cyan",
)

_PIXVERSE = _dual_mode_video_connector(
    id="pixverse",
    display_name="Pixverse",
    description="Pixverse v4,适合短动画与角色表演",
    family_prefixes=("pixverse-",),
    mode_env_key="PIXVERSE_PROVIDER_MODE",
    vendor_fields=(
        CredentialField(key="PIXVERSE_API_KEY", label="API Key", placeholder="Pixverse API Key", secret=True),
        CredentialField(
            key="PIXVERSE_BASE_URL", label="Base URL",
            placeholder="https://app-api.pixverse.ai/openapi/v2",
            required=False,
        ),
    ),
    docs_url="https://platform.pixverse.ai/",
    accent="rose",
)


_DOUBAO = VendorConnector(
    id="doubao",
    display_name="字节豆包 Seedance / Seedream",
    description="ByteDance Seedance(视频)+ Seedream(图像),火山引擎 Ark API",
    capabilities=("t2i", "i2i", "i2v", "t2v"),
    common_fields=(
        CredentialField(
            key="DOUBAO_API_KEY", label="API Key",
            placeholder="Volcano Engine API Key", secret=True,
        ),
        CredentialField(
            key="DOUBAO_BASE_URL", label="Base URL",
            placeholder="https://ark.cn-beijing.volces.com/api/v3",
            required=False,
        ),
    ),
    family_prefixes=("doubao-seedance-", "doubao-seedream-"),
    docs_url="https://www.volcengine.com/docs/82379",
    badges=("preview", "premium"),
    accent="orange",
)


_HAILUO = VendorConnector(
    id="hailuo",
    display_name="MiniMax Hailuo 海螺",
    description="MiniMax 海螺 02,日常向 i2v / t2v",
    capabilities=("i2v", "t2v"),
    common_fields=(
        CredentialField(key="HAILUO_API_KEY", label="API Key", placeholder="Hailuo API Key", secret=True),
        CredentialField(
            key="HAILUO_BASE_URL", label="Base URL",
            placeholder="https://api.minimax.chat/v1",
            required=False,
        ),
    ),
    family_prefixes=("hailuo-",),
    docs_url="https://platform.minimaxi.com/",
    badges=("preview",),
    accent="sky",
)


# ── Black Forest Labs FLUX.2 (T2I/I2I, vendor-direct) ─────────────────────
_BFL = VendorConnector(
    id="bfl",
    display_name="Black Forest Labs FLUX",
    description="FLUX.2 Pro / Max / Flash,2026 写实图像基准,支持最多 10 张参考图",
    capabilities=("t2i", "i2i"),
    common_fields=(
        CredentialField(key="BFL_API_KEY", label="API Key", placeholder="BFL API Key", secret=True),
        CredentialField(
            key="BFL_BASE_URL", label="Base URL",
            placeholder="https://api.bfl.ai/v1",
            required=False,
        ),
    ),
    family_prefixes=("flux-2",),
    docs_url="https://docs.bfl.ai/",
    badges=("new", "premium"),
    accent="indigo",
)


# ── ElevenLabs TTS (vendor-direct) ─────────────────────────────────────────
_ELEVENLABS = VendorConnector(
    id="elevenlabs",
    display_name="ElevenLabs",
    description="英语长篇朗读黄金标准,Turbo v2.5 / Multilingual v2 / v3,情感与韵律最强",
    capabilities=("tts",),
    common_fields=(
        CredentialField(
            key="ELEVENLABS_API_KEY", label="API Key",
            placeholder="ElevenLabs API Key", secret=True,
        ),
        CredentialField(
            key="ELEVENLABS_BASE_URL", label="Base URL",
            placeholder="https://api.elevenlabs.io/v1",
            required=False,
        ),
    ),
    family_prefixes=("eleven_",),
    docs_url="https://elevenlabs.io/docs/api-reference/text-to-speech",
    badges=("premium", "new"),
    accent="emerald",
)


# ── Fish Audio (open-weights TTS, voice cloning) ───────────────────────────
_FISH_AUDIO = VendorConnector(
    id="fish-audio",
    display_name="Fish Audio",
    description="Fish Audio S2,80+ 语言,15 秒克隆,公开榜单 ELO #1,价格约 ElevenLabs 的 1/10",
    capabilities=("tts",),
    common_fields=(
        CredentialField(
            key="FISH_AUDIO_API_KEY", label="API Key",
            placeholder="Fish Audio API Key", secret=True,
        ),
        CredentialField(
            key="FISH_AUDIO_BASE_URL", label="Base URL",
            placeholder="https://api.fish.audio/v1",
            required=False,
        ),
    ),
    family_prefixes=("fish-",),
    docs_url="https://docs.fish.audio/",
    badges=("new", "open-source"),
    accent="teal",
)


# ── Cartesia (low-latency TTS, voice agents) ──────────────────────────────
_CARTESIA = VendorConnector(
    id="cartesia",
    display_name="Cartesia",
    description="Sonic 3,首字节延迟 ~40-90ms,语音 Agent 实时对话首选",
    capabilities=("tts",),
    common_fields=(
        CredentialField(
            key="CARTESIA_API_KEY", label="API Key",
            placeholder="Cartesia API Key", secret=True,
        ),
        CredentialField(
            key="CARTESIA_BASE_URL", label="Base URL",
            placeholder="https://api.cartesia.ai",
            required=False,
        ),
    ),
    family_prefixes=("sonic-",),
    docs_url="https://docs.cartesia.ai/",
    badges=("new", "fast"),
    accent="lime",
)


# ── fal.ai (聚合服务,一个 key 调度数百个模型) ───────────────────────────
_FAL = VendorConnector(
    id="fal",
    display_name="fal.ai",
    description="模型聚合服务,一个 key 直连 Veo 3.1 / Sora 2 / Kling 3.0 / Seedance / FLUX.2 / Wan 2.6 等 600+ 模型",
    capabilities=("t2i", "i2i", "i2v", "t2v"),
    common_fields=(
        CredentialField(
            key="FAL_API_KEY", label="API Key",
            placeholder="fal.ai API Key (KEY_ID:KEY_SECRET)", secret=True,
        ),
        CredentialField(
            key="FAL_BASE_URL", label="Base URL",
            placeholder="https://fal.run",
            required=False,
        ),
    ),
    family_prefixes=("fal-",),
    docs_url="https://fal.ai/docs",
    badges=("new",),
    accent="fuchsia",
)


# ── LLM-only OpenAI-compatible vendors (DeepSeek / Anthropic / Kimi / ...) ──
#
# These all share the same OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
# slots, so they're surfaced as LLM presets in ``model_catalog.LLM_PRESETS``
# rather than as full vendor cards. We expose them here only as references so
# the FE can deep-link from the LLM section to docs.

# ── Remotion (local render engine — no credentials) ───────────────────────
# Not a cloud API: renders "Ken Burns" motion locally (flow A) and full
# motion-graphics videos (flow B). Surfaced as a vendor so the Settings UI can
# create an I2V instance pointing at it; it carries no credential fields.
_REMOTION = VendorConnector(
    id="remotion",
    display_name="Remotion (本地渲染)",
    description="本地 Remotion 引擎：对静图做运镜伪视频替代 i2v，零模型调用成本。需运行渲染微服务。",
    capabilities=("i2v", "t2v"),
    family_prefixes=("remotion-",),
    docs_url="https://www.remotion.dev/",
    badges=("local", "free"),
    accent="slate",
)


DEFAULT_VENDOR_CONNECTORS: Tuple[VendorConnector, ...] = (
    _DASHSCOPE,
    _KLING,
    _VIDU,
    _PIXVERSE,
    _DOUBAO,
    _HAILUO,
    _BFL,
    _ELEVENLABS,
    _FISH_AUDIO,
    _CARTESIA,
    _FAL,
    _REMOTION,
)


# ─────────────────────────────────────────────────────────────────────────
# Registry façade
# ─────────────────────────────────────────────────────────────────────────


class VendorRegistry:
    def __init__(self, connectors: Sequence[VendorConnector] = ()):
        self._connectors: Tuple[VendorConnector, ...] = tuple(connectors)

    @property
    def all(self) -> Tuple[VendorConnector, ...]:
        return self._connectors

    def by_id(self, vendor_id: str) -> Optional[VendorConnector]:
        for connector in self._connectors:
            if connector.id == vendor_id:
                return connector
        return None

    def by_capability(self, capability: str) -> List[VendorConnector]:
        cap = capability.strip().lower()
        return [c for c in self._connectors if cap in c.capabilities]

    def all_credential_keys(self) -> Tuple[str, ...]:
        """Every env key any connector might write — useful for asserting the
        ``auth.credentials.ALLOWED_KEYS`` set covers the whole registry."""
        keys: List[str] = []
        for connector in self._connectors:
            for f in connector.common_fields:
                keys.append(f.key)
            for mode in connector.modes:
                for f in mode.fields:
                    keys.append(f.key)
            if connector.mode_env_key:
                keys.append(connector.mode_env_key)
        # Dedup but keep order for readability.
        seen: set = set()
        ordered: List[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        return tuple(ordered)

    def serialize(self, capability: Optional[str] = None) -> Dict[str, object]:
        connectors = self.by_capability(capability) if capability else list(self._connectors)
        return {"connectors": [_connector_to_dict(c) for c in connectors]}


def _connector_to_dict(connector: VendorConnector) -> Dict[str, object]:
    return asdict(connector)


def get_default_vendor_registry() -> VendorRegistry:
    return VendorRegistry(DEFAULT_VENDOR_CONNECTORS)


__all__ = [
    "CredentialField",
    "VendorMode",
    "VendorConnector",
    "VendorRegistry",
    "DEFAULT_VENDOR_CONNECTORS",
    "get_default_vendor_registry",
]
