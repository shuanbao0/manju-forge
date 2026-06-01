import os
from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple


SUPPORTED_PROVIDER_BACKENDS = ("dashscope", "vendor")


@dataclass
class ProviderFamilyConfig:
    model_family: str
    backend_default: str = "dashscope"
    backend_env_key: Optional[str] = None
    credential_sources: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    supported_modalities: Tuple[str, ...] = field(default_factory=tuple)
    image_input_mode: Dict[str, str] = field(default_factory=dict)
    audio_input_mode: Dict[str, str] = field(default_factory=dict)
    reference_video_input_mode: Dict[str, str] = field(default_factory=dict)


class ProviderRegistry:
    """Data-driven provider routing registry keyed by model family prefix."""

    def __init__(self, families: Optional[Sequence[ProviderFamilyConfig]] = None):
        self._families: Dict[str, ProviderFamilyConfig] = {}
        for family in families or ():
            self.register_family(family)

    def register_family(self, config: ProviderFamilyConfig) -> None:
        family = (config.model_family or "").strip().lower()
        if not family:
            raise ValueError("model_family cannot be empty")
        backend_default = (config.backend_default or "").strip().lower()
        if backend_default not in SUPPORTED_PROVIDER_BACKENDS:
            raise ValueError(f"Unsupported backend_default: {config.backend_default}")
        self._families[family] = replace(
            config,
            model_family=family,
            backend_default=backend_default,
        )

    def get_family_config(self, model_name: str) -> ProviderFamilyConfig:
        normalized = (model_name or "").strip().lower()
        if not normalized:
            raise ValueError("model_name cannot be empty")

        for family in sorted(self._families.keys(), key=len, reverse=True):
            if normalized.startswith(family):
                return self._families[family]
        raise KeyError(f"No provider family registered for model '{model_name}'")

    def resolve_backend(self, model_name: str, env: Optional[Mapping[str, str]] = None) -> str:
        family = self.get_family_config(model_name)
        mode = ""
        if family.backend_env_key:
            env_mapping = env if env is not None else os.environ
            mode = (env_mapping.get(family.backend_env_key) or "").strip().lower()

        if mode in SUPPORTED_PROVIDER_BACKENDS:
            return mode
        return family.backend_default

    def swap_modality(self, model_name: str, target_modality: str) -> Optional[str]:
        """Return the sibling SKU of ``model_name`` for ``target_modality``.

        Works on families whose model ids follow the ``<family><modality>``
        convention (e.g. ``wan2.7-i2v`` ↔ ``wan2.7-r2v``). Returns ``None``
        when the family doesn't follow that convention or doesn't expose the
        requested modality — caller decides whether to fall back.

        Used by the video task pipeline to translate a user's picked I2V
        instance to its R2V sibling without hardcoding the model strings.
        """
        target = (target_modality or "").strip().lower()
        if not target:
            return None
        try:
            family = self.get_family_config(model_name)
        except (KeyError, ValueError):
            return None
        if target not in family.supported_modalities:
            return None
        prefix = family.model_family
        suffix = (model_name or "").strip().lower()[len(prefix):]
        if suffix in family.supported_modalities:
            if suffix == target:
                return model_name
            return f"{prefix}{target}"
        return None


DEFAULT_PROVIDER_FAMILIES: Tuple[ProviderFamilyConfig, ...] = (
    ProviderFamilyConfig(
        model_family="wan2.7-",
        backend_default="dashscope",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
        },
        supported_modalities=("t2i", "i2i", "i2v", "r2v"),
        image_input_mode={
            "dashscope": "dashscope_multimodal_message",
        },
        audio_input_mode={
            "dashscope": "dashscope_temp_file_url",
        },
        reference_video_input_mode={
            "dashscope": "dashscope_temp_file_url",
        },
    ),
    ProviderFamilyConfig(
        model_family="wan2.6-",
        backend_default="dashscope",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
        },
        supported_modalities=("t2i", "i2i", "i2v", "r2v"),
        image_input_mode={
            "dashscope": "dashscope_multimodal_message",
        },
        audio_input_mode={
            "dashscope": "dashscope_temp_file_url",
        },
        reference_video_input_mode={
            "dashscope": "dashscope_temp_file_url",
        },
    ),
    ProviderFamilyConfig(
        model_family="wan2.5-",
        backend_default="dashscope",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
        },
        supported_modalities=("t2i", "i2i", "i2v"),
        image_input_mode={
            "dashscope": "dashscope_multimodal_message",
        },
    ),
    ProviderFamilyConfig(
        model_family="wan2.2-",
        backend_default="dashscope",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
        },
        supported_modalities=("t2i", "i2v"),
    ),
    ProviderFamilyConfig(
        model_family="kling-",
        backend_default="dashscope",
        backend_env_key="KLING_PROVIDER_MODE",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
            "vendor": ("KLING_ACCESS_KEY", "KLING_SECRET_KEY"),
        },
        supported_modalities=("t2v", "i2v"),
        image_input_mode={
            "dashscope": "dashscope_image_to_video",
            "vendor": "kling_vendor_base64_image",
        },
        audio_input_mode={
            "dashscope": "dashscope_temp_file_url",
            "vendor": "kling_vendor_audio_url",
        },
        reference_video_input_mode={
            "dashscope": "dashscope_temp_file_url",
            "vendor": "kling_vendor_video_url",
        },
    ),
    ProviderFamilyConfig(
        model_family="vidu",
        backend_default="dashscope",
        backend_env_key="VIDU_PROVIDER_MODE",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
            "vendor": ("VIDU_API_KEY",),
        },
        supported_modalities=("t2v", "i2v"),
        image_input_mode={
            "dashscope": "dashscope_image_to_video",
            "vendor": "vidu_vendor_image_url",
        },
        audio_input_mode={
            "dashscope": "dashscope_temp_file_url",
            "vendor": "vidu_vendor_audio_url",
        },
        reference_video_input_mode={
            "dashscope": "dashscope_temp_file_url",
            "vendor": "vidu_vendor_video_url",
        },
    ),
    ProviderFamilyConfig(
        model_family="pixverse-",
        backend_default="dashscope",
        backend_env_key="PIXVERSE_PROVIDER_MODE",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
            "vendor": ("PIXVERSE_API_KEY",),
        },
        supported_modalities=("t2v", "i2v"),
        image_input_mode={
            "dashscope": "dashscope_image_to_video",
            "vendor": "pixverse_vendor_image_url",
        },
        audio_input_mode={
            "dashscope": "dashscope_temp_file_url",
            "vendor": "pixverse_vendor_audio_url",
        },
        reference_video_input_mode={
            "dashscope": "dashscope_temp_file_url",
            "vendor": "pixverse_vendor_video_url",
        },
    ),
    # ── DashScope-hosted image generation families (T2I / I2I) ────────────
    ProviderFamilyConfig(
        model_family="qwen-image",
        backend_default="dashscope",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
        },
        supported_modalities=("t2i", "i2i"),
        image_input_mode={
            "dashscope": "dashscope_multimodal_message",
        },
    ),
    ProviderFamilyConfig(
        model_family="flux-",
        backend_default="dashscope",
        credential_sources={
            "dashscope": ("DASHSCOPE_API_KEY",),
        },
        supported_modalities=("t2i",),
    ),
    # ── Vendor-direct only (not yet on DashScope; client wired separately) ─
    ProviderFamilyConfig(
        model_family="doubao-seedance-",
        backend_default="vendor",
        backend_env_key="DOUBAO_PROVIDER_MODE",
        credential_sources={
            "vendor": ("DOUBAO_API_KEY",),
        },
        supported_modalities=("t2v", "i2v"),
        image_input_mode={
            "vendor": "doubao_vendor_image_url",
        },
        audio_input_mode={
            "vendor": "doubao_vendor_audio_url",
        },
        reference_video_input_mode={
            "vendor": "doubao_vendor_video_url",
        },
    ),
    # Seedream — ByteDance's image companion to Seedance. Same Volcano Ark
    # account / DOUBAO_API_KEY, T2I/I2I only (no video).
    ProviderFamilyConfig(
        model_family="doubao-seedream-",
        backend_default="vendor",
        backend_env_key="DOUBAO_PROVIDER_MODE",
        credential_sources={
            "vendor": ("DOUBAO_API_KEY",),
        },
        supported_modalities=("t2i", "i2i"),
        image_input_mode={
            "vendor": "doubao_vendor_image_url",
        },
    ),
    ProviderFamilyConfig(
        model_family="hailuo-",
        backend_default="vendor",
        backend_env_key="HAILUO_PROVIDER_MODE",
        credential_sources={
            "vendor": ("MINIMAX_API_KEY", "HAILUO_API_KEY"),
        },
        supported_modalities=("t2v", "i2v"),
        image_input_mode={
            "vendor": "hailuo_vendor_image_url",
        },
    ),
    # MiniMax's official model ids carry a "MiniMax-Hailuo-" prefix; route
    # them to the same vendor-direct adapter.
    ProviderFamilyConfig(
        model_family="minimax-hailuo-",
        backend_default="vendor",
        backend_env_key="HAILUO_PROVIDER_MODE",
        credential_sources={
            "vendor": ("MINIMAX_API_KEY", "HAILUO_API_KEY"),
        },
        supported_modalities=("t2v", "i2v"),
        image_input_mode={
            "vendor": "hailuo_vendor_image_url",
        },
    ),
    # ── Black Forest Labs FLUX.2 (vendor-direct only) ──────────────────────
    ProviderFamilyConfig(
        model_family="flux-2",
        backend_default="vendor",
        backend_env_key="BFL_PROVIDER_MODE",
        credential_sources={
            "vendor": ("BFL_API_KEY",),
        },
        supported_modalities=("t2i", "i2i"),
        image_input_mode={
            "vendor": "bfl_vendor_image_url",
        },
    ),
    # ── Google (Gemini Image / Veo 3.1) — keys carried on instance row ────
    ProviderFamilyConfig(
        model_family="gemini-",
        backend_default="vendor",
        credential_sources={
            "vendor": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        },
        supported_modalities=("t2i", "i2i"),
        image_input_mode={
            "vendor": "gemini_vendor_image_part",
        },
    ),
    ProviderFamilyConfig(
        model_family="nano-banana",
        backend_default="vendor",
        credential_sources={
            "vendor": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        },
        supported_modalities=("t2i", "i2i"),
        image_input_mode={
            "vendor": "gemini_vendor_image_part",
        },
    ),
    ProviderFamilyConfig(
        model_family="veo-",
        backend_default="vendor",
        credential_sources={
            "vendor": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        },
        supported_modalities=("t2v", "i2v"),
        image_input_mode={
            "vendor": "veo_vendor_image_url",
        },
    ),
    # ── OpenAI GPT Image (vendor-direct only) ─────────────────────────────
    ProviderFamilyConfig(
        model_family="gpt-image-",
        backend_default="vendor",
        credential_sources={
            "vendor": ("OPENAI_API_KEY",),
        },
        supported_modalities=("t2i", "i2i"),
        image_input_mode={
            "vendor": "openai_vendor_image_url",
        },
    ),
    # ── ElevenLabs TTS (vendor-direct only) ───────────────────────────────
    ProviderFamilyConfig(
        model_family="eleven_",
        backend_default="vendor",
        credential_sources={
            "vendor": ("ELEVENLABS_API_KEY",),
        },
        supported_modalities=("tts",),
    ),
    # ── Fish Audio TTS (vendor-direct only) ───────────────────────────────
    ProviderFamilyConfig(
        model_family="fish-",
        backend_default="vendor",
        credential_sources={
            "vendor": ("FISH_AUDIO_API_KEY",),
        },
        supported_modalities=("tts",),
    ),
    # ── Cartesia TTS (vendor-direct only) ─────────────────────────────────
    ProviderFamilyConfig(
        model_family="sonic-",
        backend_default="vendor",
        credential_sources={
            "vendor": ("CARTESIA_API_KEY",),
        },
        supported_modalities=("tts",),
    ),
    # ── Remotion local render engine (flow A) ────────────────────────────
    # Not a cloud vendor: "remotion-*" models are rendered locally by the
    # Remotion microservice. No credentials, no env toggle — it always routes
    # to the in-process RemotionVideoAdapter. backend_default must be one of
    # the supported backends, so we reuse "vendor" (the dispatcher route is
    # keyed ("remotion-", "vendor")).
    ProviderFamilyConfig(
        model_family="remotion-",
        backend_default="vendor",
        credential_sources={},
        supported_modalities=("i2v", "t2v"),
    ),
    # ── fal.ai aggregator ────────────────────────────────────────────────
    ProviderFamilyConfig(
        model_family="fal-",
        backend_default="vendor",
        credential_sources={
            "vendor": ("FAL_API_KEY",),
        },
        supported_modalities=("t2i", "i2i", "i2v", "t2v"),
        image_input_mode={
            "vendor": "fal_vendor_image_url",
        },
        reference_video_input_mode={
            "vendor": "fal_vendor_video_url",
        },
    ),
)


def get_default_provider_registry() -> ProviderRegistry:
    return ProviderRegistry(DEFAULT_PROVIDER_FAMILIES)


def resolve_provider_backend(model_name: str, env: Optional[Mapping[str, str]] = None) -> str:
    return get_default_provider_registry().resolve_backend(model_name=model_name, env=env)


def swap_provider_modality(model_name: str, target_modality: str) -> Optional[str]:
    return get_default_provider_registry().swap_modality(model_name, target_modality)
