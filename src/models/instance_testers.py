"""Connectivity probes for each instance type — Strategy pattern.

The Settings UI's "测试连通" button calls
``POST /me/instances/{id}/test`` which dispatches here. Each tester is
keyed by :class:`InstanceType` and gets the decrypted instance to ping
the actual provider with a minimal payload.

Adding a new tester = one entry in :data:`_TESTERS`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict

from .instance import InstanceType, ModelInstance

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    ok: bool
    latency_ms: float = 0.0
    error: str = ""


Tester = Callable[[ModelInstance], TestResult]


# ─────────────────────────────────────────────────────────────────────────
# Individual probes — kept light: a single short request per type.
# ─────────────────────────────────────────────────────────────────────────


def _llm_probe(instance: ModelInstance) -> TestResult:
    """Send a tiny "ping" message via the OpenAI-compatible client.

    All current LLM vendors (DashScope, OpenAI, Anthropic-compat,
    DeepSeek, ...) speak the same chat-completion shape, so one client
    handles them all. Authoritative error code goes back to the user.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return TestResult(ok=False, error="openai package not installed")

    api_key = (
        instance.credentials.get("OPENAI_API_KEY")
        or instance.credentials.get("MINIMAX_API_KEY")
        or instance.credentials.get("DASHSCOPE_API_KEY")
    )
    if not api_key:
        return TestResult(ok=False, error="No API key configured (OPENAI_API_KEY / MINIMAX_API_KEY / DASHSCOPE_API_KEY)")

    base_url = instance.base_url or _default_llm_base_url(instance.vendor_id)
    client = OpenAI(api_key=api_key, base_url=base_url)
    started = time.time()
    try:
        client.chat.completions.create(
            model=instance.model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:  # pragma: no cover — surfaces network/auth errors
        return TestResult(ok=False, latency_ms=(time.time() - started) * 1000, error=str(e))
    return TestResult(ok=True, latency_ms=(time.time() - started) * 1000)


def _default_llm_base_url(vendor_id: str) -> str:
    """Map vendor id → default OpenAI-compatible endpoint."""
    return {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "moonshot": "https://api.moonshot.cn/v1",
        "minimax": "https://api.minimaxi.com/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
        "ollama": "http://localhost:11434/v1",
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # New 2026 vendors — not LLM, but listed for safety in case the LLM
        # tester is invoked with an unsupported vendor id.
        "bfl": "https://api.bfl.ai/v1",
        "fal": "https://fal.run",
        "elevenlabs": "https://api.elevenlabs.io/v1",
        "fish-audio": "https://api.fish.audio/v1",
        "cartesia": "https://api.cartesia.ai",
    }.get(vendor_id, "https://api.openai.com/v1")


def _dashscope_probe(instance: ModelInstance) -> TestResult:
    """Probe used for T2I / I2I / I2V / T2V / R2V instances when the
    vendor is DashScope. We just verify the API key is set — actually
    issuing a generation request would burn quota, so we treat presence
    of the key as "configured". Vendor-direct adapters get their own
    probes once they're implemented."""
    if instance.vendor_id == "dashscope":
        if instance.credentials.get("DASHSCOPE_API_KEY"):
            return TestResult(ok=True)
        return TestResult(ok=False, error="DASHSCOPE_API_KEY missing")
    # Remotion is a local render engine, not an API: health-check the service.
    if instance.vendor_id == "remotion":
        from .remotion_renderer import RemotionRenderClient

        started = time.time()
        ok = RemotionRenderClient(render_url=instance.base_url or None).health()
        latency = (time.time() - started) * 1000
        return TestResult(
            ok=ok,
            latency_ms=latency,
            error="" if ok else "Remotion render service unreachable (start remotion/server.mjs)",
        )
    # Vendor-direct: verify each required credential is set; defer real
    # network probe to provider-specific testers added later.
    expected = _vendor_direct_required_keys(instance.vendor_id)
    missing = [k for k in expected if not instance.credentials.get(k)]
    if missing:
        return TestResult(ok=False, error=f"missing credential(s): {', '.join(missing)}")
    return TestResult(ok=True)


def _vendor_direct_required_keys(vendor_id: str) -> tuple[str, ...]:
    return {
        "kling": ("KLING_ACCESS_KEY", "KLING_SECRET_KEY"),
        "vidu": ("VIDU_API_KEY",),
        "pixverse": ("PIXVERSE_API_KEY",),
        "doubao": ("DOUBAO_API_KEY",),
        "hailuo": ("HAILUO_API_KEY",),
        # MiniMax token plan covers I2V/T2I/TTS — one MINIMAX_API_KEY is enough.
        "minimax": ("MINIMAX_API_KEY",),
        # 2026 additions — vendor-direct keys validated by presence only;
        # actual network probes get wired once each adapter lands.
        "bfl": ("BFL_API_KEY",),
        "google": ("GOOGLE_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "fal": ("FAL_API_KEY",),
    }.get(vendor_id, ())


def _tts_probe(instance: ModelInstance) -> TestResult:
    """TTS supports several backends:
    - DashScope CosyVoice → needs DASHSCOPE_API_KEY
    - MiniMax T2A v2 → needs MINIMAX_API_KEY
    - ElevenLabs / Fish Audio / Cartesia → vendor-direct, presence check
    """
    vendor_required = {
        "minimax": "MINIMAX_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
        "fish-audio": "FISH_AUDIO_API_KEY",
        "cartesia": "CARTESIA_API_KEY",
    }.get(instance.vendor_id)
    if vendor_required is not None:
        if instance.credentials.get(vendor_required):
            return TestResult(ok=True)
        return TestResult(ok=False, error=f"{vendor_required} missing")
    if not instance.credentials.get("DASHSCOPE_API_KEY"):
        return TestResult(ok=False, error="DASHSCOPE_API_KEY missing")
    return TestResult(ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Strategy registry
# ─────────────────────────────────────────────────────────────────────────


_TESTERS: Dict[InstanceType, Tester] = {
    InstanceType.LLM: _llm_probe,
    InstanceType.T2I: _dashscope_probe,
    InstanceType.I2I: _dashscope_probe,
    InstanceType.I2V: _dashscope_probe,
    InstanceType.T2V: _dashscope_probe,
    InstanceType.R2V: _dashscope_probe,
    InstanceType.TTS: _tts_probe,
}


def test_instance(instance: ModelInstance) -> TestResult:
    tester = _TESTERS.get(instance.instance_type)
    if tester is None:
        return TestResult(ok=False, error=f"no tester registered for {instance.instance_type.value}")
    try:
        return tester(instance)
    except Exception as e:  # pragma: no cover
        logger.exception("instance probe failed")
        return TestResult(ok=False, error=str(e))


__all__ = ["TestResult", "test_instance"]
