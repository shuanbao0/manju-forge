"""Tests for the Remotion video flows (A: adapter, B: MG generator) + shared spec."""
import os
import tempfile
from types import SimpleNamespace

import pytest

from src.models.remotion_spec import (
    Clip,
    Focus,
    KenBurnsImageLayer,
    SubtitleCue,
    SubtitleLayer,
    VideoSpec,
)
from src.models.remotion_renderer import RemotionRenderClient
from src.models.remotion_adapter import (
    RemotionVideoAdapter,
    camera_to_kenburns,
    resolution_to_size,
)
from src.models.video_dispatcher import VideoGenerationContext, build_default_dispatcher
from src.utils.provider_registry import resolve_provider_backend, swap_provider_modality
from src.utils.vendor_connectors import get_default_vendor_registry


# ── Shared VideoSpec DTO ──────────────────────────────────────────────────


def test_videospec_roundtrip_alias_and_discriminator():
    spec = VideoSpec(
        clips=[
            Clip(
                duration_sec=5,
                audio_src="audio/f1.mp3",
                layers=[
                    KenBurnsImageLayer(src="video_inputs/t1.png", to=Focus(scale=1.15)),
                    SubtitleLayer(cues=[SubtitleCue(text="你好", to_sec=5)]),
                ],
            )
        ]
    )
    payload = spec.to_payload()
    layer0 = payload["clips"][0]["layers"][0]
    # `from_` serializes to `from` for the zod side
    assert layer0["from"]["scale"] == 1.0
    assert layer0["to"]["scale"] == 1.15
    assert payload["clips"][0]["layers"][1]["type"] == "subtitle"

    again = VideoSpec.model_validate(payload)
    assert again.clips[0].layers[0].src == "video_inputs/t1.png"
    assert again.clips[0].layers[1].cues[0].text == "你好"
    assert again.total_duration_sec == 5.0


# ── Flow A: routing + adapter ─────────────────────────────────────────────


def test_remotion_backend_is_vendor():
    assert resolve_provider_backend("remotion-kenburns") == "vendor"


def test_dispatcher_routes_remotion_to_adapter():
    disp = build_default_dispatcher(video_generator=None)
    adapter = disp.resolve("remotion-kenburns", "vendor")
    assert isinstance(adapter, RemotionVideoAdapter)
    # existing routes unaffected
    assert type(disp.resolve("wan2.6-i2v", "dashscope")).__name__ == "WanxDashScopeAdapter"
    assert type(disp.resolve("kling-v3.0", "vendor")).__name__ == "KlingVendorAdapter"


def test_remotion_vendor_has_no_credentials():
    reg = get_default_vendor_registry()
    vendor = reg.by_id("remotion")
    assert vendor is not None
    assert vendor.capabilities == ("i2v", "t2v")
    assert vendor.common_fields == ()
    assert vendor.modes == ()


@pytest.mark.parametrize(
    "movement,expect",
    [
        ("推近", lambda f0, f1: f1.scale > f0.scale),         # zoom in
        ("拉远", lambda f0, f1: f0.scale > f1.scale),         # zoom out
        ("摇右", lambda f0, f1: f0.x < f1.x),                 # pan right
        (None, lambda f0, f1: f1.scale > f0.scale),           # default gentle push
    ],
)
def test_camera_to_kenburns(movement, expect):
    f0, f1 = camera_to_kenburns(movement, None)
    assert expect(f0, f1)


def test_resolution_to_size():
    assert resolution_to_size("1080p", "9:16") == (1080, 1920)
    assert resolution_to_size("720p", "16:9") == (1280, 720)
    assert resolution_to_size("1080p", "1:1") == (1080, 1080)


def test_adapter_builds_single_clip_spec_from_extras():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "video_inputs"), exist_ok=True)
    open(os.path.join(root, "video_inputs", "t1.png"), "wb").write(b"\x89PNG")
    os.makedirs(os.path.join(root, "audio"), exist_ok=True)
    open(os.path.join(root, "audio", "f1.mp3"), "wb").write(b"ID3")

    captured = {}

    class FakeClient(RemotionRenderClient):
        def render(self, spec, output_path):
            captured["spec"] = spec.to_payload()
            return 1.23

    adapter = RemotionVideoAdapter(client=FakeClient(render_url="http://x", output_root=root))
    task = SimpleNamespace(duration=6, resolution="1080p", frame_id="fr1", model="remotion-kenburns")
    ctx = VideoGenerationContext(
        task=task,
        output_path=os.path.join(root, "video", "video_xyz.mp4"),
        img_url="video_inputs/t1.png",
        extras={
            "camera_movement": "摇右",
            "shot_size": "中景",
            "dialogue": "命运的齿轮开始转动",
            "duration_seconds": 6,
            "aspect_ratio": "9:16",
            "dialogue_audio": "audio/f1.mp3",
        },
    )
    out, secs = adapter.generate(ctx)
    assert secs == 1.23 and out.endswith("video_xyz.mp4")
    spec = captured["spec"]
    assert (spec["width"], spec["height"]) == (1080, 1920)
    clip = spec["clips"][0]
    assert clip["duration_sec"] == 6 and clip["audio_src"] == "audio/f1.mp3"
    assert [l["type"] for l in clip["layers"]] == ["kenburns_image", "subtitle"]
    assert clip["layers"][1]["cues"][0]["text"] == "命运的齿轮开始转动"


def test_adapter_requires_image():
    adapter = RemotionVideoAdapter(client=RemotionRenderClient(render_url="http://x", output_root="/tmp"))
    ctx = VideoGenerationContext(
        task=SimpleNamespace(duration=5, resolution="1080p"),
        output_path="/tmp/out.mp4",
        img_url=None,
    )
    with pytest.raises(ValueError):
        adapter.generate(ctx)


# ── Flow B: MG generator ──────────────────────────────────────────────────


def test_mg_generator_parses_validates_and_renders():
    from src.apps.comic_gen.remotion_mg import RemotionMGGenerator

    sample = """```json
    {"clips":[
      {"duration_sec":4,"transition_in":{"type":"fade","duration_sec":0.5},
       "layers":[{"type":"background","gradient":["#0f172a","#1e293b"]},
                 {"type":"title_card","text":"标题","subtitle":"副标题"}]},
      {"duration_sec":5,"layers":[{"type":"background","color":"#111"},
       {"type":"bullet_list","items":["a","b","c"]}]}
    ]}
    ```"""

    gen = RemotionMGGenerator(render_client=object())
    gen.llm = SimpleNamespace(chat=lambda messages, response_format=None: sample)

    spec = gen.generate_spec("讲讲增长", aspect_ratio="9:16")
    assert (spec.width, spec.height) == (1080, 1920)  # backfilled
    assert len(spec.clips) == 2
    assert spec.clips[0].layers[1].text == "标题"
    assert spec.clips[1].layers[1].items == ["a", "b", "c"]

    captured = {}

    def _fake_render(s, p):
        captured["p"] = p
        return 2.5

    gen._render_client = SimpleNamespace(render=_fake_render)
    assert gen.render(spec, "/tmp/out/video/remotion_x.mp4") == 2.5
    assert captured["p"].endswith("remotion_x.mp4")


def test_mg_generator_rejects_bad_json():
    from src.apps.comic_gen.remotion_mg import RemotionMGGenerator

    gen = RemotionMGGenerator(render_client=object())
    gen.llm = SimpleNamespace(chat=lambda messages, response_format=None: "not json at all")
    with pytest.raises(ValueError):
        gen.generate_spec("x")


# ── Flow B step 0: copy generation + quality (Builder + Strategy) ──────────


@pytest.mark.parametrize(
    "quality,target_sec,sections",
    [("concise", "30", "2"), ("standard", "55", "3"), ("rich", "90", "4")],
)
def test_copy_prompt_builder_encodes_quality(quality, target_sec, sections):
    from src.apps.comic_gen.remotion_mg import MGCopyPromptBuilder, MGCopyQuality

    messages = (
        MGCopyPromptBuilder()
        .with_title("AI 生成小说的优点")
        .with_existing("它能几秒扩写成章节")
        .with_quality(MGCopyQuality(quality))
        .with_style_hint("科技深色")
        .build()
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert f"{target_sec} 秒" in system          # Strategy table drives duration
    assert f"{sections} 个小节" in system         # ...and section count
    assert "科技深色" in system
    assert "AI 生成小说的优点" in user
    assert "它能几秒扩写成章节" in user


def test_spec_prompt_builder_encodes_quality_clip_hint():
    from src.apps.comic_gen.remotion_mg import MGPromptBuilder, MGCopyQuality

    system = (
        MGPromptBuilder()
        .with_topic("讲讲增长")
        .with_size(1080, 1920)
        .with_quality(MGCopyQuality.RICH)
        .build()[0]["content"]
    )
    assert "8-10 个" in system  # rich → more clips


def test_mg_generator_generate_copy_returns_text():
    from src.apps.comic_gen.remotion_mg import RemotionMGGenerator, MGCopyQuality

    seen = {}

    def _chat(messages, response_format=None):
        seen["response_format"] = response_format  # copy step must NOT force json
        return "  生成好的文案正文  "

    gen = RemotionMGGenerator(render_client=object())
    gen.llm = SimpleNamespace(chat=_chat)
    out = gen.generate_copy("标题", "素材", quality=MGCopyQuality.CONCISE)
    assert out == "生成好的文案正文"
    assert seen["response_format"] is None


# ── Renderer self-verifies the cross-process path contract ────────────────


def test_render_raises_when_output_missing(monkeypatch):
    from src.models import remotion_renderer as rr

    client = rr.RemotionRenderClient(render_url="http://x", output_root="/tmp")

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True, "seconds": 1.0}  # service claims success...

    monkeypatch.setattr(rr.requests, "post", lambda *a, **k: _Resp())
    spec = VideoSpec(clips=[Clip(duration_sec=3)])
    # ...but no file was written where the backend will serve it → hard error.
    with pytest.raises(rr.RemotionRenderError, match="未出现在"):
        client.render(spec, "/tmp/does-not-exist/remotion_z.mp4")
