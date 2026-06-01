"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Loader2, X, Eye, EyeOff } from "lucide-react";
import {
    type InstanceTypeId,
    type ModelInstanceOut,
    type ModelInstanceCreate,
    type ModelInstanceUpdate,
} from "@/lib/api";
import { useTranslation } from "@/i18n";

// ─────────────────────────────────────────────────────────────────────────
// Vendor catalog (frontend mirror; the backend is canonical via
// /registry/vendors but the wizard is fast-loading and the same metadata
// powers other paths too).
//
// To add a new vendor: append to VENDORS, ensure src/utils/vendor_connectors.py
// has the matching entry, and src/utils/model_catalog.py has the model ids.
// ─────────────────────────────────────────────────────────────────────────

interface VendorMeta {
    id: string;
    label: string;
    capabilities: InstanceTypeId[];
    /** Either a flat list (single-capability vendor) or per-type buckets
     *  (multi-capability vendor — different model ids per type). */
    suggested_models: string[] | Partial<Record<InstanceTypeId, string[]>>;
    default_base_url: string;
    credential_keys: { key: string; label: string }[];
    docs_url?: string;
}


function suggestedModelsFor(vendor: VendorMeta, type: InstanceTypeId): string[] {
    if (Array.isArray(vendor.suggested_models)) return vendor.suggested_models;
    return vendor.suggested_models[type] ?? [];
}

const VENDORS: VendorMeta[] = [
    {
        id: "dashscope",
        label: "阿里云 DashScope",
        capabilities: ["llm", "t2i", "i2i", "i2v", "t2v", "r2v", "tts"],
        suggested_models: {
            llm: ["qwen3.5-plus", "qwen3-max", "qwen-plus", "qwen-flash"],
            t2i: [
                "wan2.6-t2i", "wan2.6-image", "wan2.5-t2i-preview",
                "qwen-image", "qwen-image-plus",
                "flux-schnell", "flux-dev",
            ],
            // wan2.6-image is the unified image endpoint — accepts pure-text
            // (t2i) AND reference-image edits (i2i), so it appears in both buckets.
            i2i: ["wan2.6-image", "wan2.5-i2i-preview", "qwen-image-edit"],
            i2v: [
                "wan2.6-i2v", "wan2.6-i2v-flash", "wan2.5-i2v-preview",
                "wan2.2-i2v-plus", "wan2.2-i2v-flash",
            ],
            t2v: ["wan2.6-i2v", "wan2.5-t2v-preview"],
            r2v: ["wan2.6-i2v"],
            tts: ["cosyvoice-v3-plus", "cosyvoice-v3-flash"],
        },
        default_base_url: "https://dashscope.aliyuncs.com",
        credential_keys: [{ key: "DASHSCOPE_API_KEY", label: "API Key" }],
        docs_url: "https://help.aliyun.com/zh/model-studio/",
    },
    {
        id: "openai",
        label: "OpenAI",
        capabilities: ["llm", "t2i", "i2i"],
        suggested_models: {
            llm: ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-mini", "o3-mini"],
            // GPT Image series — Elo 榜首,通过 Images / Edits API
            t2i: ["gpt-image-2", "gpt-image-1.5", "gpt-image-1"],
            i2i: ["gpt-image-2", "gpt-image-1.5", "gpt-image-1"],
        },
        default_base_url: "https://api.openai.com/v1",
        credential_keys: [{ key: "OPENAI_API_KEY", label: "API Key" }],
        docs_url: "https://platform.openai.com/docs",
    },
    {
        id: "anthropic",
        label: "Anthropic Claude",
        capabilities: ["llm"],
        suggested_models: ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
        default_base_url: "https://api.anthropic.com/v1",
        credential_keys: [{ key: "OPENAI_API_KEY", label: "API Key" }],
        docs_url: "https://docs.anthropic.com/",
    },
    {
        id: "deepseek",
        label: "DeepSeek",
        capabilities: ["llm"],
        suggested_models: ["deepseek-chat", "deepseek-reasoner"],
        default_base_url: "https://api.deepseek.com/v1",
        credential_keys: [{ key: "OPENAI_API_KEY", label: "API Key" }],
        docs_url: "https://api-docs.deepseek.com/",
    },
    {
        id: "moonshot",
        label: "Moonshot Kimi",
        capabilities: ["llm"],
        suggested_models: ["kimi-k2", "moonshot-v1-32k", "moonshot-v1-128k"],
        default_base_url: "https://api.moonshot.cn/v1",
        credential_keys: [{ key: "OPENAI_API_KEY", label: "API Key" }],
    },
    {
        id: "minimax",
        label: "MiniMax (Token Plan)",
        // MiniMax token plan covers LLM / TTS / T2I / I2V — one API key
        // unlocks all four. Each capability gets its own ModelInstance row
        // so the project settings can route per stage.
        capabilities: ["llm", "tts", "t2i", "i2v", "t2v"],
        suggested_models: {
            llm: ["MiniMax-M2.7", "MiniMax-M2", "MiniMax-Text-01", "abab6.5s-chat"],
            tts: ["speech-2.6-hd", "speech-2.6-turbo", "speech-02-hd", "speech-01-hd"],
            t2i: ["image-01"],
            i2v: ["MiniMax-Hailuo-2.3-Fast", "MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02"],
            t2v: ["MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02"],
        },
        // 国内官方域名(注意是 minimaxi.com,不是 minimax.io / minimax.chat)
        // 海外用户改用 https://api.minimax.io/v1
        default_base_url: "https://api.minimaxi.com/v1",
        credential_keys: [{ key: "MINIMAX_API_KEY", label: "API Key" }],
        docs_url: "https://platform.minimaxi.com/",
    },
    {
        id: "zhipu",
        label: "智谱 GLM",
        capabilities: ["llm"],
        suggested_models: ["glm-5", "glm-4.5", "glm-4-flash"],
        default_base_url: "https://open.bigmodel.cn/api/paas/v4",
        credential_keys: [{ key: "OPENAI_API_KEY", label: "API Key" }],
    },
    {
        id: "google",
        label: "Google Gemini",
        // Gemini family covers LLM (Gemini 2.5/3) + Image (Nano Banana) + Video (Veo 3.1).
        capabilities: ["llm", "t2i", "i2i", "i2v", "t2v"],
        suggested_models: {
            llm: ["gemini-3-pro", "gemini-3-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
            // 2026 SOTA 多参考图角色一致性
            t2i: ["gemini-3.1-flash-image", "nano-banana-pro", "nano-banana-2"],
            i2i: ["gemini-3.1-flash-image", "nano-banana-pro", "nano-banana-2"],
            // Veo 3.1 — 综合第一,原生 4K + 原生音频
            i2v: ["veo-3.1", "veo-3.1-fast"],
            t2v: ["veo-3.1", "veo-3.1-fast"],
        },
        default_base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        credential_keys: [{ key: "GOOGLE_API_KEY", label: "API Key" }],
        docs_url: "https://ai.google.dev/",
    },
    {
        id: "ollama",
        label: "本地 Ollama",
        capabilities: ["llm"],
        suggested_models: ["qwen2.5:72b", "llama3.3:70b", "deepseek-r1:32b"],
        default_base_url: "http://localhost:11434/v1",
        credential_keys: [{ key: "OPENAI_API_KEY", label: "API Key (一般填 ollama)" }],
        docs_url: "https://ollama.com/",
    },
    {
        id: "kling",
        label: "Kling AI (vendor-direct)",
        capabilities: ["i2v", "t2v"],
        suggested_models: ["kling-v3.0", "kling-v3", "kling-2.1-master"],
        default_base_url: "https://api-beijing.klingai.com/v1",
        credential_keys: [
            { key: "KLING_ACCESS_KEY", label: "Access Key" },
            { key: "KLING_SECRET_KEY", label: "Secret Key" },
        ],
    },
    {
        id: "vidu",
        label: "Vidu (vendor-direct)",
        capabilities: ["i2v", "t2v"],
        suggested_models: ["viduq3-pro", "viduq3-turbo"],
        default_base_url: "https://api.vidu.cn/ent/v2",
        credential_keys: [{ key: "VIDU_API_KEY", label: "API Key" }],
    },
    {
        id: "pixverse",
        label: "Pixverse (vendor-direct)",
        capabilities: ["i2v", "t2v"],
        suggested_models: ["pixverse-v4"],
        default_base_url: "https://app-api.pixverse.ai/openapi/v2",
        credential_keys: [{ key: "PIXVERSE_API_KEY", label: "API Key" }],
    },
    {
        id: "doubao",
        label: "字节豆包 Seedance / Seedream",
        // Volcano Engine Ark 同账号同时覆盖 Seedance 视频 + Seedream 图像。
        capabilities: ["t2i", "i2i", "i2v", "t2v"],
        suggested_models: {
            t2i: ["doubao-seedream-5.0", "doubao-seedream-4.5"],
            i2i: ["doubao-seedream-5.0", "doubao-seedream-4.5"],
            i2v: [
                "doubao-seedance-2.0-pro",
                "doubao-seedance-2.0",
                "doubao-seedance-1.5-pro",
                "doubao-seedance-1.0-pro",
            ],
            t2v: [
                "doubao-seedance-2.0-pro",
                "doubao-seedance-2.0",
                "doubao-seedance-1.5-pro",
                "doubao-seedance-1.0-pro",
            ],
        },
        default_base_url: "https://ark.cn-beijing.volces.com/api/v3",
        credential_keys: [{ key: "DOUBAO_API_KEY", label: "API Key" }],
        docs_url: "https://www.volcengine.com/docs/82379",
    },
    // (Hailuo merged into the unified `minimax` entry above so a single
    // MINIMAX_API_KEY covers LLM / TTS / T2I / I2V on the same plan.)
    // ── 2026 新增 ────────────────────────────────────────────────────
    {
        id: "bfl",
        label: "Black Forest Labs FLUX",
        // FLUX.2 Pro / Max / Flash — 2026 写实摄影基准,支持最多 10 张参考图
        capabilities: ["t2i", "i2i"],
        suggested_models: ["flux-2-pro", "flux-2-max", "flux-2-flash"],
        default_base_url: "https://api.bfl.ai/v1",
        credential_keys: [{ key: "BFL_API_KEY", label: "API Key" }],
        docs_url: "https://docs.bfl.ai/",
    },
    {
        id: "elevenlabs",
        label: "ElevenLabs",
        // 英语长篇朗读黄金标准
        capabilities: ["tts"],
        suggested_models: ["eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_v3"],
        default_base_url: "https://api.elevenlabs.io/v1",
        credential_keys: [{ key: "ELEVENLABS_API_KEY", label: "API Key" }],
        docs_url: "https://elevenlabs.io/docs/api-reference/text-to-speech",
    },
    {
        id: "fish-audio",
        label: "Fish Audio",
        // 80+ 语言,15 秒克隆,公开榜单 ELO #1
        capabilities: ["tts"],
        suggested_models: ["fish-s2", "fish-s1"],
        default_base_url: "https://api.fish.audio/v1",
        credential_keys: [{ key: "FISH_AUDIO_API_KEY", label: "API Key" }],
        docs_url: "https://docs.fish.audio/",
    },
    {
        id: "cartesia",
        label: "Cartesia",
        // 首字节 ~40-90ms,语音 Agent 实时对话最佳
        capabilities: ["tts"],
        suggested_models: ["sonic-3", "sonic-2"],
        default_base_url: "https://api.cartesia.ai",
        credential_keys: [{ key: "CARTESIA_API_KEY", label: "API Key" }],
        docs_url: "https://docs.cartesia.ai/",
    },
    {
        id: "fal",
        label: "fal.ai (聚合)",
        // 一个 key 直连 600+ 模型 — Veo 3.1 / Sora 2 / Kling 3.0 / Seedance / FLUX.2 等
        capabilities: ["t2i", "i2i", "i2v", "t2v"],
        suggested_models: {
            t2i: ["fal-flux-2-pro", "fal-seedream-5.0", "fal-nano-banana-pro"],
            i2i: ["fal-flux-2-pro", "fal-gemini-3.1-flash-image"],
            i2v: ["fal-veo-3.1", "fal-kling-3.0", "fal-seedance-1.5-pro"],
            t2v: ["fal-veo-3.1", "fal-kling-3.0", "fal-seedance-1.5-pro"],
        },
        default_base_url: "https://fal.run",
        credential_keys: [{ key: "FAL_API_KEY", label: "API Key (KEY_ID:KEY_SECRET)" }],
        docs_url: "https://fal.ai/docs",
    },
    {
        id: "remotion",
        label: "Remotion (本地渲染 · 零成本)",
        // Local render engine — turns stills into Ken Burns "pseudo-motion"
        // clips instead of calling a paid i2v API. Needs the render microservice
        // running. No credentials.
        capabilities: ["i2v", "t2v"],
        suggested_models: {
            i2v: ["remotion-kenburns"],
            t2v: ["remotion-kenburns"],
        },
        default_base_url: "",
        credential_keys: [],
        docs_url: "https://www.remotion.dev/",
    },
];

// Translation keys for the seven instance type labels surfaced in the wizard.
// Resolved via `t(...)` at render time so they follow the active locale.
const TYPE_LABEL_KEYS: Record<InstanceTypeId, { key: string; fallback: string }> = {
    llm: { key: "instances.typeLLMLong", fallback: "LLM (剧本/润色)" },
    t2i: { key: "instances.typeT2ILong", fallback: "Text-to-Image" },
    i2i: { key: "instances.typeI2ILong", fallback: "Image-to-Image (storyboard)" },
    i2v: { key: "instances.typeI2VLong", fallback: "Image-to-Video" },
    t2v: { key: "instances.typeT2VLong", fallback: "Text-to-Video" },
    r2v: { key: "instances.typeR2VLong", fallback: "Reference-to-Video" },
    tts: { key: "instances.typeTTSLong", fallback: "Text-to-Speech" },
};


export interface InstanceWizardProps {
    initialType?: InstanceTypeId;
    /** When provided, the wizard runs in "edit" mode for that instance. */
    editing?: ModelInstanceOut | null;
    onClose: () => void;
    onSave: (
        payload: ModelInstanceCreate | ModelInstanceUpdate,
        editingId: string | null,
    ) => Promise<void>;
}


export function InstanceWizard({ initialType, editing, onClose, onSave }: InstanceWizardProps) {
    const { t } = useTranslation();
    const typeLabel = (id: InstanceTypeId) => {
        const meta = TYPE_LABEL_KEYS[id];
        return t(meta.key, undefined, meta.fallback);
    };
    const isEdit = !!editing;
    // When the user opened the wizard from a specific type section (e.g.
    // "添加 LLM 实例"), skip step 1 — type is already known.
    const [step, setStep] = useState<1 | 2 | 3>(isEdit ? 3 : initialType ? 2 : 1);
    const [type, setType] = useState<InstanceTypeId>(initialType ?? "llm");
    const [vendorId, setVendorId] = useState<string>(editing?.vendor_id ?? "");
    const [modelName, setModelName] = useState<string>(editing?.model_name ?? "");
    const [displayName, setDisplayName] = useState<string>(editing?.display_name ?? "");
    const [baseUrl, setBaseUrl] = useState<string>(editing?.base_url ?? "");
    const [credentials, setCredentials] = useState<Record<string, string>>({});
    const [revealed, setRevealed] = useState<Record<string, boolean>>({});
    const [setAsDefault, setSetAsDefault] = useState(false);
    const [saving, setSaving] = useState(false);
    const [saveError, setSaveError] = useState("");

    const eligibleVendors = useMemo(() => VENDORS.filter((v) => v.capabilities.includes(type)), [type]);
    const vendorMeta = useMemo(() => VENDORS.find((v) => v.id === vendorId), [vendorId]);
    const typeSuggestions = useMemo(
        () => (vendorMeta ? suggestedModelsFor(vendorMeta, type) : []),
        [vendorMeta, type],
    );

    // When type changes, reset vendor selection if the current vendor no longer fits.
    useEffect(() => {
        if (vendorId && !eligibleVendors.find((v) => v.id === vendorId)) {
            setVendorId("");
        }
    }, [type, eligibleVendors, vendorId]);

    // When vendor changes, prefill suggested model (for current type) + base url.
    useEffect(() => {
        if (!vendorMeta) return;
        if (!modelName) setModelName(typeSuggestions[0] ?? "");
        if (!baseUrl) setBaseUrl(vendorMeta.default_base_url);
    }, [vendorId]); // eslint-disable-line react-hooks/exhaustive-deps

    const requiredCredKeys = vendorMeta?.credential_keys ?? [];
    const credsValid = requiredCredKeys.every((k) => (credentials[k.key] || "").trim().length > 0);
    const canSubmit = vendorId && modelName && displayName && (isEdit || credsValid);

    const handleSave = async () => {
        if (!canSubmit) return;
        setSaving(true);
        setSaveError("");
        try {
            if (isEdit) {
                const update: ModelInstanceUpdate = {
                    display_name: displayName,
                    model_name: modelName,
                    base_url: baseUrl,
                };
                if (Object.keys(credentials).length > 0) {
                    update.credentials = credentials;
                }
                await onSave(update, editing!.id);
            } else {
                const create: ModelInstanceCreate = {
                    instance_type: type,
                    vendor_id: vendorId,
                    model_name: modelName,
                    display_name: displayName,
                    credentials,
                    base_url: baseUrl,
                    is_default: setAsDefault,
                };
                await onSave(create, null);
            }
            onClose();
        } catch (e) {
            setSaveError(e instanceof Error ? e.message : String(e));
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-zinc-950 border border-white/10 rounded-2xl shadow-2xl">
                <button
                    type="button"
                    onClick={onClose}
                    className="absolute top-4 right-4 p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/5 transition-colors z-10"
                >
                    <X size={18} />
                </button>

                <div className="p-6">
                    <h2 className="text-lg font-bold text-white">
                        {isEdit
                            ? t("instances.wizardEditTitle", { name: editing?.display_name ?? "" }, `编辑实例 · ${editing?.display_name}`)
                            : initialType
                                ? t("instances.wizardAddTypedTitle", { label: typeLabel(initialType) }, `添加 ${typeLabel(initialType)} 实例`)
                                : t("instances.wizardAddTitle", undefined, "添加模型实例")}
                    </h2>
                    <p className="text-xs text-gray-500 mt-1">
                        {isEdit
                            ? t("instances.wizardEditHint", undefined, "改完保存即可。留空凭证字段会保留原值。")
                            : initialType
                                ? t("instances.wizardAddTypedHint", undefined, "两步完成:选厂商 → 填凭证")
                                : t("instances.wizardAddHint", undefined, "三步完成:选类型 → 选厂商 → 填凭证")}
                    </p>

                    {/* Stepper — only the visible steps. When initialType
                        was provided, step 1 is suppressed so the bar shows
                        two segments instead of three. */}
                    {!isEdit && (
                        <div className="flex items-center gap-2 mt-4 mb-6">
                            {(initialType ? [2, 3] : [1, 2, 3]).map((n) => (
                                <div
                                    key={n}
                                    className={`flex-1 h-1 rounded-full transition-colors ${step >= n ? "bg-amber-500" : "bg-white/10"}`}
                                />
                            ))}
                        </div>
                    )}

                    {/* Step 1: Type */}
                    {step === 1 && !isEdit && (
                        <div className="space-y-3">
                            <div className="text-xs uppercase tracking-wider text-gray-500">{t("instances.wizardStep1Heading", undefined, "第 1 步 · 选类型")}</div>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                                {(Object.keys(TYPE_LABEL_KEYS) as InstanceTypeId[]).map((id) => (
                                    <button
                                        key={id}
                                        type="button"
                                        onClick={() => setType(id)}
                                        className={`p-3 rounded-lg border text-left transition-colors ${type === id ? "border-amber-500/60 bg-amber-500/10 text-amber-200" : "border-white/10 bg-white/5 text-gray-300 hover:border-white/20"}`}
                                    >
                                        <div className="text-xs font-mono uppercase">{id}</div>
                                        <div className="text-[10px] text-gray-500 mt-0.5">{typeLabel(id)}</div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Step 2: Vendor */}
                    {step === 2 && !isEdit && (
                        <div className="space-y-3">
                            <div className="text-xs uppercase tracking-wider text-gray-500">{t("instances.wizardStep2Heading", undefined, "第 2 步 · 选厂商")}</div>
                            <div className="grid grid-cols-2 gap-2">
                                {eligibleVendors.map((v) => (
                                    <button
                                        key={v.id}
                                        type="button"
                                        onClick={() => setVendorId(v.id)}
                                        className={`p-3 rounded-lg border text-left transition-colors ${vendorId === v.id ? "border-amber-500/60 bg-amber-500/10 text-amber-200" : "border-white/10 bg-white/5 text-gray-300 hover:border-white/20"}`}
                                    >
                                        <div className="text-sm font-medium text-white">{v.label}</div>
                                        <div className="text-[10px] text-gray-500 mt-0.5 truncate font-mono">{suggestedModelsFor(v, type).slice(0, 3).join(", ")}</div>
                                    </button>
                                ))}
                            </div>
                            {eligibleVendors.length === 0 && (
                                <p className="text-xs text-gray-500">{t("instances.noVendorsForType", undefined, "该类型暂无可用厂商。")}</p>
                            )}
                        </div>
                    )}

                    {/* Step 3: Details */}
                    {step === 3 && (
                        <div className="space-y-4">
                            <div className="text-xs uppercase tracking-wider text-gray-500">{t("instances.wizardStep3Heading", undefined, "第 3 步 · 填详情")}</div>

                            <div>
                                <label className="block text-xs font-medium text-gray-300 mb-1.5">{t("instances.aliasLabel", undefined, "别名")} <span className="text-rose-500">*</span></label>
                                <input
                                    type="text"
                                    value={displayName}
                                    onChange={(e) => setDisplayName(e.target.value)}
                                    placeholder={t("instances.aliasPlaceholder", undefined, "例如:工作室主力 LLM")}
                                    className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/30"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-gray-300 mb-1.5">{t("instances.modelLabel", undefined, "模型")} <span className="text-rose-500">*</span></label>
                                <input
                                    type="text"
                                    value={modelName}
                                    onChange={(e) => setModelName(e.target.value)}
                                    placeholder={t("instances.modelIdPlaceholder", undefined, "模型 ID")}
                                    className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/30 font-mono"
                                />
                                {typeSuggestions.length > 0 && (
                                    <div className="flex flex-wrap gap-1 mt-2">
                                        {typeSuggestions.map((m) => (
                                            <button
                                                key={m}
                                                type="button"
                                                onClick={() => setModelName(m)}
                                                className={`px-2 py-1 text-[11px] rounded border transition-colors ${modelName === m ? "border-amber-500/60 bg-amber-500/15 text-amber-200" : "border-white/10 bg-white/5 text-gray-400 hover:text-gray-200"}`}
                                            >
                                                {m}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div>
                                <label className="flex items-center justify-between text-xs font-medium text-gray-300 mb-1.5">
                                    <span>{t("instances.baseUrlLabel")}</span>
                                    <span className="text-[10px] text-gray-600">{t("instances.baseUrlHint", undefined, "留空使用厂商默认")}</span>
                                </label>
                                <input
                                    type="text"
                                    value={baseUrl}
                                    onChange={(e) => setBaseUrl(e.target.value)}
                                    placeholder={vendorMeta?.default_base_url ?? ""}
                                    className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/30 font-mono"
                                />
                            </div>

                            {requiredCredKeys.map((ck) => (
                                <div key={ck.key}>
                                    <label className="block text-xs font-medium text-gray-300 mb-1.5">
                                        {ck.label} {!isEdit && <span className="text-rose-500">*</span>}
                                        {isEdit && <span className="text-[10px] text-gray-600 ml-1">{t("instances.credentialKeepHint", undefined, "留空保留原值")}</span>}
                                    </label>
                                    <div className="relative">
                                        <input
                                            type={revealed[ck.key] ? "text" : "password"}
                                            value={credentials[ck.key] || ""}
                                            onChange={(e) => setCredentials((c) => ({ ...c, [ck.key]: e.target.value }))}
                                            placeholder="••••••••"
                                            className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 pr-9 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/30"
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setRevealed((r) => ({ ...r, [ck.key]: !r[ck.key] }))}
                                            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded text-gray-500 hover:text-gray-300 hover:bg-white/5"
                                        >
                                            {revealed[ck.key] ? <EyeOff size={14} /> : <Eye size={14} />}
                                        </button>
                                    </div>
                                </div>
                            ))}

                            {!isEdit && (
                                <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={setAsDefault}
                                        onChange={(e) => setSetAsDefault(e.target.checked)}
                                        className="rounded border-white/20"
                                    />
                                    {t("instances.setAsTypeDefault", undefined, "设为该类型的默认实例")}
                                </label>
                            )}

                            {vendorMeta?.docs_url && (
                                <a
                                    href={vendorMeta.docs_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-[11px] text-gray-500 hover:text-gray-300"
                                >
                                    {t("instances.viewVendorDocs", { vendor: vendorMeta.label }, `查看 ${vendorMeta.label} 文档`)}
                                </a>
                            )}

                            {saveError && (
                                <div className="px-3 py-2 rounded bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300">
                                    {saveError}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Footer */}
                    <div className="flex items-center justify-between mt-6 pt-4 border-t border-white/10">
                        {/* "上一步" only available when there's a previous
                            visible step. With initialType, step 1 is hidden
                            so step 2 is the first visible step. */}
                        {!isEdit && step > (initialType ? 2 : 1) ? (
                            <button
                                type="button"
                                onClick={() => setStep((s) => (s - 1) as 1 | 2 | 3)}
                                className="flex items-center gap-1 px-3 py-2 text-xs text-gray-400 hover:text-white transition-colors"
                            >
                                <ChevronLeft size={14} />
                                {t("instances.wizardPrev")}
                            </button>
                        ) : (
                            <div />
                        )}

                        {step < 3 && !isEdit ? (
                            <button
                                type="button"
                                onClick={() => setStep((s) => (s + 1) as 1 | 2 | 3)}
                                disabled={(step === 1 && !type) || (step === 2 && !vendorId)}
                                className="flex items-center gap-1 px-4 py-2 text-xs font-medium bg-amber-600 hover:bg-amber-500 text-white rounded-lg transition-colors disabled:opacity-50"
                            >
                                {t("instances.wizardNext")}
                                <ChevronRight size={14} />
                            </button>
                        ) : (
                            <button
                                type="button"
                                onClick={handleSave}
                                disabled={!canSubmit || saving}
                                className="flex items-center gap-1 px-4 py-2 text-xs font-medium bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white rounded-lg transition-colors disabled:opacity-50"
                            >
                                {saving && <Loader2 size={14} className="animate-spin" />}
                                {isEdit ? t("instances.save") : t("instances.createInstance", undefined, "创建实例")}
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
