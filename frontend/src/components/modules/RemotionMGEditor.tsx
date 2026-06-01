"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Sparkles, Film, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { getAssetUrl } from "@/lib/utils";

/**
 * Flow B editor — chat-only motion-graphics / explainer videos rendered by
 * Remotion. Three steps, all self-contained (no zustand coupling):
 *   1. 输入文案 → 创建项目 (createMGProject)
 *   2. 生成动效脚本 (generateMGSpec, LLM 出 VideoSpec)
 *   3. 渲染视频 (renderMGVideo) → 播放
 *
 * Inline WYSIWYG preview via @remotion/player is a planned follow-up; for now
 * the rendered MP4 is shown with a plain <video>.
 */
export default function RemotionMGEditor({ id }: { id: string | null }) {
    const [project, setProject] = useState<any>(null);
    const [title, setTitle] = useState("");
    const [text, setText] = useState("");
    const [aspect, setAspect] = useState("9:16");
    const [styleHint, setStyleHint] = useState("");
    const [busy, setBusy] = useState<null | "create" | "spec" | "render">(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!id) return;
        api.getProject(id)
            .then((p) => {
                setProject(p);
                setTitle(p.title || "");
                setText(p.originalText || "");
                setAspect(p.model_settings?.storyboard_aspect_ratio || "9:16");
            })
            .catch((e) => setError(String(e?.message || e)));
    }, [id]);

    const goHome = () => { window.location.hash = "#/"; };

    const handleGenerate = async () => {
        setError(null);
        try {
            let p = project;
            // Create the project first if this is a fresh editor.
            if (!p) {
                if (!title.trim() || !text.trim()) {
                    setError("请填写标题和文案");
                    return;
                }
                setBusy("create");
                p = await api.createMGProject(title.trim(), text.trim(), aspect);
                setProject(p);
                window.location.hash = `#/mg/${p.id}`;
            }
            setBusy("spec");
            const updated = await api.generateMGSpec(p.id, styleHint || undefined);
            setProject(updated);
        } catch (e: any) {
            setError(e?.response?.data?.detail || String(e?.message || e));
        } finally {
            setBusy(null);
        }
    };

    const handleRender = async () => {
        if (!project) return;
        setError(null);
        setBusy("render");
        try {
            const updated = await api.renderMGVideo(project.id);
            setProject(updated);
        } catch (e: any) {
            setError(e?.response?.data?.detail || String(e?.message || e));
        } finally {
            setBusy(null);
        }
    };

    const specClips = project?.mg_spec?.clips?.length ?? 0;

    return (
        <div className="min-h-screen bg-gray-950 text-white">
            <div className="border-b border-white/10 px-6 py-3 flex items-center gap-3">
                <button onClick={goHome} className="p-2 rounded-lg hover:bg-white/10">
                    <ArrowLeft size={18} />
                </button>
                <Film size={18} className="text-emerald-400" />
                <h1 className="font-display font-bold">图文 / 解说视频（Remotion · 本地零成本）</h1>
            </div>

            <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
                {error && (
                    <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 px-4 py-3 text-sm">
                        {error}
                    </div>
                )}

                {/* Step 1: input */}
                <section className="space-y-3">
                    <label className="block text-sm text-gray-400">标题</label>
                    <input
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        disabled={!!project}
                        placeholder="例如：三个增长黑客技巧"
                        className="w-full bg-gray-900 border border-white/10 rounded-lg px-3 py-2 text-sm disabled:opacity-60"
                    />
                    <label className="block text-sm text-gray-400">文案 / 主题</label>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        disabled={!!project}
                        rows={6}
                        placeholder="粘贴一段要讲解的内容，或描述视频主题。LLM 会把它转成动效脚本。"
                        className="w-full bg-gray-900 border border-white/10 rounded-lg px-3 py-2 text-sm disabled:opacity-60"
                    />
                    <div className="flex items-center gap-3">
                        <label className="text-sm text-gray-400">画幅</label>
                        <select
                            value={aspect}
                            onChange={(e) => setAspect(e.target.value)}
                            disabled={!!project}
                            className="bg-gray-900 border border-white/10 rounded-lg px-3 py-2 text-sm disabled:opacity-60"
                        >
                            <option value="9:16">9:16 竖屏</option>
                            <option value="16:9">16:9 横屏</option>
                            <option value="1:1">1:1 方形</option>
                        </select>
                        <input
                            value={styleHint}
                            onChange={(e) => setStyleHint(e.target.value)}
                            placeholder="风格倾向（可选）：科技深色 / 明亮活泼…"
                            className="flex-1 bg-gray-900 border border-white/10 rounded-lg px-3 py-2 text-sm"
                        />
                    </div>
                </section>

                {/* Step 2: generate spec */}
                <button
                    onClick={handleGenerate}
                    disabled={busy !== null}
                    className="w-full bg-primary hover:bg-primary/90 disabled:opacity-50 rounded-lg px-4 py-2.5 font-medium flex items-center justify-center gap-2"
                >
                    {busy === "create" || busy === "spec" ? (
                        <Loader2 size={16} className="animate-spin" />
                    ) : (
                        <Sparkles size={16} />
                    )}
                    {project ? "重新生成动效脚本" : "创建并生成动效脚本"}
                </button>

                {/* Spec preview */}
                {project?.mg_spec && (
                    <section className="space-y-2">
                        <div className="text-sm text-gray-400">
                            动效脚本：{specClips} 个镜头 · {project.mg_spec.width}×{project.mg_spec.height}
                        </div>
                        <pre className="max-h-72 overflow-auto bg-gray-900 border border-white/10 rounded-lg p-3 text-xs text-gray-300">
                            {JSON.stringify(project.mg_spec, null, 2)}
                        </pre>
                        <button
                            onClick={handleRender}
                            disabled={busy !== null}
                            className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg px-4 py-2.5 font-medium flex items-center justify-center gap-2"
                        >
                            {busy === "render" ? (
                                <Loader2 size={16} className="animate-spin" />
                            ) : (
                                <Film size={16} />
                            )}
                            渲染视频
                        </button>
                    </section>
                )}

                {/* Rendered output */}
                {project?.remotion_video_url && (
                    <section className="space-y-2">
                        <div className="text-sm text-gray-400">成片</div>
                        <video
                            key={project.remotion_video_url}
                            src={getAssetUrl(project.remotion_video_url)}
                            controls
                            className="w-full rounded-lg border border-white/10 bg-black"
                        />
                    </section>
                )}
            </div>
        </div>
    );
}
