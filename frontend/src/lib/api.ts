import axios from "axios";
import { clearSession, getToken, type SetupStatus, type TokenResponse, type CurrentUser, persistSession, persistUser } from "./auth";

// Dynamic API URL detection:
// 1. Docker build: nginx proxies API requests on the same origin (host port 3000)
// 2. Packaged desktop app: frontend is served by the backend, use same origin
// 3. Next.js dev server (port 3000/3001): API runs separately on backend port 17177
const getApiUrl = (): string => {
    // Allow explicit override (e.g. when the FastAPI backend lives on a
    // different host/port than the Next.js dev server). Set this in your
    // .env.local or via the shell before `npm run dev`.
    if (process.env.NEXT_PUBLIC_API_URL) {
        return process.env.NEXT_PUBLIC_API_URL.replace(/\/+$/, "");
    }

    if (typeof window !== 'undefined') {
        const { protocol, hostname, port } = window.location;
        const sameOrigin = `${protocol}//${hostname}${port ? ':' + port : ''}`;

        // Docker builds bake NEXT_PUBLIC_DOCKER_BUILD=true into the bundle.
        // The frontend container (nginx) proxies /auth, /me, /projects, ... to the
        // backend container, so all API calls must stay on the same origin.
        if (process.env.NEXT_PUBLIC_DOCKER_BUILD === 'true') {
            return sameOrigin;
        }

        // Dev mode heuristic: any non-prod build that isn't on the FastAPI
        // backend port (17177) is assumed to be the Next.js dev server.
        // Don't pin to 3000/3001 — Next picks 3002+ when those are busy,
        // which used to silently fall through to sameOrigin and serve the
        // SPA HTML for /registry/* requests.
        const isProdBuild = process.env.NODE_ENV === 'production';
        if (!isProdBuild && port !== '17177') {
            return `${protocol}//${hostname}:17177`;
        }

        // Packaged desktop app / production export: backend serves the
        // static frontend on its own port (17177).
        return sameOrigin;
    }

    // SSR fallback
    return 'http://localhost:17177';
};

export const API_URL = getApiUrl();

// ── Global auth interceptors ──────────────────────────────────────────────
// All `axios.X(${API_URL}/...)` calls below benefit automatically from these
// interceptors on the default axios instance.

const AUTH_FREE_PATHS = ["/auth/setup", "/auth/setup-status", "/auth/login"];

const LOCALE_LS_KEY = "manju_forge_locale";

/** Read the user's chosen UI locale from localStorage. Mirrors the value the
 *  i18n provider persists; kept as a plain helper so non-React code (this
 *  axios interceptor, fetch wrappers) doesn't have to import the React layer.
 */
function readUiLocale(): string {
    if (typeof window === "undefined") return "zh-CN";
    try {
        const v = window.localStorage.getItem(LOCALE_LS_KEY);
        if (v === "zh-CN" || v === "en-US") return v;
    } catch {
        /* ignore */
    }
    return "zh-CN";
}

let _interceptorsInstalled = false;
function installAuthInterceptors() {
    if (_interceptorsInstalled || typeof window === "undefined") return;
    _interceptorsInstalled = true;

    axios.interceptors.request.use((config) => {
        const url = (config.url ?? "").toString();
        if (!url.startsWith(API_URL)) return config; // leave third-party requests alone
        // Tag every backend request with the user's chosen locale so the
        // FastAPI LocaleMiddleware can return error messages in that language.
        config.headers = config.headers ?? {};
        (config.headers as Record<string, string>)["Accept-Language"] = readUiLocale();
        const path = url.slice(API_URL.length);
        if (AUTH_FREE_PATHS.some((p) => path.startsWith(p))) return config;
        const token = getToken();
        if (token) {
            (config.headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
        }
        return config;
    });

    axios.interceptors.response.use(
        (resp) => resp,
        (error) => {
            const status = error?.response?.status;
            const url = (error?.config?.url ?? "").toString();
            if (status === 401 && url.startsWith(API_URL)) {
                const path = url.slice(API_URL.length);
                if (!AUTH_FREE_PATHS.some((p) => path.startsWith(p))) {
                    clearSession();
                    if (window.location.hash !== "#/login") {
                        window.location.hash = "#/login";
                    }
                }
            }
            return Promise.reject(error);
        }
    );
}

installAuthInterceptors();

/** Wrap raw fetch() with the same Authorization + locale header behavior. */
export async function authedFetch(input: string, init: RequestInit = {}): Promise<Response> {
    const token = typeof window === "undefined" ? null : getToken();
    const headers = new Headers(init.headers ?? {});
    const isAuthFree = AUTH_FREE_PATHS.some((p) => input.startsWith(`${API_URL}${p}`));
    if (input.startsWith(API_URL)) {
        headers.set("Accept-Language", readUiLocale());
    }
    if (token && input.startsWith(API_URL) && !isAuthFree) {
        headers.set("Authorization", `Bearer ${token}`);
    }
    const resp = await fetch(input, { ...init, headers });
    if (resp.status === 401 && input.startsWith(API_URL) && !isAuthFree && typeof window !== "undefined") {
        clearSession();
        if (window.location.hash !== "#/login") {
            window.location.hash = "#/login";
        }
    }
    return resp;
}

export type ProviderMode = "dashscope" | "vendor";
export type LLMProvider = "dashscope" | "openai";

export interface EnvConfigPayload {
    DASHSCOPE_API_KEY?: string;
    ALIBABA_CLOUD_ACCESS_KEY_ID?: string;
    ALIBABA_CLOUD_ACCESS_KEY_SECRET?: string;
    OSS_BUCKET_NAME?: string;
    OSS_ENDPOINT?: string;
    OSS_BASE_PATH?: string;
    KLING_PROVIDER_MODE?: ProviderMode;
    VIDU_PROVIDER_MODE?: ProviderMode;
    PIXVERSE_PROVIDER_MODE?: ProviderMode;
    KLING_ACCESS_KEY?: string;
    KLING_SECRET_KEY?: string;
    VIDU_API_KEY?: string;
    LLM_PROVIDER?: LLMProvider;
    OPENAI_API_KEY?: string;
    OPENAI_BASE_URL?: string;
    OPENAI_MODEL?: string;
    API_HOST?: string;
    API_PORT?: string;
    endpoint_overrides?: Record<string, string>;
    [key: string]: string | Record<string, string> | undefined;
}

export interface VideoTask {
    id: string;
    project_id: string;
    image_url: string;
    prompt: string;
    status: "pending" | "processing" | "completed" | "failed";
    video_url?: string;
    duration: number;
    seed?: number;
    resolution: string;
    generate_audio: boolean;
    audio_url?: string;
    prompt_extend: boolean;
    negative_prompt?: string;
    created_at: number;
    model?: string;
    frame_id?: string;
    generation_mode?: string;
    reference_video_urls?: string[];
}

// ── Per-user credentials ─────────────────────────────────────────────────

export interface MyCredentialsOut {
    values: Record<string, string>;
    masked: boolean;
}

export const me = {
    getCredentials: async (reveal = false): Promise<MyCredentialsOut> => {
        const res = await axios.get<MyCredentialsOut>(`${API_URL}/me/credentials`, {
            params: { reveal },
        });
        return res.data;
    },
    replaceCredentials: async (values: Record<string, string>): Promise<MyCredentialsOut> => {
        const res = await axios.put<MyCredentialsOut>(`${API_URL}/me/credentials`, { values });
        return res.data;
    },
    patchCredentials: async (values: Record<string, string>): Promise<MyCredentialsOut> => {
        const res = await axios.patch<MyCredentialsOut>(`${API_URL}/me/credentials`, { values });
        return res.data;
    },
    deleteCredential: async (key: string): Promise<void> => {
        await axios.delete(`${API_URL}/me/credentials/${encodeURIComponent(key)}`);
    },
    listCredentialKeys: async (): Promise<string[]> => {
        const res = await axios.get<string[]>(`${API_URL}/me/credentials/keys`);
        return res.data;
    },
};

// ── Admin / auth typed responses ──────────────────────────────────────────

export interface AdminStats {
    user_count: number;
    active_user_count: number;
    admin_count: number;
    disabled_user_count: number;
    last_login_at: string | null;
}

export interface AdminSettings {
    registration_enabled: boolean;
    invitation_required: boolean;
    default_user_role: "admin" | "user";
}

export interface AuditLogEntry {
    id: number;
    action: string;
    actor_user_id: number | null;
    actor_email: string;
    target_user_id: number | null;
    target_email: string;
    detail: string;
    ip: string;
    created_at: string;
}

export const auth = {
    setupStatus: async (): Promise<SetupStatus> => {
        const res = await axios.get<SetupStatus>(`${API_URL}/auth/setup-status`);
        return res.data;
    },
    setup: async (email: string, password: string, displayName?: string): Promise<TokenResponse> => {
        const res = await axios.post<TokenResponse>(`${API_URL}/auth/setup`, {
            email, password, display_name: displayName ?? "",
        });
        persistSession(res.data);
        return res.data;
    },
    login: async (email: string, password: string): Promise<TokenResponse> => {
        const res = await axios.post<TokenResponse>(`${API_URL}/auth/login`, { email, password });
        persistSession(res.data);
        return res.data;
    },
    logout: async (): Promise<void> => {
        try {
            await axios.post(`${API_URL}/auth/logout`);
        } catch {
            // Even if server-side rejected (already revoked), drop client state.
        } finally {
            clearSession();
        }
    },
    me: async (): Promise<CurrentUser> => {
        const res = await axios.get<CurrentUser>(`${API_URL}/auth/me`);
        persistUser(res.data);
        return res.data;
    },
    changePassword: async (currentPassword: string, newPassword: string): Promise<TokenResponse> => {
        const res = await axios.post<TokenResponse>(`${API_URL}/auth/password`, {
            current_password: currentPassword, new_password: newPassword,
        });
        persistSession(res.data);
        return res.data;
    },
};

export const admin = {
    listUsers: async (): Promise<CurrentUser[]> => {
        const res = await axios.get<CurrentUser[]>(`${API_URL}/admin/users`);
        return res.data;
    },
    getUser: async (id: number): Promise<CurrentUser> => {
        const res = await axios.get<CurrentUser>(`${API_URL}/admin/users/${id}`);
        return res.data;
    },
    createUser: async (payload: {
        email: string;
        password: string;
        role?: "admin" | "user";
        display_name?: string;
    }): Promise<CurrentUser> => {
        const res = await axios.post<CurrentUser>(`${API_URL}/admin/users`, {
            email: payload.email,
            password: payload.password,
            role: payload.role ?? "user",
            display_name: payload.display_name ?? "",
        });
        return res.data;
    },
    updateUser: async (
        id: number,
        payload: Partial<{
            role: "admin" | "user";
            status: "active" | "disabled";
            display_name: string;
            new_password: string;
        }>
    ): Promise<CurrentUser> => {
        const res = await axios.patch<CurrentUser>(`${API_URL}/admin/users/${id}`, payload);
        return res.data;
    },
    deleteUser: async (id: number): Promise<void> => {
        await axios.delete(`${API_URL}/admin/users/${id}`);
    },
    forceLogout: async (id: number): Promise<CurrentUser> => {
        const res = await axios.post<CurrentUser>(`${API_URL}/admin/users/${id}/force-logout`);
        return res.data;
    },
    stats: async (): Promise<AdminStats> => {
        const res = await axios.get<AdminStats>(`${API_URL}/admin/stats`);
        return res.data;
    },
    getSettings: async (): Promise<AdminSettings> => {
        const res = await axios.get<AdminSettings>(`${API_URL}/admin/settings`);
        return res.data;
    },
    updateSettings: async (payload: Partial<AdminSettings>): Promise<AdminSettings> => {
        const res = await axios.put<AdminSettings>(`${API_URL}/admin/settings`, payload);
        return res.data;
    },
    auditLogs: async (limit = 100, offset = 0): Promise<AuditLogEntry[]> => {
        const res = await axios.get<AuditLogEntry[]>(`${API_URL}/admin/audit-logs`, {
            params: { limit, offset },
        });
        return res.data;
    },
};

export const api = {
    createProject: async (title: string, text: string, skipAnalysis: boolean = false) => {
        const res = await axios.post(`${API_URL}/projects`, { title, text }, {
            params: { skip_analysis: skipAnalysis }
        });
        return { ...res.data, originalText: res.data.original_text };
    },

    getProjects: async () => {
        const res = await axios.get(`${API_URL}/projects/`);
        return res.data.map((p: any) => ({ ...p, originalText: p.original_text }));
    },

    getProject: async (scriptId: string) => {
        const res = await axios.get(`${API_URL}/projects/${scriptId}`);
        return { ...res.data, originalText: res.data.original_text };
    },

    deleteProject: async (scriptId: string) => {
        const res = await axios.delete(`${API_URL}/projects/${scriptId}`);
        return res.data;
    },

    reparseProject: async (scriptId: string, text: string) => {
        const res = await axios.put(`${API_URL}/projects/${scriptId}/reparse`, { text });
        return { ...res.data, originalText: res.data.original_text };
    },

    // ── Remotion MG engine (flow B): chat-only motion-graphics videos ──────
    createMGProject: async (title: string, text: string, aspectRatio: string = "9:16") => {
        const res = await axios.post(`${API_URL}/projects/mg`, {
            title, text, aspect_ratio: aspectRatio,
        });
        return { ...res.data, originalText: res.data.original_text };
    },

    generateMGSpec: async (scriptId: string, styleHint?: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/remotion/spec`, {
            style_hint: styleHint ?? null,
        });
        return { ...res.data, originalText: res.data.original_text };
    },

    renderMGVideo: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/remotion/render`);
        return { ...res.data, originalText: res.data.original_text };
    },

    syncDescriptions: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/sync_descriptions`);
        return res.data;
    },

    generateAssets: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/generate_assets`);
        return res.data;
    },

    createVideoTask: async (
        id: string,
        image_url: string,
        prompt: string,
        duration: number = 5,
        seed?: number,
        resolution: string = "720p",
        generateAudio: boolean = false,
        audioUrl?: string,
        promptExtend: boolean = true,
        negativePrompt?: string,
        batchSize: number = 1,
        model: string = "wan2.6-i2v",
        frameId?: string,
        shotType: string = "single",  // 'single' or 'multi' (only for wan2.6-i2v)
        generationMode: string = "i2v",  // 'i2v' or 'r2v'
        referenceVideoUrls: string[] = [],  // Reference videos for R2V (max 3)
        // Kling params
        mode?: string,
        sound?: boolean,
        cfgScale?: number,
        // Vidu params
        viduAudio?: boolean,
        movementAmplitude?: string,
        // ModelInstance picked in the UI (per-task override).
        i2vInstanceId?: string | null
    ) => {
        const res = await axios.post(`${API_URL}/projects/${id}/video_tasks`, {
            image_url,
            prompt,
            duration,
            seed,
            resolution,
            generate_audio: generateAudio,
            audio_url: audioUrl,
            prompt_extend: promptExtend,
            negative_prompt: negativePrompt,
            batch_size: batchSize,
            model,
            frame_id: frameId,
            shot_type: shotType,
            generation_mode: generationMode,
            reference_video_urls: referenceVideoUrls,
            // Kling
            mode,
            sound: sound != null ? (sound ? "on" : "off") : undefined,
            cfg_scale: cfgScale,
            // Vidu
            vidu_audio: viduAudio,
            movement_amplitude: movementAmplitude,
            i2v_instance_id: i2vInstanceId,
        });
        return res.data;
    },


    uploadFile: async (file: File) => {
        const formData = new FormData();
        formData.append("file", file);
        const response = await authedFetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error("Failed to upload file");
        return response.json();
    },

    /**
     * Upload an asset image as a new variant.
     * The uploaded image will be marked as the 'upload source' for reverse generation.
     */
    uploadAsset: async (
        scriptId: string,
        assetType: string,
        assetId: string,
        file: File,
        uploadType: string,
        description?: string
    ) => {
        const formData = new FormData();
        formData.append("file", file);

        const params = new URLSearchParams({
            upload_type: uploadType,
        });
        if (description) {
            params.append("description", description);
        }

        const response = await fetch(
            `${API_URL}/projects/${scriptId}/assets/${assetType}/${assetId}/upload?${params.toString()}`,
            {
                method: "POST",
                body: formData,
            }
        );

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Failed to upload asset");
        }

        return response.json();
    },

    generateAsset: async (scriptId: string, assetId: string, assetType: string, stylePreset: string, stylePrompt?: string, generationType: string = "all", prompt: string = "", applyStyle: boolean = true, negativePrompt: string = "", batchSize: number = 1, modelName?: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/generate`, {
            asset_id: assetId,
            asset_type: assetType,
            style_preset: stylePreset,
            style_prompt: stylePrompt,
            generation_type: generationType,
            prompt: prompt,
            apply_style: applyStyle,
            negative_prompt: negativePrompt,
            batch_size: batchSize,
            model_name: modelName
        });
        // Now returns { ...script, _task_id: string }
        return res.data;
    },

    getTaskStatus: async (taskId: string) => {
        const res = await axios.get(`${API_URL}/tasks/${taskId}`);
        return res.data;
    },

    generateAssetVideo: async (scriptId: string, assetType: string, assetId: string, data: { prompt?: string, duration?: number, aspect_ratio?: string }) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/${assetType}/${assetId}/generate_video`, data);
        return res.data;
    },

    /**
     * Generate Motion Reference video for an asset (Character Full Body/Headshot, Scene, or Prop).
     * This is part of Asset Activation v2.
     */
    generateMotionRef: async (
        scriptId: string,
        assetId: string,
        assetType: 'full_body' | 'head_shot' | 'scene' | 'prop',
        prompt?: string,
        audioUrl?: string,
        duration: number = 5,
        batchSize: number = 1
    ): Promise<any & { _task_id?: string }> => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/generate_motion_ref`, {
            asset_id: assetId,
            asset_type: assetType,
            prompt,
            audio_url: audioUrl,
            duration,
            batch_size: batchSize
        });
        return res.data;
    },

    deleteAssetVideo: async (scriptId: string, assetType: string, assetId: string, videoId: string) => {
        const res = await axios.delete(`${API_URL}/projects/${scriptId}/assets/${assetType}/${assetId}/videos/${videoId}`);
        return res.data;
    },

    toggleAssetLock: async (scriptId: string, assetId: string, assetType: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/toggle_lock`, {
            asset_id: assetId,
            asset_type: assetType
        });
        return res.data;
    },

    updateAssetImage: async (scriptId: string, assetId: string, assetType: string, imageUrl: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/update_image`, {
            asset_id: assetId,
            asset_type: assetType,
            image_url: imageUrl
        });
        return res.data;
    },

    selectAssetVariant: async (scriptId: string, assetId: string, assetType: string, variantId: string, generationType?: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/variant/select`, {
            asset_id: assetId,
            asset_type: assetType,
            variant_id: variantId,
            generation_type: generationType
        });
        return res.data;
    },

    deleteAssetVariant: async (scriptId: string, assetId: string, assetType: string, variantId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/variant/delete`, {
            asset_id: assetId,
            asset_type: assetType,
            variant_id: variantId
        });
        return res.data;
    },

    favoriteAssetVariant: async (scriptId: string, assetId: string, assetType: string, variantId: string, isFavorited: boolean, generationType?: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/variant/favorite`, {
            asset_id: assetId,
            asset_type: assetType,
            variant_id: variantId,
            is_favorited: isFavorited,
            generation_type: generationType
        });
        return res.data;
    },

    updateModelSettings: async (
        scriptId: string,
        payload: {
            llm_instance_id?: string | null;
            t2i_instance_id?: string | null;
            i2i_instance_id?: string | null;
            i2v_instance_id?: string | null;
            tts_instance_id?: string | null;
            character_aspect_ratio?: string;
            scene_aspect_ratio?: string;
            prop_aspect_ratio?: string;
            storyboard_aspect_ratio?: string;
        },
    ) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/model_settings`, payload);
        return res.data;
    },

    getPromptConfig: async (scriptId: string) => {
        const res = await axios.get(`${API_URL}/projects/${scriptId}/prompt_config`);
        return res.data;
    },

    updatePromptConfig: async (scriptId: string, config: { storyboard_polish?: string; video_polish?: string; r2v_polish?: string }) => {
        const res = await axios.put(`${API_URL}/projects/${scriptId}/prompt_config`, config);
        return res.data;
    },

    selectVideo: async (scriptId: string, frameId: string, videoId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/frames/${frameId}/select_video`, {
            video_id: videoId
        });
        return res.data;
    },

    mergeVideos: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/merge`);
        return res.data;
    },

    // Art Direction APIs
    analyzeScriptForStyles: async (scriptId: string, scriptText: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/art_direction/analyze`, {
            script_text: scriptText
        });
        return res.data;
    },

    saveArtDirection: async (scriptId: string, selectedStyleId: string, styleConfig: any, customStyles: any[] = [], aiRecommendations: any[] = []) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/art_direction/save`, {
            selected_style_id: selectedStyleId,
            style_config: styleConfig,
            custom_styles: customStyles,
            ai_recommendations: aiRecommendations
        });
        return res.data;
    },

    getStylePresets: async () => {
        const res = await axios.get(`${API_URL}/art_direction/presets`);
        return res.data;
    },

    // NOTE: polishPrompt removed - use refineFramePrompt for storyboard prompts
    polishVideoPrompt: async (draftPrompt: string, feedback: string = "", scriptId: string = "") => {
        const res = await axios.post(`${API_URL}/video/polish_prompt`, {
            draft_prompt: draftPrompt,
            feedback: feedback,
            script_id: scriptId,
        });
        return res.data;
    },
    polishR2VPrompt: async (draftPrompt: string, slots: { description: string }[], feedback: string = "", scriptId: string = "") => {
        const res = await axios.post(`${API_URL}/video/polish_r2v_prompt`, {
            draft_prompt: draftPrompt,
            slots: slots,
            feedback: feedback,
            script_id: scriptId,
        });
        return res.data;
    },
    updateAssetDescription: async (scriptId: string, assetId: string, assetType: string, description: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/update_description`, {
            asset_id: assetId,
            asset_type: assetType,
            description: description
        });
        return res.data;
    },

    updateAssetAttributes: async (scriptId: string, assetId: string, assetType: string, attributes: any) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/assets/update_attributes`, {
            asset_id: assetId,
            asset_type: assetType,
            attributes: attributes
        });
        return res.data;
    },

    toggleFrameLock: async (scriptId: string, frameId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/frames/toggle_lock`, {
            frame_id: frameId
        });
        return res.data;
    },

    updateFrame: async (scriptId: string, frameId: string, data: {
        image_prompt?: string;
        action_description?: string;
        dialogue?: string;
        camera_angle?: string;
        scene_id?: string;
        character_ids?: string[];
        title?: string | null;
        duration_seconds?: number | null;
    }) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/frames/update`, {
            frame_id: frameId,
            ...data
        });
        return res.data;
    },

    updateProjectStyle: async (scriptId: string, stylePreset: string, stylePrompt?: string) => {
        const res = await axios.patch(`${API_URL}/projects/${scriptId}/style`, {
            style_preset: stylePreset,
            style_prompt: stylePrompt
        });
        return res.data;
    },

    renderFrame: async (scriptId: string, frameId: string, compositionData: any, prompt: string, batchSize: number = 1) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/storyboard/render`, {
            frame_id: frameId,
            composition_data: compositionData,
            prompt: prompt,
            batch_size: batchSize
        });
        return res.data;
    },

    // === STORYBOARD DRAMATIZATION v2 ===

    /**
     * Analyzes script text and generates storyboard frames using AI.
     * Replaces existing frames with newly generated ones.
     *
     * Async: returns ``{ ...script, _task_id }`` immediately. Caller polls
     * ``getTaskStatus(task_id)`` until status === "completed" / "failed", then
     * refetches the project to get the new frames.
     */
    analyzeToStoryboard: async (scriptId: string, text: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/storyboard/analyze`, {
            text: text
        });
        return res.data;
    },

    /**
     * Refines a raw prompt into bilingual (CN/EN) prompts using AI.
     * Returns { prompt_cn, prompt_en, frame_updated }.
     */
    /**
     * Sync refine_prompt — kept for back-compat. Prefer
     * :pyfunc:`refineFramePromptAsync` so long LLM calls don't trip
     * upstream gateway timeouts.
     */
    refineFramePrompt: async (scriptId: string, frameId: string, rawPrompt: string, assets: any[] = [], feedback: string = "") => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/storyboard/refine_prompt`, {
            frame_id: frameId,
            raw_prompt: rawPrompt,
            assets: assets,
            feedback: feedback
        });
        return res.data;
    },

    /** Async variant — returns ``{ _task_id }``. Poll task and read ``status.result``. */
    refineFramePromptAsync: async (scriptId: string, frameId: string, rawPrompt: string, assets: any[] = [], feedback: string = "") => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/storyboard/refine_prompt_async`, {
            frame_id: frameId,
            raw_prompt: rawPrompt,
            assets: assets,
            feedback: feedback
        });
        return res.data;
    },

    /**
     * One-shot helper: submit + poll for a one-shot task's result.
     *
     * Use when you need the result inline inside an async handler (i.e.
     * outside of a React component where ``useAsyncTask`` would be the
     * better choice). Polls ``/tasks/{id}`` every ``intervalMs`` (default
     * 1.5s); resolves with the task's ``result`` payload on completion;
     * rejects with the task's ``error`` on failure.
     */
    pollAsyncTaskResult: async <T = any>(submitResponse: { _task_id?: string; [k: string]: any }, intervalMs = 1500): Promise<T> => {
        const taskId = submitResponse?._task_id;
        if (!taskId) {
            // Backwards-compat: server returned the result inline.
            return submitResponse as unknown as T;
        }
        // eslint-disable-next-line no-constant-condition
        while (true) {
            await new Promise((r) => setTimeout(r, intervalMs));
            // eslint-disable-next-line @typescript-eslint/no-use-before-define
            const status: any = await api.getTaskStatus(taskId);
            if (status.status === "completed") return status.result as T;
            if (status.status === "failed") throw new Error(status.error || "task failed");
        }
    },

    generateStoryboard: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/generate_storyboard`);
        return res.data;
    },

    /**
     * Async batch render of every eligible storyboard frame.
     * Returns ``{ ...script, _task_id }`` immediately. Caller polls
     * ``getTaskStatus(task_id)`` for granular progress
     * (``completed_count`` / ``failed_count`` / ``current_frame_id`` / ``errors``).
     * ``force=true`` re-renders frames that already have images;
     * locked frames are always skipped.
     */
    renderAllStoryboard: async (scriptId: string, force: boolean = false) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/storyboard/render_all`, { force });
        return res.data;
    },

    getVoices: async () => {
        const response = await authedFetch(`${API_URL}/voices`);
        if (!response.ok) throw new Error("Failed to fetch voices");
        return response.json();
    },

    bindVoice: async (scriptId: string, charId: string, voiceId: string, voiceName: string) => {
        const response = await authedFetch(`${API_URL}/projects/${scriptId}/characters/${charId}/voice`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ voice_id: voiceId, voice_name: voiceName }),
        });
        if (!response.ok) throw new Error("Failed to bind voice");
        return response.json();
    },

    generateAudio: async (scriptId: string) => {
        const response = await authedFetch(`${API_URL}/projects/${scriptId}/generate_audio`, {
            method: "POST",
        });
        if (!response.ok) throw new Error("Failed to generate audio");
        return response.json();
    },

    generateLineAudio: async (scriptId: string, frameId: string, speed: number, pitch: number, volume: number = 50) => {
        const response = await authedFetch(`${API_URL}/projects/${scriptId}/frames/${frameId}/audio`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ speed, pitch, volume }),
        });
        if (!response.ok) throw new Error("Failed to generate line audio");
        return response.json();
    },

    updateVoiceParams: async (scriptId: string, charId: string, speed: number, pitch: number, volume: number) => {
        const response = await authedFetch(`${API_URL}/projects/${scriptId}/characters/${charId}/voice_params`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ speed, pitch, volume }),
        });
        if (!response.ok) throw new Error("Failed to update voice params");
        return response.json();
    },

    /**
     * Async export. Returns ``{ ...script, _task_id }`` immediately.
     * Caller polls ``getTaskStatus(task_id)`` until ``status === 'completed'``;
     * the merged video URL is then in ``output_url`` on the status response.
     * Pass ``options.force = true`` to re-merge even if a cached result exists.
     */
    exportProject: async (scriptId: string, options: any) => {
        const response = await authedFetch(`${API_URL}/projects/${scriptId}/export`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(options),
        });
        if (!response.ok) throw new Error("Failed to export project");
        return response.json();
    },

    generateVideo: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/generate_video`);
        return res.data;
    },

    /** Async batch i2v: every eligible frame gets a video. Mirrors renderAllStoryboard. */
    renderAllVideos: async (scriptId: string, force: boolean = false) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/video/render_all`, { force });
        return res.data;
    },

    /** Async batch audio synth: dialogue + SFX for every eligible frame. */
    renderAllAudio: async (scriptId: string, force: boolean = false) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/audio/render_all`, { force });
        return res.data;
    },

    getEnvConfig: async (): Promise<EnvConfigPayload> => {
        const res = await axios.get<EnvConfigPayload>(`${API_URL}/config/env`);
        return res.data;
    },

    saveEnvConfig: async (config: EnvConfigPayload) => {
        const res = await axios.post(`${API_URL}/config/env`, config, {
            timeout: 60000, // 60 seconds timeout
        });
        return res.data;
    },

    extractLastFrame: async (scriptId: string, frameId: string, videoTaskId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/frames/${frameId}/extract_last_frame`, {
            video_task_id: videoTaskId,
        });
        return res.data;
    },

    uploadFrameImage: async (scriptId: string, frameId: string, file: File) => {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch(
            `${API_URL}/projects/${scriptId}/frames/${frameId}/upload_image`,
            { method: "POST", body: formData }
        );
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Failed to upload frame image");
        }
        return response.json();
    },

    // ============================================
    // Series APIs
    // ============================================

    // Series CRUD
    createSeries: async (title: string, description?: string) => {
        const response = await axios.post(`${API_URL}/series`, { title, description });
        return response.data;
    },
    listSeries: async () => {
        const response = await axios.get(`${API_URL}/series`);
        return response.data;
    },
    getSeries: async (seriesId: string) => {
        const response = await axios.get(`${API_URL}/series/${seriesId}`);
        return response.data;
    },
    updateSeries: async (seriesId: string, data: { title?: string; description?: string }) => {
        const response = await axios.put(`${API_URL}/series/${seriesId}`, data);
        return response.data;
    },
    deleteSeries: async (seriesId: string) => {
        const response = await axios.delete(`${API_URL}/series/${seriesId}`);
        return response.data;
    },

    // Series Episodes
    getSeriesEpisodes: async (seriesId: string) => {
        const response = await axios.get(`${API_URL}/series/${seriesId}/episodes`);
        return response.data;
    },
    addEpisodeToSeries: async (seriesId: string, scriptId: string, episodeNumber?: number) => {
        const response = await axios.post(`${API_URL}/series/${seriesId}/episodes`, { script_id: scriptId, episode_number: episodeNumber });
        return response.data;
    },
    removeEpisodeFromSeries: async (seriesId: string, scriptId: string) => {
        const response = await axios.delete(`${API_URL}/series/${seriesId}/episodes/${scriptId}`);
        return response.data;
    },

    // Series Assets
    getSeriesAssets: async (seriesId: string) => {
        const response = await axios.get(`${API_URL}/series/${seriesId}/assets`);
        return response.data;
    },
    importSeriesAssets: async (seriesId: string, sourceSeriesId: string, assetIds: string[]) => {
        const response = await axios.post(`${API_URL}/series/${seriesId}/assets/import`, { source_series_id: sourceSeriesId, asset_ids: assetIds });
        return response.data;
    },

    // Series Prompt Config
    getSeriesPromptConfig: async (seriesId: string) => {
        const response = await axios.get(`${API_URL}/series/${seriesId}/prompt_config`);
        return response.data;
    },
    updateSeriesPromptConfig: async (seriesId: string, config: { storyboard_polish?: string; video_polish?: string; r2v_polish?: string }) => {
        const response = await axios.put(`${API_URL}/series/${seriesId}/prompt_config`, config);
        return response.data;
    },
    getSeriesModelSettings: async (seriesId: string) => {
        const response = await axios.get(`${API_URL}/series/${seriesId}/model_settings`);
        return response.data;
    },
    updateSeriesModelSettings: async (seriesId: string, settings: {
        llm_instance_id?: string | null;
        t2i_instance_id?: string | null;
        i2i_instance_id?: string | null;
        i2v_instance_id?: string | null;
        tts_instance_id?: string | null;
        character_aspect_ratio?: string;
        scene_aspect_ratio?: string;
        prop_aspect_ratio?: string;
        storyboard_aspect_ratio?: string;
    }) => {
        const response = await axios.put(`${API_URL}/series/${seriesId}/model_settings`, settings);
        return response.data;
    },

    // Helper: create a project and add it as an episode to a series
    createEpisodeForSeries: async (seriesId: string, title: string, episodeNumber: number) => {
        const project = await api.createProject(title, "", true);
        await api.addEpisodeToSeries(seriesId, project.id, episodeNumber);
        return project;
    },

    // File Import
    importFilePreview: async (file: File, suggestedEpisodes: number = 3) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await axios.post(`${API_URL}/series/import/preview?suggested_episodes=${suggestedEpisodes}`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },
    importFileConfirm: async (data: { title: string; description?: string; text: string; episodes: any[] }) => {
        const response = await axios.post(`${API_URL}/series/import/confirm`, data);
        return response.data;
    },

    // ── huobao-pipeline parity (script_rewriter / extractor / voice_assigner /
    //    video timeline slicer / batch keyframes). Each endpoint returns the
    //    raw backend payload — callers map fields they need.

    /** Novel → screenplay-format rewrite. Persists onto ``script.formatted_text``;
     *  ``original_text`` stays untouched so users can revert / re-rewrite. */
    rewriteToScreenplay: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/script/rewrite-to-screenplay`);
        return res.data;
    },

    /** Incremental entity extraction. Default ``"incremental"`` reuses
     *  matching Series characters/scenes/props; ``"full"`` re-extracts from scratch. */
    extractEntities: async (scriptId: string, strategy: 'incremental' | 'full' = 'incremental') => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/entities/extract`, { strategy });
        return res.data as {
            strategy: string;
            created: { characters: string[]; scenes: string[]; props: string[] };
            reused: { characters: string[]; scenes: string[]; props: string[] };
        };
    },

    /** Auto-assign voices via the chain (Locked → SeriesReuse → LLMMatch → DefaultPool).
     *  Characters with an existing ``voice_id`` are left untouched. */
    autoAssignVoices: async (scriptId: string) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/voices/auto-assign`);
        return res.data as { assignments: Record<string, string>; skipped: string[] };
    },

    /** Slice a frame's ``video_prompt`` into time-axis form for Seedance/Veo-style
     *  long-shot backends. Persists to ``frame.video_prompt_timeline``. */
    sliceVideoTimeline: async (
        scriptId: string,
        frameId: string,
        opts: { segmentSeconds?: number; durationOverride?: number } = {}
    ) => {
        const res = await axios.post(
            `${API_URL}/projects/${scriptId}/frames/${frameId}/video/slice-timeline`,
            {
                segment_seconds: opts.segmentSeconds ?? 3,
                duration_override: opts.durationOverride ?? null,
            }
        );
        return res.data;
    },

    /** Batch keyframe generation. On grid-capable T2I vendors (Seedream)
     *  one composite call yields N tiles; other vendors fall back to per-shot.
     *  Same response shape on either path. */
    batchKeyframes: async (
        scriptId: string,
        frameIds: string[],
        opts: { mode?: 'first_frame' | 'first_last' | 'multi_ref'; forcePerShot?: boolean } = {}
    ) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/frames/keyframes/batch`, {
            frame_ids: frameIds,
            mode: opts.mode ?? 'first_frame',
            force_per_shot: opts.forcePerShot ?? false,
        });
        return res.data as {
            method: 'grid' | 'per_shot';
            vendor: string | null;
            assignments: Record<string, string[]>;
        };
    },
};

// ============================================
// CRUD APIs for Assets and Frames
// ============================================

export const crudApi = {
    // Character CRUD
    createCharacter: async (scriptId: string, data: {
        name: string;
        description?: string;
        age?: string;
        gender?: string;
        clothing?: string;
    }) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/characters`, data);
        return res.data;
    },

    deleteCharacter: async (scriptId: string, characterId: string) => {
        const res = await axios.delete(`${API_URL}/projects/${scriptId}/characters/${characterId}`);
        return res.data;
    },

    // Scene CRUD
    createScene: async (scriptId: string, data: {
        name: string;
        description?: string;
        time_of_day?: string;
        lighting_mood?: string;
    }) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/scenes`, data);
        return res.data;
    },

    deleteScene: async (scriptId: string, sceneId: string) => {
        const res = await axios.delete(`${API_URL}/projects/${scriptId}/scenes/${sceneId}`);
        return res.data;
    },

    // Prop CRUD
    createProp: async (scriptId: string, data: {
        name: string;
        description?: string;
    }) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/props`, data);
        return res.data;
    },

    deleteProp: async (scriptId: string, propId: string) => {
        const res = await axios.delete(`${API_URL}/projects/${scriptId}/props/${propId}`);
        return res.data;
    },

    // Frame CRUD
    createFrame: async (scriptId: string, data: {
        scene_id: string;
        action_description: string;
        character_ids?: string[];
        prop_ids?: string[];
        dialogue?: string;
        speaker?: string;
        camera_angle?: string;
        insert_at?: number;
    }) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/frames`, data);
        return res.data;
    },

    deleteFrame: async (scriptId: string, frameId: string) => {
        const res = await axios.delete(`${API_URL}/projects/${scriptId}/frames/${frameId}`);
        return res.data;
    },

    copyFrame: async (scriptId: string, frameId: string, insertAt?: number) => {
        const res = await axios.post(`${API_URL}/projects/${scriptId}/frames/copy`, {
            frame_id: frameId,
            insert_at: insertAt
        });
        return res.data;
    },

    reorderFrames: async (scriptId: string, frameIds: string[]) => {
        const res = await axios.put(`${API_URL}/projects/${scriptId}/frames/reorder`, {
            frame_ids: frameIds
        });
        return res.data;
    }
};


// ── Model registry (single-source-of-truth catalog) ──────────────────────
//
// Backed by GET /registry/models on the backend (see src/utils/model_catalog.py).
// The dropdowns in the Settings page render these. Hardcoded fallbacks live in
// frontend/src/store/projectStore.ts so the UI still works offline / on first
// load before the network round-trip.

export type ModelCapability = "t2i" | "i2i" | "i2v" | "t2v" | "r2v" | "tts" | "llm";

export interface ModelCardDTO {
    id: string;
    family: string;
    display_name: string;
    description: string;
    capabilities: ModelCapability[];
    provider_key: string;
    requires_credentials: string[];
    params?: Record<string, unknown>;
    available: boolean;
    badges: string[];
}

export interface LLMPresetDTO {
    id: string;
    provider: "dashscope" | "openai";
    display_name: string;
    description: string;
    base_url: string;
    suggested_models: string[];
    api_key_env: string;
    docs_url: string;
    badges: string[];
}

export interface AspectRatioDTO {
    id: string;
    name: string;
    description: string;
}

export interface ModelCatalogDTO {
    cards: ModelCardDTO[];
    presets: LLMPresetDTO[];
    aspect_ratios: AspectRatioDTO[];
}

// Vendor connectors (Kling / Vidu / Doubao / Hailuo / ...). Backed by
// /registry/vendors. Each entry drives one expandable VendorCard in Settings.

export interface CredentialFieldDTO {
    key: string;
    label: string;
    placeholder: string;
    secret: boolean;
    help_text: string;
    required: boolean;
}

export interface VendorModeDTO {
    id: string; // "dashscope" | "vendor"
    label: string;
    description: string;
    fields: CredentialFieldDTO[];
}

export interface VendorConnectorDTO {
    id: string;
    display_name: string;
    description: string;
    capabilities: ModelCapability[];
    common_fields: CredentialFieldDTO[];
    modes: VendorModeDTO[];
    mode_env_key: string | null;
    family_prefixes: string[];
    docs_url: string;
    badges: string[];
    accent: string;
}

// ── Model instances ──────────────────────────────────────────────────────
//
// User-configured model instances (one per Vendor × Type combo, optionally
// many per type). Backed by /me/instances endpoints — Settings UI renders a
// list of these grouped by type.

export type InstanceTypeId = "llm" | "t2i" | "i2i" | "i2v" | "t2v" | "r2v" | "tts";

export interface ModelInstanceOut {
    id: string;
    instance_type: InstanceTypeId;
    vendor_id: string;
    model_name: string;
    display_name: string;
    credential_keys: string[];   // presence-only; secrets stay server-side
    base_url: string;
    extra_params: Record<string, unknown>;
    is_default: boolean;
    created_at: number;
    updated_at: number;
}

export interface ModelInstanceCreate {
    instance_type: InstanceTypeId;
    vendor_id: string;
    model_name: string;
    display_name: string;
    credentials?: Record<string, string>;
    base_url?: string;
    extra_params?: Record<string, unknown>;
    is_default?: boolean;
}

export interface ModelInstanceUpdate {
    display_name?: string;
    model_name?: string;
    credentials?: Record<string, string>;
    base_url?: string;
    extra_params?: Record<string, unknown>;
}

export interface InstanceTestResult {
    ok: boolean;
    latency_ms: number;
    error: string;
}

export const instances = {
    list: async (type?: InstanceTypeId): Promise<ModelInstanceOut[]> => {
        const url = type ? `${API_URL}/me/instances?type=${type}` : `${API_URL}/me/instances`;
        const res = await authedFetch(url);
        if (!res.ok) throw new Error("Failed to list instances");
        return res.json();
    },
    get: async (id: string): Promise<ModelInstanceOut> => {
        const res = await authedFetch(`${API_URL}/me/instances/${id}`);
        if (!res.ok) throw new Error("Failed to fetch instance");
        return res.json();
    },
    create: async (payload: ModelInstanceCreate): Promise<ModelInstanceOut> => {
        const res = await axios.post<ModelInstanceOut>(`${API_URL}/me/instances`, payload);
        return res.data;
    },
    update: async (id: string, payload: ModelInstanceUpdate): Promise<ModelInstanceOut> => {
        const res = await axios.put<ModelInstanceOut>(`${API_URL}/me/instances/${id}`, payload);
        return res.data;
    },
    delete: async (id: string): Promise<void> => {
        await axios.delete(`${API_URL}/me/instances/${id}`);
    },
    setDefault: async (id: string): Promise<ModelInstanceOut> => {
        const res = await axios.post<ModelInstanceOut>(`${API_URL}/me/instances/${id}/set-default`, {});
        return res.data;
    },
    test: async (id: string): Promise<InstanceTestResult> => {
        const res = await axios.post<InstanceTestResult>(`${API_URL}/me/instances/${id}/test`, {});
        return res.data;
    },
};


export const registry = {
    getCatalog: async (capability?: ModelCapability): Promise<ModelCatalogDTO> => {
        const url = capability
            ? `${API_URL}/registry/models?capability=${capability}`
            : `${API_URL}/registry/models`;
        const response = await authedFetch(url);
        if (!response.ok) throw new Error("Failed to fetch model catalog");
        return response.json();
    },
    getLLMPresets: async (): Promise<{ presets: LLMPresetDTO[] }> => {
        const response = await authedFetch(`${API_URL}/registry/llm-presets`);
        if (!response.ok) throw new Error("Failed to fetch LLM presets");
        return response.json();
    },
    getVendors: async (capability?: ModelCapability): Promise<{ connectors: VendorConnectorDTO[] }> => {
        const url = capability
            ? `${API_URL}/registry/vendors?capability=${capability}`
            : `${API_URL}/registry/vendors`;
        const response = await authedFetch(url);
        if (!response.ok) throw new Error("Failed to fetch vendor connectors");
        return response.json();
    },
};
