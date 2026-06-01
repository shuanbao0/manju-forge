import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { api, API_URL, authedFetch } from '@/lib/api';

export interface ImageVariant {
    id: string;
    url: string;
    created_at: number;
    prompt_used?: string;
}

export interface ImageAsset {
    selected_id: string | null;
    variants: ImageVariant[];
}

export interface VideoTask {
    id: string;
    project_id: string;
    asset_id?: string;
    frame_id?: string;
    image_url: string;
    prompt: string;
    status: string;
    video_url?: string;
    duration?: number;
    created_at: number;
    model?: string;
    generation_mode?: string;  // 'i2v' or 'r2v'
    reference_video_urls?: string[];  // Reference videos for R2V
}

export interface Character {
    id: string;
    name: string;
    description?: string;
    age?: string;
    gender?: string;
    clothing?: string;
    visual_weight?: number;

    // Legacy fields
    image_url?: string;
    avatar_url?: string;
    full_body_image_url?: string;
    three_view_image_url?: string;
    headshot_image_url?: string;

    // New Asset Containers
    full_body_asset?: ImageAsset;
    three_view_asset?: ImageAsset;
    headshot_asset?: ImageAsset;

    // Video Assets
    video_assets?: VideoTask[];
    video_prompt?: string;

    voice_id?: string;
    voice_name?: string;
    locked?: boolean;
    status?: string;
    is_consistent?: boolean;
    full_body_updated_at?: number;
    three_view_updated_at?: number;
    headshot_updated_at?: number;
}

export interface Scene {
    id: string;
    name: string;
    description: string;
    image_url?: string;
    image_asset?: ImageAsset;
    video_assets?: VideoTask[];
    video_prompt?: string;
    status?: string;
    locked?: boolean;
    time_of_day?: string;
    lighting_mood?: string;
}

export interface Prop {
    id: string;
    name: string;
    description: string;
    image_url?: string;
    image_asset?: ImageAsset;
    video_assets?: VideoTask[];
    video_prompt?: string;
    status?: string;
    locked?: boolean;
}

export interface StoryboardFrame {
    id: string;
    // huobao-parity metadata: short UI label + planned duration (seconds).
    // Both optional so older project.json deserialises cleanly.
    title?: string;
    duration_seconds?: number;
    scene_id: string;
    image_url?: string;
    image_asset?: ImageAsset;
    rendered_image_url?: string;
    rendered_image_asset?: ImageAsset;
    // Closing keyframe (populated by FirstLastMode batch keyframe generation).
    end_frame_asset?: ImageAsset;
    // Audio prompts emitted by the storyboard analyzer alongside visuals.
    bgm_prompt?: string;
    sfx_prompt?: string;
    bgm_url?: string;
    sfx_url?: string;
    // Time-axis form of ``video_prompt`` for Seedance/Veo-style long-shot i2v.
    video_prompt_timeline?: string;
    status?: string;
    locked?: boolean;
    // ... other fields
}

export interface StylePreset {
    id: string;
    name: string;
    color: string;
    prompt: string;
    negative_prompt?: string;
}

export interface StyleConfig {
    id: string;
    name: string;
    description?: string;
    positive_prompt: string;
    negative_prompt: string;
    thumbnail_url?: string;
    is_custom: boolean;
    reason?: string; // For AI recommendations
}

export interface ArtDirection {
    selected_style_id: string;
    style_config: StyleConfig;
    custom_styles: StyleConfig[];
    ai_recommendations: StyleConfig[];
}

export interface ModelSettings {
    // ModelInstance id references — null means "use my default for this type".
    llm_instance_id: string | null;
    t2i_instance_id: string | null;
    i2i_instance_id: string | null;
    i2v_instance_id: string | null;
    tts_instance_id: string | null;
    character_aspect_ratio: string;
    scene_aspect_ratio: string;
    prop_aspect_ratio: string;
    storyboard_aspect_ratio: string;
}

// Model options for dropdowns.
//
// These arrays act as **fallbacks only**: the canonical catalog is fetched
// from the user's ModelInstance list (see useInstances). We keep
// hardcoded copies here so the Settings page renders something reasonable
// during SSR / before the network round-trip resolves. Adding a new model
// belongs in `src/utils/model_catalog.py` on the backend, not here.
export const T2I_MODELS = [
    { id: 'wan2.6-t2i', name: 'Wan 2.6 T2I', description: 'modelCatalog.descriptions.wan26t2i' },
    { id: 'wan2.5-t2i-preview', name: 'Wan 2.5 T2I Preview', description: 'modelCatalog.descriptions.wan25t2ipreview' },
    { id: 'wan2.2-t2i-plus', name: 'Wan 2.2 T2I Plus', description: 'modelCatalog.descriptions.wan22t2iplus' },
    { id: 'wan2.2-t2i-flash', name: 'Wan 2.2 T2I Flash', description: 'modelCatalog.descriptions.wan22t2iflash' },
    { id: 'qwen-image', name: 'Qwen-Image', description: 'modelCatalog.descriptions.qwenimage' },
    { id: 'qwen-image-plus', name: 'Qwen-Image Plus', description: 'modelCatalog.descriptions.qwenimageplus' },
    { id: 'flux-schnell', name: 'FLUX.1 Schnell', description: 'modelCatalog.descriptions.fluxschnell' },
    { id: 'flux-dev', name: 'FLUX.1 Dev', description: 'modelCatalog.descriptions.fluxdev' },
];

export const I2I_MODELS = [
    { id: 'wan2.6-image', name: 'Wan 2.6 Image', description: 'modelCatalog.descriptions.wan26image' },
    { id: 'wan2.5-i2i-preview', name: 'Wan 2.5 I2I Preview', description: 'modelCatalog.descriptions.wan25i2ipreview' },
    { id: 'qwen-image-edit', name: 'Qwen-Image Edit', description: 'modelCatalog.descriptions.qwenimageedit' },
];

// I2V capability metadata (DurationConfig / ModelParamSupport / per-family
// schemas) was moved to ``@/lib/i2vFamilies``. Selection is now driven by
// ``ModelInstance``s — see ``ModelInstancePicker`` + ``useI2VInstances``.

export const ASPECT_RATIOS = [
    { id: '9:16', name: '9:16', description: 'modelCatalog.aspectRatios.portrait' },
    { id: '16:9', name: '16:9', description: 'modelCatalog.aspectRatios.landscape' },
    { id: '1:1', name: '1:1', description: 'modelCatalog.aspectRatios.square' },
];

export interface VideoParams {
    resolution: string;
    duration: number;
    seed: number | undefined;
    generateAudio: boolean;
    audioUrl: string;
    promptExtend: boolean;
    negativePrompt: string;
    batchSize: number;
    cameraMovement: string;
    subjectMotion: string;
    /** ``model_name`` of the selected ``ModelInstance``; drives capability lookup. */
    model: string;
    /** ``ModelInstance.id`` chosen in the picker; ``null`` falls back to user default. */
    i2vInstanceId: string | null;
    shotType: string;
    generationMode: string;
    referenceVideoUrls: string[];
    // Kling
    mode: string;
    sound: boolean;
    cfgScale: number;
    // Vidu
    viduAudio: boolean;
    movementAmplitude: string;
}

export interface PromptConfig {
    storyboard_polish: string;
    video_polish: string;
    r2v_polish: string;
}

export interface Series {
    id: string;
    title: string;
    description: string;
    characters: Character[];
    scenes: Scene[];
    props: Prop[];
    art_direction?: ArtDirection;
    prompt_config?: PromptConfig;
    model_settings?: ModelSettings;
    episode_ids: string[];
    created_at: number;
    updated_at: number;
}

export interface Project {
    id: string;
    title: string;
    originalText: string;
    // Screenplay-format rewrite produced by ``api.rewriteToScreenplay``.
    // Downstream stages (analyze_to_storyboard) prefer this when present.
    formatted_text?: string;
    // Episode-scoped references into the Series catalog, populated by the
    // incremental extractor (api.extractEntities, strategy="incremental").
    used_character_ids?: string[];
    used_scene_ids?: string[];
    used_prop_ids?: string[];
    characters: Character[];
    scenes: Scene[];
    props: Prop[];
    frames: any[]; // Keeping as any for now to avoid breaking too much, but ideally StoryboardFrame[]
    video_tasks?: any[];
    status: string;
    createdAt: string;
    updatedAt: string;
    aspectRatio?: string;
    style_preset?: string;
    art_direction?: ArtDirection;
    model_settings?: ModelSettings;
    prompt_config?: PromptConfig;
    merged_video_url?: string;
    series_id?: string;
    episode_number?: number;
    // Remotion MG engine (flow B)
    generation_engine?: 'comic_ai' | 'remotion_mg';
    mg_spec?: any;
    remotion_video_url?: string;
}

interface ProjectStore {
    projects: Project[];
    currentProject: Project | null;
    isLoading: boolean;
    isAnalyzing: boolean;
    isAnalyzingArtStyle: boolean;



    // Global Selection State
    selectedFrameId: string | null;

    // Actions
    setProjects: (projects: Project[]) => void;  // For syncing from backend
    createProject: (title: string, text: string, skipAnalysis?: boolean) => Promise<void>;
    analyzeProject: (script: string) => Promise<void>;
    analyzeArtStyle: (scriptId: string, text: string) => Promise<void>;
    loadProjects: () => void;
    selectProject: (id: string) => Promise<void>;
    updateProject: (id: string, data: Partial<Project>) => void;
    deleteProject: (id: string) => Promise<void>;
    clearCurrentProject: () => void;



    // Selection Actions
    // Selection Actions
    setSelectedFrameId: (id: string | null) => void;

    // Asset Generation State
    generatingTasks: { assetId: string; generationType: string; batchSize: number }[];
    addGeneratingTask: (assetId: string, generationType: string, batchSize: number) => void;
    removeGeneratingTask: (assetId: string, generationType: string) => void;

    // Storyboard Frame Rendering State
    renderingFrames: Set<string>;  // Set of frame IDs currently being rendered
    addRenderingFrame: (frameId: string) => void;
    removeRenderingFrame: (frameId: string) => void;

    // Storyboard Analysis State (persists across tab switches)
    isAnalyzingStoryboard: boolean;
    setIsAnalyzingStoryboard: (value: boolean) => void;

    // Series State
    seriesList: Series[];
    currentSeries: Series | null;
    fetchSeriesList: () => Promise<void>;
    fetchSeries: (id: string) => Promise<void>;
    createSeries: (title: string, description?: string) => Promise<Series>;
    deleteSeries: (id: string) => Promise<void>;
    setCurrentSeries: (series: Series | null) => void;
}

export const useProjectStore = create<ProjectStore>()(
    persist(
        (set, get) => ({
            projects: [],
            currentProject: null,
            isLoading: false,
            isAnalyzing: false,
            selectedFrameId: null,

            // Sync projects from backend
            setProjects: (projects: Project[]) => set({ projects }),

            createProject: async (title: string, text: string, skipAnalysis: boolean = false) => {
                set({ isLoading: true });
                try {
                    const project = await api.createProject(title, text, skipAnalysis);
                    set((state) => ({
                        projects: [...state.projects, project],
                        currentProject: project,
                        isLoading: false,
                    }));
                } catch (error) {
                    console.error('Failed to create project:', error);
                    set({ isLoading: false });
                    throw error;
                }
            },

            analyzeProject: async (script: string) => {
                const { currentProject, updateProject, createProject } = get();
                set({ isAnalyzing: true });

                try {
                    let project: Project;
                    if (currentProject && currentProject.id) {
                        project = await api.reparseProject(currentProject.id, script);
                        // Update the store with the new/updated project
                        set((state) => ({
                            projects: state.projects.map((p) =>
                                p.id === project.id ? { ...project, updatedAt: new Date().toISOString() } : p
                            ),
                            currentProject: { ...project, updatedAt: new Date().toISOString() }
                        }));
                    } else {
                        // If no current project, create one (assuming title is available or default)
                        // This case might be rare if we always create project first, but handling it just in case
                        await createProject(currentProject?.title || "New Project", script);
                    }
                } catch (error) {
                    console.error("Failed to analyze script:", error);
                    throw error;
                } finally {
                    set({ isAnalyzing: false });
                }
            },

            loadProjects: () => {
                // Projects are already loaded from localStorage via persist middleware
                // This is mainly for future API sync if needed
            },

            selectProject: async (id: string) => {
                // First, try to set from local cache for immediate feedback
                const cachedProject = get().projects.find((p) => p.id === id);
                if (cachedProject) {
                    set({ currentProject: cachedProject });
                }

                // Then fetch latest data from backend
                try {
                    const response = await authedFetch(`${API_URL}/projects/${id}`);
                    if (response.ok) {
                        const rawData = await response.json();
                        // Transform data to match frontend model (snake_case -> camelCase for specific fields)
                        const latestProject = {
                            ...rawData,
                            originalText: rawData.original_text
                        };

                        // Update both currentProject and projects array with latest data
                        set((state) => ({
                            currentProject: latestProject,
                            projects: state.projects.map((p) =>
                                p.id === id ? latestProject : p
                            ),
                        }));
                    }
                } catch (error) {
                    console.error('Failed to fetch latest project data:', error);
                    // Keep using cached version if fetch fails
                }
            },

            updateProject: (id: string, data: Partial<Project>) => {
                set((state) => ({
                    projects: state.projects.map((p) =>
                        p.id === id ? { ...p, ...data, updatedAt: new Date().toISOString() } : p
                    ),
                    currentProject:
                        state.currentProject?.id === id
                            ? { ...state.currentProject, ...data, updatedAt: new Date().toISOString() }
                            : state.currentProject,
                }));
            },

            deleteProject: async (id: string) => {
                try {
                    // Delete from backend first
                    await api.deleteProject(id);
                    // Then remove from local state
                    set((state) => ({
                        projects: state.projects.filter((p) => p.id !== id),
                        currentProject: state.currentProject?.id === id ? null : state.currentProject
                    }));
                } catch (error) {
                    console.error('Failed to delete project from backend:', error);
                    // Still remove from local state for UX, but warn user
                    set((state) => ({
                        projects: state.projects.filter((p) => p.id !== id),
                        currentProject: state.currentProject?.id === id ? null : state.currentProject
                    }));
                }
            },

            isAnalyzingArtStyle: false,

            analyzeArtStyle: async (scriptId: string, text: string) => {
                set({ isAnalyzingArtStyle: true });
                try {
                    const data = await api.analyzeScriptForStyles(scriptId, text);

                    // Update the project with new recommendations
                    // We need to fetch the latest project state to ensure we don't overwrite other changes
                    // But for now, let's assume we just want to update the recommendations

                    // Actually, analyzeScriptForStyles just returns recommendations, it doesn't save them to the project yet
                    // The user needs to select one.
                    // BUT, to persist them, we should probably save them to the project immediately if possible?
                    // Or just return them?
                    // The issue is: if we navigate away, we lose the return value.
                    // So we MUST save them to the project or store them in the store.

                    // Let's store them in the current project in the store
                    const current = get().currentProject;
                    if (current) {
                        const updatedArtDirection = {
                            ...current.art_direction,
                            ai_recommendations: data.recommendations
                        } as ArtDirection;

                        // Update local state
                        set((state) => ({
                            currentProject: state.currentProject ? {
                                ...state.currentProject,
                                art_direction: updatedArtDirection
                            } : null
                        }));

                        // Also try to save to backend if we have an active art direction
                        // If not, we just keep it in memory until user saves
                    }

                } catch (error) {
                    console.error("Failed to analyze art style:", error);
                    // We could add an error state here if needed
                } finally {
                    set({ isAnalyzingArtStyle: false });
                }
            },

            clearCurrentProject: () => {
                set({ currentProject: null });
            },



            setSelectedFrameId: (id) => set({ selectedFrameId: id }),

            // Asset Generation State
            generatingTasks: [],
            addGeneratingTask: (assetId: string, generationType: string, batchSize: number) => set((state) => ({
                generatingTasks: [...state.generatingTasks, { assetId, generationType, batchSize }]
            })),
            removeGeneratingTask: (assetId: string, generationType: string) => set((state) => ({
                generatingTasks: state.generatingTasks.filter((t) => !(t.assetId === assetId && t.generationType === generationType))
            })),

            // Storyboard Frame Rendering State
            renderingFrames: new Set<string>(),
            addRenderingFrame: (frameId: string) => set((state) => {
                const newSet = new Set(state.renderingFrames);
                newSet.add(frameId);
                return { renderingFrames: newSet };
            }),
            removeRenderingFrame: (frameId: string) => set((state) => {
                const newSet = new Set(state.renderingFrames);
                newSet.delete(frameId);
                return { renderingFrames: newSet };
            }),

            // Storyboard Analysis State
            isAnalyzingStoryboard: false,
            setIsAnalyzingStoryboard: (value: boolean) => set({ isAnalyzingStoryboard: value }),

            // Series State
            seriesList: [],
            currentSeries: null,

            fetchSeriesList: async () => {
                try {
                    const seriesList = await api.listSeries();
                    set({ seriesList });
                } catch (error) {
                    console.error('Failed to fetch series list:', error);
                }
            },

            fetchSeries: async (id: string) => {
                try {
                    const series = await api.getSeries(id);
                    set({ currentSeries: series });
                } catch (error) {
                    console.error('Failed to fetch series:', error);
                }
            },

            createSeries: async (title: string, description?: string) => {
                try {
                    const series = await api.createSeries(title, description);
                    set((state) => ({
                        seriesList: [...state.seriesList, series],
                    }));
                    return series;
                } catch (error) {
                    console.error('Failed to create series:', error);
                    throw error;
                }
            },

            deleteSeries: async (id: string) => {
                try {
                    await api.deleteSeries(id);
                    set((state) => ({
                        seriesList: state.seriesList.filter((s) => s.id !== id),
                        currentSeries: state.currentSeries?.id === id ? null : state.currentSeries,
                    }));
                } catch (error) {
                    console.error('Failed to delete series:', error);
                    throw error;
                }
            },

            setCurrentSeries: (series: Series | null) => set({ currentSeries: series }),
        }),
        {
            name: 'project-storage',
            partialize: (state) => ({
                projects: state.projects,

                generatingTasks: state.generatingTasks // Now persisting this to maintain state across refreshes
            }),
        }
    )
);
