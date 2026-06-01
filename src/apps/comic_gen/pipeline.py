from typing import Dict, Any, List, Optional, Tuple, Callable, Iterator
from contextlib import contextmanager
import json
import os
import re
import time
import uuid
import subprocess
import threading
import platform
from urllib.parse import quote
from .models import Script, GenerationStatus, VideoTask, Character, Scene, StoryboardFrame, Series, PromptConfig, ImageAsset, ImageVariant, ModelSettings
from .llm import ScriptProcessor
from .assets import AssetGenerator
from .storyboard import StoryboardGenerator
from .video import VideoGenerator
from .audio import AudioGenerator
from .export import ExportManager
from .instance_resolver import scoped_instance, require_instance
from ...models.instance import InstanceType
from ...utils import get_logger
from ...utils.oss_utils import is_object_key
from ...utils.provider_media import MediaResolver
from ...utils.provider_registry import resolve_provider_backend
from ...utils.system_check import get_ffmpeg_path, get_ffmpeg_install_instructions
from ...i18n import t as _t

logger = get_logger(__name__)

# --- Security helpers ---

# Allowed pattern for IDs used in file paths (UUID hex + hyphens)
_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _validate_safe_id(value: str, label: str = "id") -> str:
    """Ensure a value is safe to embed in file paths / command args (UUID-like)."""
    if not value or not _SAFE_ID_RE.match(value):
        raise ValueError(f"Invalid {label}: contains unsafe characters")
    return value


def _safe_resolve_path(base_dir: str, untrusted_rel: str) -> str:
    """Resolve *untrusted_rel* under *base_dir* and ensure the result stays inside it.

    Prevents path-traversal attacks (e.g. ``../../etc/passwd``).
    Returns the resolved absolute path; raises ValueError on escape attempts.
    """
    base = os.path.realpath(base_dir)
    resolved = os.path.realpath(os.path.join(base, untrusted_rel))
    if not resolved.startswith(base + os.sep) and resolved != base:
        raise ValueError(f"Path escapes base directory: {untrusted_rel}")
    return resolved

class ComicGenPipeline:
    # Shared default so instances created via ``__new__`` (used by some tests)
    # still expose a sensible ``data_root`` even when ``__init__`` is skipped.
    data_root: str = "output"

    def __init__(self, config: Dict[str, Any] = None, data_root: str = "output"):
        self.config = config or {}
        # Per-tenant data root. Each user's pipeline points at
        # ``output/users/<uid>``; the legacy single-tenant default stays
        # ``output``. All "output" path literals below are read off this
        # field so multi-user instances can keep state physically isolated.
        self.data_root = data_root
        os.makedirs(self.data_root, exist_ok=True)

        def _sub_config(name: str) -> Dict[str, Any]:
            base = self.config.get(name, {}) or {}
            return {**base, "data_root": self.data_root}

        self.script_processor = ScriptProcessor()
        self.asset_generator = AssetGenerator(_sub_config('assets'))
        self.storyboard_generator = StoryboardGenerator(_sub_config('storyboard'))
        self.video_generator = VideoGenerator(_sub_config('video'))
        self.audio_generator = AudioGenerator(_sub_config('audio'))
        self.export_manager = ExportManager(_sub_config('export'))
        self._mg_generator = None  # flow B (Remotion MG), built lazily

        self.data_file = os.path.join(self.data_root, "projects.json")
        self.series_data_file = os.path.join(self.data_root, "series.json")
        self._save_lock = threading.RLock()  # Reentrant lock to prevent concurrent file writes
        self.scripts: Dict[str, Script] = self._load_data()
        self.series_store: Dict[str, Series] = self._load_series_data()
        
        # Task management for async asset generation
        # Format: { task_id: { status: str, progress: int, error: str, script_id: str, asset_id: str, created_at: float } }
        self.asset_generation_tasks: Dict[str, Dict[str, Any]] = {}
        self.video_generation_tasks: Dict[str, Dict[str, Any]] = {}
        # Temporary cache for file import previews (import_id -> text)
        self._import_cache: Dict[str, str] = {}
        # Cached model instances for Kling/Vidu (lazily initialized).
        # Kept for backward-compat with tests that monkeypatch these slots
        # directly; the actual dispatch goes through ``_video_dispatcher``.
        self._kling_model = None
        self._vidu_model = None
        self._video_dispatcher = None  # built lazily in process_video_task

    def _resolve_video_backend(self, model_name: str) -> str:
        try:
            return resolve_provider_backend(model_name)
        except (KeyError, ValueError):
            logger.debug(
                "Provider backend not registered for video model %s, defaulting to dashscope.",
                model_name,
            )
            return "dashscope"
        except Exception as e:
            logger.warning(
                "Unexpected error resolving provider backend for video model %s: %s. "
                "Falling back to dashscope.",
                model_name,
                e,
            )
            return "dashscope"

    def _build_video_extras(self, script: "Script", task: "VideoTask") -> Dict[str, Any]:
        """Collect frame-level hints for locally-composing video adapters.

        Only the Remotion engine reads these today (camera_movement → Ken Burns,
        dialogue → burned-in subtitle, dialogue audio → clip audio). Returns an
        empty dict for asset videos or when the frame can't be found, so cloud
        adapters keep getting a no-op ``extras``.
        """
        extras: Dict[str, Any] = {}
        if not getattr(task, "frame_id", None) or not script:
            return extras
        frame = next((f for f in script.frames if f.id == task.frame_id), None)
        if frame is None:
            return extras
        extras["camera_movement"] = frame.camera_movement
        extras["shot_size"] = frame.shot_size
        extras["dialogue"] = frame.dialogue
        extras["duration_seconds"] = frame.duration_seconds
        extras["dialogue_audio"] = frame.audio_url
        ms = getattr(script, "model_settings", None)
        if ms is not None:
            extras["aspect_ratio"] = getattr(ms, "storyboard_aspect_ratio", None)
        return extras

    def export_project(self, script_id: str, options: Dict[str, Any]) -> str:
        """Step 7: Export project to final video."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        export_url = self.export_manager.render_project(script, options)
        return export_url

    def get_script(self, script_id: str) -> Optional[Script]:
        return self.scripts.get(script_id)

    def _load_data(self) -> Dict[str, Script]:
        if not os.path.exists(self.data_file):
            return {}
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                return {k: Script(**v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {}

    def _save_data(self):
        """Save data with thread lock to prevent concurrent write issues."""
        with self._save_lock:
            try:
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                with open(self.data_file, 'w') as f:
                    json.dump({k: v.dict() for k, v in self.scripts.items()}, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save data: {e}")

    def create_project(self, title: str, text: str, skip_analysis: bool = False) -> Script:
        """Step 1: Parse novel and create project."""
        if skip_analysis:
            script = self.script_processor.create_draft_script(title, text)
        else:
            # Use the user's default LLM instance for the initial parse —
            # there is no project-scoped instance yet at this point.
            with scoped_instance(None, InstanceType.LLM):
                script = self.script_processor.parse_novel(title, text)

        self.scripts[script.id] = script
        self._save_data()
        return script

    def reparse_project(self, script_id: str, text: str) -> Script:
        """Re-parse the text for an existing project, replacing all entities."""
        existing_script = self.scripts.get(script_id)
        if not existing_script:
            raise ValueError("Script not found")

        # Parse the new text under the project's configured LLM instance.
        with scoped_instance(existing_script.model_settings.llm_instance_id, InstanceType.LLM):
            new_script = self.script_processor.parse_novel(existing_script.title, text)
        
        # Preserve the original script ID and timestamps
        new_script.id = existing_script.id
        new_script.created_at = existing_script.created_at
        new_script.updated_at = time.time()
        
        # Preserve project-level settings
        new_script.art_direction = existing_script.art_direction
        new_script.model_settings = existing_script.model_settings
        new_script.style_preset = existing_script.style_preset
        new_script.style_prompt = existing_script.style_prompt
        new_script.merged_video_url = existing_script.merged_video_url
        
        # Replace the script in memory
        self.scripts[script_id] = new_script
        self._save_data()
        return new_script

    # ── Flow B: Remotion motion-graphics engine (chat-only) ───────────────

    @property
    def mg_generator(self):
        """Lazily-built generator for the Remotion MG engine (flow B)."""
        if self._mg_generator is None:
            from .remotion_mg import RemotionMGGenerator
            self._mg_generator = RemotionMGGenerator()
        return self._mg_generator

    def create_mg_project(self, title: str, text: str, aspect_ratio: str = "9:16") -> Script:
        """Create a chat-only motion-graphics project (flow B).

        No novel parse / no entities — the project's whole video is authored as
        a VideoSpec by the LLM later (``generate_mg_spec``) and rendered by
        Remotion (``render_mg_video``). The漫剧 stages are never invoked.
        """
        now = time.time()
        script = Script(
            id=str(uuid.uuid4()),
            title=title,
            original_text=text,
            generation_engine="remotion_mg",
            model_settings=ModelSettings(storyboard_aspect_ratio=aspect_ratio),
            created_at=now,
            updated_at=now,
        )
        self.scripts[script.id] = script
        self._save_data()
        return script

    def generate_mg_spec(self, script_id: str, style_hint: Optional[str] = None) -> Script:
        """Flow B step 1: LLM authors the VideoSpec and stores it on the script."""
        script = self.get_script(script_id)
        if not script:
            raise ValueError("Script not found")
        aspect = getattr(script.model_settings, "storyboard_aspect_ratio", "9:16") or "9:16"
        with scoped_instance(script.model_settings.llm_instance_id, InstanceType.LLM):
            spec = self.mg_generator.generate_spec(
                script.original_text, aspect_ratio=aspect, style_hint=style_hint
            )
        script.mg_spec = spec.to_payload()
        script.updated_at = time.time()
        self._save_data()
        return script

    def render_mg_video(self, script_id: str) -> Script:
        """Flow B step 2: render the stored VideoSpec to an MP4 via Remotion."""
        from ...models.remotion_spec import VideoSpec

        script = self.get_script(script_id)
        if not script:
            raise ValueError("Script not found")
        if not script.mg_spec:
            raise ValueError("No mg_spec on this project — run generate_mg_spec first")

        spec = VideoSpec.model_validate(script.mg_spec)
        output_filename = f"remotion_{script_id}_{int(time.time())}.mp4"
        output_path = _safe_resolve_path(
            os.path.join(self.data_root, "video"), output_filename
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.mg_generator.render(spec, output_path)
        # Mirror merged_video_url's "videos/<file>" alias convention.
        script.remotion_video_url = f"videos/{output_filename}"
        script.updated_at = time.time()
        self._save_data()
        return script

    def generate_assets(self, script_id: str) -> Script:
        """Step 2: Generate character and scene assets (Batch)."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        logger.info(f"Generating assets for script {script.id}")
        
        # Sort characters: Base characters first (those without base_character_id)
        sorted_chars = sorted(script.characters, key=lambda c: 0 if not c.base_character_id else 1)

        for char in sorted_chars:
            self.generate_asset(script_id, char.id, "character")
            
        for scene in script.scenes:
            self.generate_asset(script_id, scene.id, "scene")
            
        for prop in script.props:
            self.generate_asset(script_id, prop.id, "prop")
            
        self._save_data()
        return script

    def generate_asset(self, script_id: str, asset_id: str, asset_type: str, style_preset: str = None, reference_image_url: str = None, style_prompt: str = None, generation_type: str = "all", prompt: str = None, apply_style: bool = True, negative_prompt: str = None, batch_size: int = 1, model_name: str = None) -> Script:
        """Step 2: Generate a specific asset (character/scene/prop).
        If style_preset is None, uses the project's global style."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        # Get effective size based on asset type
        from .assets import ASPECT_RATIO_TO_SIZE
        if asset_type == "character":
            aspect_ratio = script.model_settings.character_aspect_ratio
            default_size = "576*1024"  # Portrait
        elif asset_type == "scene":
            aspect_ratio = script.model_settings.scene_aspect_ratio
            default_size = "1024*576"  # Landscape
        elif asset_type == "prop":
            aspect_ratio = script.model_settings.prop_aspect_ratio
            default_size = "1024*1024"  # Square
        else:
            aspect_ratio = "9:16"
            default_size = "576*1024"

        effective_size = ASPECT_RATIO_TO_SIZE.get(aspect_ratio, default_size)
        
        # Determine effective style: Art Direction > passed style > legacy style
        effective_positive_prompt = ""
        effective_negative_prompt = negative_prompt or "" # Use passed negative prompt if available
        
        # Only calculate style prompt if apply_style is True
        if apply_style:
            if script.art_direction and script.art_direction.style_config:
                # Use Art Direction (highest priority)
                effective_positive_prompt = script.art_direction.style_config.get('positive_prompt', '')
                # Append global negative prompt if not overridden or append to it?
                # Let's append global negative prompt to the specific one for better results
                global_neg = script.art_direction.style_config.get('negative_prompt', '')
                if global_neg:
                    effective_negative_prompt = f"{effective_negative_prompt}, {global_neg}" if effective_negative_prompt else global_neg
            elif style_prompt:
                # Use passed style_prompt (for manual override)
                effective_positive_prompt = style_prompt
            elif style_preset:
                # Use passed style_preset (legacy)
                effective_positive_prompt = f"{style_preset} style"
            elif script.style_preset:
                # Fallback to script's legacy style_preset
                effective_positive_prompt = f"{script.style_preset} style"
                if script.style_prompt:
                    effective_positive_prompt += f", {script.style_prompt}"
        
        asset_list = []
        target_asset = None
        
        if asset_type == "character":
            asset_list = script.characters
        elif asset_type == "scene":
            asset_list = script.scenes
        elif asset_type == "prop":
            asset_list = script.props
        else:
            raise ValueError(f"Invalid asset_type: {asset_type}")
        
        target_asset = next((a for a in asset_list if a.id == asset_id), None)
        if not target_asset:
            raise ValueError(f"{asset_type.capitalize()} {asset_id} not found")
        
        target_asset.status = GenerationStatus.PROCESSING
        self._save_data()

        try:
            # Bind T2I and I2I instances to this call so credentials and
            # model_name flow from the user's configured ModelInstance rows.
            with scoped_instance(script.model_settings.t2i_instance_id, InstanceType.T2I) as t2i_inst, \
                 scoped_instance(script.model_settings.i2i_instance_id, InstanceType.I2I) as i2i_inst:
                # Caller-provided ``model_name`` (rare per-call override) wins over
                # the instance's; otherwise the instance must exist.
                t2i_model = model_name or require_instance(t2i_inst, InstanceType.T2I).model_name
                # I2I is only used inside character generation here; if the user
                # hasn't configured an I2I instance we still want the T2I path to
                # work, so we lazily require I2I only when the asset code asks
                # for it (passed to ``i2i_model_name=`` below).
                i2i_model = i2i_inst.model_name if i2i_inst else None
                # Generate with Art Direction style injected
                if asset_type == "character":
                    self.asset_generator.generate_character(
                        target_asset,
                        generation_type=generation_type,
                        prompt=prompt,
                        positive_prompt=effective_positive_prompt,
                        negative_prompt=effective_negative_prompt,
                        batch_size=batch_size,
                        model_name=t2i_model,
                        i2i_model_name=i2i_model,
                        size=effective_size
                    )
                elif asset_type == "scene":
                    self.asset_generator.generate_scene(target_asset, effective_positive_prompt, effective_negative_prompt, batch_size=batch_size, model_name=t2i_model, size=effective_size)
                elif asset_type == "prop":
                    self.asset_generator.generate_prop(target_asset, effective_positive_prompt, effective_negative_prompt, batch_size=batch_size, model_name=t2i_model, size=effective_size)
                
            target_asset.status = GenerationStatus.COMPLETED
        except Exception as e:
            target_asset.status = GenerationStatus.FAILED
            raise e
        finally:
            self._save_data()
        
        return script

    def create_asset_generation_task(self, script_id: str, asset_id: str, asset_type: str, 
                                      style_preset: str = None, reference_image_url: str = None, 
                                      style_prompt: str = None, generation_type: str = "all", 
                                      prompt: str = None, apply_style: bool = True, 
                                      negative_prompt: str = None, batch_size: int = 1, 
                                      model_name: str = None) -> Tuple[Script, str]:
        """Creates an async asset generation task and returns (script, task_id) immediately."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        # Find the asset and set to PROCESSING
        asset_list = []
        if asset_type == "character":
            asset_list = script.characters
        elif asset_type == "scene":
            asset_list = script.scenes
        elif asset_type == "prop":
            asset_list = script.props
        else:
            raise ValueError(f"Invalid asset_type: {asset_type}")
        
        target_asset = next((a for a in asset_list if a.id == asset_id), None)
        if not target_asset:
            raise ValueError(f"{asset_type.capitalize()} {asset_id} not found")
        
        target_asset.status = GenerationStatus.PROCESSING
        
        # Create task
        task_id = str(uuid.uuid4())
        self.asset_generation_tasks[task_id] = {
            "status": "pending",  # pending -> processing -> completed/failed
            "progress": 0,
            "error": None,
            "script_id": script_id,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "created_at": time.time(),
            # Store all params for later processing
            "params": {
                "style_preset": style_preset,
                "reference_image_url": reference_image_url,
                "style_prompt": style_prompt,
                "generation_type": generation_type,
                "prompt": prompt,
                "apply_style": apply_style,
                "negative_prompt": negative_prompt,
                "batch_size": batch_size,
                "model_name": model_name
            }
        }
        
        self._save_data()
        return script, task_id

    def process_asset_generation_task(self, task_id: str):
        """Processes an asset generation task in the background."""
        task = self.asset_generation_tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        task["status"] = "processing"

        try:
            params = task["params"]
            if task.get("is_series"):
                # Series asset generation — operate on series_store
                self._process_series_asset_task(task, params)
            else:
                # Project asset generation — existing logic
                self.generate_asset(
                    task["script_id"],
                    task["asset_id"],
                    task["asset_type"],
                    params["style_preset"],
                    params["reference_image_url"],
                    params["style_prompt"],
                    params["generation_type"],
                    params["prompt"],
                    params["apply_style"],
                    params["negative_prompt"],
                    params["batch_size"],
                    params["model_name"]
                )
            task["status"] = "completed"
            task["progress"] = 100
            logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            logger.error(f"Task {task_id} failed: {e}")

    def _process_series_asset_task(self, task: Dict, params: Dict):
        """Process a Series asset generation task."""
        series_id = task["script_id"]  # stored as script_id for compatibility
        series = self.series_store.get(series_id)
        if not series:
            raise ValueError("Series not found")

        asset_id = task["asset_id"]
        asset_type = task["asset_type"]
        positive_prompt = params.get("effective_positive_prompt", "")
        negative_prompt = params.get("effective_negative_prompt", "")
        t2i_model = params.get("t2i_model")
        if not t2i_model:
            from .instance_resolver import InstanceNotConfiguredError
            raise InstanceNotConfiguredError(InstanceType.T2I)
        effective_size = params.get("effective_size", "576*1024")
        batch_size = params.get("batch_size", 1)
        generation_type = params.get("generation_type", "all")
        prompt = params.get("prompt")
        reference_image_url = params.get("reference_image_url")

        # Re-bind instance scope on the worker thread (ContextVars are
        # carried by run_in_executor; this is a safety net for paths that
        # didn't go through it).
        with scoped_instance(params.get("t2i_instance_id"), InstanceType.T2I), \
             scoped_instance(params.get("i2i_instance_id"), InstanceType.I2I):
            if asset_type == "character":
                target = next((c for c in series.characters if c.id == asset_id), None)
                if not target:
                    raise ValueError(f"Character {asset_id} not found in series")
                self.asset_generator.generate_character(
                    target, generation_type=generation_type, prompt=prompt or "",
                    positive_prompt=positive_prompt, negative_prompt=negative_prompt,
                    batch_size=batch_size, model_name=t2i_model, size=effective_size,
                )
            elif asset_type == "scene":
                target = next((s for s in series.scenes if s.id == asset_id), None)
                if not target:
                    raise ValueError(f"Scene {asset_id} not found in series")
                self.asset_generator.generate_scene(
                    target, positive_prompt=positive_prompt, negative_prompt=negative_prompt,
                    batch_size=batch_size, model_name=t2i_model, size=effective_size,
                )
            elif asset_type == "prop":
                target = next((p for p in series.props if p.id == asset_id), None)
                if not target:
                    raise ValueError(f"Prop {asset_id} not found in series")
                self.asset_generator.generate_prop(
                    target, positive_prompt=positive_prompt, negative_prompt=negative_prompt,
                    batch_size=batch_size, model_name=t2i_model, size=effective_size,
                )
            else:
                raise ValueError(f"Unknown asset type: {asset_type}")

        self._save_series_data()

    def get_asset_generation_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Returns the status of an asset generation task."""
        # Check image tasks first
        task = self.asset_generation_tasks.get(task_id)
        if not task:
            # Then check video tasks
            task = self.video_generation_tasks.get(task_id)

        if not task:
            return None

        result = {
            "task_id": task_id,
            "status": task["status"],
            "progress": task.get("progress", 0),
            "error": task.get("error"),
            "asset_id": task.get("asset_id"),
            "asset_type": task.get("asset_type"),
            "script_id": task.get("script_id"),
            "created_at": task.get("created_at"),
        }
        # Surface batch-render + one-shot fields when present so the frontend
        # can render granular progress / pull results without a separate
        # endpoint. Keys absent from the task dict are simply omitted.
        for key in (
            "task_type",
            "total_count",
            "pending_count",
            "completed_count",
            "failed_count",
            "current_item_id",
            "current_frame_id",  # legacy alias kept for storyboard batch
            "current_stage",     # used by export pipeline
            "errors",
            "result",            # one-shot return value (LLM tasks)
            "output_url",        # export tasks
        ):
            if key in task:
                result[key] = task[key]
        return result

    # === Async-task primitives =============================================
    #
    # Three building blocks every async endpoint composes from. They exist
    # because we found ourselves rewriting the same status-machine
    # boilerplate (pending→processing→completed/failed) in every new
    # ``process_X_task`` method. Centralizing means each new background
    # job is ~10 lines of business logic instead of 30 lines of plumbing.
    #
    # Pattern in use:
    #   task_id = self._register_task(task_type="...", script_id=sid, ...)
    #   # later, on the worker thread:
    #   def process_X_task(self, task_id):
    #       with self._task_lifecycle(task_id) as task:
    #           # business logic; raise to mark failed, normal return = ok
    #           ...

    def _register_task(
        self,
        *,
        task_type: str,
        script_id: Optional[str] = None,
        store: str = "asset",
        **fields: Any,
    ) -> str:
        """Create + register a task dict and return its UUID.

        Single source of truth for the task envelope shape. Every batch
        field (``completed_count`` / ``failed_count`` / ``errors`` /
        ``current_item_id``) is initialized here so ``/tasks/{id}`` can
        always read them without ``KeyError``.

        Pass extra task-specific keys via ``**fields`` (e.g. ``params``,
        ``asset_id``, ``asset_type``).
        """
        task_id = str(uuid.uuid4())
        envelope: Dict[str, Any] = {
            "task_type": task_type,
            "status": "pending",
            "progress": 0,
            "error": None,
            "script_id": script_id,
            "created_at": time.time(),
            "completed_count": 0,
            "failed_count": 0,
            "errors": {},
            "current_item_id": None,
        }
        envelope.update(fields)
        if store == "video":
            self.video_generation_tasks[task_id] = envelope
        else:
            self.asset_generation_tasks[task_id] = envelope
        return task_id

    def _get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return (
            self.asset_generation_tasks.get(task_id)
            or self.video_generation_tasks.get(task_id)
        )

    @contextmanager
    def _task_lifecycle(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Status state-machine: pending→processing→completed/failed.

        Catches and records any exception so workers focus on business
        logic only. A worker that wants to *succeed early* (no work to
        do) can ``return`` normally; one that wants to *fail* can raise.
        """
        task = self._get_task(task_id)
        if task is None:
            # Defensive: a vanished task ID is a programmer error, not a
            # user-facing failure. Yield a transient throwaway dict so the
            # ``with`` block doesn't explode, and log loudly.
            logger.error(f"Task {task_id} not found")
            yield {"status": "failed", "error": "task vanished", "progress": 0}
            return
        task["status"] = "processing"
        try:
            yield task
            # Worker may have already set status (e.g. "completed" after
            # a partial-failure batch). Only finalize if still in flight.
            if task["status"] == "processing":
                task["status"] = "completed"
                task["progress"] = 100
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            logger.exception(f"Task {task_id} failed: {e}")

    def _run_per_item_batch(
        self,
        task_id: str,
        items: List[Any],
        work: Callable[[Any], None],
        item_id: Callable[[Any], str] = lambda x: x.id,
    ) -> None:
        """Iterate items, isolate per-item failures, persist after each.

        Generic worker shared by storyboard/video/audio batch tasks.
        ``work`` is the per-item Strategy: it raises on failure, returns
        on success. Exceptions are recorded into ``task['errors'][iid]``
        so the batch keeps marching even if one item explodes.
        """
        task = self._get_task(task_id)
        if task is None:
            return
        total = len(items)
        if total == 0:
            return
        for idx, item in enumerate(items, start=1):
            iid = item_id(item)
            task["current_item_id"] = iid
            try:
                work(item)
                task["completed_count"] += 1
            except Exception as e:
                task["failed_count"] += 1
                task["errors"][iid] = str(e)
                logger.exception(f"Batch {task_id}: item {iid} failed: {e}")
            task["progress"] = int(idx / total * 100)
            try:
                self._save_data()
            except Exception:
                logger.exception("Batch: _save_data failed (continuing)")
        task["current_item_id"] = None

    def _run_one_shot(
        self,
        task_id: str,
        work: Callable[[], Any],
    ) -> None:
        """Run ``work()`` once inside the task lifecycle; stash its return
        value into ``task['result']`` for the client to fetch via
        ``/tasks/{id}``.

        Used for LLM polish / analyze calls — short single operations
        whose **return value** is what the caller needs (vs. batch tasks
        which mutate persisted project state).
        """
        with self._task_lifecycle(task_id) as task:
            task["result"] = work()

    def create_motion_ref_task(self, script_id: str, asset_id: str, asset_type: str, 
                                prompt: Optional[str] = None, audio_url: Optional[str] = None, 
                                duration: int = 5, batch_size: int = 1) -> Tuple[Script, str]:
        """Creates an async motion reference generation task."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        task_id = str(uuid.uuid4())
        self.video_generation_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "error": None,
            "script_id": script_id,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "created_at": time.time(),
            "params": {
                "prompt": prompt,
                "audio_url": audio_url,
                "duration": duration,
                "batch_size": batch_size
            }
        }
        
        self._save_data()
        return script, task_id

    def process_motion_ref_task(self, script_id: str, task_id: str):
        """Processes a video generation task in the background."""
        task = self.video_generation_tasks.get(task_id)
        if not task:
            logger.error(f"Video task {task_id} not found")
            return
            
        task["status"] = "processing"
        
        try:
            params = task["params"]
            # Call the synchronous generate_motion_ref method
            self.generate_motion_ref(
                script_id=script_id,
                asset_id=task["asset_id"],
                asset_type=task["asset_type"],
                prompt=params["prompt"],
                audio_url=params["audio_url"],
                duration=params["duration"],
                batch_size=params["batch_size"]
            )
            task["status"] = "completed"
            task["progress"] = 100
            logger.info(f"Video task {task_id} completed successfully")
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            logger.error(f"Video task {task_id} failed: {e}")

    def sync_descriptions_from_script_entities(self, script_id: str) -> Script:
        """
        Syncs entity descriptions from ScriptProcessor parsed entities.
        This clears saved prompts so the UI will regenerate them from the current description.
        
        Note: This only updates prompts, not generated images/videos.
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        # Clear saved prompts for all characters so UI will regenerate from description
        for character in script.characters:
            character.full_body_prompt = None
            character.three_view_prompt = None
            character.headshot_prompt = None
            character.video_prompt = None
        
        # Scenes and props might also have prompts to clear (if applicable)
        for scene in script.scenes:
            if hasattr(scene, 'prompt'):
                scene.prompt = None
        
        for prop in script.props:
            if hasattr(prop, 'prompt'):
                prop.prompt = None
        
        self._save_data()
        logger.info(f"Descriptions synced for script {script_id}: cleared prompts for {len(script.characters)} characters, {len(script.scenes)} scenes, {len(script.props)} props")
        return script

    def add_character(self, script_id: str, name: str, description: str) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        new_char = Character(
            id=f"char_{uuid.uuid4().hex[:8]}",
            name=name,
            description=description
        )
        script.characters.append(new_char)
        self._save_data()
        return script

    def delete_character(self, script_id: str, char_id: str) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        script.characters = [c for c in script.characters if c.id != char_id]
        self._save_data()
        return script

    def add_scene(self, script_id: str, name: str, description: str) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        new_scene = Scene(
            id=f"scene_{uuid.uuid4().hex[:8]}",
            name=name,
            description=description
        )
        script.scenes.append(new_scene)
        self._save_data()
        return script

    def delete_scene(self, script_id: str, scene_id: str) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        script.scenes = [s for s in script.scenes if s.id != scene_id]
        self._save_data()
        return script
    
    def toggle_asset_lock(self, script_id: str, asset_id: str, asset_type: str) -> Script:
        """Toggle the locked status of an asset."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            
        if not target_asset:
            raise ValueError(f"Asset {asset_id} of type {asset_type} not found")
            
        # Toggle the locked status
        target_asset.locked = not target_asset.locked
        self._save_data()
        return script

    def toggle_frame_lock(self, script_id: str, frame_id: str) -> Script:
        """Toggle the locked status of a frame."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        target_frame = next((f for f in script.frames if f.id == frame_id), None)
        if not target_frame:
            raise ValueError(f"Frame {frame_id} not found")
            
        # Toggle the locked status
        target_frame.locked = not target_frame.locked
        self._save_data()
        return script

    def update_asset_image(self, script_id: str, asset_id: str, asset_type: str, image_url: str) -> Script:
        """Updates the image URL of an asset manually."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            
        if not target_asset:
            raise ValueError(f"Asset {asset_id} of type {asset_type} not found")
            
        target_asset.image_url = image_url
        # For characters, also update avatar if it's not set or if we want to sync them
        # For now, let's assume the uploaded image is the main reference. 
        # If it's a character, we might want to set avatar_url to the same image for simplicity
        if asset_type == "character":
            target_asset.avatar_url = image_url
            
        self._save_data()
        return script

    def update_asset_description(self, script_id: str, asset_id: str, asset_type: str, description: str) -> Script:
        """Updates the description of an asset."""
        return self.update_asset_attributes(script_id, asset_id, asset_type, {"description": description})

    def update_asset_attributes(self, script_id: str, asset_id: str, asset_type: str, attributes: Dict[str, Any]) -> Script:
        """Updates arbitrary attributes of an asset."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            
        if not target_asset:
            raise ValueError(f"Asset {asset_id} of type {asset_type} not found")
            
        # Update attributes
        for key, value in attributes.items():
            if hasattr(target_asset, key):
                setattr(target_asset, key, value)
            else:
                logger.warning(f"Attribute {key} not found in {asset_type} model")
        
        self._save_data()
        return script

    def add_uploaded_asset_variant(
        self, 
        script_id: str, 
        asset_type: str, 
        asset_id: str, 
        upload_type: str, 
        image_url: str, 
        description: Optional[str] = None
    ) -> Script:
        """
        Adds an uploaded image as a new variant to an asset.
        The uploaded image is marked with is_uploaded_source=True.
        
        Args:
            script_id: The project ID
            asset_type: "character", "scene", or "prop"
            asset_id: The asset ID
            upload_type: "full_body", "head_shot", "three_views", or "image"
            image_url: URL of the uploaded image (OSS Object Key)
            description: Optional modified description for reverse generation
        """
        from .models import ImageVariant, AssetUnit
        
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        # Find target asset
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
        
        if not target_asset:
            raise ValueError(f"Asset {asset_id} of type {asset_type} not found")
        
        # Create new variant with upload source flag
        new_variant = ImageVariant(
            id=str(uuid.uuid4()),
            url=image_url,
            prompt_used=description or target_asset.description,
            is_uploaded_source=True,
            upload_type=upload_type
        )
        
        # Update description if provided
        if description:
            target_asset.description = description
        
        # Add variant to the appropriate asset unit
        if asset_type == "character":
            # Map upload_type to the correct asset unit
            if upload_type == "full_body":
                target_unit = target_asset.full_body
            elif upload_type == "head_shot":
                target_unit = target_asset.head_shot
            elif upload_type == "three_views":
                target_unit = target_asset.three_views
            else:
                raise ValueError(f"Invalid upload_type for character: {upload_type}")
            
            # Ensure AssetUnit exists
            if target_unit is None:
                target_unit = AssetUnit()
                if upload_type == "full_body":
                    target_asset.full_body = target_unit
                elif upload_type == "head_shot":
                    target_asset.head_shot = target_unit
                elif upload_type == "three_views":
                    target_asset.three_views = target_unit
            
            # Add variant and select it
            target_unit.image_variants.append(new_variant)
            target_unit.selected_image_id = new_variant.id
            target_unit.image_updated_at = time.time()
            
            # === ALSO UPDATE LEGACY FIELDS for frontend compatibility ===
            # Create variant for legacy ImageAsset structure
            legacy_variant = ImageVariant(
                id=new_variant.id,
                url=image_url,
                prompt_used=description or target_asset.description,
                is_uploaded_source=True,
                upload_type=upload_type
            )
            
            if upload_type == "full_body":
                # Ensure full_body_asset exists
                if target_asset.full_body_asset is None:
                    from .models import ImageAsset
                    target_asset.full_body_asset = ImageAsset()
                target_asset.full_body_asset.variants.append(legacy_variant)
                target_asset.full_body_asset.selected_id = new_variant.id
                target_asset.full_body_image_url = image_url
            elif upload_type == "head_shot":
                # Ensure headshot_asset exists
                if target_asset.headshot_asset is None:
                    from .models import ImageAsset
                    target_asset.headshot_asset = ImageAsset()
                target_asset.headshot_asset.variants.append(legacy_variant)
                target_asset.headshot_asset.selected_id = new_variant.id
                target_asset.headshot_image_url = image_url
            elif upload_type == "three_views":
                # Ensure three_view_asset exists
                if target_asset.three_view_asset is None:
                    from .models import ImageAsset
                    target_asset.three_view_asset = ImageAsset()
                target_asset.three_view_asset.variants.append(legacy_variant)
                target_asset.three_view_asset.selected_id = new_variant.id
                target_asset.three_view_image_url = image_url
            
            logger.info(f"Added uploaded variant {new_variant.id} to character {asset_id} {upload_type}")
            
        elif asset_type in ["scene", "prop"]:
            # Scene and Prop have a single 'image' asset unit
            if not hasattr(target_asset, 'image') or target_asset.image is None:
                target_asset.image = AssetUnit()
            
            target_asset.image.image_variants.append(new_variant)
            target_asset.image.selected_image_id = new_variant.id
            target_asset.image.image_updated_at = time.time()
            
            # Also update legacy image_url field
            target_asset.image_url = image_url
            
            logger.info(f"Added uploaded variant {new_variant.id} to {asset_type} {asset_id}")
        
        self._save_data()
        return script

    def update_project_style(self, script_id: str, style_preset: str, style_prompt: Optional[str] = None) -> Script:
        """Updates the global style settings for a project."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        script.style_preset = style_preset
        script.style_prompt = style_prompt
        script.updated_at = time.time()
        self._save_data()
        return script
    
    def save_art_direction(self, script_id: str, selected_style_id: str, style_config: Dict[str, Any], custom_styles: List[Dict[str, Any]] = None, ai_recommendations: List[Dict[str, Any]] = None) -> Script:
        """Saves the Art Direction configuration."""
        from .models import ArtDirection
        
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        # Create Art Direction object
        art_direction = ArtDirection(
            selected_style_id=selected_style_id,
            style_config=style_config,
            custom_styles=custom_styles or [],
            ai_recommendations=ai_recommendations or []
        )
        
        script.art_direction = art_direction
        script.updated_at = time.time()
        self._save_data()
        return script

    # === SCRIPT REWRITE (novel → screenplay) ===

    def rewrite_script_to_screenplay(self, script_id: str) -> Script:
        """Run the screenplay rewriter on ``script.original_text`` and persist.

        Result lands on ``script.formatted_text``; ``original_text`` is
        preserved so the user can re-rewrite or revert. Downstream
        analyzers should prefer ``formatted_text`` when present.
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        if not script.original_text:
            raise ValueError("Script has no original_text to rewrite")

        with scoped_instance(script.model_settings.llm_instance_id, InstanceType.LLM):
            formatted = self.script_processor.rewrite_to_screenplay(script.original_text)

        script.formatted_text = formatted
        script.updated_at = time.time()
        self._save_data()
        return script

    # === ENTITY EXTRACTION (incremental / full) ===

    def process_script_entities(
        self, script_id: str, strategy: str = "incremental"
    ) -> Dict[str, Any]:
        """Run entity extraction and write results into the Series catalog.

        Uses ``"incremental"`` by default — the LLM is shown the existing
        Series catalog and asked to reuse entities by id rather than
        invent duplicates. Pass ``strategy="full"`` to force a clean
        re-extraction (entities still get deduped by the MatchSpec safety
        net, but no LLM-side reuse hint is provided).

        Returns a summary suitable for the API response::

            {"created": {"characters": [...], "scenes": [...], "props": [...]},
             "reused":  {"characters": [...], "scenes": [...], "props": [...]}}
        """
        from .extraction import EntityCatalog, get_strategy

        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        series = self.series_store.get(script.series_id) if script.series_id else None
        catalog = EntityCatalog(script, series)
        strat = get_strategy(strategy)

        with scoped_instance(script.model_settings.llm_instance_id, InstanceType.LLM):
            result = strat.extract(script.original_text, catalog, self.script_processor)

        # Persist: both the (possibly mutated) Series and the script (for used_*_ids).
        script.updated_at = time.time()
        self._save_data()
        if series is not None:
            self._save_series_data()

        return {
            "strategy": strategy,
            "created": {
                "characters": result.created_character_ids,
                "scenes": result.created_scene_ids,
                "props": result.created_prop_ids,
            },
            "reused": {
                "characters": result.reused_character_ids,
                "scenes": result.reused_scene_ids,
                "props": result.reused_prop_ids,
            },
        }

    # === BATCH KEYFRAME GENERATION (vendor-aware grid) ===

    def batch_generate_frame_keyframes(
        self,
        script_id: str,
        frame_ids: List[str],
        *,
        mode: str = "first_frame",
        force_per_shot: bool = False,
    ) -> Dict[str, Any]:
        """Render keyframes for multiple frames in one batch.

        On a grid-capable T2I vendor (currently only Seedream via the
        ``doubao`` vendor id), one composite image is requested and
        split into ``N`` tiles — each frame gets one tile written to
        its ``rendered_image_asset``. On other vendors, falls back to
        per-shot parallel calls through ``StoryboardGenerator``.

        Use ``force_per_shot=True`` to bypass the grid path even when
        the vendor supports it (e.g. when iterating on a single frame
        and you want fine control). Result shape::

            {"method": "grid"|"per_shot",
             "vendor": "doubao"|...,
             "assignments": {frame_id: [tile_url_rel, ...]}}
        """
        from .keyframes import get_grid_capability, get_mode

        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        if not frame_ids:
            raise ValueError("frame_ids must not be empty")

        id_to_frame = {f.id: f for f in script.frames}
        missing = [fid for fid in frame_ids if fid not in id_to_frame]
        if missing:
            raise ValueError(f"Frames not found: {missing}")
        frames = [id_to_frame[fid] for fid in frame_ids]

        strategy = get_mode(mode)
        total_panels = len(frames) * strategy.panels_per_frame()

        with scoped_instance(
            script.model_settings.t2i_instance_id, InstanceType.T2I
        ) as t2i_inst:
            vendor_id = t2i_inst.vendor_id if t2i_inst else None
            capability = get_grid_capability(vendor_id)
            use_grid = (
                capability.supports_native_grid
                and not force_per_shot
                and total_panels > 1
                and total_panels <= capability.max_panels
            )

            if use_grid:
                method = "grid"
                assignments = self._render_keyframes_grid(
                    script, frames, strategy, capability,
                )
            else:
                method = "per_shot"
                assignments = self._render_keyframes_per_shot(script, frames)

        script.updated_at = time.time()
        self._save_data()

        return {
            "method": method,
            "vendor": vendor_id,
            "assignments": assignments,
        }

    def _render_keyframes_grid(
        self,
        script: Script,
        frames: List[StoryboardFrame],
        strategy: Any,
        capability: Any,
    ) -> Dict[str, List[str]]:
        """Native-grid path: one composite call → split → distribute tiles.

        The vendor adapter is reached via :meth:`StoryboardGenerator._route_for_call`
        so the same instance/credential plumbing is reused; we just pass
        a composite prompt and a square output size. Per-panel storage
        slot (``main`` / ``end_frame`` / ``ref``) is decided by the
        strategy via ``tile_slot(panel_index)`` so this method stays
        mode-agnostic.
        """
        from .keyframes import pick_layout, split_grid_image

        layout = pick_layout(len(frames) * strategy.panels_per_frame())
        style_prompt = ""
        if script.art_direction and script.art_direction.style_config:
            style_prompt = script.art_direction.style_config.get("positive_prompt", "")
        composite_prompt = strategy.build_composite_prompt(frames, style_prompt)

        grid_dir = os.path.join(self.data_root, "storyboard", "grid")
        tile_dir = os.path.join(self.data_root, "storyboard", "tiles")
        os.makedirs(grid_dir, exist_ok=True)
        composite_id = str(uuid.uuid4())
        composite_path = os.path.join(grid_dir, f"{composite_id}.png")

        adapter = self.storyboard_generator._route_for_call()
        adapter.generate(
            composite_prompt,
            composite_path,
            size=capability.recommended_composite_size,
        )

        tile_paths = split_grid_image(
            composite_path, layout.rows, layout.cols, tile_dir, composite_id,
        )
        assignments_by_frame = strategy.tiles_to_assignments(tile_paths, frames)

        result: Dict[str, List[str]] = {}
        for frame in frames:
            tile_list = assignments_by_frame.get(frame.id, [])
            rel_urls: List[str] = []
            for panel_idx, tile_path in enumerate(tile_list):
                slot = strategy.tile_slot(panel_idx)
                rel_url = os.path.relpath(tile_path, self.data_root)
                self._write_keyframe_tile_to_slot(
                    frame, slot, rel_url, composite_prompt,
                )
                rel_urls.append(rel_url)
            if rel_urls:
                frame.updated_at = time.time()
                frame.status = GenerationStatus.COMPLETED
            result[frame.id] = rel_urls
        return result

    @staticmethod
    def _write_keyframe_tile_to_slot(
        frame: StoryboardFrame,
        slot: str,
        tile_url: str,
        prompt: str,
    ) -> None:
        """Append a new ImageVariant to the slot's asset container.

        Slot map (kept here, not in modes, so future schema changes
        only touch one place):

        * ``main``      → ``rendered_image_asset`` (canonical seed,
          also syncs the legacy ``rendered_image_url`` / ``image_url``).
        * ``end_frame`` → ``end_frame_asset`` (closing keyframe for
          first+last i2v).
        * ``ref``       → ``rendered_image_asset.variants`` pool
          (alternative angles; user picks one as the seed).
        """
        from .keyframes import SLOT_END_FRAME, SLOT_MAIN, SLOT_REF

        if slot == SLOT_END_FRAME:
            if not frame.end_frame_asset:
                frame.end_frame_asset = ImageAsset()
            target = frame.end_frame_asset
        else:
            # main and ref share rendered_image_asset
            if not frame.rendered_image_asset:
                frame.rendered_image_asset = ImageAsset()
            target = frame.rendered_image_asset

        variant_id = str(uuid.uuid4())
        target.variants.append(
            ImageVariant(
                id=variant_id,
                url=tile_url,
                prompt_used=prompt,
                created_at=time.time(),
            )
        )
        target.selected_id = variant_id

        # Legacy URL fields track the canonical seed only.
        if slot == SLOT_MAIN:
            frame.rendered_image_url = tile_url
            frame.image_url = tile_url

    def _render_keyframes_per_shot(
        self,
        script: Script,
        frames: List[StoryboardFrame],
    ) -> Dict[str, List[str]]:
        """Fallback path: call StoryboardGenerator.generate_frame per shot.

        Loses the cross-shot consistency the grid path provides, but
        works on every vendor. Errors on a single frame don't abort
        the batch — that frame just maps to an empty list.
        """
        result: Dict[str, List[str]] = {}
        for frame in frames:
            scene = next((s for s in script.scenes if s.id == frame.scene_id), None)
            try:
                self.storyboard_generator.generate_frame(
                    frame, script.characters, scene,
                )
            except Exception as e:
                logger.error(f"per-shot keyframe render failed for {frame.id}: {e}")
                result[frame.id] = []
                continue
            variant = None
            if frame.rendered_image_asset and frame.rendered_image_asset.selected_id:
                variant = next(
                    (v for v in frame.rendered_image_asset.variants
                     if v.id == frame.rendered_image_asset.selected_id),
                    None,
                )
            result[frame.id] = [variant.url] if variant else []
        return result

    # === VIDEO PROMPT TIMELINE SLICING ===

    def slice_frame_video_timeline(
        self,
        script_id: str,
        frame_id: str,
        *,
        segment_seconds: int = 3,
        duration_override: Optional[int] = None,
    ) -> StoryboardFrame:
        """Generate ``video_prompt_timeline`` for a single frame.

        Pulls the duration from the frame's selected ``VideoTask`` when
        available, otherwise falls back to ``duration_override``
        (default 5s, matching ``VideoTask.duration``). Persists the
        sliced result onto ``frame.video_prompt_timeline`` and returns
        the updated frame.

        The slicer is opt-in per frame — calling it again overwrites
        the previous timeline but never mutates ``frame.video_prompt``.
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        frame = next((f for f in script.frames if f.id == frame_id), None)
        if frame is None:
            raise ValueError(f"Frame {frame_id} not found")
        if not frame.video_prompt:
            raise ValueError(f"Frame {frame_id} has no video_prompt to slice")

        # Resolve duration: selected video task → first video task → override → 5
        duration = duration_override
        if duration is None:
            selected = next(
                (vt for vt in script.video_tasks
                 if vt.frame_id == frame_id and (frame.selected_video_id is None or vt.id == frame.selected_video_id)),
                None,
            )
            duration = (selected.duration if selected else None) or 5

        scene = next((s for s in script.scenes if s.id == frame.scene_id), None)
        characters = [
            c.name for c in script.characters if c.id in frame.character_ids
        ]

        with scoped_instance(script.model_settings.llm_instance_id, InstanceType.LLM):
            sliced = self.script_processor.slice_video_prompt_timeline(
                frame.video_prompt,
                duration,
                segment_seconds=segment_seconds,
                location=scene.name if scene else None,
                characters=characters or None,
                sound=frame.sfx_prompt,
            )

        frame.video_prompt_timeline = sliced
        frame.updated_at = time.time()
        script.updated_at = time.time()
        self._save_data()
        return frame

    # === AUTO VOICE ASSIGNMENT ===

    def auto_assign_voices(self, script_id: str) -> Dict[str, Any]:
        """Run the voice-assignment chain over a script's characters.

        Returns ``{"assignments": {char_id: voice_id}, "skipped": [char_id, ...]}``
        — ``skipped`` covers characters where every rule returned None
        (typically because the character is locked with no prior voice).

        Manual overrides win: characters with an existing ``voice_id``
        are kept as-is by :class:`LockedRule` at the head of the chain.
        Newly-assigned ``voice_name`` is looked up from the available
        catalog so the Settings UI shows the human-readable label
        without an extra round-trip.
        """
        from .voice import (
            AssignContext,
            DefaultPoolRule,
            LLMMatchRule,
            LockedRule,
            SeriesReuseRule,
            VoiceAssigner,
        )

        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        series = self.series_store.get(script.series_id) if script.series_id else None
        available = self.audio_generator.get_available_voices()
        id_to_name = {v.get("id"): v.get("name") for v in available}

        chain = [
            LockedRule(),
            SeriesReuseRule(),
            LLMMatchRule(self.script_processor),
            DefaultPoolRule(),
        ]
        assigner = VoiceAssigner(chain)

        # The LLMMatchRule talks to the LLM — bind the project's instance
        # so credentials / model_name come from the right ModelInstance.
        with scoped_instance(script.model_settings.llm_instance_id, InstanceType.LLM):
            mapping = assigner.assign_all(
                script.characters, available, AssignContext(series=series)
            )

        skipped: List[str] = []
        for ch in script.characters:
            vid = mapping.get(ch.id)
            if vid is None:
                skipped.append(ch.id)
                continue
            if ch.voice_id == vid:
                continue  # nothing to write back
            ch.voice_id = vid
            name = id_to_name.get(vid)
            if name:
                ch.voice_name = name

        script.updated_at = time.time()
        self._save_data()
        # SeriesReuseRule may have read voice_id off series.characters, but
        # we don't write back to series here — series-level voice mgmt is
        # a separate concern handled by the series character endpoints.

        return {"assignments": mapping, "skipped": skipped}

    # === STORYBOARD DRAMATIZATION v2 ===

    def analyze_text_to_frames(self, script_id: str, text: str) -> Script:
        """
        Analyzes script text and generates storyboard frames using LLM.
        Replaces existing frames with newly generated ones.
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        logger.info(f"Analyzing text to frames for project {script_id}")

        # Resolve assets (merge Series + Episode if applicable)
        resolved = self.resolve_episode_assets(script)
        all_characters = resolved["characters"]
        all_scenes = resolved["scenes"]
        all_props = resolved["props"]

        # Build entities JSON from resolved characters, scenes, props
        entities_json = {
            "characters": [{"id": c.id, "name": c.name, "description": c.description} for c in all_characters],
            "scenes": [{"id": s.id, "name": s.name, "description": s.description} for s in all_scenes],
            "props": [{"id": p.id, "name": p.name, "description": p.description} for p in all_props],
        }

        # Call LLM to analyze text (bound to project's LLM instance).
        with scoped_instance(script.model_settings.llm_instance_id, InstanceType.LLM):
            raw_frames = self.script_processor.analyze_to_storyboard(text, entities_json)

        if not raw_frames:
            raise RuntimeError(_t("errors.storyboard_no_frames"))

        # Convert raw frame dicts to StoryboardFrame objects
        new_frames = []
        for idx, frame_data in enumerate(raw_frames):
            # Resolve scene ID by name
            scene_ref_name = frame_data.get("scene_ref_name", "")
            scene_id = None
            for scene in all_scenes:
                if scene.name == scene_ref_name or scene_ref_name in scene.name:
                    scene_id = scene.id
                    break
            if not scene_id and all_scenes:
                scene_id = all_scenes[0].id  # Fallback to first scene
            elif not scene_id:
                scene_id = str(uuid.uuid4())  # Generate a placeholder ID

            # Resolve character IDs by names
            char_ref_names = frame_data.get("character_ref_names", [])
            character_ids = []
            for char_name in char_ref_names:
                for char in all_characters:
                    if char.name == char_name or char_name in char.name:
                        character_ids.append(char.id)
                        break

            # Resolve prop IDs by names
            prop_ref_names = frame_data.get("prop_ref_names", [])
            prop_ids = []
            for prop_name in prop_ref_names:
                for prop in all_props:
                    if prop.name == prop_name or prop_name in prop.name:
                        prop_ids.append(prop.id)
                        break
            
            # Defensive parse for duration_seconds — LLMs occasionally return
            # a string ("4") or a float ("4.0"); clamp to the model's range.
            raw_dur = frame_data.get("duration_seconds")
            duration_seconds: Optional[int] = None
            if raw_dur is not None:
                try:
                    duration_seconds = max(1, min(60, int(float(raw_dur))))
                except (TypeError, ValueError):
                    duration_seconds = None

            frame = StoryboardFrame(
                id=str(uuid.uuid4()),
                # Display metadata (rule 6 of the builder's visual-atoms block)
                title=frame_data.get("title") or None,
                duration_seconds=duration_seconds,
                scene_id=scene_id,
                character_ids=character_ids,
                prop_ids=prop_ids,
                # Action description - now a unified field combining character acting and physics
                action_description=frame_data.get("action_description", ""),
                # Visual atmosphere
                visual_atmosphere=frame_data.get("visual_atmosphere"),
                # Camera parameters
                shot_size=frame_data.get("shot_size"),
                camera_angle=frame_data.get("camera_angle", "平视"),
                camera_movement=frame_data.get("camera_movement"),
                # Dialogue
                dialogue=frame_data.get("dialogue"),
                speaker=frame_data.get("speaker"),
                # Audio prompts (populated by builder.with_audio_atoms; may be
                # absent on older mock fixtures, hence .get with default None)
                bgm_prompt=frame_data.get("bgm_prompt"),
                sfx_prompt=frame_data.get("sfx_prompt"),
                # Status
                status=GenerationStatus.PENDING
            )
            new_frames.append(frame)
        
        # Replace existing frames with new ones
        script.frames = new_frames
        script.updated_at = time.time()
        
        logger.info(f"Generated {len(new_frames)} frames from text analysis")
        self._save_data()
        return script

    def create_storyboard_analysis_task(self, script_id: str, text: str) -> Tuple[Script, str]:
        """Creates an async storyboard analysis task and returns (script, task_id) immediately.

        The actual LLM call (which can take 1–3 minutes for long scripts) runs in a
        background worker; the client polls ``/tasks/{task_id}`` for completion. This
        avoids 5xx timeouts at upstream gateways (Cloudflare hard-limits responses to
        ~100s on free/pro plans) when scripts are long.
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        # ``add_background_task`` copies ContextVars into the worker, so the
        # per-user RequestContext is preserved; ``analyze_text_to_frames``
        # then re-enters its own ``scoped_instance`` from ``script.model_settings``.
        task_id = str(uuid.uuid4())
        self.asset_generation_tasks[task_id] = {
            "task_type": "storyboard_analysis",
            "status": "pending",
            "progress": 0,
            "error": None,
            "script_id": script_id,
            "asset_id": None,
            "asset_type": None,
            "created_at": time.time(),
            "params": {"text": text},
        }
        return script, task_id

    def process_storyboard_analysis_task(self, task_id: str):
        """Processes a storyboard analysis task in the background."""
        task = self.asset_generation_tasks.get(task_id)
        if not task:
            logger.error(f"Storyboard analysis task {task_id} not found")
            return

        task["status"] = "processing"
        try:
            self.analyze_text_to_frames(task["script_id"], task["params"]["text"])
            task["status"] = "completed"
            task["progress"] = 100
            logger.info(f"Storyboard analysis task {task_id} completed successfully")
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            logger.error(f"Storyboard analysis task {task_id} failed: {e}", exc_info=True)

    def refine_frame_prompt(self, script_id: str, frame_id: str, raw_prompt: str, assets: List[Dict[str, Any]], feedback: str = "") -> Dict[str, Any]:
        """
        Refines a raw prompt into bilingual (CN/EN) prompts using LLM.
        Also updates the frame with the refined prompts.
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        logger.debug(f"Refining prompt for frame {frame_id}")

        # Read custom prompt config with 3-level fallback (Episode → Series → default)
        series = self.series_store.get(script.series_id) if script.series_id else None
        custom_prompt = self.get_effective_prompt("storyboard_polish", script, series)
        # If it's the system default, pass empty so the LLM method uses its built-in default
        from .llm import DEFAULT_STORYBOARD_POLISH_PROMPT
        if custom_prompt == DEFAULT_STORYBOARD_POLISH_PROMPT:
            custom_prompt = ""

        # Call LLM to refine prompt (bound to project's LLM instance).
        with scoped_instance(script.model_settings.llm_instance_id, InstanceType.LLM):
            result = self.script_processor.polish_storyboard_prompt(raw_prompt, assets, feedback, custom_prompt)
        
        # Find and update the frame
        frame_found = False
        for frame in script.frames:
            if frame.id == frame_id:
                frame.image_prompt_cn = result.get("prompt_cn")
                frame.image_prompt_en = result.get("prompt_en")
                frame.image_prompt = result.get("prompt_en")  # Also update legacy field
                frame.updated_at = time.time()
                frame_found = True
                break
        
        if frame_found:
            self._save_data()
        
        return {
            "prompt_cn": result.get("prompt_cn"),
            "prompt_en": result.get("prompt_en"),
            "frame_updated": frame_found
        }

    def generate_storyboard(self, script_id: str) -> Script:
        """Step 3: Generate storyboard images (Initial/Batch).

        Synchronous full-batch render kept for back-compat. New code should
        use ``create_storyboard_batch_render_task`` (async, skips frames the
        user already has, uses each frame's tuned prompt + selected refs).
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        # Bind I2I instance so per-frame image-conditioned generation picks up
        # the user's credentials + model_name (parity with the async batch path
        # at create_storyboard_batch_render_task).
        i2i_instance_id = getattr(getattr(script, "model_settings", None), "i2i_instance_id", None)
        with scoped_instance(i2i_instance_id, InstanceType.I2I):
            script = self.storyboard_generator.generate_storyboard(script)
        self._save_data()
        return script

    # === Storyboard batch render (async) ==================================
    #
    # Design notes
    # ------------
    # We deliberately split *eligibility* (``_should_render_frame``) from
    # *execution* (``_render_single_frame``) so the rule for skipping
    # already-rendered / locked frames can be unit-tested in isolation
    # and reused by future endpoints (e.g. "render selected frames"). The
    # batch worker iterates frames, applies the predicate, and on each
    # success/failure updates the task dict so the frontend's existing
    # ``/tasks/{id}`` poll surfaces granular progress without WebSockets.

    @staticmethod
    def _should_render_frame(frame: StoryboardFrame, force: bool = False) -> bool:
        """Predicate: should the batch worker render this frame?

        Locked frames are *never* rendered (the lock is the user's "don't
        touch" signal). When ``force=False`` we additionally skip frames
        that already have an output image.
        """
        if frame.locked:
            return False
        if force:
            return True
        has_image = bool(getattr(frame, "rendered_image_url", None) or frame.image_url)
        return not has_image

    def _render_single_frame(self, script: Script, frame: StoryboardFrame) -> None:
        """Render one frame using the prompts + composition stored on the
        frame itself.

        This is the batch path's per-frame work. It mirrors what
        :meth:`generate_storyboard_render` does for the single-frame
        endpoint, but pulls inputs from frame state instead of HTTP body
        — so frames the user has already tuned (refined prompts, selected
        composition refs) get rendered with their tuned values rather
        than auto-defaults.
        """
        # Prompt priority: refined EN > legacy image_prompt > auto-built fallback.
        # The model expects English; CN is for user display only.
        prompt = frame.image_prompt_en or frame.image_prompt or ""

        # Reference paths: prefer composition_data (canvas-selected refs)
        # if the user has touched composition; else pass empty so
        # ``StoryboardGenerator.generate_frame`` auto-collects from the
        # frame's character_ids + scene_id.
        ref_image_paths: List[str] = []
        composition = frame.composition_data or {}
        ref_urls: List[str] = list(composition.get("reference_image_urls") or [])
        single = composition.get("reference_image_url")
        if single and single not in ref_urls:
            ref_urls.insert(0, single)

        for url in ref_urls:
            if not url:
                continue
            if is_object_key(url) or url.startswith("http"):
                ref_image_paths.append(url)
            else:
                potential = _safe_resolve_path(self.data_root, url)
                if os.path.exists(potential):
                    ref_image_paths.append(potential)

        scene = next((s for s in script.scenes if s.id == frame.scene_id), None)

        from .assets import ASPECT_RATIO_TO_SIZE
        aspect = script.model_settings.storyboard_aspect_ratio
        effective_size = ASPECT_RATIO_TO_SIZE.get(aspect, "1024*576")

        with scoped_instance(script.model_settings.i2i_instance_id, InstanceType.I2I) as i2i_inst:
            i2i_model = require_instance(i2i_inst, InstanceType.I2I).model_name
            self.storyboard_generator.generate_frame(
                frame,
                script.characters,
                scene,
                ref_image_path=ref_image_paths[0] if ref_image_paths else None,
                ref_image_paths=ref_image_paths,
                prompt=prompt or None,
                batch_size=1,
                size=effective_size,
                model_name=i2i_model,
            )

    def create_storyboard_batch_render_task(
        self, script_id: str, force: bool = False
    ) -> Tuple[Script, str]:
        """Register an async batch-render task and return ``(script, task_id)``.

        The actual rendering runs in :meth:`process_storyboard_batch_render_task`
        on a BackgroundTask worker. The endpoint returns immediately so no
        upstream gateway can time the connection out.
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        pending_ids = [
            f.id for f in script.frames if self._should_render_frame(f, force=force)
        ]
        task_id = self._register_task(
            task_type="storyboard_batch_render",
            script_id=script_id,
            params={"force": force},
            total_count=len(script.frames),
            pending_count=len(pending_ids),
        )
        return script, task_id

    def process_storyboard_batch_render_task(self, task_id: str) -> None:
        """Worker for ``create_storyboard_batch_render_task``.

        Renders every eligible frame using each frame's own tuned
        ``image_prompt`` + ``composition_data``. Per-frame failures are
        isolated; partial progress is persisted after each frame so a
        crash mid-batch loses at most the in-flight frame.
        """
        with self._task_lifecycle(task_id) as task:
            script = self.scripts.get(task["script_id"])
            if not script:
                raise ValueError("Script not found")
            force = bool(task["params"].get("force", False))
            pending = [
                f for f in script.frames if self._should_render_frame(f, force=force)
            ]
            self._run_per_item_batch(
                task_id, pending, lambda f: self._render_single_frame(script, f)
            )

    def update_frame(self, script_id: str, frame_id: str, **kwargs) -> Script:
        """Update frame data (prompt, scene_id, character_ids, etc.)."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        frame = next((f for f in script.frames if f.id == frame_id), None)
        if not frame:
            raise ValueError(f"Frame {frame_id} not found")
        
        # Update only provided fields
        if kwargs.get('image_prompt') is not None:
            frame.image_prompt = kwargs['image_prompt']
        if kwargs.get('action_description') is not None:
            frame.action_description = kwargs['action_description']
        if kwargs.get('dialogue') is not None:
            frame.dialogue = kwargs['dialogue']
        if kwargs.get('camera_angle') is not None:
            frame.camera_angle = kwargs['camera_angle']
        if kwargs.get('scene_id') is not None:
            frame.scene_id = kwargs['scene_id']
        if kwargs.get('character_ids') is not None:
            frame.character_ids = kwargs['character_ids']
        # huobao-parity display metadata. Empty-string title → None so the
        # storyboard list shows the fallback (frame id snippet) instead of a
        # blank label. Pydantic's ``ge=1, le=60`` guards out-of-range values.
        if 'title' in kwargs:
            raw_title = kwargs.get('title')
            frame.title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else None
        if 'duration_seconds' in kwargs:
            frame.duration_seconds = kwargs.get('duration_seconds')

        self._save_data()
        return script

    def add_frame(self, script_id: str, scene_id: str = None, action_description: str = "", camera_angle: str = "medium_shot", insert_at: int = None) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        new_frame = StoryboardFrame(
            id=f"frame_{uuid.uuid4().hex[:8]}",
            scene_id=scene_id or (script.scenes[0].id if script.scenes else ""),
            character_ids=[],
            action_description=action_description,
            camera_angle=camera_angle
        )
        
        if insert_at is not None and 0 <= insert_at <= len(script.frames):
            script.frames.insert(insert_at, new_frame)
        else:
            script.frames.append(new_frame)
            
        self._save_data()
        return script

    def copy_frame(self, script_id: str, frame_id: str, insert_at: int = None) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        original_frame = next((f for f in script.frames if f.id == frame_id), None)
        if not original_frame:
            raise ValueError(f"Frame {frame_id} not found")
            
        # Create a deep copy with new ID
        new_frame = original_frame.copy()
        new_frame.id = f"frame_{uuid.uuid4().hex[:8]}"
        new_frame.updated_at = time.time()
        # Reset generation status and URLs for the copy? 
        # Usually copy implies copying content, but maybe we want to keep the image?
        # Let's keep the image/content but reset status if it was processing?
        # Actually, if we copy, we probably want the same image reference initially.
        # But we should reset the "locked" status maybe?
        new_frame.locked = False
        
        if insert_at is not None and 0 <= insert_at <= len(script.frames):
            script.frames.insert(insert_at, new_frame)
        else:
            # Insert after the original frame by default
            try:
                original_index = script.frames.index(original_frame)
                script.frames.insert(original_index + 1, new_frame)
            except ValueError:
                script.frames.append(new_frame)
                
        self._save_data()
        return script

    def delete_frame(self, script_id: str, frame_id: str) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        script.frames = [f for f in script.frames if f.id != frame_id]
        self._save_data()
        return script

    def reorder_frames(self, script_id: str, frame_ids: List[str]) -> Script:
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        frame_map = {f.id: f for f in script.frames}
        new_frames = []
        for fid in frame_ids:
            if fid in frame_map:
                new_frames.append(frame_map[fid])
        
        script.frames = new_frames
        self._save_data()
        return script

    def generate_motion_ref(
        self,
        script_id: str,
        asset_id: str,
        asset_type: str,  # 'full_body' | 'head_shot' for characters; 'scene' | 'prop' for scenes and props
        prompt: Optional[str] = None,
        audio_url: Optional[str] = None,
        duration: int = 5,
        batch_size: int = 1
    ) -> Script:
        """Generate Motion Reference video for an asset (Character Full Body/Headshot, Scene, or Prop).

        Args:
            script_id: ID of the project/script
            asset_id: ID of the asset (character, scene, or prop)
            asset_type: 'full_body' | 'head_shot' for characters; 'scene' or 'prop' for scenes and props
            prompt: Custom prompt for motion generation
            audio_url: URL of driving audio for lip-sync
            duration: Video duration in seconds (5 or 10)
            batch_size: Number of videos to generate
        """
        from .models import VideoVariant, AssetUnit, VideoTask

        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        # Find the target asset based on type
        target_asset = None
        asset_display_name = ""

        if asset_type in ["full_body", "head_shot"]:
            # Find the character
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
            asset_display_name = "Character"
        elif asset_type == "scene":
            # Find the scene
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
            asset_display_name = "Scene"
        elif asset_type == "prop":
            # Find the prop
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            asset_display_name = "Prop"
        else:
            raise ValueError(f"Invalid asset_type: {asset_type}. Must be 'full_body', 'head_shot', 'scene', or 'prop'")

        if not target_asset:
            raise ValueError(f"{asset_display_name} {asset_id} not found")

        # Get the appropriate AssetUnit or image URL based on the asset type
        asset_unit = None  # For characters with AssetUnit
        generated_videos = []  # Store generated videos

        if asset_type in ["full_body", "head_shot"]:
            # Handle character asset
            asset_unit = getattr(target_asset, asset_type, None)
            # Get source image from the AssetUnit or legacy field
            if asset_unit and asset_unit.selected_image_id:
                source_img = next(
                    (v for v in asset_unit.image_variants if v.id == asset_unit.selected_image_id),
                    None
                )
                source_image_url = source_img.url if source_img else (
                    target_asset.full_body_image_url if asset_type == "full_body" else target_asset.headshot_image_url
                )
            else:
                source_image_url = (
                    target_asset.full_body_image_url if asset_type == "full_body"
                    else target_asset.headshot_image_url
                )

            # Default prompt for character
            if not prompt:
                if audio_url:
                    prompt = f"{asset_type.replace('_', ' ').title()} character reference video. {target_asset.description}. The character is speaking naturally matching the audio, with accurate lip-sync and facial expressions. Stable camera, high quality, 4k."
                else:
                    prompt = f"{asset_type.replace('_', ' ').title()} character reference video. {target_asset.description}. Looking around, breathing, slight movement, subtle gestures. Stable camera, high quality, 4k."
        else:
            # Handle scene or prop assets
            source_image_url = target_asset.image_url
            # Default prompt for scene and prop
            if not prompt:
                if asset_type == "scene":
                    if audio_url:
                        prompt = f"Cinematic scene video reference of {target_asset.name}. {target_asset.description}. Ambient motion, lighting changes, natural elements moving, birds, clouds. Soundscape matching the audio. High quality, 4k."
                    else:
                        prompt = f"Cinematic scene video reference of {target_asset.name}. {target_asset.description}. Ambient motion, lighting changes, natural elements moving, birds, clouds. Slow pan across the scene. High quality, 4k."
                else:  # prop
                    if audio_url:
                        prompt = f"Cinematic prop video reference of {target_asset.name}. {target_asset.description}. Rotating object, detailed textures visible, ambient motion, subtle movements matching audio. High quality, 4k."
                    else:
                        prompt = f"Cinematic prop video reference of {target_asset.name}. {target_asset.description}. Rotating object, detailed textures visible, ambient motion, subtle movements. High quality, 4k."

        # Check if source image exists
        if not source_image_url:
            raise ValueError(f"No source image available for {asset_type}. Please generate a static image first.")

        # Bind the user's I2V ModelInstance for the duration of this call so
        # ``WanxModel.api_key`` resolves from the instance's encrypted creds and
        # the model name comes from the instance (not the wanx hardcoded default).
        # Without this scope, Motion Reference would silently fall back to env
        # vars + ``wan2.6-i2v`` regardless of what the user picked in Settings.
        i2v_instance_id = getattr(getattr(script, "model_settings", None), "i2v_instance_id", None)
        with scoped_instance(i2v_instance_id, InstanceType.I2V) as i2v_inst:
            resolved_model_name = require_instance(i2v_inst, InstanceType.I2V).model_name

            # Generate videos based on the asset type
            for i in range(batch_size):
                try:
                    # Call video generator (I2V)
                    video_result = self.video_generator.generate_i2v(
                        image_url=source_image_url,
                        prompt=prompt,
                        duration=duration,
                        audio_url=audio_url,
                        model_name=resolved_model_name,
                    )

                    if video_result and video_result.get("video_url"):
                        if asset_type in ["full_body", "head_shot"]:
                            # For characters, create VideoVariant in AssetUnit
                            video_variant = VideoVariant(
                                id=f"video_{uuid.uuid4().hex[:8]}",
                                url=video_result["video_url"],
                                prompt_used=prompt,
                                audio_url=audio_url,
                                source_image_id=None  # Don't set this to avoid complications
                            )
                            asset_unit.video_variants.append(video_variant)

                            # Auto-select the first generated video
                            if not asset_unit.selected_video_id:
                                asset_unit.selected_video_id = video_variant.id

                            generated_videos.append(video_variant)
                            logger.info(f"Generated motion ref video: {video_variant.id}")
                        else:
                            # For scenes and props, create VideoTask and add to asset's video_assets
                            video_task = VideoTask(
                                id=f"video_{uuid.uuid4().hex[:8]}",
                                project_id=script_id,
                                asset_id=asset_id,
                                image_url=source_image_url,
                                prompt=prompt,
                                status="completed",  # Since generation is done in this step
                                video_url=video_result["video_url"],
                                duration=duration,
                                created_at=time.time(),
                                generate_audio=bool(audio_url),
                                model=resolved_model_name,
                                generation_mode="i2v"  # Image to video (motion reference)
                            )

                            # Add to the asset's video_assets
                            target_asset.video_assets.append(video_task)
                            generated_videos.append(video_task)
                            logger.info(f"Generated motion ref video for {asset_type}: {video_task.id}")
                except Exception as e:
                    logger.error(f"Failed to generate motion ref video for {asset_type}: {e}")

        # For character assets, update the AssetUnit
        if asset_type in ["full_body", "head_shot"]:
            # Ensure AssetUnit exists
            if asset_unit is None:
                asset_unit = AssetUnit()
                setattr(target_asset, asset_type, asset_unit)

            asset_unit.video_prompt = prompt
            asset_unit.video_updated_at = time.time()
        # For scene and prop assets, the video tasks are already added in the generation loop above

        if batch_size > 0 and not generated_videos:
            raise RuntimeError(f"Failed to generate any motion reference videos for {asset_type}")

        self._save_data()
        return script

    def generate_storyboard_render(self, script_id: str, frame_id: str, composition_data: Optional[Dict[str, Any]], prompt: str, batch_size: int = 1) -> Script:
        """Step 3b: Render a specific frame from composition data."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        frame = next((f for f in script.frames if f.id == frame_id), None)
        if not frame:
            raise ValueError(f"Frame {frame_id} not found")
            
        frame.status = GenerationStatus.PROCESSING
        if composition_data:
            frame.composition_data = composition_data
        frame.image_prompt = prompt
        self._save_data()
        
        try:
            # Extract reference image URL from composition data if available
            ref_image_url = None
            ref_image_urls = []
            
            if composition_data:
                ref_image_url = composition_data.get('reference_image_url')
                ref_image_urls = composition_data.get('reference_image_urls', [])
            
            ref_image_paths = []
            
            # Resolve multiple paths
            for url in ref_image_urls:
                if not url:
                    continue
                if is_object_key(url) or url.startswith("http"):
                    ref_image_paths.append(url)
                else:
                    potential_path = _safe_resolve_path(self.data_root, url)
                    if os.path.exists(potential_path):
                        ref_image_paths.append(potential_path)
            
            # Also handle single path if provided (legacy support)
            if ref_image_url and ref_image_url not in ref_image_urls:
                if is_object_key(ref_image_url) or ref_image_url.startswith("http"):
                    if ref_image_url not in ref_image_paths:
                        ref_image_paths.append(ref_image_url)
                else:
                    potential_path = _safe_resolve_path(self.data_root, ref_image_url)
                    if os.path.exists(potential_path):
                        if potential_path not in ref_image_paths:
                            ref_image_paths.append(potential_path)
            
            # Use the first path as ref_image_path for legacy generator support if needed
            ref_image_path = ref_image_paths[0] if ref_image_paths else None
            
            # Use the prompt as-is from frontend (already contains style)
            final_prompt = prompt
            
            # Update frame with final prompt
            frame.image_prompt = final_prompt
            
            # Find scene for this frame
            scene = next((s for s in script.scenes if s.id == frame.scene_id), None)

            # Get effective size from storyboard_aspect_ratio
            from .assets import ASPECT_RATIO_TO_SIZE
            storyboard_aspect_ratio = script.model_settings.storyboard_aspect_ratio
            effective_size = ASPECT_RATIO_TO_SIZE.get(storyboard_aspect_ratio, "1024*576")  # Default to landscape
            
            # Resolve I2I instance and bind it for this call.
            with scoped_instance(script.model_settings.i2i_instance_id, InstanceType.I2I) as i2i_inst:
                i2i_model = require_instance(i2i_inst, InstanceType.I2I).model_name
                logger.info(f"Rendering frame {frame_id} using model {i2i_model} with {len(ref_image_paths)} reference images")
                if len(ref_image_urls) > 0:
                    logger.debug(f"Original reference URLs from frontend: {ref_image_urls}")

                self.storyboard_generator.generate_frame(
                    frame,
                    script.characters,
                    scene,
                    ref_image_path=ref_image_path,
                    ref_image_paths=ref_image_paths,
                    prompt=final_prompt,
                    batch_size=batch_size,
                    size=effective_size,
                    model_name=i2i_model
                )
            
            self._save_data()
            return script
        except Exception as e:
            frame.status = GenerationStatus.FAILED
            self._save_data()
            raise e
            # 1. Take the composition_data (positions of assets)
            # 2. Construct a composite image (ControlNet input)
            # 3. Call Img2Img with the composite + prompt
            
            logger.debug(f"Rendering frame {frame_id} with prompt: {prompt}")
            time.sleep(1.5) # Simulate processing
            
            # Mock Result
            mock_url = f"https://placehold.co/1280x720/2a2a2a/FFF?text=Rendered+Frame+{frame_id}"
            frame.rendered_image_url = mock_url
            frame.image_url = mock_url # Update main image too
            frame.status = GenerationStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Frame rendering failed: {e}")
            frame.status = GenerationStatus.FAILED
            
        self._save_data()
        return script

    def generate_video(self, script_id: str) -> Script:
        """Step 4: Generate video clips."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        script = self.video_generator.generate_video(script)
        self._save_data()
        return script

    def create_video_task(self, script_id: str, image_url: str, prompt: str, duration: int = 5, seed: int = None, resolution: str = "720p", generate_audio: bool = False, audio_url: str = None, prompt_extend: bool = True, negative_prompt: str = None, model: Optional[str] = None, frame_id: str = None, shot_type: str = "single", generation_mode: str = "i2v", reference_video_urls: list = None, mode: str = None, sound: str = None, cfg_scale: float = None, vidu_audio: bool = None, movement_amplitude: str = None, i2v_instance_id: Optional[str] = None) -> Tuple[Script, str]:
        """Creates a new video generation task."""
        script = self.get_script(script_id)
        if not script:
            raise ValueError("Script not found")
        
        task_id = str(uuid.uuid4())

        # Resolve `model` from the user-configured I2V ModelInstance when the
        # caller didn't supply one. Override chain mirrors process_video_task:
        # per-task pick → script-level setting → user default. Without this,
        # the batch path (_render_single_video, which never passes `model`)
        # would feed `None` into VideoTask and trip a Pydantic str-required
        # validation error before the task ever reaches process_video_task.
        if not model:
            script_instance_id = getattr(
                getattr(script, "model_settings", None), "i2v_instance_id", None
            )
            resolved_instance_id = i2v_instance_id or script_instance_id
            with scoped_instance(resolved_instance_id, InstanceType.I2V) as i2v_inst:
                model = require_instance(i2v_inst, InstanceType.I2V).model_name

        # If R2V mode is selected, swap the I2V model id for its R2V sibling
        # via the provider registry (e.g. wan2.7-i2v → wan2.7-r2v). Each
        # family declares the I2V/R2V SKU pair; the swap is data-driven, not
        # a hardcoded mapping. Without this, R2V mode would still ship the
        # I2V SKU and the request shape mismatch breaks the call. If the
        # current model has no R2V sibling registered, leave it as-is and
        # let the dispatcher reject it loudly rather than silently picking
        # an unrelated default.
        if generation_mode == "r2v":
            from ...utils.provider_registry import swap_provider_modality
            swapped = swap_provider_modality(model, "r2v")
            if swapped:
                model = swapped
        
        # Snapshot the input image to ensure consistency
        snapshot_url = image_url
        try:
            # Resolve source path
            if image_url and not image_url.startswith("http"):
                # Assume relative to output dir
                src_path = _safe_resolve_path(self.data_root, image_url)
                if os.path.exists(src_path) and os.path.isfile(src_path):
                    # Create snapshot dir
                    snapshot_dir = os.path.join(self.data_root, "video_inputs")
                    os.makedirs(snapshot_dir, exist_ok=True)

                    # Define snapshot path
                    ext = os.path.splitext(os.path.basename(image_url))[1] or ".png"
                    _validate_safe_id(task_id, "task_id")
                    snapshot_filename = f"{task_id}{ext}"
                    snapshot_path = _safe_resolve_path(snapshot_dir, snapshot_filename)
                    
                    # Copy file
                    import shutil
                    shutil.copy2(src_path, snapshot_path)
                    
                    # Update URL to relative path
                    snapshot_url = f"video_inputs/{snapshot_filename}"
        except Exception as e:
            logger.error(f"Failed to snapshot input image: {e}")
            # Fallback to original URL

        task = VideoTask(
            id=task_id,
            project_id=script_id,
            frame_id=frame_id,
            image_url=snapshot_url,
            prompt=prompt,
            status="pending",
            duration=duration,
            seed=seed,
            resolution=resolution,
            generate_audio=generate_audio,
            audio_url=audio_url,
            prompt_extend=prompt_extend,
            negative_prompt=negative_prompt,
            model=model,
            shot_type=shot_type,
            generation_mode=generation_mode,
            reference_video_urls=reference_video_urls or [],
            mode=mode,
            sound=sound,
            cfg_scale=cfg_scale,
            vidu_audio=vidu_audio,
            movement_amplitude=movement_amplitude,
            i2v_instance_id=i2v_instance_id,
            created_at=time.time()
        )
        
        if not script.video_tasks:
            script.video_tasks = []
        script.video_tasks.append(task)
        
        self._save_data()
        return script, task_id

    def extract_last_frame(self, script_id: str, frame_id: str, video_task_id: str) -> Script:
        """Extract the last frame from a video task and add it as a variant of the frame's rendered_image_asset."""
        from .models import ImageVariant, ImageAsset

        script = self.get_script(script_id)
        if not script:
            raise ValueError("Script not found")

        frame = next((f for f in script.frames if f.id == frame_id), None)
        if not frame:
            raise ValueError("Frame not found")

        # Find the video task
        video_task = next((t for t in script.video_tasks if t.id == video_task_id), None)
        if not video_task or video_task.status != "completed" or not video_task.video_url:
            raise ValueError("Video task not found or not completed")

        # Materialize video on local disk (handles local paths, OSS object
        # keys, and http URLs uniformly).
        import tempfile as _tempfile
        try:
            video_path = MediaResolver().to_local_file(video_task.video_url, suffix=".mp4")
        except Exception as e:
            raise ValueError(f"Video file not accessible: {e}")
        cleanup_video = video_path.startswith(_tempfile.gettempdir())

        # Extract last frame using FFmpeg
        ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg is required for frame extraction but was not found.")

        output_dir = os.path.join(self.data_root, "storyboard")
        os.makedirs(output_dir, exist_ok=True)
        _validate_safe_id(frame_id, "frame_id")
        output_filename = f"frame_{frame_id}_lastframe_{uuid.uuid4().hex[:8]}.jpg"
        output_path = _safe_resolve_path(output_dir, output_filename)

        cmd = [
            ffmpeg_path, "-sseof", "-0.1",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            "-y", output_path
        ]

        try:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg error: {result.stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError("FFmpeg frame extraction timed out")

            if not os.path.exists(output_path):
                raise RuntimeError("Failed to extract last frame from video")
        finally:
            if cleanup_video:
                try:
                    os.remove(video_path)
                except OSError:
                    pass

        # Upload to OSS if configured
        from ...utils.oss_utils import OSSImageUploader
        uploader = OSSImageUploader()
        oss_url = uploader.upload_image(output_path)
        image_url = oss_url if oss_url else os.path.relpath(output_path, self.data_root)

        # Create new variant
        variant = ImageVariant(
            id=str(uuid.uuid4()),
            url=image_url,
            prompt_used="Extracted last frame from video",
            is_uploaded_source=True,
            upload_type="image",
        )

        # Initialize rendered_image_asset if needed
        if not frame.rendered_image_asset:
            frame.rendered_image_asset = ImageAsset()

        frame.rendered_image_asset.variants.append(variant)
        frame.rendered_image_asset.selected_id = variant.id
        # Also update rendered_image_url so VideoCreator can pick it up
        frame.rendered_image_url = image_url

        script.updated_at = time.time()
        self._save_data()
        return script

    def upload_frame_image(self, script_id: str, frame_id: str, image_path: str) -> Script:
        """Upload an image as a variant of the frame's rendered_image_asset."""
        from .models import ImageVariant, ImageAsset

        # Validate that image_path is inside the output directory
        safe_path = _safe_resolve_path(self.data_root, os.path.relpath(image_path, self.data_root) if os.path.isabs(image_path) else image_path)

        script = self.get_script(script_id)
        if not script:
            raise ValueError("Script not found")

        frame = next((f for f in script.frames if f.id == frame_id), None)
        if not frame:
            raise ValueError("Frame not found")

        # Upload to OSS if configured
        from ...utils.oss_utils import OSSImageUploader
        uploader = OSSImageUploader()
        oss_url = uploader.upload_image(safe_path)
        image_url = oss_url if oss_url else os.path.relpath(safe_path, self.data_root)

        # Create new variant
        variant = ImageVariant(
            id=str(uuid.uuid4()),
            url=image_url,
            prompt_used="User uploaded image",
            is_uploaded_source=True,
            upload_type="image",
        )

        if not frame.rendered_image_asset:
            frame.rendered_image_asset = ImageAsset()

        frame.rendered_image_asset.variants.append(variant)
        frame.rendered_image_asset.selected_id = variant.id
        # Also update rendered_image_url so VideoCreator can pick it up
        frame.rendered_image_url = image_url

        script.updated_at = time.time()
        self._save_data()
        return script

    def select_video_for_frame(self, script_id: str, frame_id: str, video_id: str) -> Script:
        """Step 5a: Select a video variant for a frame."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        frame = next((f for f in script.frames if f.id == frame_id), None)
        if not frame:
            raise ValueError("Frame not found")
            
        # Verify video exists and belongs to project
        video = next((v for v in script.video_tasks if v.id == video_id), None)
        if not video:
            raise ValueError("Video task not found")
            
        frame.selected_video_id = video_id
        
        # Also update the frame's video_url to point to this video for easy access
        frame.video_url = video.video_url
        
        self._save_data()
        return script

    def merge_videos(self, script_id: str) -> Script:
        """Step 5b: Merge selected videos into a single file."""
        _validate_safe_id(script_id, "script_id")
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        logger.info(f"[MERGE] Starting video merge for script {script_id}")
        
        # Check if ffmpeg is available (prioritize bundled version)
        ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path:
            install_instructions = get_ffmpeg_install_instructions()
            error_msg = (
                "FFmpeg is required for video merging but was not found.\n\n"
                f"{install_instructions}\n\n"
                "After installation, restart the application."
            )
            logger.error(f"[MERGE] FFmpeg not found. {error_msg}")
            raise RuntimeError(error_msg)
        
        # Log ffmpeg version for debugging
        try:
            version_result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if version_result.returncode == 0:
                version_line = version_result.stdout.split('\n')[0] if version_result.stdout else "Unknown"
                logger.debug(f"[MERGE] Using FFmpeg: {version_line}")
                logger.debug(f"[MERGE] FFmpeg path: {ffmpeg_path}")
            else:
                logger.warning(f"[MERGE] Could not get FFmpeg version (exit code {version_result.returncode})")
        except Exception as e:
            logger.warning(f"[MERGE] Could not get FFmpeg version: {e}")
            
        # Collect video paths
        video_paths = []
        for i, frame in enumerate(script.frames):
            logger.info(f"[MERGE] Processing frame {i+1}/{len(script.frames)}: {frame.id}")
            
            if not frame.selected_video_id:
                # Try to find a default completed video
                default_video = next((v for v in script.video_tasks if v.frame_id == frame.id and v.status == "completed"), None)
                if default_video and default_video.video_url:
                    logger.debug(f"[MERGE]   -> Using default video: {default_video.video_url}")
                    video_paths.append(default_video.video_url)
                else:
                    logger.warning(f"[MERGE]   -> No video selected or available, skipping")
                continue
                
            video = next((v for v in script.video_tasks if v.id == frame.selected_video_id), None)
            if video and video.video_url:
                logger.debug(f"[MERGE]   -> Selected video: {video.video_url}")
                video_paths.append(video.video_url)
            else:
                logger.warning(f"[MERGE]   -> Selected video {frame.selected_video_id} not found or has no URL")
                
        if not video_paths:
            logger.error("[MERGE] No videos found to merge!")
            raise ValueError("No videos selected to merge. Please select videos for each frame first.")
        
        logger.info(f"[MERGE] Found {len(video_paths)} videos to merge")
            
        # Materialize each clip on local disk for ffmpeg's concat demuxer.
        # Local paths under output/ resolve to themselves; object keys and
        # remote URLs are downloaded to temp files we clean up at the end.
        import tempfile as _tempfile
        list_path = _safe_resolve_path(self.data_root, f"merge_list_{script_id}.txt")
        abs_video_paths = []
        resolver = MediaResolver()
        tmp_root = _tempfile.gettempdir()
        temp_files: List[str] = []

        with open(list_path, "w") as f:
            for path in video_paths:
                try:
                    abs_path = resolver.to_local_file(path, suffix=".mp4")
                except Exception as e:
                    logger.warning(f"[MERGE] Could not materialize video {path}: {e}")
                    continue
                if abs_path.startswith(tmp_root):
                    temp_files.append(abs_path)
                f.write(f"file '{abs_path}'\n")
                abs_video_paths.append(abs_path)
                logger.debug(f"[MERGE] Added to list: {abs_path}")
                        
        if not abs_video_paths:
            logger.error("[MERGE] No valid video files found on disk!")
            raise ValueError("No valid video files found. The video files may have been deleted or moved.")
        
        logger.info(f"[MERGE] Merge list created with {len(abs_video_paths)} videos")

        # Output path
        output_filename = f"merged_{script_id}_{int(time.time())}.mp4"
        output_path = _safe_resolve_path(os.path.join(self.data_root, "video"), output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.debug(f"[MERGE] Output path: {output_path}")
        
        # Log video file details for debugging
        for i, path in enumerate(abs_video_paths):
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.debug(f"[MERGE] Input video {i+1}: {os.path.basename(path)} ({size_mb:.2f} MB)")
            except Exception as e:
                logger.warning(f"[MERGE] Could not get size for video {i+1}: {e}")
        
        # Run ffmpeg
        # Use re-encoding for better compatibility (slower but more reliable)
        # -c:v libx264 -c:a aac ensures consistent output format
        cmd = [
            ffmpeg_path, "-y",  # Use the detected ffmpeg path
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",  # Re-encode video with H.264
            "-crf", "23",       # Quality (lower = better, 23 is default)
            "-preset", "fast",  # Encoding speed
            "-c:a", "aac",      # Re-encode audio with AAC
            "-b:a", "128k",     # Audio bitrate
            "-movflags", "+faststart",  # Web optimization
            output_path
        ]
        
        logger.debug(f"[MERGE] Running FFmpeg command: {' '.join(cmd)}")
        logger.debug(f"[MERGE] Platform: {platform.system()} {platform.release()}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=600)  # 10 min timeout for re-encoding
            logger.debug(f"[MERGE] FFmpeg stdout: {result.stdout.decode()[:500] if result.stdout else 'empty'}")
            logger.info(f"[MERGE] FFmpeg completed successfully")

            # Update script with merged video path
            # Use 'videos/' (plural) to match the /files/videos route
            script.merged_video_url = f"videos/{output_filename}"

            # Verify file was created and log details
            if os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"[MERGE] ✅ Merged video created successfully: {output_filename} ({file_size_mb:.2f} MB)")
                logger.info(f"[MERGE] ✅ Video accessible at: /files/videos/{output_filename}")
            else:
                logger.error(f"[MERGE] ❌ Merged video file NOT found at: {output_path}")
                raise RuntimeError(f"Video merge completed but output file not found: {output_path}")

            self._save_data()
            return script
        except subprocess.TimeoutExpired:
            logger.error("[MERGE] FFmpeg timed out after 600 seconds")
            raise RuntimeError("FFmpeg timed out. The videos may be too large.")
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode() if e.stderr else "No error output"
            stdout_msg = e.stdout.decode() if e.stdout else "No output"

            # Log full details for debugging
            logger.error(f"[MERGE] FFmpeg failed with exit code {e.returncode}")
            logger.error(f"[MERGE] FFmpeg command: {' '.join(cmd)}")
            logger.error(f"[MERGE] FFmpeg stderr: {stderr_msg}")
            logger.error(f"[MERGE] FFmpeg stdout: {stdout_msg}")
            logger.error(f"[MERGE] Video files attempted: {[os.path.basename(p) for p in abs_video_paths]}")

            # Extract user-friendly error message
            user_msg = self._extract_ffmpeg_error_message(stderr_msg, abs_video_paths)
            raise RuntimeError(user_msg)
        finally:
            if os.path.exists(list_path):
                try:
                    os.remove(list_path)
                except OSError:
                    pass
            for tmp in temp_files:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    
    # === Video batch render (async) =======================================

    @staticmethod
    def _should_render_video(frame: StoryboardFrame, force: bool = False) -> bool:
        """Predicate: should the batch worker render a clip for this frame?

        i2v needs a source image. A frame whose image hasn't been
        generated yet is silently skipped — the user has to render the
        storyboard image first. Already-rendered videos
        (``frame.video_url`` set, or ``frame.selected_video_id`` set)
        are skipped unless ``force=True``.
        """
        # Cannot generate video without a source image.
        if not (frame.image_url or getattr(frame, "rendered_image_url", None)):
            return False
        if force:
            return True
        if frame.video_url:
            return False
        if getattr(frame, "selected_video_id", None):
            return False
        return True

    def _render_single_video(self, script: Script, frame: StoryboardFrame) -> None:
        """Per-frame video render used by the batch worker.

        Enqueues a VideoTask using sensible defaults pulled from the
        frame (image, prompt, duration), then runs the existing
        single-task worker (``process_video_task``) inline. Each frame
        keeps its own VideoTask record so the regular Video panel UI
        shows live status during the batch.
        """
        image_url = frame.image_url or getattr(frame, "rendered_image_url", None)
        if not image_url:
            raise ValueError("frame has no source image")
        prompt = frame.video_prompt or frame.image_prompt_en or frame.image_prompt or frame.action_description or ""
        # Inherit duration default from CreateVideoTaskRequest (5s) — anything
        # the user customized lives on a per-task creation flow that doesn't
        # apply to batch.
        _, vtask_id = self.create_video_task(
            script.id,
            image_url=image_url,
            prompt=prompt,
            frame_id=frame.id,
        )
        self.process_video_task(script.id, vtask_id)

    def create_video_batch_render_task(
        self, script_id: str, force: bool = False
    ) -> Tuple[Script, str]:
        """Register an async batch i2v task and return ``(script, task_id)``."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        pending_ids = [
            f.id for f in script.frames if self._should_render_video(f, force=force)
        ]
        task_id = self._register_task(
            task_type="video_batch_render",
            script_id=script_id,
            params={"force": force},
            total_count=len(script.frames),
            pending_count=len(pending_ids),
        )
        return script, task_id

    def process_video_batch_render_task(self, task_id: str) -> None:
        with self._task_lifecycle(task_id) as task:
            script = self.scripts.get(task["script_id"])
            if not script:
                raise ValueError("Script not found")
            force = bool(task["params"].get("force", False))
            pending = [
                f for f in script.frames if self._should_render_video(f, force=force)
            ]
            self._run_per_item_batch(
                task_id, pending, lambda f: self._render_single_video(script, f)
            )

    # === Audio batch render (async) =======================================

    @staticmethod
    def _should_render_audio(frame: StoryboardFrame, force: bool = False) -> bool:
        """Predicate: should the batch worker render audio for this frame?

        A frame is "pending audio" if it has source material (dialogue
        text or action description) but the corresponding output is
        missing. With ``force=True``, any frame with source material
        re-renders. Frames with neither dialogue nor action are silently
        skipped — there's nothing to synthesize.
        """
        has_dialogue_source = bool(frame.dialogue)
        has_sfx_source = bool(frame.action_description)
        if not (has_dialogue_source or has_sfx_source):
            return False
        if force:
            return True
        if has_dialogue_source and not frame.audio_url:
            return True
        if has_sfx_source and not frame.sfx_url:
            return True
        return False

    def _render_single_audio(self, script: Script, frame: StoryboardFrame) -> None:
        """Per-frame audio synth (dialogue + SFX) used by the batch worker.

        Bound to the script's TTS instance. Mirrors the inline body of
        the legacy synchronous ``generate_audio`` so behavior matches
        what users got from the old path.
        """
        with scoped_instance(script.model_settings.tts_instance_id, InstanceType.TTS):
            if frame.dialogue:
                speaker = None
                if frame.character_ids:
                    speaker = next(
                        (c for c in script.characters if c.id == frame.character_ids[0]),
                        None,
                    )
                if speaker:
                    self.audio_generator.generate_dialogue(
                        frame, speaker,
                        speed=speaker.voice_speed,
                        pitch=speaker.voice_pitch,
                        volume=speaker.voice_volume,
                    )
            if frame.action_description:
                self.audio_generator.generate_sfx(frame)
            if frame.video_url:
                self.audio_generator.generate_sfx_from_video(frame)

    def create_audio_batch_render_task(
        self, script_id: str, force: bool = False
    ) -> Tuple[Script, str]:
        """Register an async batch audio synth task."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        pending_ids = [
            f.id for f in script.frames if self._should_render_audio(f, force=force)
        ]
        task_id = self._register_task(
            task_type="audio_batch_render",
            script_id=script_id,
            params={"force": force},
            total_count=len(script.frames),
            pending_count=len(pending_ids),
        )
        return script, task_id

    def process_audio_batch_render_task(self, task_id: str) -> None:
        with self._task_lifecycle(task_id) as task:
            script = self.scripts.get(task["script_id"])
            if not script:
                raise ValueError("Script not found")
            force = bool(task["params"].get("force", False))
            pending = [
                f for f in script.frames if self._should_render_audio(f, force=force)
            ]
            self._run_per_item_batch(
                task_id, pending, lambda f: self._render_single_audio(script, f)
            )

    # === Export pipeline (async wrapper around merge_videos) ==============
    #
    # Coarse 3-stage progress (preflight → ffmpeg → finalize) keeps the
    # contract simple: granular FFmpeg progress requires parsing -progress
    # pipe output and is fiddly enough to defer until users ask. Each
    # stage transition just bumps ``progress`` and ``current_stage``.

    def create_export_task(
        self, script_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Script, str]:
        """Register an async export task. Returns ``(script, task_id)``.

        If the project already has a ``merged_video_url`` AND the caller
        didn't ask to re-export (``params['force']`` falsy), the task
        completes synchronously with the cached URL — saves users from
        re-running FFmpeg on every re-export click.
        """
        _validate_safe_id(script_id, "script_id")
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        params = params or {}
        task_id = self._register_task(
            task_type="export",
            script_id=script_id,
            params=params,
            current_stage="pending",
            output_url=None,
        )

        # Fast-path: cached merge result, no FFmpeg needed.
        if script.merged_video_url and not params.get("force"):
            task = self._get_task(task_id)
            assert task is not None  # we just registered it
            task["status"] = "completed"
            task["progress"] = 100
            task["current_stage"] = "cached"
            task["output_url"] = script.merged_video_url

        return script, task_id

    def process_export_task(self, task_id: str) -> None:
        """Worker for ``create_export_task``. Runs the FFmpeg merge in
        three observable stages so the frontend can render a real
        progress UI instead of an opaque spinner.
        """
        # Cached merge: ``create_export_task`` already finalized it, so
        # there's nothing to do. Check *before* entering the lifecycle
        # context — entering would reset status to "processing".
        existing = self._get_task(task_id)
        if existing and existing["status"] == "completed":
            return

        with self._task_lifecycle(task_id) as task:
            script_id = task["script_id"]

            task["current_stage"] = "preflight"
            task["progress"] = 5
            # merge_videos itself validates ffmpeg presence + frame video
            # availability — let it raise so the lifecycle records error.

            task["current_stage"] = "ffmpeg"
            task["progress"] = 30
            merged = self.merge_videos(script_id)

            task["current_stage"] = "finalize"
            task["progress"] = 95
            task["output_url"] = merged.merged_video_url

    def _extract_ffmpeg_error_message(self, stderr: str, video_paths: List[str]) -> str:
        """
        Extract a user-friendly error message from ffmpeg stderr output.
        
        Args:
            stderr: The stderr output from ffmpeg
            video_paths: List of video file paths that were being processed
            
        Returns:
            A user-friendly error message
        """
        if not stderr:
            return "FFmpeg merge failed with no error output. Please check the log files."
        
        stderr_lower = stderr.lower()
        
        # Common error patterns with user-friendly messages
        if "no such file or directory" in stderr_lower:
            return (
                "One or more video files could not be found.\n"
                "The videos may have been deleted or moved.\n"
                "Please try regenerating the missing videos."
            )
        
        if "invalid data found" in stderr_lower or "invalid file" in stderr_lower or "moov atom not found" in stderr_lower:
            return (
                "One or more video files are corrupted or incomplete.\n"
                "This can happen if video generation was interrupted.\n"
                "Please try regenerating the affected videos."
            )
        
        if ("codec" in stderr_lower and ("not supported" in stderr_lower or "unknown" in stderr_lower)):
            return (
                "Video codec compatibility issue detected.\n"
                "The video format may not be supported by your FFmpeg installation.\n"
                "Try updating FFmpeg to the latest version."
            )
        
        if "permission denied" in stderr_lower or "access is denied" in stderr_lower:
            return (
                "Permission denied when accessing video files.\n"
                "Please check that the application has read/write permissions\n"
                "for the output directory."
            )
        
        if "disk full" in stderr_lower or "no space" in stderr_lower:
            return (
                "Insufficient disk space to create the merged video.\n"
                "Please free up some space and try again."
            )
        
        if "height not divisible" in stderr_lower or "width not divisible" in stderr_lower:
            return (
                "Video resolution compatibility issue.\n"
                "The videos have incompatible dimensions.\n"
                "This should not happen - please report this issue."
            )
        
        if "invalid argument" in stderr_lower:
            # Check if it's related to file list
            if any("filelist" in line.lower() or "concat" in line.lower() for line in stderr.split('\n')):
                return (
                    "FFmpeg could not read the video file list.\n"
                    "This might be a file path encoding issue.\n"
                    "Please ensure video filenames don't contain special characters."
                )
        
        # Fallback: extract the most relevant error line
        # Usually the last non-empty line before the final summary
        error_lines = [line.strip() for line in stderr.split('\n') if line.strip()]
        if error_lines:
            # Look for lines that seem like actual errors (contain "error", "failed", etc.)
            for line in reversed(error_lines):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['error', 'failed', 'invalid', 'cannot', 'unable']):
                    # Truncate if too long
                    if len(line) > 200:
                        line = line[:200] + "..."
                    return f"FFmpeg error: {line}\n\nPlease check the application logs for more details."
            
            # If no error keyword found, use last line
            last_line = error_lines[-1]
            if len(last_line) > 200:
                last_line = last_line[:200] + "..."
            return f"FFmpeg merge failed: {last_line}\n\nPlease check the application logs for more details."
        
        return "FFmpeg merge failed with unknown error. Please check the application logs for details."

    def create_asset_video_task(self, script_id: str, asset_id: str, asset_type: str, prompt: str, duration: int = 5, aspect_ratio: str = None) -> Tuple[Script, str]:
        """Creates a new video generation task for an asset (R2V)."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        # Find asset
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            
        if not target_asset:
            raise ValueError(f"Asset {asset_id} of type {asset_type} not found")
            
        # Use main image as reference
        image_url = target_asset.image_url
        if not image_url:
             # Try fallback for character
             if asset_type == "character":
                 image_url = target_asset.full_body_image_url or target_asset.avatar_url
        
        if not image_url:
            raise ValueError("Asset has no reference image")

        # Save prompt to asset
        if prompt:
            target_asset.video_prompt = prompt
            
        task_id = str(uuid.uuid4())
        
        # Create VideoTask
        task = VideoTask(
            id=task_id,
            project_id=script_id,
            asset_id=asset_id, # Link to asset
            image_url=image_url,
            prompt=prompt or f"Cinematic shot of {target_asset.name}",
            status="pending",
            duration=duration,
            generation_mode="r2v",  # process_video_task resolves the actual SKU from the I2V instance
            created_at=time.time()
        )
        
        # Add to script.video_tasks for global tracking
        if not script.video_tasks:
            script.video_tasks = []
        script.video_tasks.append(task)
        
        # Add to asset's video_assets list
        if not target_asset.video_assets:
            target_asset.video_assets = []
        target_asset.video_assets.append(task)
        
        self._save_data()
        return script, task_id

    def process_video_task(self, script_id: str, task_id: str):
        """Processes a video task."""
        script = self.get_script(script_id)
        if not script:
            logger.error(f"Script {script_id} not found for task {task_id}")
            return
            
        task = next((t for t in script.video_tasks if t.id == task_id), None)
        
        if not task:
            logger.error(f"Task {task_id} not found in script {script_id}")
            return

        try:
            # Update status to processing
            task.status = "processing"
            self._save_data()

            # Generate video
            output_filename = f"video_{task_id}.mp4"
            output_path = os.path.join(self.data_root, "video", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Handle Audio Logic
            # 1. Silent: audio_url=None, audio=False
            # 2. AI Sound: audio_url=None, audio=True
            # 3. Sound Driven: audio_url=URL (audio param ignored)
            
            final_audio_url = None
            final_generate_audio = False
            
            if task.audio_url:
                # Sound Driven Mode
                final_audio_url = task.audio_url
                final_generate_audio = False # API says audio param ignored if url present, but let's be explicit
            elif task.generate_audio:
                # AI Sound Mode
                final_audio_url = None
                final_generate_audio = True
            else:
                # Silent Mode
                final_audio_url = None
                final_generate_audio = False

            # Ensure img_url is passed correctly for OSS
            img_url = task.image_url

            # Route to the appropriate model via the dispatcher (strategy
            # registry). Bind the I2V instance so credentials + base_url
            # flow through the runtime context.
            from ...models.video_dispatcher import (
                VideoGenerationContext,
                build_default_dispatcher,
            )
            # Tests may construct the pipeline via ``__new__`` (skipping
            # ``__init__``), so use ``getattr`` for a safe lazy build.
            if getattr(self, "_video_dispatcher", None) is None:
                self._video_dispatcher = build_default_dispatcher(self.video_generator)
            # Override chain: task-level pick (Settings sidebar) → script-level
            # default (Settings page) → user's default instance (loaded inside
            # scoped_instance when both are None).
            ms = getattr(script, "model_settings", None) if script else None
            script_instance_id = getattr(ms, "i2v_instance_id", None) if ms else None
            i2v_instance_id = task.i2v_instance_id or script_instance_id
            with scoped_instance(i2v_instance_id, InstanceType.I2V) as i2v_inst:
                # The bound instance is the source of truth for the model
                # name. R2V mode swaps to the family's R2V sibling via the
                # provider registry. No legacy hardcoded fallback — if the
                # user hasn't configured an I2V instance the call must fail
                # loudly so they know to set one up.
                # Outside an authenticated request (CLI, tests), DB lookup
                # returns None even when the caller has pre-bound an instance
                # via ``with_instance``; honor that pre-binding before erroring.
                if i2v_inst is None:
                    from ...runtime import current_instance
                    i2v_inst = current_instance()
                inst = require_instance(i2v_inst, InstanceType.I2V)
                base_model = inst.model_name
                if (task.generation_mode or "").lower() == "r2v":
                    from ...utils.provider_registry import swap_provider_modality
                    swapped = swap_provider_modality(base_model, "r2v")
                    if swapped:
                        base_model = swapped
                task.model = base_model
                model_name = task.model
                backend = self._resolve_video_backend(model_name)
                adapter = self._video_dispatcher.resolve(model_name, backend)
                # Frame-derived motion hints for adapters that compose locally
                # (the Remotion engine maps these to Ken Burns + subtitle). The
                # generic ``extras`` channel exists for exactly this — cloud
                # adapters ignore it, so existing i2v/r2v routes are unaffected.
                extras = self._build_video_extras(script, task)
                # Adapters resolve their own bytes/URL via MediaResolver from
                # the stable img_url ref — no per-call temp materialization
                # here. (`video.py::generate_clip` still passes img_path for
                # legacy in-output references; that path is unaffected.)
                video_path, _ = adapter.generate(
                    VideoGenerationContext(
                        task=task,
                        output_path=output_path,
                        img_url=img_url,
                        audio_url=final_audio_url,
                        generate_audio=final_generate_audio,
                        extras=extras,
                    )
                )
            
            task.video_url = os.path.relpath(output_path, self.data_root)
            task.status = "completed"

            # Sync with asset if this is an asset video
            if task.asset_id:
                self._sync_asset_video_task(script, task)
            # Auto-link the first successful render back to its storyboard
            # frame so the "pending video" count drops as renders complete.
            # Mirrors the asset auto-select at L1622; never overwrites a
            # user's explicit selection.
            if task.frame_id:
                self._sync_frame_video_task(script, task)

        except Exception as e:
            import traceback
            logger.exception("Failed to process video task")
            logger.error(f"Video generation failed: {e}")
            task.status = "failed"
            if task.asset_id:
                self._sync_asset_video_task(script, task)
            
        self._save_data()

    def _sync_asset_video_task(self, script: Script, task: VideoTask):
        """Syncs the updated task status/url back to the asset's video_assets list."""
        target_asset = None
        # Search in all asset types
        for char in script.characters:
            if char.id == task.asset_id:
                target_asset = char
                break
        if not target_asset:
            for scene in script.scenes:
                if scene.id == task.asset_id:
                    target_asset = scene
                    break
        if not target_asset:
            for prop in script.props:
                if prop.id == task.asset_id:
                    target_asset = prop
                    break
        
        if target_asset:
            # Find and update the task in the asset's list
            for i, t in enumerate(target_asset.video_assets):
                if t.id == task.id:
                    target_asset.video_assets[i] = task
                    break
            else:
                # Not found, append it (shouldn't happen if created correctly, but good fallback)
                target_asset.video_assets.append(task)

    def _sync_frame_video_task(self, script: Script, task: VideoTask) -> None:
        """Auto-link a completed video task back to its storyboard frame.

        Counterpart to :meth:`_sync_asset_video_task`. The first successful
        render for a frame becomes its selected variant; later renders for
        the same frame are kept on ``script.video_tasks`` but never
        silently overwrite an existing ``selected_video_id``.
        """
        frame = next((f for f in script.frames if f.id == task.frame_id), None)
        if frame is None or frame.selected_video_id:
            return
        frame.selected_video_id = task.id
        frame.video_url = task.video_url

    def create_asset_video_task(self, script_id: str, asset_id: str, asset_type: str, prompt: str = None, duration: int = 5, aspect_ratio: str = None) -> Tuple[Script, str]:
        """Creates a video generation task for an asset (I2V)."""
        script = self.get_script(script_id)
        if not script:
            raise ValueError("Script not found")
            
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
            # Use full body image for character video
            image_url = target_asset.full_body_image_url or target_asset.image_url
            if not prompt:
                prompt = f"A cinematic shot of {target_asset.name}, {target_asset.description}, looking around, breathing, slight movement, high quality, 4k"
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
            image_url = target_asset.image_url
            if not prompt:
                prompt = f"A cinematic shot of {target_asset.name}, {target_asset.description}, ambient motion, lighting change, high quality, 4k"
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            image_url = target_asset.image_url
            if not prompt:
                prompt = f"A cinematic shot of {target_asset.name}, {target_asset.description}, rotating slowly, high quality, 4k"
        else:
            raise ValueError(f"Invalid asset_type: {asset_type}")
            
        if not target_asset:
            raise ValueError(f"Asset {asset_id} not found")
            
        if not image_url:
            raise ValueError(f"Asset {asset_id} has no image to generate video from")

        # Create task using existing method logic but with asset_id
        task_id = str(uuid.uuid4())
        
        # Snapshot logic (duplicated from create_video_task for now, or could refactor)
        snapshot_url = image_url
        try:
            if not image_url.startswith("http"):
                src_path = os.path.join(self.data_root, image_url)
                if os.path.exists(src_path):
                    snapshot_dir = os.path.join(self.data_root, "video_inputs")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    ext = os.path.splitext(image_url)[1] or ".png"
                    snapshot_filename = f"{task_id}{ext}"
                    snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                    import shutil
                    shutil.copy2(src_path, snapshot_path)
                    snapshot_url = f"video_inputs/{snapshot_filename}"
        except Exception:
            pass

        # Determine resolution from aspect ratio or default
        resolution = "720p" # Default
        # TODO: Map aspect_ratio to resolution if needed
        
        task = VideoTask(
            id=task_id,
            project_id=script_id,
            asset_id=asset_id,
            image_url=snapshot_url,
            prompt=prompt,
            status="pending",
            duration=duration,
            resolution=resolution,
            generation_mode="i2v",  # process_video_task resolves the actual SKU from the I2V instance
            created_at=time.time()
        )
        
        # Add to global list
        if not script.video_tasks:
            script.video_tasks = []
        script.video_tasks.append(task)
        
        # Add to asset list
        target_asset.video_assets.append(task)
        
        self._save_data()
        return script, task_id

    def delete_asset_video(self, script_id: str, asset_id: str, asset_type: str, video_id: str) -> Script:
        """Deletes a video from an asset."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        # Find asset
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
        
        if not target_asset:
            raise ValueError(f"Asset {asset_id} of type {asset_type} not found")
        
        # Find the task first to get video_url for file deletion
        video_task_to_delete = None
        if script.video_tasks:
            video_task_to_delete = next((v for v in script.video_tasks if v.id == video_id), None)
        
        # Remove from asset's video_assets
        if target_asset.video_assets:
            original_len = len(target_asset.video_assets)
            target_asset.video_assets = [v for v in target_asset.video_assets if v.id != video_id]
            if len(target_asset.video_assets) == original_len and not video_task_to_delete:
                 # Only raise if not found in either place, or just log warning?
                 # If found in global list but not asset list, it's weird but we should proceed.
                 pass

        # Also remove from script.video_tasks
        if script.video_tasks:
            script.video_tasks = [v for v in script.video_tasks if v.id != video_id]
        
        # Try to delete the video file
        try:
            if video_task_to_delete and video_task_to_delete.video_url:
                video_path = os.path.join(self.data_root, video_task_to_delete.video_url)
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"Deleted video file: {video_path}")
        except Exception as e:
            logger.warning(f"Failed to delete video file: {e}")
        
        self._save_data()
        return script

    def generate_audio(self, script_id: str) -> Script:
        """Step 5: Generate audio (Dialogue & SFX)."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        logger.info(f"Generating audio for script {script.id}")

        with scoped_instance(script.model_settings.tts_instance_id, InstanceType.TTS):
            for frame in script.frames:
                # Generate Dialogue
                if frame.dialogue:
                    speaker = None
                    if frame.character_ids:
                        speaker = next((c for c in script.characters if c.id == frame.character_ids[0]), None)

                    if speaker:
                        self.audio_generator.generate_dialogue(
                            frame, speaker,
                            speed=speaker.voice_speed,
                            pitch=speaker.voice_pitch,
                            volume=speaker.voice_volume
                        )

                # Generate SFX (Text-to-Audio)
                if frame.action_description:
                    self.audio_generator.generate_sfx(frame)

                # Generate SFX (Video-to-Audio) - if video exists
                if frame.video_url:
                    self.audio_generator.generate_sfx_from_video(frame)

                # Generate BGM
                self.audio_generator.generate_bgm(frame)

        self._save_data()
        return script

    def generate_dialogue_line(self, script_id: str, frame_id: str, speed: float = 1.0, pitch: float = 1.0, volume: int = 50) -> Script:
        """Generates audio for a specific line with parameters."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        frame = next((f for f in script.frames if f.id == frame_id), None)
        if not frame:
            raise ValueError("Frame not found")

        if frame.dialogue:
            speaker = None
            if frame.character_ids:
                speaker = next((c for c in script.characters if c.id == frame.character_ids[0]), None)

            if speaker:
                with scoped_instance(script.model_settings.tts_instance_id, InstanceType.TTS):
                    self.audio_generator.generate_dialogue(frame, speaker, speed, pitch, volume)

        self._save_data()
        return script

    def bind_voice(self, script_id: str, char_id: str, voice_id: str, voice_name: str) -> Script:
        """Binds a voice to a character."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        char = next((c for c in script.characters if c.id == char_id), None)
        if not char:
            raise ValueError("Character not found")
            
        char.voice_id = voice_id
        char.voice_name = voice_name
        self._save_data()
        return script

    def get_script(self, script_id: str) -> Optional[Script]:
        return self.scripts.get(script_id)

    def _select_variant_in_asset(self, image_asset: Any, variant_id: str) -> Any:
        """Helper to select a variant in an ImageAsset. Returns the selected variant if found."""
        if not image_asset or not image_asset.variants:
            return None
            
        for variant in image_asset.variants:
            if variant.id == variant_id:
                image_asset.selected_id = variant_id
                return variant
        return None

    def _delete_variant_in_asset(self, image_asset: Any, variant_id: str) -> bool:
        """Helper to delete a variant in an ImageAsset. Returns True if found and deleted."""
        if not image_asset or not image_asset.variants:
            return False
            
        initial_len = len(image_asset.variants)
        image_asset.variants = [v for v in image_asset.variants if v.id != variant_id]
        
        if len(image_asset.variants) < initial_len:
            # If we deleted the selected one, select the last one or None
            if image_asset.selected_id == variant_id:
                if image_asset.variants:
                    image_asset.selected_id = image_asset.variants[-1].id
                else:
                    image_asset.selected_id = None
            return True
        return False

    def select_asset_variant(self, script_id: str, asset_id: str, asset_type: str, variant_id: str, generation_type: str = None) -> Script:
        """Selects a specific variant for an asset."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
            if target_asset:
                # If generation_type is specified, only select from that specific asset
                if generation_type == "full_body":
                    variant = self._select_variant_in_asset(target_asset.full_body_asset, variant_id)
                    if variant:
                        target_asset.full_body_image_url = variant.url
                        target_asset.image_url = variant.url  # Legacy sync
                elif generation_type == "three_view":
                    variant = self._select_variant_in_asset(target_asset.three_view_asset, variant_id)
                    if variant:
                        target_asset.three_view_image_url = variant.url
                elif generation_type == "headshot":
                    variant = self._select_variant_in_asset(target_asset.headshot_asset, variant_id)
                    if variant:
                        target_asset.headshot_image_url = variant.url
                        target_asset.avatar_url = variant.url  # Sync avatar
                else:
                    # Legacy fallback: search all assets (for backward compatibility)
                    variant = self._select_variant_in_asset(target_asset.full_body_asset, variant_id)
                    if variant:
                        target_asset.full_body_image_url = variant.url
                        target_asset.image_url = variant.url
                    
                    if not variant:
                        variant = self._select_variant_in_asset(target_asset.three_view_asset, variant_id)
                        if variant:
                            target_asset.three_view_image_url = variant.url
                    
                    if not variant:
                        variant = self._select_variant_in_asset(target_asset.headshot_asset, variant_id)
                        if variant:
                            target_asset.headshot_image_url = variant.url
                            target_asset.avatar_url = variant.url
                        
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
            if target_asset:
                variant = self._select_variant_in_asset(target_asset.image_asset, variant_id)
                if variant:
                    target_asset.image_url = variant.url

        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            if target_asset:
                variant = self._select_variant_in_asset(target_asset.image_asset, variant_id)
                if variant:
                    target_asset.image_url = variant.url

        elif asset_type == "storyboard_frame":
            target_asset = next((f for f in script.frames if f.id == asset_id), None)
            if target_asset:
                # Check rendered_image_asset
                variant = self._select_variant_in_asset(target_asset.rendered_image_asset, variant_id)
                if variant:
                    target_asset.rendered_image_url = variant.url
                    target_asset.image_url = variant.url # Main image is rendered one
                
                # Also check image_asset (sketch)?
                if not variant:
                    variant = self._select_variant_in_asset(target_asset.image_asset, variant_id)
                    # If sketch, maybe don't update main image_url if rendered exists?
                    # For now, let's assume we only select rendered variants for frames usually.
        
        self._save_data()
        return script

    def delete_asset_variant(self, script_id: str, asset_id: str, asset_type: str, variant_id: str) -> Script:
        """Deletes a specific variant from an asset."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
            
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
            if target_asset:
                if self._delete_variant_in_asset(target_asset.full_body_asset, variant_id):
                    # Sync legacy if needed
                    if target_asset.full_body_asset.selected_id:
                        selected = next((v for v in target_asset.full_body_asset.variants if v.id == target_asset.full_body_asset.selected_id), None)
                        target_asset.image_url = selected.url if selected else None
                    else:
                        target_asset.image_url = None
                
                elif self._delete_variant_in_asset(target_asset.three_view_asset, variant_id):
                    if target_asset.three_view_asset.selected_id:
                        selected = next((v for v in target_asset.three_view_asset.variants if v.id == target_asset.three_view_asset.selected_id), None)
                        target_asset.three_view_image_url = selected.url if selected else None
                    else:
                        target_asset.three_view_image_url = None

                elif self._delete_variant_in_asset(target_asset.headshot_asset, variant_id):
                    if target_asset.headshot_asset.selected_id:
                        selected = next((v for v in target_asset.headshot_asset.variants if v.id == target_asset.headshot_asset.selected_id), None)
                        target_asset.headshot_image_url = selected.url if selected else None
                    else:
                        target_asset.headshot_image_url = None

        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
            if target_asset and self._delete_variant_in_asset(target_asset.image_asset, variant_id):
                if target_asset.image_asset.selected_id:
                    selected = next((v for v in target_asset.image_asset.variants if v.id == target_asset.image_asset.selected_id), None)
                    target_asset.image_url = selected.url if selected else None
                else:
                    target_asset.image_url = None

        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            if target_asset and self._delete_variant_in_asset(target_asset.image_asset, variant_id):
                if target_asset.image_asset.selected_id:
                    selected = next((v for v in target_asset.image_asset.variants if v.id == target_asset.image_asset.selected_id), None)
                    target_asset.image_url = selected.url if selected else None
                else:
                    target_asset.image_url = None

        elif asset_type == "storyboard_frame":
            target_asset = next((f for f in script.frames if f.id == asset_id), None)
            if target_asset:
                if self._delete_variant_in_asset(target_asset.rendered_image_asset, variant_id):
                    if target_asset.rendered_image_asset.selected_id:
                        selected = next((v for v in target_asset.rendered_image_asset.variants if v.id == target_asset.rendered_image_asset.selected_id), None)
                        target_asset.rendered_image_url = selected.url if selected else None
                        target_asset.image_url = selected.url if selected else None
                    else:
                        target_asset.rendered_image_url = None
                        # Don't clear image_url if it might fall back to sketch? 
                        # For now, clear it if rendered is cleared.
                        target_asset.image_url = None

        self._save_data()
        return script

    def update_model_settings(
        self,
        script_id: str,
        llm_instance_id: Optional[str] = None,
        t2i_instance_id: Optional[str] = None,
        i2i_instance_id: Optional[str] = None,
        i2v_instance_id: Optional[str] = None,
        tts_instance_id: Optional[str] = None,
        character_aspect_ratio: Optional[str] = None,
        scene_aspect_ratio: Optional[str] = None,
        prop_aspect_ratio: Optional[str] = None,
        storyboard_aspect_ratio: Optional[str] = None,
    ) -> Script:
        """Updates the project's model-instance references and aspect ratios.

        Each id should be a uuid matching one of the user's ``ModelInstance``
        rows; an empty string is treated as "clear / use default".
        """
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")

        if llm_instance_id is not None:
            script.model_settings.llm_instance_id = llm_instance_id or None
        if t2i_instance_id is not None:
            script.model_settings.t2i_instance_id = t2i_instance_id or None
        if i2i_instance_id is not None:
            script.model_settings.i2i_instance_id = i2i_instance_id or None
        if i2v_instance_id is not None:
            script.model_settings.i2v_instance_id = i2v_instance_id or None
        if tts_instance_id is not None:
            script.model_settings.tts_instance_id = tts_instance_id or None
        if character_aspect_ratio:
            script.model_settings.character_aspect_ratio = character_aspect_ratio
        if scene_aspect_ratio:
            script.model_settings.scene_aspect_ratio = scene_aspect_ratio
        if prop_aspect_ratio:
            script.model_settings.prop_aspect_ratio = prop_aspect_ratio
        if storyboard_aspect_ratio:
            script.model_settings.storyboard_aspect_ratio = storyboard_aspect_ratio

        self._save_data()
        return script

    def _set_variant_favorite(self, image_asset: Any, variant_id: str, is_favorited: bool) -> bool:
        """Helper to set favorite status of a variant. Returns True if found."""
        if not image_asset or not image_asset.variants:
            return False
        for v in image_asset.variants:
            if v.id == variant_id:
                v.is_favorited = is_favorited
                return True
        return False

    def toggle_variant_favorite(self, script_id: str, asset_id: str, asset_type: str, variant_id: str, is_favorited: bool, generation_type: str = None) -> Script:
        """Toggles the favorite status of a variant."""
        script = self.scripts.get(script_id)
        if not script:
            raise ValueError("Script not found")
        
        found = False
        if asset_type == "character":
            target_asset = next((c for c in script.characters if c.id == asset_id), None)
            if target_asset:
                if generation_type == "full_body":
                    found = self._set_variant_favorite(target_asset.full_body_asset, variant_id, is_favorited)
                elif generation_type == "three_view":
                    found = self._set_variant_favorite(target_asset.three_view_asset, variant_id, is_favorited)
                elif generation_type == "headshot":
                    found = self._set_variant_favorite(target_asset.headshot_asset, variant_id, is_favorited)
                else:
                    # Try all character assets
                    found = self._set_variant_favorite(target_asset.full_body_asset, variant_id, is_favorited) or \
                            self._set_variant_favorite(target_asset.three_view_asset, variant_id, is_favorited) or \
                            self._set_variant_favorite(target_asset.headshot_asset, variant_id, is_favorited)
        
        elif asset_type == "scene":
            target_asset = next((s for s in script.scenes if s.id == asset_id), None)
            if target_asset:
                found = self._set_variant_favorite(target_asset.image_asset, variant_id, is_favorited)
        
        elif asset_type == "prop":
            target_asset = next((p for p in script.props if p.id == asset_id), None)
            if target_asset:
                found = self._set_variant_favorite(target_asset.image_asset, variant_id, is_favorited)
        
        elif asset_type == "storyboard_frame":
            target_asset = next((f for f in script.frames if f.id == asset_id), None)
            if target_asset:
                found = self._set_variant_favorite(target_asset.rendered_image_asset, variant_id, is_favorited) or \
                        self._set_variant_favorite(target_asset.image_asset, variant_id, is_favorited)
        
        if not found:
            raise ValueError(f"Variant {variant_id} not found")

        self._save_data()
        return script

    # ============================================================
    # Series Storage & CRUD
    # ============================================================

    def _load_series_data(self) -> Dict[str, Series]:
        if not os.path.exists(self.series_data_file):
            return {}
        try:
            with open(self.series_data_file, 'r') as f:
                data = json.load(f)
                return {k: Series(**v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load series data: {e}")
            return {}

    def _save_series_data_unlocked(self):
        """Save series data without acquiring the lock (caller must hold self._save_lock)."""
        try:
            os.makedirs(os.path.dirname(self.series_data_file) or ".", exist_ok=True)
            with open(self.series_data_file, 'w') as f:
                json.dump({k: v.model_dump() for k, v in self.series_store.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save series data: {e}")

    def _save_series_data(self):
        """Save series data with thread lock."""
        with self._save_lock:
            self._save_series_data_unlocked()

    def create_series(self, title: str, description: str = "") -> Series:
        """Create a new Series."""
        with self._save_lock:
            series = Series(
                id=str(uuid.uuid4()),
                title=title,
                description=description,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self.series_store[series.id] = series
            self._save_series_data_unlocked()
            return series

    def get_series(self, series_id: str) -> Optional[Series]:
        return self.series_store.get(series_id)

    def list_series(self) -> List[Series]:
        return list(self.series_store.values())

    def update_series(self, series_id: str, updates: Dict[str, Any]) -> Series:
        """Update Series fields (title, description, etc.)."""
        with self._save_lock:
            series = self.series_store.get(series_id)
            if not series:
                raise ValueError("Series not found")
            for key, value in updates.items():
                if hasattr(series, key) and key not in ("id", "created_at", "episode_ids"):
                    setattr(series, key, value)
            series.updated_at = time.time()
            self.series_store[series_id] = series
            self._save_series_data_unlocked()
            return series

    def delete_series(self, series_id: str) -> None:
        """Delete a Series and disassociate its episodes."""
        with self._save_lock:
            series = self.series_store.get(series_id)
            if not series:
                raise ValueError("Series not found")
            # Disassociate episodes
            for ep_id in series.episode_ids:
                script = self.scripts.get(ep_id)
                if script:
                    script.series_id = None
                    script.episode_number = None
            self._save_data()
            del self.series_store[series_id]
            self._save_series_data_unlocked()

    def add_episode_to_series(self, series_id: str, script_id: str, episode_number: Optional[int] = None) -> Series:
        """Add an existing Script/Project as an Episode to a Series."""
        with self._save_lock:
            series = self.series_store.get(series_id)
            if not series:
                raise ValueError("Series not found")
            script = self.scripts.get(script_id)
            if not script:
                raise ValueError("Script not found")
            # If script already belongs to another series, remove it from the old one
            if script.series_id and script.series_id != series_id:
                old_series = self.series_store.get(script.series_id)
                if old_series and script_id in old_series.episode_ids:
                    old_series.episode_ids.remove(script_id)
            if script_id not in series.episode_ids:
                series.episode_ids.append(script_id)
            script.series_id = series_id
            script.episode_number = episode_number or len(series.episode_ids)
            series.updated_at = time.time()
            self._save_data()
            self._save_series_data_unlocked()
            return series

    def remove_episode_from_series(self, series_id: str, script_id: str) -> Series:
        """Remove an Episode from a Series (does not delete the project)."""
        with self._save_lock:
            series = self.series_store.get(series_id)
            if not series:
                raise ValueError("Series not found")
            if script_id in series.episode_ids:
                series.episode_ids.remove(script_id)
            script = self.scripts.get(script_id)
            if script:
                script.series_id = None
                script.episode_number = None
            series.updated_at = time.time()
            self._save_data()
            self._save_series_data_unlocked()
            return series

    def get_series_episodes(self, series_id: str) -> List[Script]:
        """Get all Episodes belonging to a Series, in order."""
        series = self.series_store.get(series_id)
        if not series:
            raise ValueError("Series not found")
        episodes = []
        for ep_id in series.episode_ids:
            script = self.scripts.get(ep_id)
            if script:
                episodes.append(script)
        return episodes

    def resolve_episode_assets(self, episode: Script, series: Optional[Series] = None) -> Dict[str, List]:
        """Merge Episode-local assets with Series shared assets.
        Episode-local assets take priority (by ID) over Series assets."""
        if not series:
            # Auto-lookup series if episode has series_id
            if episode.series_id:
                series = self.series_store.get(episode.series_id)
        if not series:
            return {
                "characters": episode.characters,
                "scenes": episode.scenes,
                "props": episode.props,
            }
        # Build lookup by ID for episode-local assets
        ep_char_ids = {c.id for c in episode.characters}
        ep_scene_ids = {s.id for s in episode.scenes}
        ep_prop_ids = {p.id for p in episode.props}

        merged_characters = list(episode.characters) + [c for c in series.characters if c.id not in ep_char_ids]
        merged_scenes = list(episode.scenes) + [s for s in series.scenes if s.id not in ep_scene_ids]
        merged_props = list(episode.props) + [p for p in series.props if p.id not in ep_prop_ids]

        return {
            "characters": merged_characters,
            "scenes": merged_scenes,
            "props": merged_props,
        }

    # ============================================================
    # File Import & Episode Splitting
    # ============================================================

    def import_file_and_split(self, text: str, suggested_episodes: int = 3) -> List[Dict]:
        """Split text into episodes using LLM. Returns episode preview data."""
        with scoped_instance(None, InstanceType.LLM):
            return self.script_processor.split_into_episodes(text, suggested_episodes)

    def create_series_from_import(self, title: str, text: str, episodes_data: List[Dict],
                                   description: str = "") -> Dict:
        """Create a Series with Episodes from import data.
        episodes_data: list of dicts with episode_number, title, start_marker, end_marker."""
        # Create the Series (already acquires lock internally)
        series = self.create_series(title, description)

        # Split text into episode chunks based on markers
        episode_texts = self._split_text_by_markers(text, episodes_data)

        with self._save_lock:
            # Create Episode (Script) for each chunk
            created_episodes = []
            for idx, ep_data in enumerate(episodes_data):
                ep_text = episode_texts[idx] if idx < len(episode_texts) else ""
                ep_title = ep_data.get("title", f"第{idx+1}集")
                episode_number = ep_data.get("episode_number", idx + 1)

                # Create draft script (no LLM analysis yet — user can trigger later)
                script = self.script_processor.create_draft_script(ep_title, ep_text)
                script.series_id = series.id
                script.episode_number = episode_number
                self.scripts[script.id] = script

                series.episode_ids.append(script.id)
                created_episodes.append({
                    "id": script.id,
                    "title": ep_title,
                    "episode_number": episode_number,
                    "text_length": len(ep_text),
                })

            self._save_data()
            self._save_series_data_unlocked()

        return {
            "series": series.model_dump(),
            "episodes": created_episodes,
        }

    def _split_text_by_markers(self, text: str, episodes_data: List[Dict]) -> List[str]:
        """Split text into chunks using start/end markers from LLM.
        Searches sequentially to avoid overlapping chunks."""
        chunks = []
        search_from = 0  # Track position to avoid overlap

        for ep in episodes_data:
            start_marker = ep.get("start_marker", "")
            end_marker = ep.get("end_marker", "")

            start_idx = search_from
            end_idx = len(text)

            if start_marker:
                found = text.find(start_marker, search_from)
                if found >= 0:
                    start_idx = found

            if end_marker:
                found = text.find(end_marker, start_idx)
                if found >= 0:
                    end_idx = found + len(end_marker)

            chunks.append(text[start_idx:end_idx])
            search_from = end_idx  # Next episode starts after this one

        # Fallback: if markers produced empty/overlapping chunks, do equal split
        if not chunks or all(len(c.strip()) == 0 for c in chunks):
            chunk_size = max(1, len(text) // len(episodes_data))
            chunks = []
            for i in range(len(episodes_data)):
                start = i * chunk_size
                end = start + chunk_size if i < len(episodes_data) - 1 else len(text)
                chunks.append(text[start:end])

        return chunks

    # ============================================================
    # Series Asset Operations
    # ============================================================

    def _find_series_asset(self, series_id: str, asset_id: str, asset_type: str):
        """Find an asset in a Series. Returns (series, asset) tuple."""
        if asset_type not in ("character", "scene", "prop"):
            raise ValueError(f"Invalid asset type: {asset_type}")
        series = self.series_store.get(series_id)
        if not series:
            raise ValueError("Series not found")
        target_asset = None
        if asset_type == "character":
            target_asset = next((c for c in series.characters if c.id == asset_id), None)
        elif asset_type == "scene":
            target_asset = next((s for s in series.scenes if s.id == asset_id), None)
        elif asset_type == "prop":
            target_asset = next((p for p in series.props if p.id == asset_id), None)
        if not target_asset:
            raise ValueError(f"Asset {asset_id} of type {asset_type} not found in series")
        return series, target_asset

    def toggle_series_asset_lock(self, series_id: str, asset_id: str, asset_type: str) -> Series:
        """Toggle the locked status of a Series asset."""
        with self._save_lock:
            series, target_asset = self._find_series_asset(series_id, asset_id, asset_type)
            target_asset.locked = not target_asset.locked
            self._save_series_data_unlocked()
            return series

    def update_series_asset_image(self, series_id: str, asset_id: str, asset_type: str, image_url: str) -> Series:
        """Updates the image URL of a Series asset."""
        with self._save_lock:
            series, target_asset = self._find_series_asset(series_id, asset_id, asset_type)
            target_asset.image_url = image_url
            if asset_type == "character":
                target_asset.avatar_url = image_url
            self._save_series_data_unlocked()
            return series

    def update_series_asset_attributes(self, series_id: str, asset_id: str, asset_type: str, attributes: Dict[str, Any]) -> Series:
        """Updates arbitrary attributes of a Series asset."""
        with self._save_lock:
            series, target_asset = self._find_series_asset(series_id, asset_id, asset_type)
            for key, value in attributes.items():
                if hasattr(target_asset, key) and key not in ("id", "status", "locked"):
                    setattr(target_asset, key, value)
            series.updated_at = time.time()
            self._save_series_data_unlocked()
            return series

    def generate_series_asset(self, series_id: str, asset_id: str, asset_type: str,
                              style_preset: str = None, reference_image_url: str = None,
                              style_prompt: str = None, generation_type: str = "all",
                              prompt: str = None, apply_style: bool = True,
                              negative_prompt: str = None, batch_size: int = 1,
                              model_name: str = None) -> tuple:
        """Generate a Series asset. Creates an async task like project asset generation.
        Returns (series, task_id)."""
        series = self.series_store.get(series_id)
        if not series:
            raise ValueError("Series not found")

        # Resolve T2I instance now (within request scope) so task params can
        # carry the resolved model_name + instance_id forward to the worker.
        from .instance_resolver import load_instance as _load_inst
        t2i_inst = require_instance(
            _load_inst(series.model_settings.t2i_instance_id, InstanceType.T2I),
            InstanceType.T2I,
        )
        i2i_inst = _load_inst(series.model_settings.i2i_instance_id, InstanceType.I2I)
        t2i_model = model_name or t2i_inst.model_name
        t2i_instance_id = t2i_inst.id
        i2i_instance_id = i2i_inst.id if i2i_inst else None

        from .assets import ASPECT_RATIO_TO_SIZE
        if asset_type == "character":
            aspect_ratio = series.model_settings.character_aspect_ratio
            default_size = "576*1024"
        elif asset_type == "scene":
            aspect_ratio = series.model_settings.scene_aspect_ratio
            default_size = "1024*576"
        elif asset_type == "prop":
            aspect_ratio = series.model_settings.prop_aspect_ratio
            default_size = "1024*1024"
        else:
            aspect_ratio = "9:16"
            default_size = "576*1024"
        effective_size = ASPECT_RATIO_TO_SIZE.get(aspect_ratio, default_size)

        effective_positive_prompt = ""
        effective_negative_prompt = negative_prompt or ""
        if apply_style:
            if series.art_direction and series.art_direction.style_config:
                effective_positive_prompt = series.art_direction.style_config.get('positive_prompt', '')
                global_neg = series.art_direction.style_config.get('negative_prompt', '')
                if global_neg:
                    effective_negative_prompt = f"{effective_negative_prompt}, {global_neg}" if effective_negative_prompt else global_neg
            elif style_prompt:
                effective_positive_prompt = style_prompt
            elif style_preset:
                effective_positive_prompt = f"{style_preset} style"

        task_id = str(uuid.uuid4())
        self.asset_generation_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "error": None,
            "script_id": series_id,  # reuse field name for task lookup
            "asset_id": asset_id,
            "asset_type": asset_type,
            "created_at": time.time(),
            "is_series": True,
            "params": {
                "style_preset": style_preset,
                "reference_image_url": reference_image_url,
                "effective_positive_prompt": effective_positive_prompt,
                "effective_negative_prompt": effective_negative_prompt,
                "generation_type": generation_type,
                "prompt": prompt,
                "apply_style": apply_style,
                "batch_size": batch_size,
                "t2i_model": t2i_model,
                "t2i_instance_id": t2i_instance_id,
                "i2i_instance_id": i2i_instance_id,
                "effective_size": effective_size,
            }
        }
        return series, task_id

    def import_assets_from_series(self, target_series_id: str, source_series_id: str, asset_ids: List[str]) -> Tuple[Series, List[str], List[str]]:
        """Deep-copy selected assets from source Series to target Series.
        Returns (target_series, imported_ids, skipped_ids)."""
        with self._save_lock:
            target = self.series_store.get(target_series_id)
            if not target:
                raise ValueError("Target series not found")
            source = self.series_store.get(source_series_id)
            if not source:
                raise ValueError("Source series not found")

            # Build lookup of all source assets
            source_assets = {}
            for c in source.characters:
                source_assets[c.id] = ("character", c)
            for s in source.scenes:
                source_assets[s.id] = ("scene", s)
            for p in source.props:
                source_assets[p.id] = ("prop", p)

            imported_ids = []
            skipped_ids = []
            for aid in asset_ids:
                if aid not in source_assets:
                    skipped_ids.append(aid)
                    continue
                asset_type, asset = source_assets[aid]
                # Deep copy with new ID
                import copy
                new_asset = copy.deepcopy(asset)
                new_asset.id = str(uuid.uuid4())
                if asset_type == "character":
                    target.characters.append(new_asset)
                elif asset_type == "scene":
                    target.scenes.append(new_asset)
                elif asset_type == "prop":
                    target.props.append(new_asset)
                imported_ids.append(aid)

            target.updated_at = time.time()
            self._save_series_data_unlocked()
            return target, imported_ids, skipped_ids

    def get_effective_prompt(self, prompt_type: str, episode: Script, series: Optional[Series] = None) -> str:
        """Three-level fallback: Episode -> Series -> system default."""
        valid_prompt_types = ("storyboard_polish", "video_polish", "r2v_polish")
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be one of {valid_prompt_types}")
        from .llm import DEFAULT_STORYBOARD_POLISH_PROMPT, DEFAULT_VIDEO_POLISH_PROMPT, DEFAULT_R2V_POLISH_PROMPT
        defaults = {
            "storyboard_polish": DEFAULT_STORYBOARD_POLISH_PROMPT,
            "video_polish": DEFAULT_VIDEO_POLISH_PROMPT,
            "r2v_polish": DEFAULT_R2V_POLISH_PROMPT,
        }
        episode_value = getattr(episode.prompt_config, prompt_type, "")
        if episode_value.strip():
            return episode_value
        if series:
            series_value = getattr(series.prompt_config, prompt_type, "")
            if series_value.strip():
                return series_value
        return defaults.get(prompt_type, "")
