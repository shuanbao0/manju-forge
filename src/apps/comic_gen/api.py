from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import shutil
import uuid
import logging
import traceback
from .pipeline import ComicGenPipeline
from .models import (
    PromptConfig,
    ProviderBackend,
    ProviderRoutingConfig,
    Script,
    Series,
    VideoTask,
)
from .llm import ScriptProcessor, DEFAULT_STORYBOARD_POLISH_PROMPT, DEFAULT_VIDEO_POLISH_PROMPT, DEFAULT_R2V_POLISH_PROMPT
from ...utils.oss_utils import OSSImageUploader, sign_oss_urls_in_data
from ...utils import setup_logging
from ...auth import routes as auth_routes
from ...auth import me_routes as me_routes_module
from ...auth.db import get_engine as _ensure_auth_db
from ...auth.middleware import AuthContextMiddleware
from ...admin import routes as admin_routes
from .pipeline_factory import pipeline_proxy, current_pipeline as _current_pipeline_for_user
from ...auth.deps import require_admin
from ...auth.models import User as _AuthUser
from ... import runtime as _ctx_runtime  # carries request context across executor / bg tasks
from ...i18n import get_locale, t as _t
from ...i18n.middleware import LocaleMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, set_key

app = FastAPI(title="AI Comic Gen API")
logger = logging.getLogger(__name__)

# Setup logging to user directory
setup_logging()


from ...models.instance import InstanceNotConfiguredError as _InstanceNotConfiguredError


@app.exception_handler(_InstanceNotConfiguredError)
async def _instance_not_configured_handler(request: Request, exc: _InstanceNotConfiguredError):
    """Surface "no usable model instance" as a 400 with a structured code so
    the frontend can route the user to Settings → 模型实例 instead of showing
    a 500 stack trace."""
    return JSONResponse(
        status_code=400,
        content={
            "detail": {
                "code": "INSTANCE_NOT_CONFIGURED",
                "instance_type": exc.instance_type.value,
                "message": str(exc),
            }
        },
    )

# Use absolute path for .env file (api.py is in src/apps/comic_gen/)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
env_path = os.path.join(_project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# Debug: Print OSS configuration at startup
logger.info(f"STARTUP: OSS_ENDPOINT={os.getenv('OSS_ENDPOINT')}, OSS_BUCKET_NAME={os.getenv('OSS_BUCKET_NAME')}, OSS_BASE_PATH={os.getenv('OSS_BASE_PATH')}")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # ``allow_credentials=True`` would be invalid per CORS spec when origin
    # is ``*`` and the browser would reject the response. Auth uses Bearer
    # tokens in the Authorization header (no cookies), so we don't need
    # credentialed requests anyway.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # Allow browsers to access Content-Disposition for downloads
)

# Initialize the auth/users SQLite database eagerly so first request doesn't pay
# the schema-creation cost and so misconfiguration surfaces at startup.
_ensure_auth_db()

# Authenticate every business-route request and bind a request-scoped
# RequestContext containing user + decrypted credentials. Auth/admin/docs
# paths are passed through; their FastAPI dependencies handle auth themselves.
app.add_middleware(AuthContextMiddleware)

# Resolve the request's locale from Accept-Language and bind it to the
# i18n contextvar so downstream service-layer code can call ``t(key)``
# without threading the locale through every signature.
app.add_middleware(LocaleMiddleware)

app.include_router(auth_routes.router)
app.include_router(admin_routes.router)
app.include_router(me_routes_module.router)

# Middleware to add cache headers on user file responses (the multi-user
# /me/files route below produces these — public CDN paths are kept short-TTL
# by default to avoid stale-data leakage between users).
@app.middleware("http")
async def add_cache_control_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/me/files/"):
        response.headers["Cache-Control"] = "private, max-age=300"
    return response

# Create the legacy-tenant root + the multi-tenant root. Per-user dirs
# (``output/users/<uid>/...``) are created lazily on first pipeline use.
os.makedirs("output", exist_ok=True)
os.makedirs("output/users", exist_ok=True)

# NOTE: the public ``/files/*`` static mounts that existed pre-P2 have been
# removed. Multi-user instances cannot expose the entire ``output/`` tree
# without authentication, so reads now go through ``GET /me/files/{path}``
# which validates the caller and resolves the path inside their per-user
# directory (P4 will redirect to presigned URLs when object storage is on).


# The proxy resolves to a per-user pipeline at attribute access time. Existing
# route handlers that did ``pipeline.X(...)`` keep compiling, but now reach
# the right user's data instead of a single global ``output/`` directory.
pipeline = pipeline_proxy


# ── Authenticated file serving ────────────────────────────────────────────
#
# Replaces the legacy public ``app.mount("/files", StaticFiles(...))``. Same
# URL shape, but every read now has to come from the calling user's own
# ``output/users/<uid>/`` directory. Path traversal is blocked via
# ``_safe_resolve_path``. P4 will swap the local-disk read for a presigned
# redirect when object storage is configured.

from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402  (must come after app init)
from .pipeline import _safe_resolve_path  # noqa: E402
from ...utils.object_storage import ObjectStorageClient  # noqa: E402
from ...utils.oss_utils import is_object_key as _is_object_key  # noqa: E402

# Legacy path families that older ``projects.json`` rows store in URLs.
# Keep a single resolver that strips them so all of these work transparently.
_FILE_PATH_PREFIXES = ("outputs/videos/", "outputs/assets/", "outputs/", "videos/", "assets/")


def _strip_legacy_prefix(rel: str) -> str:
    rel = rel.lstrip("/")
    for prefix in _FILE_PATH_PREFIXES:
        if rel.startswith(prefix):
            # Map ``outputs/videos/x.mp4`` → ``video/x.mp4`` (singular)
            tail = rel[len(prefix):]
            if prefix == "outputs/videos/":
                return os.path.join("video", tail)
            if prefix == "videos/":
                return os.path.join("video", tail)
            if prefix == "outputs/assets/":
                return os.path.join("assets", tail)
            if prefix == "outputs/":
                return tail
            return os.path.join("assets", tail)
    return rel


@app.get("/files/{rel_path:path}")
async def serve_user_file(rel_path: str):
    """Authenticated counterpart of the old ``/files/*`` static mount.

    If ``rel_path`` looks like an object-storage Object Key and the user
    has S3/MinIO configured: when the storage endpoint is publicly
    reachable we redirect to a short-lived presigned URL so the browser
    bypasses FastAPI; when the endpoint is internal-only (self-hosted
    MinIO at ``http://minio:9000``) we proxy the bytes through this
    endpoint instead, since a redirect URL the browser can't resolve is
    worse than a slightly-slower proxy. Otherwise streams the file from
    the user's local ``output/users/<uid>/`` dir.
    """
    pipe = _current_pipeline_for_user()

    try:
        if _is_object_key(rel_path):
            client = ObjectStorageClient.for_current_user()
            if client.is_configured:
                # Defense in depth: the path_prefix already includes
                # ``users/<uid>`` so signing is scoped, but reject
                # cross-tenant keys explicitly to avoid surprising callers
                # if the prefix logic is ever changed.
                user_prefix = client.config.path_prefix.rstrip("/")
                normalized = rel_path.lstrip("/")
                if user_prefix and not normalized.startswith(user_prefix + "/") and normalized != user_prefix:
                    raise HTTPException(status_code=403, detail="object key outside current user's namespace")

                if client.endpoint_is_internal:
                    data = client.download_bytes(rel_path)
                    if data is not None:
                        import mimetypes as _mimetypes
                        from fastapi.responses import Response
                        media_type, _ = _mimetypes.guess_type(rel_path)
                        return Response(
                            content=data,
                            media_type=media_type or "application/octet-stream",
                            headers={"Cache-Control": "private, max-age=300"},
                        )
                else:
                    url = client.presigned_get_url(rel_path)
                    if url:
                        return RedirectResponse(url, status_code=302)
    except HTTPException:
        raise
    except Exception:
        # Fall through to local-disk handling below.
        pass

    safe_rel = _strip_legacy_prefix(rel_path)
    try:
        abs_path = _safe_resolve_path(pipe.data_root, safe_rel)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid path")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(abs_path)


@app.get("/me/files/{rel_path:path}")
async def serve_user_file_alias(rel_path: str):
    """Modern alias of /files; encourages clients to be explicit about the
    multi-user nature. Behavior identical to /files."""
    return await serve_user_file(rel_path)

# ── Model catalog registry (single source of truth for FE dropdowns) ─────
# Backed by ``src.utils.model_catalog`` — adding a new model is just a card.

@app.get("/registry/models")
async def list_registered_models(capability: Optional[str] = None):
    """Return the curated model catalog.

    Optional ``capability`` filter: ``t2i`` | ``i2i`` | ``i2v`` | ``t2v`` |
    ``r2v`` | ``tts`` | ``llm``. When omitted, returns the full catalog.
    """
    from ...utils.model_catalog import get_default_catalog
    return get_default_catalog().serialize(capability=capability)


@app.get("/registry/llm-presets")
async def list_llm_presets():
    """Return the LLM provider presets shown in the Settings dropdown."""
    from ...utils.model_catalog import get_default_catalog
    return {"presets": get_default_catalog().serialize()["presets"]}


@app.get("/registry/vendors")
async def list_vendor_connectors(capability: Optional[str] = None):
    """Return the configurable vendor connectors (Kling / Vidu / Doubao / ...).

    The Settings UI renders one card per connector. Optional ``capability``
    filter narrows by ``llm`` | ``i2v`` | ``t2v`` | ``r2v`` | ``t2i`` | ``i2i``.
    """
    from ...utils.vendor_connectors import get_default_vendor_registry
    return get_default_vendor_registry().serialize(capability=capability)


@app.get("/debug/config")
async def debug_config(_admin: _AuthUser = Depends(require_admin)):
    """Admin-only diagnostic endpoint to check OSS and path configuration."""
    uploader = OSSImageUploader()
    return {
        "oss_configured": uploader.is_configured,
        "oss_bucket_initialized": uploader.bucket is not None,
        "oss_base_path": os.getenv("OSS_BASE_PATH", "manju-forge"),
        "output_dir_exists": os.path.exists("output"),
        "output_contents": os.listdir("output") if os.path.exists("output") else [],
        "cwd": os.getcwd(),
        "env_vars_present": {
            "OSS_ENDPOINT": bool(os.getenv("OSS_ENDPOINT")),
            "OSS_BUCKET_NAME": bool(os.getenv("OSS_BUCKET_NAME")),
            "ALIBABA_CLOUD_ACCESS_KEY_ID": bool(os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")),
        }
    }

def signed_response(data):
    """Helper to sign OSS URLs in data before returning to frontend.
    
    Handles Pydantic models, lists of models, and dicts.
    Returns a JSONResponse with signed URLs.
    """
    if data is None:
        return JSONResponse(content=None)
    
    # Convert Pydantic models to dict
    if hasattr(data, "model_dump"):
        processed_data = data.model_dump()
    elif isinstance(data, list):
        processed_data = [item.model_dump() if hasattr(item, "model_dump") else item for item in data]
    else:
        processed_data = data
    
    # Check if OSS is configured
    uploader = OSSImageUploader()
    if uploader.is_configured:
        # OSS mode: sign URLs in the data
        processed_data = sign_oss_urls_in_data(processed_data, uploader)
    
    # Return JSONResponse directly to avoid Pydantic re-validation stripping fields
    return JSONResponse(content=processed_data)


# ============================================================
# Shared Request Models (used by both Project and Series endpoints)
# ============================================================

class GenerateAssetRequest(BaseModel):
    asset_id: str
    asset_type: str
    style_preset: str = "Cinematic"
    reference_image_url: Optional[str] = None
    style_prompt: Optional[str] = None
    generation_type: str = "all"  # 'full_body', 'three_view', 'headshot', 'all'
    prompt: Optional[str] = None
    apply_style: bool = True
    negative_prompt: Optional[str] = None
    batch_size: int = 1
    model_name: Optional[str] = None

class ToggleLockRequest(BaseModel):
    asset_id: str
    asset_type: str

class UpdateAssetImageRequest(BaseModel):
    asset_id: str
    asset_type: str
    image_url: str

class UpdateAssetAttributesRequest(BaseModel):
    asset_id: str
    asset_type: str
    attributes: Dict[str, Any]


@app.get("/system/check")
async def check_system(_admin: _AuthUser = Depends(require_admin)):
    """Admin-only: check system dependencies (ffmpeg, etc.) and configuration."""
    from utils.system_check import run_system_checks
    return run_system_checks()





@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a file and returns its URL (OSS if configured, else local)."""
    try:
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(_current_pipeline_for_user().data_root, "uploads", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Try uploading to OSS
        oss_url = OSSImageUploader().upload_image(file_path)
        if oss_url:
            return signed_response({"url": oss_url})

        # Fallback to local URL (relative path for frontend getAssetUrl)
        return {"url": f"uploads/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UploadAssetRequest(BaseModel):
    upload_type: str  # "full_body" | "head_shot" | "three_views" | "image"
    description: Optional[str] = None  # User-modified description for reverse generation


@app.post("/projects/{script_id}/assets/{asset_type}/{asset_id}/upload")
async def upload_asset(
    script_id: str,
    asset_type: str,
    asset_id: str,
    upload_type: str,
    description: Optional[str] = None,
    file: UploadFile = File(...)
):
    """
    Uploads an image as a new variant for an asset.
    The uploaded image is marked as the 'upload source' for reverse generation.
    
    - asset_type: "character", "scene", or "prop"
    - upload_type: "full_body", "head_shot", "three_views", or "image" (for scene/prop)
    - description: Optional modified description for the asset
    """
    try:
        # 1. Save file locally first
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(_current_pipeline_for_user().data_root, "uploads", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Upload to OSS
        uploader = OSSImageUploader()
        oss_url = uploader.upload_image(file_path)
        if not oss_url:
            oss_url = f"uploads/{filename}"  # Fallback to local path
        
        # 3. Update asset with new variant
        updated_script = pipeline.add_uploaded_asset_variant(
            script_id=script_id,
            asset_type=asset_type,
            asset_id=asset_id,
            upload_type=upload_type,
            image_url=oss_url,
            description=description
        )
        
        if not updated_script:
            raise HTTPException(status_code=404, detail="Script or asset not found")
        
        return signed_response(updated_script)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error uploading asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CreateProjectRequest(BaseModel):
    title: str
    text: str


@app.post("/projects", response_model=Script)
async def create_project(request: CreateProjectRequest, skip_analysis: bool = False):
    """Creates a new project from a novel text."""
    # Run in thread pool to avoid blocking event loop during LLM analysis.
    # ``_ctx_runtime.run_in_executor`` carries the per-request user context
    # across the thread boundary so the LLM picks up the caller's API key.
    result = await _ctx_runtime.run_in_executor(
        None,
        pipeline.create_project, request.title, request.text, skip_analysis,
    )
    return signed_response(result)



class ReparseProjectRequest(BaseModel):
    text: str


@app.put("/projects/{script_id}/reparse", response_model=Script)
async def reparse_project(script_id: str, request: ReparseProjectRequest):
    """Re-parses the text for an existing project, replacing all entities."""
    try:
        # Carry user context into the worker thread (per-user creds).
        result = await _ctx_runtime.run_in_executor(
            None, pipeline.reparse_project, script_id, request.text,
        )
        return signed_response(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================
# Remotion MG engine (flow B) — chat-only motion-graphics videos.
# Nested under /projects/* so the existing nginx allowlist covers it.
# ============================================================


class CreateMGProjectRequest(BaseModel):
    title: str
    text: str
    aspect_ratio: str = "9:16"


@app.post("/projects/mg", response_model=Script)
async def create_mg_project(request: CreateMGProjectRequest):
    """Create a chat-only Remotion motion-graphics project (flow B)."""
    result = await _ctx_runtime.run_in_executor(
        None,
        pipeline.create_mg_project, request.title, request.text, request.aspect_ratio,
    )
    return signed_response(result)


class GenerateMGSpecRequest(BaseModel):
    style_hint: Optional[str] = None


@app.post("/projects/{script_id}/remotion/spec", response_model=Script)
async def generate_mg_spec(script_id: str, request: GenerateMGSpecRequest):
    """Flow B step 1: LLM authors the VideoSpec for this project."""
    try:
        result = await _ctx_runtime.run_in_executor(
            None, pipeline.generate_mg_spec, script_id, request.style_hint,
        )
        return signed_response(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/remotion/render", response_model=Script)
async def render_mg_video(script_id: str):
    """Flow B step 2: render the stored VideoSpec to an MP4 via Remotion."""
    try:
        result = await _ctx_runtime.run_in_executor(
            None, pipeline.render_mg_video, script_id,
        )
        return signed_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/", response_model=List[dict])
async def list_projects():
    """Lists all projects from backend storage."""
    scripts = list(pipeline.scripts.values())
    return signed_response(scripts)


# ============================================================
# Series CRUD
# ============================================================

class CreateSeriesRequest(BaseModel):
    title: str
    description: str = ""


class UpdateSeriesRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


@app.post("/series")
async def create_series(request: CreateSeriesRequest):
    """Create a new Series."""
    series = pipeline.create_series(request.title, request.description)
    return signed_response(series)


@app.get("/series")
async def list_series():
    """List all Series."""
    series_list = pipeline.list_series()
    return signed_response(series_list)


@app.get("/series/{series_id}")
async def get_series(series_id: str):
    """Get Series details including assets and episode list."""
    series = pipeline.get_series(series_id)
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    # Include episode summaries
    episodes = pipeline.get_series_episodes(series_id)
    result = series.model_dump()
    result["episodes"] = [
        {
            "id": ep.id,
            "title": ep.title,
            "episode_number": ep.episode_number,
            "created_at": ep.created_at,
            "updated_at": ep.updated_at,
        }
        for ep in episodes
    ]
    return signed_response(result)


@app.put("/series/{series_id}")
async def update_series(series_id: str, request: UpdateSeriesRequest):
    """Update Series title/description."""
    try:
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        series = pipeline.update_series(series_id, updates)
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/series/{series_id}")
async def delete_series(series_id: str):
    """Delete a Series and disassociate its episodes."""
    try:
        pipeline.delete_series(series_id)
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class AddEpisodeRequest(BaseModel):
    script_id: str
    episode_number: Optional[int] = None


@app.post("/series/{series_id}/episodes")
async def add_episode_to_series(series_id: str, request: AddEpisodeRequest):
    """Add an existing project as an episode to a Series."""
    try:
        series = pipeline.add_episode_to_series(series_id, request.script_id, request.episode_number)
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/series/{series_id}/episodes/{script_id}")
async def remove_episode_from_series(series_id: str, script_id: str):
    """Remove an episode from a Series (does not delete the project)."""
    try:
        series = pipeline.remove_episode_from_series(series_id, script_id)
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/series/{series_id}/episodes")
async def get_series_episodes(series_id: str):
    """Get all episodes in a Series."""
    try:
        episodes = pipeline.get_series_episodes(series_id)
        return signed_response(episodes)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/series/{series_id}/prompt_config")
async def get_series_prompt_config(series_id: str):
    """Get Series prompt config with system defaults."""
    series = pipeline.get_series(series_id)
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    return {
        "prompt_config": series.prompt_config.model_dump(),
        "defaults": {
            "storyboard_polish": DEFAULT_STORYBOARD_POLISH_PROMPT,
            "video_polish": DEFAULT_VIDEO_POLISH_PROMPT,
            "r2v_polish": DEFAULT_R2V_POLISH_PROMPT,
        },
    }


@app.put("/series/{series_id}/prompt_config")
async def update_series_prompt_config(series_id: str, config: PromptConfig):
    """Update Series-level prompt config."""
    try:
        series = pipeline.update_series(series_id, {"prompt_config": config})
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================
# Series Model Settings
# ============================================================

class UpdateModelSettingsRequest(BaseModel):
    """References to the user's ModelInstance rows (uuids)."""

    llm_instance_id: Optional[str] = None
    t2i_instance_id: Optional[str] = None
    i2i_instance_id: Optional[str] = None
    i2v_instance_id: Optional[str] = None
    tts_instance_id: Optional[str] = None
    character_aspect_ratio: Optional[str] = None
    scene_aspect_ratio: Optional[str] = None
    prop_aspect_ratio: Optional[str] = None
    storyboard_aspect_ratio: Optional[str] = None

@app.get("/series/{series_id}/model_settings")
async def get_series_model_settings(series_id: str):
    """Get Series model settings."""
    series = pipeline.get_series(series_id)
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    return series.model_settings.model_dump()


@app.put("/series/{series_id}/model_settings")
async def update_series_model_settings(series_id: str, settings: UpdateModelSettingsRequest):
    """Update Series-level model settings."""
    updates = {k: v for k, v in settings.model_dump().items() if v is not None}
    if not updates:
        series = pipeline.get_series(series_id)
        if not series:
            raise HTTPException(status_code=404, detail="Series not found")
        return signed_response(series)
    try:
        current_series = pipeline.get_series(series_id)
        if not current_series:
            raise HTTPException(status_code=404, detail="Series not found")
        ms = current_series.model_settings.model_copy(update=updates)
        series = pipeline.update_series(series_id, {"model_settings": ms})
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================
# Series Asset Operations
# ============================================================

@app.get("/series/{series_id}/assets")
async def get_series_assets(series_id: str):
    """Get all shared assets from a Series."""
    series = pipeline.get_series(series_id)
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    return signed_response({
        "characters": [c.model_dump() for c in series.characters],
        "scenes": [s.model_dump() for s in series.scenes],
        "props": [p.model_dump() for p in series.props],
    })


@app.post("/series/{series_id}/assets/generate")
async def generate_series_asset(series_id: str, request: GenerateAssetRequest, background_tasks: BackgroundTasks):
    """Generate a single asset for a Series (async)."""
    try:
        series, task_id = pipeline.generate_series_asset(
            series_id,
            request.asset_id,
            request.asset_type,
            request.style_preset,
            request.reference_image_url,
            request.style_prompt,
            request.generation_type,
            request.prompt,
            request.apply_style,
            request.negative_prompt,
            request.batch_size,
            request.model_name
        )
        _ctx_runtime.add_background_task(background_tasks, pipeline.process_asset_generation_task, task_id)
        response_data = series.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/series/{series_id}/assets/toggle_lock")
async def toggle_series_asset_lock(series_id: str, request: ToggleLockRequest):
    """Toggle the locked status of a Series asset."""
    try:
        series = pipeline.toggle_series_asset_lock(series_id, request.asset_id, request.asset_type)
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/series/{series_id}/assets/update_image")
async def update_series_asset_image(series_id: str, request: UpdateAssetImageRequest):
    """Update a Series asset's image URL."""
    try:
        series = pipeline.update_series_asset_image(series_id, request.asset_id, request.asset_type, request.image_url)
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/series/{series_id}/assets/update_attributes")
async def update_series_asset_attributes(series_id: str, request: UpdateAssetAttributesRequest):
    """Update arbitrary attributes of a Series asset."""
    try:
        series = pipeline.update_series_asset_attributes(
            series_id, request.asset_id, request.asset_type, request.attributes
        )
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ImportAssetsRequest(BaseModel):
    source_series_id: str
    asset_ids: List[str]


@app.post("/series/{series_id}/assets/import")
async def import_series_assets(series_id: str, request: ImportAssetsRequest):
    """Deep-copy assets from another Series into this one."""
    try:
        series, imported_ids, skipped_ids = pipeline.import_assets_from_series(series_id, request.source_series_id, request.asset_ids)
        return signed_response(series)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# File Import & Episode Splitting
# ============================================================

@app.post("/series/import/preview")
async def import_file_preview(
    file: UploadFile = File(...),
    suggested_episodes: int = 3,
    locale: str = Depends(get_locale),
):
    """Upload a txt/md file and get LLM episode split preview."""
    if suggested_episodes < 1 or suggested_episodes > 50:
        raise HTTPException(status_code=400, detail=_t("errors.import_episodes_out_of_range", locale))
    try:
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8")
        if not text.strip():
            raise HTTPException(status_code=400, detail=_t("errors.import_file_empty", locale))

        episodes = await _ctx_runtime.run_in_executor(
            None, pipeline.import_file_and_split, text, suggested_episodes,
        )
        # Store text in pipeline cache, return import_id instead of full text
        import_id = str(uuid.uuid4())
        pipeline._import_cache[import_id] = text
        return {
            "filename": file.filename,
            "text_length": len(text),
            "suggested_episodes": suggested_episodes,
            "episodes": episodes,
            "import_id": import_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("File import preview failed")
        raise HTTPException(status_code=500, detail=str(e))


class ConfirmImportRequest(BaseModel):
    title: str
    description: str = ""
    import_id: str = ""
    text: Optional[str] = None
    episodes: List[Dict[str, Any]]  # episode_number, title, start_marker, end_marker, ...


@app.post("/series/import/confirm")
async def import_file_confirm(request: ConfirmImportRequest):
    """Confirm the episode split and create Series + Episodes (sync).

    Sync; LLM split + persistence can take 10–60s. Prefer the
    ``_async`` variant for production.
    """
    try:
        # Prefer import_id from cache, fallback to request.text
        text = None
        if request.import_id:
            text = pipeline._import_cache.pop(request.import_id, None)
        if not text:
            text = request.text
        if not text:
            raise ValueError("No text available. Provide import_id or text.")
        result = await _ctx_runtime.run_in_executor(
            None,
            pipeline.create_series_from_import,
            request.title,
            text,
            request.episodes,
            request.description,
        )
        return signed_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Import confirm failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/series/import/confirm_async")
async def import_file_confirm_async(
    request: ConfirmImportRequest, background_tasks: BackgroundTasks
):
    """Async variant of :pyfunc:`import_file_confirm`.

    Note ``_import_cache.pop`` runs *eagerly* on the request thread to
    grab the cached text before the background worker starts —
    otherwise a slow worker scheduling could let the cache TTL expire.
    """
    text = None
    if request.import_id:
        text = pipeline._import_cache.pop(request.import_id, None)
    if not text:
        text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text available. Provide import_id or text.")

    return _async_oneshot(
        task_type="series_import_confirm",
        script_id=None,
        work=lambda: pipeline.create_series_from_import(
            request.title, text, request.episodes, request.description,
        ),
        background_tasks=background_tasks,
    )


class EnvConfig(ProviderRoutingConfig):
    DASHSCOPE_API_KEY: Optional[str] = None
    ALIBABA_CLOUD_ACCESS_KEY_ID: Optional[str] = None
    ALIBABA_CLOUD_ACCESS_KEY_SECRET: Optional[str] = None
    OSS_BUCKET_NAME: Optional[str] = None
    OSS_ENDPOINT: Optional[str] = None
    OSS_BASE_PATH: Optional[str] = None
    KLING_ACCESS_KEY: Optional[str] = None
    KLING_SECRET_KEY: Optional[str] = None
    VIDU_API_KEY: Optional[str] = None
    LLM_PROVIDER: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    API_HOST: Optional[str] = None
    API_PORT: Optional[str] = None
    endpoint_overrides: Dict[str, str] = Field(default_factory=dict)


def _normalize_provider_mode(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower()
    if normalized in (ProviderBackend.DASHSCOPE.value, ProviderBackend.VENDOR.value):
        return normalized
    return ProviderBackend.DASHSCOPE.value


def get_user_config_path() -> str:
    """
    Returns the path to the user config file.
    - Development mode: Uses .env in project root
    - Packaged app mode: Uses ~/.manju-forge/config.json
    """
    from ...utils import get_user_data_dir
    
    # Check if running in packaged mode (e.g., via environment variable or frozen check)
    is_packaged = os.getenv("MANJU_FORGE_PACKAGED", "false").lower() == "true" or getattr(sys, 'frozen', False)
    
    if is_packaged:
        # Use user home directory for packaged app
        config_dir = get_user_data_dir()
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "config.json")
    else:
        # Use .env in project root for development
        # Get absolute path to project root (api.py is in src/apps/comic_gen/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(project_root, ".env")



def load_user_config():
    """Loads user config from file and applies to environment."""
    config_path = get_user_config_path()
    
    if config_path.endswith(".json"):
        # JSON config for packaged app
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key, value in config.items():
                    if value:
                        os.environ[key] = value
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
    # .env is already loaded at startup via dotenv


def save_user_config(config_dict: dict):
    """Saves user config to file."""
    config_path = get_user_config_path()

    if config_path.endswith(".json"):
        # JSON config for packaged app
        import json
        existing_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    existing_config = json.load(f)
            except:
                pass
        existing_config.update(config_dict)
        with open(config_path, "w") as f:
            json.dump(existing_config, f, indent=2)
    else:
        # .env for development
        for key, value in config_dict.items():
            if value is not None:
                set_key(config_path, key, value)


def remove_user_config_keys(keys: list):
    """Removes keys from the persisted config file."""
    if not keys:
        return
    config_path = get_user_config_path()

    if config_path.endswith(".json"):
        import json
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    existing_config = json.load(f)
                for key in keys:
                    existing_config.pop(key, None)
                with open(config_path, "w") as f:
                    json.dump(existing_config, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to remove keys from config: {e}")
    else:
        from dotenv import unset_key
        for key in keys:
            try:
                unset_key(config_path, key)
            except Exception as e:
                logger.warning(f"Failed to unset key {key} from .env: {e}")


# Load user config on startup
import sys
load_user_config()



@app.get("/config/info")
async def get_config_info(_admin: _AuthUser = Depends(require_admin)):
    """Admin-only: information about the instance's config storage mode."""
    config_path = get_user_config_path()
    is_packaged = os.getenv("MANJU_FORGE_PACKAGED", "false").lower() == "true" or getattr(sys, 'frozen', False)
    return {
        "mode": "packaged" if is_packaged else "development",
        "config_path": config_path,
        "config_exists": os.path.exists(config_path)
    }


@app.post("/config/env")
async def update_env_config(config: EnvConfig, _admin: _AuthUser = Depends(require_admin)):
    """Admin-only: update instance-wide environment configuration.

    Per-user secrets (DASHSCOPE_API_KEY, OSS keys, Kling/Vidu, OpenAI) now
    live on the user record and should be edited via ``/me/credentials``.
    This endpoint is preserved for the small set of instance-level settings
    (``API_HOST`` / ``API_PORT`` / global LLM defaults) that admins still
    set in the ``.env`` file.
    """
    try:
        raw_config = config.dict(exclude_unset=True)

        # Extract endpoint_overrides and flatten into config_dict
        endpoint_overrides = raw_config.pop("endpoint_overrides", {})

        # Filter out None values and serialize enum values as plain strings.
        config_dict: Dict[str, str] = {}
        for key, value in raw_config.items():
            if value is None:
                continue
            if isinstance(value, ProviderBackend):
                config_dict[key] = value.value
            else:
                config_dict[key] = value

        # Process endpoint overrides: validate keys against known providers
        from ...utils.endpoints import PROVIDER_DEFAULTS
        allowed_keys = {f"{p}_BASE_URL" for p in PROVIDER_DEFAULTS}
        keys_to_remove = []
        for env_key, value in endpoint_overrides.items():
            if env_key not in allowed_keys:
                logger.warning(f"Ignoring unknown endpoint key: {env_key}")
                continue
            if value and value.strip():
                config_dict[env_key] = value.strip()
            else:
                # Clear override: remove from env and config file
                os.environ.pop(env_key, None)
                keys_to_remove.append(env_key)

        # Update current process env
        for key, value in config_dict.items():
            os.environ[key] = value

        # Save to file
        save_user_config(config_dict)
        remove_user_config_keys(keys_to_remove)

        # Reset OSS singleton to pick up new config (non-blocking)
        try:
            OSSImageUploader.reset_instance()
            logger.info("OSS instance reset successfully")
        except Exception as oss_e:
            # OSS reset failure should not block config saving
            logger.warning(f"OSS reset failed (non-critical): {oss_e}")

        config_path = get_user_config_path()
        return {"status": "success", "message": f"Configuration saved to {config_path}"}
    except Exception as e:
        logger.exception("Failed to save environment configuration")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/projects/{script_id}", response_model=Script)
async def get_project(script_id: str):
    """Retrieves a project by ID."""
    script = pipeline.get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Project not found")
    return signed_response(script)



@app.delete("/projects/{script_id}")
async def delete_project(script_id: str):
    """Deletes a project by ID. WARNING: This permanently removes the project from backend storage."""
    script = pipeline.get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        # If project belongs to a Series, remove from episode_ids
        if script.series_id:
            series = pipeline.get_series(script.series_id)
            if series and script_id in series.episode_ids:
                series.episode_ids.remove(script_id)
                pipeline._save_series_data()

        # Remove from pipeline scripts
        del pipeline.scripts[script_id]
        pipeline._save_data()
        return {"status": "deleted", "id": script_id, "title": script.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/sync_descriptions", response_model=Script)
async def sync_descriptions(script_id: str):
    """
    Syncs entity descriptions from Script module to Assets module.
    
    This endpoint forces a refresh of the project data, ensuring that any
    description changes made in the Script module are reflected in Assets.
    
    Note: This only syncs descriptions; generated images/videos are preserved.
    """
    try:
        updated_script = pipeline.sync_descriptions_from_script_entities(script_id)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AddCharacterRequest(BaseModel):
    name: str
    description: str

@app.post("/projects/{script_id}/characters", response_model=Script)
async def add_character(script_id: str, request: AddCharacterRequest):
    """Adds a new character."""
    try:
        updated_script = pipeline.add_character(script_id, request.name, request.description)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/projects/{script_id}/characters/{char_id}", response_model=Script)
async def delete_character(script_id: str, char_id: str):
    """Deletes a character."""
    try:
        updated_script = pipeline.delete_character(script_id, char_id)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AddSceneRequest(BaseModel):
    name: str
    description: str

@app.post("/projects/{script_id}/scenes", response_model=Script)
async def add_scene(script_id: str, request: AddSceneRequest):
    """Adds a new scene."""
    try:
        updated_script = pipeline.add_scene(script_id, request.name, request.description)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/projects/{script_id}/scenes/{scene_id}", response_model=Script)
async def delete_scene(script_id: str, scene_id: str):
    """Deletes a scene."""
    try:
        updated_script = pipeline.delete_scene(script_id, scene_id)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UpdateStyleRequest(BaseModel):
    style_preset: str
    style_prompt: Optional[str] = None


@app.patch("/projects/{script_id}/style", response_model=Script)
async def update_project_style(script_id: str, request: UpdateStyleRequest):
    """Updates the global style settings for a project."""
    try:
        updated_script = pipeline.update_project_style(
            script_id,
            request.style_preset,
            request.style_prompt
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/generate_assets", response_model=Script)
async def generate_assets(script_id: str, background_tasks: BackgroundTasks):
    """Triggers asset generation."""
    script = pipeline.get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Project not found")

    # Run in background to avoid blocking
    # For simplicity in this demo, we run synchronously or use background tasks
    # pipeline.generate_assets(script_id) 
    # But since we want to return the updated status, we might want to run it and return.
    # Given the mock nature, it's fast.

    try:
        updated_script = pipeline.generate_assets(script_id)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class GenerateMotionRefRequest(BaseModel):
    """Request model for generating Motion Reference videos."""
    asset_id: str
    asset_type: str  # 'full_body' | 'head_shot' for characters; 'scene' | 'prop' for scenes and props
    prompt: Optional[str] = None
    audio_url: Optional[str] = None  # Driving audio for lip-sync
    duration: int = 5
    batch_size: int = 1


@app.post("/projects/{script_id}/assets/generate_motion_ref")
async def generate_motion_ref(script_id: str, request: GenerateMotionRefRequest, background_tasks: BackgroundTasks):
    """Generates a Motion Reference video for an asset (Character Full Body/Headshot, Scene, or Prop)."""
    try:
        script, task_id = pipeline.create_motion_ref_task(
            script_id=script_id,
            asset_id=request.asset_id,
            asset_type=request.asset_type,
            prompt=request.prompt,
            audio_url=request.audio_url,
            duration=request.duration,
            batch_size=request.batch_size
        )
        
        # Add background processing
        _ctx_runtime.add_background_task(background_tasks, pipeline.process_motion_ref_task, script_id, task_id)
        
        # Return script with task_id for frontend polling
        response_data = script.model_dump() if hasattr(script, 'model_dump') else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === SCRIPT REWRITE (novel → screenplay) ===


@app.post("/projects/{script_id}/script/rewrite-to-screenplay")
async def rewrite_script_to_screenplay(script_id: str):
    """Rewrite ``original_text`` into structured screenplay format.

    Result is persisted to ``script.formatted_text`` while
    ``original_text`` stays intact. Downstream storyboard analysis can
    then use the formatted version for cleaner shot decomposition.
    """
    try:
        script = pipeline.rewrite_script_to_screenplay(script_id)
        return signed_response(script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in rewrite_script_to_screenplay: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === ENTITY EXTRACTION (incremental / full) ===


class ExtractEntitiesRequest(BaseModel):
    """Request to (re-)extract entities from a script.

    ``strategy`` defaults to ``"incremental"`` — for Series episodes the
    LLM is fed the existing Series catalog and instructed to reuse
    matching characters/scenes/props instead of creating duplicates.
    Pass ``"full"`` to force a clean re-extraction.
    """

    strategy: str = Field("incremental", description="'incremental' or 'full'")


@app.post("/projects/{script_id}/entities/extract")
async def extract_script_entities(script_id: str, request: ExtractEntitiesRequest):
    """Run entity extraction and persist results into the Series catalog."""
    try:
        result = pipeline.process_script_entities(script_id, strategy=request.strategy)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in extract_script_entities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === BATCH KEYFRAME GENERATION (vendor-aware grid) ===


class BatchKeyframesRequest(BaseModel):
    """Request to batch-render keyframes for several storyboard frames.

    ``force_per_shot`` bypasses the grid path even on grid-capable
    vendors — useful when the user is iterating on a single frame and
    doesn't want it bundled with siblings.
    """

    frame_ids: List[str] = Field(..., min_length=1)
    mode: str = Field("first_frame")
    force_per_shot: bool = Field(False)


@app.post("/projects/{script_id}/frames/keyframes/batch")
async def batch_generate_frame_keyframes(
    script_id: str, request: BatchKeyframesRequest
):
    """Render keyframes for multiple frames in one batch.

    Uses native grid composition on grid-capable vendors (Seedream),
    falls back to per-shot parallel generation otherwise. Same return
    shape on either path — caller doesn't need to know which was used.
    """
    try:
        return pipeline.batch_generate_frame_keyframes(
            script_id, request.frame_ids,
            mode=request.mode,
            force_per_shot=request.force_per_shot,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch_generate_frame_keyframes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === VIDEO PROMPT TIMELINE SLICING ===


class SliceTimelineRequest(BaseModel):
    """Request to slice a frame's ``video_prompt`` into a time-axis timeline.

    ``segment_seconds`` defaults to 3 (huobao convention). ``duration_override``
    lets the caller pin a duration when the frame has no associated video task yet.
    """

    segment_seconds: int = Field(3, ge=1, le=10)
    duration_override: Optional[int] = Field(None, ge=1, le=60)


@app.post("/projects/{script_id}/frames/{frame_id}/video/slice-timeline")
async def slice_frame_video_timeline(
    script_id: str, frame_id: str, request: SliceTimelineRequest
):
    """Generate and persist ``video_prompt_timeline`` for one frame.

    Opt-in: the legacy ``video_prompt`` is left untouched so video
    generation calls that don't consume the timeline format keep working.
    """
    try:
        frame = pipeline.slice_frame_video_timeline(
            script_id, frame_id,
            segment_seconds=request.segment_seconds,
            duration_override=request.duration_override,
        )
        return signed_response(frame)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in slice_frame_video_timeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === AUTO VOICE ASSIGNMENT ===


@app.post("/projects/{script_id}/voices/auto-assign")
async def auto_assign_voices(script_id: str):
    """Run the voice-assignment chain over the script's characters.

    Characters with an existing ``voice_id`` are left untouched
    (manual overrides win). Returns the new mapping + IDs that were
    intentionally skipped (locked-with-blank-voice cases).
    """
    try:
        return pipeline.auto_assign_voices(script_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in auto_assign_voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === STORYBOARD DRAMATIZATION v2 ===

class AnalyzeToStoryboardRequest(BaseModel):
    """Request to analyze script text into storyboard frames."""
    text: str


@app.post("/projects/{script_id}/storyboard/analyze")
async def analyze_to_storyboard(
    script_id: str,
    request: AnalyzeToStoryboardRequest,
    background_tasks: BackgroundTasks,
):
    """
    Analyzes script text and generates storyboard frames using AI (Prompt B).
    Replaces existing frames with newly generated ones.

    Async: returns immediately with ``_task_id``; client polls ``/tasks/{task_id}``
    for completion. The LLM call regularly takes 30–180s for long scripts and was
    previously hitting upstream gateway timeouts (Cloudflare ~100s) when run
    synchronously.
    """
    try:
        script, task_id = pipeline.create_storyboard_analysis_task(script_id, request.text)
        _ctx_runtime.add_background_task(
            background_tasks, pipeline.process_storyboard_analysis_task, task_id
        )
        response_data = script.model_dump() if hasattr(script, "model_dump") else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_to_storyboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class RefinePromptRequest(BaseModel):
    """Request to refine a frame's prompt using AI."""
    frame_id: str
    raw_prompt: str
    assets: list = []  # List of asset references
    feedback: str = Field("", max_length=2000)  # User feedback for iterative refinement


@app.post("/projects/{script_id}/storyboard/refine_prompt")
async def refine_storyboard_prompt(script_id: str, request: RefinePromptRequest):
    """
    Refines a raw prompt into bilingual (CN/EN) prompts using AI (Prompt C).
    Returns the refined prompts and optionally updates the frame.

    Synchronous version kept for back-compat. New clients should prefer
    the ``_async`` variant which avoids upstream gateway timeouts.
    """
    try:
        result = pipeline.refine_frame_prompt(
            script_id,
            request.frame_id,
            request.raw_prompt,
            request.assets,
            request.feedback,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in refine_storyboard_prompt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _async_oneshot(
    *,
    task_type: str,
    script_id: Optional[str],
    work,
    background_tasks: BackgroundTasks,
):
    """Helper: register a one-shot task and dispatch the work in
    background. Returns ``{"_task_id": ...}`` so the client knows what
    to poll. Centralizes the create→register→dispatch dance.
    """
    task_id = pipeline._register_task(task_type=task_type, script_id=script_id)
    _ctx_runtime.add_background_task(
        background_tasks, pipeline._run_one_shot, task_id, work
    )
    return {"_task_id": task_id}


@app.post("/projects/{script_id}/storyboard/refine_prompt_async")
async def refine_storyboard_prompt_async(
    script_id: str, request: RefinePromptRequest, background_tasks: BackgroundTasks
):
    """Async variant of :pyfunc:`refine_storyboard_prompt`. Returns
    ``{ _task_id }``; client polls ``/tasks/{id}`` for the polished
    bilingual prompt in ``status['result']``.
    """
    if not pipeline.get_script(script_id):
        raise HTTPException(status_code=404, detail="Script not found")
    return _async_oneshot(
        task_type="storyboard_refine_prompt",
        script_id=script_id,
        work=lambda: pipeline.refine_frame_prompt(
            script_id, request.frame_id, request.raw_prompt, request.assets, request.feedback,
        ),
        background_tasks=background_tasks,
    )


@app.post("/projects/{script_id}/generate_storyboard", response_model=Script)
async def generate_storyboard(script_id: str):
    """Triggers storyboard generation (legacy synchronous batch).

    Prefer ``POST /projects/{id}/storyboard/render_all`` — same effect but
    async with progress polling, won't trip upstream gateway timeouts.
    """
    try:
        updated_script = pipeline.generate_storyboard(script_id)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RenderAllRequest(BaseModel):
    force: bool = False  # True = re-render frames that already have images


@app.post("/projects/{script_id}/storyboard/render_all")
async def render_all_storyboard(
    script_id: str,
    request: RenderAllRequest,
    background_tasks: BackgroundTasks,
):
    """Render every eligible storyboard frame asynchronously.

    Eligible = not locked, and (no image yet | ``force=true``). The endpoint
    returns immediately with ``_task_id``; the client polls
    ``/tasks/{task_id}`` for granular progress
    (``completed_count`` / ``failed_count`` / ``current_frame_id`` / ``errors``).

    Each frame is rendered using its own tuned ``image_prompt`` and
    ``composition_data`` — this is the batch counterpart of
    ``/storyboard/render`` and shares the same per-frame logic.
    """
    try:
        script, task_id = pipeline.create_storyboard_batch_render_task(
            script_id, force=request.force
        )
        _ctx_runtime.add_background_task(
            background_tasks,
            pipeline.process_storyboard_batch_render_task,
            task_id,
        )
        response_data = script.model_dump() if hasattr(script, "model_dump") else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in render_all_storyboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/projects/{script_id}/generate_video", response_model=Script)
async def generate_video(script_id: str):
    """Triggers video generation (legacy synchronous batch).

    Prefer ``POST /projects/{id}/video/render_all`` — same effect but
    async with progress polling. Kept for back-compat with old clients.
    """
    try:
        updated_script = pipeline.generate_video(script_id)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RenderAllVideoRequest(BaseModel):
    force: bool = False


@app.post("/projects/{script_id}/video/render_all")
async def render_all_videos(
    script_id: str,
    request: RenderAllVideoRequest,
    background_tasks: BackgroundTasks,
):
    """Generate i2v videos for every eligible frame asynchronously.

    Eligible = has source image AND (no video yet | ``force=true``).
    Returns ``_task_id``; client polls ``/tasks/{id}`` for granular
    progress (each frame produces a normal VideoTask record so the
    Video panel UI shows live status alongside the batch progress bar).
    """
    try:
        script, task_id = pipeline.create_video_batch_render_task(
            script_id, force=request.force
        )
        _ctx_runtime.add_background_task(
            background_tasks, pipeline.process_video_batch_render_task, task_id
        )
        response_data = script.model_dump() if hasattr(script, "model_dump") else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in render_all_videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/generate_audio", response_model=Script)
async def generate_audio(script_id: str):
    """Triggers audio generation (legacy synchronous batch).

    Prefer ``POST /projects/{id}/audio/render_all`` — same effect but
    async with progress polling. Kept for back-compat with old clients.
    """
    try:
        updated_script = pipeline.generate_audio(script_id)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RenderAllAudioRequest(BaseModel):
    force: bool = False


@app.post("/projects/{script_id}/audio/render_all")
async def render_all_audio(
    script_id: str,
    request: RenderAllAudioRequest,
    background_tasks: BackgroundTasks,
):
    """Synthesize dialogue + SFX for every eligible frame asynchronously.

    Eligible = has source material (dialogue or action_description) AND
    (corresponding output missing | ``force=true``). Returns
    ``_task_id``; client polls ``/tasks/{id}``.
    """
    try:
        script, task_id = pipeline.create_audio_batch_render_task(
            script_id, force=request.force
        )
        _ctx_runtime.add_background_task(
            background_tasks, pipeline.process_audio_batch_render_task, task_id
        )
        response_data = script.model_dump() if hasattr(script, "model_dump") else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in render_all_audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



class CreateVideoTaskRequest(BaseModel):
    image_url: str
    prompt: str
    frame_id: Optional[str] = None
    duration: int = 5
    seed: Optional[int] = None
    resolution: str = "720p"
    generate_audio: bool = False
    audio_url: Optional[str] = None
    prompt_extend: bool = True
    negative_prompt: Optional[str] = None
    batch_size: int = 1
    # Optional override; backend resolves from i2v_instance_id when omitted.
    model: Optional[str] = None
    shot_type: str = "single"  # 'single' or 'multi' (only for wan2.6-i2v)
    generation_mode: str = "i2v"  # 'i2v' (image-to-video) or 'r2v' (reference-to-video)
    reference_video_urls: List[str] = []  # Reference video URLs for R2V (max 3)
    # Kling params
    mode: Optional[str] = None
    sound: Optional[str] = None
    cfg_scale: Optional[float] = None
    # Vidu params
    vidu_audio: Optional[bool] = None
    movement_amplitude: Optional[str] = None
    # ModelInstance picked in the UI for this submit (per-task override).
    i2v_instance_id: Optional[str] = None


async def process_video_task(script_id: str, task_id: str):
    """Background task to generate video."""
    try:
        pipeline.process_video_task(script_id, task_id)
    except Exception as e:
        logger.error(f"Error processing video task {task_id}: {e}")


@app.post("/projects/{script_id}/video_tasks", response_model=List[VideoTask])
async def create_video_task(script_id: str, request: CreateVideoTaskRequest, background_tasks: BackgroundTasks):
    """Creates new video generation tasks."""
    try:
        tasks = []
        for _ in range(request.batch_size):
            script, task_id = pipeline.create_video_task(
                script_id=script_id,
                image_url=request.image_url,
                prompt=request.prompt,
                frame_id=request.frame_id,
                duration=request.duration,
                seed=request.seed,
                resolution=request.resolution,
                generate_audio=request.generate_audio,
                audio_url=request.audio_url,
                prompt_extend=request.prompt_extend,
                negative_prompt=request.negative_prompt,
                model=request.model,
                shot_type=request.shot_type,
                generation_mode=request.generation_mode,
                reference_video_urls=request.reference_video_urls,
                mode=request.mode,
                sound=request.sound,
                cfg_scale=request.cfg_scale,
                vidu_audio=request.vidu_audio,
                movement_amplitude=request.movement_amplitude,
                i2v_instance_id=request.i2v_instance_id,
            )

            # Find the created task object
            created_task = next((t for t in script.video_tasks if t.id == task_id), None)
            if created_task:
                tasks.append(created_task)

            # Add background processing
            _ctx_runtime.add_background_task(background_tasks, pipeline.process_video_task, script_id, task_id)

        return signed_response(tasks)

    except Exception as e:
        import traceback
        logger.exception("An error occurred")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/assets/generate")
async def generate_single_asset(script_id: str, request: GenerateAssetRequest, background_tasks: BackgroundTasks):
    """Generates a single asset with specific options (async).
    Returns immediately with task_id for polling progress."""
    try:
        script, task_id = pipeline.create_asset_generation_task(
            script_id,
            request.asset_id,
            request.asset_type,
            request.style_preset,
            request.reference_image_url,
            request.style_prompt,
            request.generation_type,
            request.prompt,
            request.apply_style,
            request.negative_prompt,
            request.batch_size,
            request.model_name
        )
        
        # Add background processing
        _ctx_runtime.add_background_task(background_tasks, pipeline.process_asset_generation_task, task_id)
        
        # Return script with task_id for frontend polling
        response_data = script.model_dump() if hasattr(script, 'model_dump') else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Returns the status of an asset generation task for polling."""
    status = pipeline.get_asset_generation_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # If completed, return the updated script as well
    if status["status"] == "completed":
        script = pipeline.get_script(status["script_id"])
        if script:
            status["script"] = signed_response(script).body.decode('utf-8')
    
    return status


class GenerateAssetVideoRequest(BaseModel):
    prompt: Optional[str] = None
    duration: int = 5
    aspect_ratio: Optional[str] = None


@app.post("/projects/{script_id}/assets/{asset_type}/{asset_id}/generate_video", response_model=Script)
async def generate_asset_video(script_id: str, asset_type: str, asset_id: str, request: GenerateAssetVideoRequest, background_tasks: BackgroundTasks):
    """Generates a video for a specific asset (I2V)."""
    try:
        script, task_id = pipeline.create_asset_video_task(
            script_id,
            asset_id,
            asset_type,
            request.prompt,
            request.duration,
            request.aspect_ratio
        )
        
        # Add background processing
        _ctx_runtime.add_background_task(background_tasks, pipeline.process_video_task, script_id, task_id)
        
        return signed_response(script)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/projects/{script_id}/assets/{asset_type}/{asset_id}/videos/{video_id}", response_model=Script)
async def delete_asset_video(script_id: str, asset_type: str, asset_id: str, video_id: str):
    """Deletes a video from an asset."""
    try:
        updated_script = pipeline.delete_asset_video(
            script_id,
            asset_id,
            asset_type,
            video_id
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/projects/{script_id}/assets/toggle_lock", response_model=Script)
async def toggle_asset_lock(script_id: str, request: ToggleLockRequest):
    """Toggles the locked status of an asset."""
    try:
        updated_script = pipeline.toggle_asset_lock(
            script_id,
            request.asset_id,
            request.asset_type
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/projects/{script_id}/assets/update_image", response_model=Script)
async def update_asset_image(script_id: str, request: UpdateAssetImageRequest):
    """Updates an asset's image URL manually."""
    try:
        updated_script = pipeline.update_asset_image(
            script_id,
            request.asset_id,
            request.asset_type,
            request.image_url
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/projects/{script_id}/assets/update_attributes", response_model=Script)
async def update_asset_attributes(script_id: str, request: UpdateAssetAttributesRequest):
    """Updates arbitrary attributes of an asset."""
    try:
        updated_script = pipeline.update_asset_attributes(
            script_id,
            request.asset_id,
            request.asset_type,
            request.attributes
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class UpdateAssetDescriptionRequest(BaseModel):
    asset_id: str
    asset_type: str
    description: str


@app.post("/projects/{script_id}/assets/update_description", response_model=Script)
async def update_asset_description(script_id: str, request: UpdateAssetDescriptionRequest):
    """Updates an asset's description."""
    try:
        updated_script = pipeline.update_asset_description(
            script_id,
            request.asset_id,
            request.asset_type,
            request.description
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class SelectVariantRequest(BaseModel):
    asset_id: str
    asset_type: str
    variant_id: str
    generation_type: str = None  # For character: "full_body", "three_view", "headshot"

@app.post("/projects/{script_id}/assets/variant/select", response_model=Script)
async def select_asset_variant(script_id: str, request: SelectVariantRequest):
    """Selects a specific variant for an asset."""
    try:
        updated_script = pipeline.select_asset_variant(
            script_id,
            request.asset_id,
            request.asset_type,
            request.variant_id,
            request.generation_type
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DeleteVariantRequest(BaseModel):
    asset_id: str
    asset_type: str
    variant_id: str

@app.post("/projects/{script_id}/assets/variant/delete", response_model=Script)
async def delete_asset_variant(script_id: str, request: DeleteVariantRequest):
    """Deletes a specific variant from an asset."""
    try:
        updated_script = pipeline.delete_asset_variant(
            script_id,
            request.asset_id,
            request.asset_type,
            request.variant_id
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FavoriteVariantRequest(BaseModel):
    asset_id: str
    asset_type: str
    variant_id: str
    generation_type: Optional[str] = None  # For character: 'full_body', 'three_view', 'headshot'
    is_favorited: bool

@app.post("/projects/{script_id}/assets/variant/favorite", response_model=Script)
async def toggle_variant_favorite(script_id: str, request: FavoriteVariantRequest):
    """Toggles the favorite status of a variant. Favorited variants won't be auto-deleted when limit is reached."""
    try:
        updated_script = pipeline.toggle_variant_favorite(
            script_id,
            request.asset_id,
            request.asset_type,
            request.variant_id,
            request.is_favorited,
            request.generation_type
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{script_id}/model_settings", response_model=Script)
async def update_model_settings(script_id: str, request: UpdateModelSettingsRequest):
    """Update a project's model-instance references and aspect ratios."""
    try:
        updated_script = pipeline.update_model_settings(
            script_id,
            llm_instance_id=request.llm_instance_id,
            t2i_instance_id=request.t2i_instance_id,
            i2i_instance_id=request.i2i_instance_id,
            i2v_instance_id=request.i2v_instance_id,
            tts_instance_id=request.tts_instance_id,
            character_aspect_ratio=request.character_aspect_ratio,
            scene_aspect_ratio=request.scene_aspect_ratio,
            prop_aspect_ratio=request.prop_aspect_ratio,
            storyboard_aspect_ratio=request.storyboard_aspect_ratio,
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdatePromptConfigRequest(BaseModel):
    storyboard_polish: str = ""
    video_polish: str = ""
    r2v_polish: str = ""


@app.get("/projects/{script_id}/prompt_config")
async def get_prompt_config(script_id: str):
    """Returns project prompt_config and system default prompts for reference."""
    try:
        script = pipeline.get_script(script_id)
        if not script:
            raise HTTPException(status_code=404, detail="Project not found")
        config = script.prompt_config if hasattr(script, 'prompt_config') else PromptConfig()
        return {
            "prompt_config": config.model_dump(),
            "defaults": {
                "storyboard_polish": DEFAULT_STORYBOARD_POLISH_PROMPT,
                "video_polish": DEFAULT_VIDEO_POLISH_PROMPT,
                "r2v_polish": DEFAULT_R2V_POLISH_PROMPT,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/projects/{script_id}/prompt_config")
async def update_prompt_config(script_id: str, request: UpdatePromptConfigRequest):
    """Updates project custom prompt configuration. Empty string = use system default."""
    try:
        script = pipeline.get_script(script_id)
        if not script:
            raise HTTPException(status_code=404, detail="Project not found")
        script.prompt_config = PromptConfig(
            storyboard_polish=request.storyboard_polish,
            video_polish=request.video_polish,
            r2v_polish=request.r2v_polish,
        )
        pipeline._save_data()
        return {"prompt_config": script.prompt_config.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BindVoiceRequest(BaseModel):
    voice_id: str
    voice_name: str


@app.post("/projects/{script_id}/characters/{char_id}/voice", response_model=Script)
async def bind_voice(script_id: str, char_id: str, request: BindVoiceRequest):
    """Binds a voice to a character."""
    try:
        updated_script = pipeline.bind_voice(script_id, char_id, request.voice_id, request.voice_name)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateVoiceParamsRequest(BaseModel):
    speed: float = 1.0
    pitch: float = 1.0
    volume: int = 50


@app.put("/projects/{script_id}/characters/{char_id}/voice_params", response_model=Script)
async def update_voice_params(script_id: str, char_id: str, request: UpdateVoiceParamsRequest):
    """Updates voice parameters for a character."""
    script = pipeline.get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Script not found")
    char = next((c for c in script.characters if c.id == char_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    char.voice_speed = request.speed
    char.voice_pitch = request.pitch
    char.voice_volume = request.volume
    pipeline._save_data()
    return signed_response(script)


@app.get("/voices")
async def get_voices():
    """Returns list of available voices."""
    return pipeline.audio_generator.get_available_voices()


class GenerateLineAudioRequest(BaseModel):
    speed: float = 1.0
    pitch: float = 1.0
    volume: int = 50


@app.post("/projects/{script_id}/frames/{frame_id}/audio", response_model=Script)
async def generate_line_audio(script_id: str, frame_id: str, request: GenerateLineAudioRequest):
    """Generates audio for a specific frame with parameters."""
    try:
        updated_script = pipeline.generate_dialogue_line(script_id, frame_id, request.speed, request.pitch, request.volume)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/mix/generate_sfx", response_model=Script)
async def generate_mix_sfx(script_id: str):
    """Triggers Video-to-Audio SFX generation for all frames."""
    # Re-using generate_audio for now as it covers everything, 
    # but ideally we'd have granular methods in pipeline.
    # Let's just call generate_audio again, it's idempotent-ish.
    try:
        updated_script = pipeline.generate_audio(script_id)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/mix/generate_bgm", response_model=Script)
async def generate_mix_bgm(script_id: str):
    """Triggers BGM generation."""
    try:
        updated_script = pipeline.generate_audio(script_id)
        return signed_response(updated_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ToggleFrameLockRequest(BaseModel):
    frame_id: str


@app.post("/projects/{script_id}/frames/toggle_lock", response_model=Script)
async def toggle_frame_lock(script_id: str, request: ToggleFrameLockRequest):
    """Toggles the locked status of a frame."""
    try:
        updated_script = pipeline.toggle_frame_lock(
            script_id,
            request.frame_id
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateFrameRequest(BaseModel):
    frame_id: str
    image_prompt: Optional[str] = None
    action_description: Optional[str] = None
    dialogue: Optional[str] = None
    camera_angle: Optional[str] = None
    scene_id: Optional[str] = None
    character_ids: Optional[List[str]] = None
    # huobao-parity display metadata. ``model_fields_set`` is consulted in
    # the pipeline so the client can clear ``title`` (""→None) without
    # accidentally clearing it via "field not provided".
    title: Optional[str] = None
    duration_seconds: Optional[int] = Field(None, ge=1, le=60)


@app.post("/projects/{script_id}/frames/update", response_model=Script)
async def update_frame(script_id: str, request: UpdateFrameRequest):
    """Updates frame data (prompt, scene, characters, etc.)."""
    try:
        # Only forward keys the client actually set so partial updates work
        # for both the legacy fields and the new title / duration_seconds.
        forwarded = {
            k: getattr(request, k)
            for k in (
                "image_prompt", "action_description", "dialogue",
                "camera_angle", "scene_id", "character_ids",
                "title", "duration_seconds",
            )
            if k in request.model_fields_set
        }
        updated_script = pipeline.update_frame(script_id, request.frame_id, **forwarded)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AddFrameRequest(BaseModel):
    scene_id: Optional[str] = None
    action_description: str = ""
    camera_angle: str = "medium_shot"
    insert_at: Optional[int] = None

@app.post("/projects/{script_id}/frames", response_model=Script)
async def add_frame(script_id: str, request: AddFrameRequest):
    """Adds a new storyboard frame."""
    try:
        updated_script = pipeline.add_frame(
            script_id, 
            request.scene_id, 
            request.action_description, 
            request.camera_angle,
            request.insert_at
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/projects/{script_id}/frames/{frame_id}", response_model=Script)
async def delete_frame(script_id: str, frame_id: str):
    """Deletes a storyboard frame."""
    try:
        updated_script = pipeline.delete_frame(script_id, frame_id)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CopyFrameRequest(BaseModel):
    frame_id: str
    insert_at: Optional[int] = None

@app.post("/projects/{script_id}/frames/copy", response_model=Script)
async def copy_frame(script_id: str, request: CopyFrameRequest):
    """Copies a storyboard frame."""
    try:
        updated_script = pipeline.copy_frame(script_id, request.frame_id, request.insert_at)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ReorderFramesRequest(BaseModel):
    frame_ids: List[str]

@app.put("/projects/{script_id}/frames/reorder", response_model=Script)
async def reorder_frames(script_id: str, request: ReorderFramesRequest):
    """Reorders storyboard frames."""
    try:
        updated_script = pipeline.reorder_frames(script_id, request.frame_ids)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RenderFrameRequest(BaseModel):
    frame_id: str
    composition_data: Optional[Dict[str, Any]] = None
    prompt: str
    batch_size: int = 1


@app.post("/projects/{script_id}/storyboard/render", response_model=Script)
async def render_frame(script_id: str, request: RenderFrameRequest):
    """Renders a specific frame using composition data (I2I).

    Sync; can take 10–60s. Prefer the ``_async`` variant for production
    deployments behind short-timeout gateways.
    """
    try:
        logger.info(f"Rendering frame {request.frame_id}")

        updated_script = pipeline.generate_storyboard_render(
            script_id,
            request.frame_id,
            request.composition_data,
            request.prompt,
            request.batch_size
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error rendering frame {request.frame_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/storyboard/render_async")
async def render_frame_async(
    script_id: str, request: RenderFrameRequest, background_tasks: BackgroundTasks
):
    """Async variant of :pyfunc:`render_frame`. Returns ``_task_id``;
    client polls ``/tasks/{id}`` until completed (no result payload —
    on completion, refetch the project to get the new image)."""
    if not pipeline.get_script(script_id):
        raise HTTPException(status_code=404, detail="Script not found")
    return _async_oneshot(
        task_type="storyboard_render_single",
        script_id=script_id,
        work=lambda: pipeline.generate_storyboard_render(
            script_id, request.frame_id, request.composition_data,
            request.prompt, request.batch_size,
        ),
        background_tasks=background_tasks,
    )


class SelectVideoRequest(BaseModel):
    video_id: str


@app.post("/projects/{script_id}/frames/{frame_id}/select_video", response_model=Script)
async def select_video(script_id: str, frame_id: str, request: SelectVideoRequest):
    """Selects a video variant for a specific frame."""
    try:
        updated_script = pipeline.select_video_for_frame(script_id, frame_id, request.video_id)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExtractLastFrameRequest(BaseModel):
    video_task_id: str


@app.post("/projects/{script_id}/frames/{frame_id}/extract_last_frame")
async def extract_last_frame(script_id: str, frame_id: str, request: ExtractLastFrameRequest):
    """Extract the last frame from a completed video and add it as a variant to the frame's rendered_image_asset."""
    try:
        updated_script = pipeline.extract_last_frame(script_id, frame_id, request.video_task_id)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Error extracting last frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/frames/{frame_id}/upload_image")
async def upload_frame_image(script_id: str, frame_id: str, file: UploadFile = File(...)):
    """Upload an image as a variant for a frame's rendered_image_asset."""
    try:
        # Save file locally first
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(_current_pipeline_for_user().data_root, "uploads", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        updated_script = pipeline.upload_frame_image(script_id, frame_id, file_path)
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error uploading frame image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/merge")
async def merge_videos(script_id: str, background_tasks: BackgroundTasks):
    """Merge all selected frame videos into final output (async).

    FFmpeg can take 30 s – several minutes; running it synchronously
    used to trip upstream gateways with 504. The endpoint now returns
    ``_task_id`` immediately; client polls ``/tasks/{task_id}`` until
    ``status == 'completed'`` then reads ``output_url``.
    """
    try:
        script, task_id = pipeline.create_export_task(script_id, params={})
        _ctx_runtime.add_background_task(
            background_tasks, pipeline.process_export_task, task_id
        )
        response_data = script.model_dump() if hasattr(script, "model_dump") else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[MERGE ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")


# ===== Export Endpoint =====

class ExportRequest(BaseModel):
    resolution: str = "1080p"
    format: str = "mp4"
    subtitles: str = "none"
    force: bool = False  # True = re-merge even if cached merged_video_url exists


@app.post("/projects/{script_id}/export")
async def export_project(
    script_id: str, request: ExportRequest, background_tasks: BackgroundTasks
):
    """Export the project video (async).

    Wraps :pyfunc:`merge_videos` with the same task-polling pattern.
    ``resolution`` / ``format`` / ``subtitles`` are accepted for forward
    compatibility but not yet applied to the FFmpeg command.

    Returns ``{ ..., _task_id }``. Client polls ``/tasks/{task_id}``;
    when ``status == 'completed'``, ``output_url`` holds the merged file
    URL (relative path, sign via OSS if needed on the client).
    """
    try:
        script, task_id = pipeline.create_export_task(
            script_id, params={"force": request.force}
        )
        _ctx_runtime.add_background_task(
            background_tasks, pipeline.process_export_task, task_id
        )
        response_data = script.model_dump() if hasattr(script, "model_dump") else script.dict()
        response_data["_task_id"] = task_id
        return signed_response(response_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[EXPORT ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ===== Art Direction Endpoints =====

class AnalyzeStyleRequest(BaseModel):
    script_text: str


class SaveArtDirectionRequest(BaseModel):
    selected_style_id: str
    style_config: Dict[str, Any]
    custom_styles: List[Dict[str, Any]] = []
    ai_recommendations: List[Dict[str, Any]] = []


@app.post("/projects/{script_id}/art_direction/analyze")
async def analyze_script_for_styles(script_id: str, request: AnalyzeStyleRequest):
    """Analyze script content and recommend visual styles using LLM"""
    try:
        # Get the script to ensure it exists
        script = pipeline.get_script(script_id)
        if not script:
            raise HTTPException(status_code=404, detail="Script not found")

        recommendations = await _ctx_runtime.run_in_executor(
            None, pipeline.script_processor.analyze_script_for_styles, request.script_text,
        )

        return {"recommendations": recommendations}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.exception("An error occurred")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{script_id}/art_direction/analyze_async")
async def analyze_script_for_styles_async(
    script_id: str, request: AnalyzeStyleRequest, background_tasks: BackgroundTasks
):
    """Async variant of art-direction analyze. Returns ``_task_id``;
    poll ``/tasks/{id}`` for ``{recommendations: [...]}`` in ``status['result']``."""
    if not pipeline.get_script(script_id):
        raise HTTPException(status_code=404, detail="Script not found")
    return _async_oneshot(
        task_type="art_direction_analyze",
        script_id=script_id,
        work=lambda: {
            "recommendations": pipeline.script_processor.analyze_script_for_styles(
                request.script_text
            )
        },
        background_tasks=background_tasks,
    )


@app.post("/projects/{script_id}/art_direction/save", response_model=Script)
async def save_art_direction(script_id: str, request: SaveArtDirectionRequest):
    """Save Art Direction configuration to the project"""
    try:
        updated_script = pipeline.save_art_direction(
            script_id,
            request.selected_style_id,
            request.style_config,
            request.custom_styles,
            request.ai_recommendations
        )
        return signed_response(updated_script)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        logger.exception("An error occurred")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/art_direction/presets")
async def get_style_presets():
    """Get built-in style presets"""
    try:
        import json
        import os
        preset_file = os.path.join(os.path.dirname(__file__), "style_presets.json")
        logger.debug(f"Loading presets from {preset_file}")
        logger.debug(f"File exists: {os.path.exists(preset_file)}")

        if not os.path.exists(preset_file):
            logger.debug("DEBUG: Preset file not found!")
            return {"presets": []}

        with open(preset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {"presets": data}
    except Exception as e:
        import traceback
        logger.exception("An error occurred")
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: /storyboard/polish_prompt removed - use /storyboard/refine_prompt instead


def _get_custom_prompt(script_id: str, field: str) -> str:
    """Read a custom prompt with 3-level fallback: Episode → Series → system default.
    Returns empty string if result equals system default (so LLM method uses its built-in)."""
    if not script_id:
        return ""
    script = pipeline.get_script(script_id)
    if not script:
        return ""
    series = pipeline.get_series(script.series_id) if script.series_id else None
    effective = pipeline.get_effective_prompt(field, script, series)
    # If it's the system default, return empty so the LLM method uses its built-in default
    from .llm import DEFAULT_STORYBOARD_POLISH_PROMPT, DEFAULT_VIDEO_POLISH_PROMPT, DEFAULT_R2V_POLISH_PROMPT
    defaults = {
        "storyboard_polish": DEFAULT_STORYBOARD_POLISH_PROMPT,
        "video_polish": DEFAULT_VIDEO_POLISH_PROMPT,
        "r2v_polish": DEFAULT_R2V_POLISH_PROMPT,
    }
    if effective == defaults.get(field, ""):
        return ""
    return effective


class PolishVideoPromptRequest(BaseModel):
    draft_prompt: str
    feedback: str = Field("", max_length=2000)  # User feedback for iterative refinement
    script_id: str = ""  # Optional: project ID to load custom prompt config


@app.post("/video/polish_prompt")
async def polish_video_prompt(request: PolishVideoPromptRequest):
    """Polishes a video generation prompt using LLM. Returns bilingual prompts."""
    try:
        custom_prompt = _get_custom_prompt(request.script_id, "video_polish")
        processor = ScriptProcessor()
        result = processor.polish_video_prompt(request.draft_prompt, request.feedback, custom_prompt)
        return {
            "prompt_cn": result.get("prompt_cn", ""),
            "prompt_en": result.get("prompt_en", "")
        }
    except Exception as e:
        import traceback
        logger.exception("An error occurred")
        raise HTTPException(status_code=500, detail=str(e))


def _polish_video_work(req: "PolishVideoPromptRequest"):
    custom_prompt = _get_custom_prompt(req.script_id, "video_polish")
    processor = ScriptProcessor()
    result = processor.polish_video_prompt(req.draft_prompt, req.feedback, custom_prompt)
    return {
        "prompt_cn": result.get("prompt_cn", ""),
        "prompt_en": result.get("prompt_en", ""),
    }


@app.post("/video/polish_prompt_async")
async def polish_video_prompt_async(
    request: PolishVideoPromptRequest, background_tasks: BackgroundTasks
):
    """Async variant — poll ``/tasks/{id}`` for ``status['result']``."""
    return _async_oneshot(
        task_type="video_polish_prompt",
        script_id=request.script_id or None,
        work=lambda: _polish_video_work(request),
        background_tasks=background_tasks,
    )


class RefSlot(BaseModel):
    description: str  # Character name, e.g., "雷震", "白兔"


class PolishR2VPromptRequest(BaseModel):
    draft_prompt: str
    slots: List[RefSlot]
    feedback: str = Field("", max_length=2000)  # User feedback for iterative refinement
    script_id: str = ""  # Optional: project ID to load custom prompt config


@app.post("/video/polish_r2v_prompt")
async def polish_r2v_prompt(request: PolishR2VPromptRequest):
    """Polishes a R2V (Reference-to-Video) prompt using LLM. Returns bilingual prompts."""
    try:
        custom_prompt = _get_custom_prompt(request.script_id, "r2v_polish")
        processor = ScriptProcessor()
        slot_info = [{"description": s.description} for s in request.slots]
        result = processor.polish_r2v_prompt(request.draft_prompt, slot_info, request.feedback, custom_prompt)
        return {
            "prompt_cn": result.get("prompt_cn", ""),
            "prompt_en": result.get("prompt_en", "")
        }
    except Exception as e:
        import traceback
        logger.exception("An error occurred")
        raise HTTPException(status_code=500, detail=str(e))


def _polish_r2v_work(req: "PolishR2VPromptRequest"):
    custom_prompt = _get_custom_prompt(req.script_id, "r2v_polish")
    processor = ScriptProcessor()
    slot_info = [{"description": s.description} for s in req.slots]
    result = processor.polish_r2v_prompt(req.draft_prompt, slot_info, req.feedback, custom_prompt)
    return {
        "prompt_cn": result.get("prompt_cn", ""),
        "prompt_en": result.get("prompt_en", ""),
    }


@app.post("/video/polish_r2v_prompt_async")
async def polish_r2v_prompt_async(
    request: PolishR2VPromptRequest, background_tasks: BackgroundTasks
):
    """Async variant — poll ``/tasks/{id}`` for ``status['result']``."""
    return _async_oneshot(
        task_type="video_polish_r2v_prompt",
        script_id=request.script_id or None,
        work=lambda: _polish_r2v_work(request),
        background_tasks=background_tasks,
    )


# ===== Environment Configuration Endpoints =====

@app.get("/config/env")
async def get_env_config(_admin: _AuthUser = Depends(require_admin)):
    """Admin-only: read instance-wide environment configuration.

    For per-user keys (DashScope, OSS, Kling, Vidu, OpenAI), use
    ``GET /me/credentials`` instead.
    """
    try:
        from ...utils.endpoints import PROVIDER_DEFAULTS
        endpoint_overrides = {}
        for provider in PROVIDER_DEFAULTS:
            env_key = f"{provider}_BASE_URL"
            value = os.getenv(env_key)
            if value:
                endpoint_overrides[env_key] = value

        llm_provider_raw = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        llm_provider = llm_provider_raw if llm_provider_raw in ("dashscope", "openai") else "dashscope"

        return {
            "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY", ""),
            "ALIBABA_CLOUD_ACCESS_KEY_ID": os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", ""),
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET": os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", ""),
            "OSS_BUCKET_NAME": os.getenv("OSS_BUCKET_NAME", ""),
            "OSS_ENDPOINT": os.getenv("OSS_ENDPOINT", ""),
            "OSS_BASE_PATH": os.getenv("OSS_BASE_PATH", ""),
            "KLING_ACCESS_KEY": os.getenv("KLING_ACCESS_KEY", ""),
            "KLING_SECRET_KEY": os.getenv("KLING_SECRET_KEY", ""),
            "VIDU_API_KEY": os.getenv("VIDU_API_KEY", ""),
            "KLING_PROVIDER_MODE": _normalize_provider_mode(os.getenv("KLING_PROVIDER_MODE")),
            "VIDU_PROVIDER_MODE": _normalize_provider_mode(os.getenv("VIDU_PROVIDER_MODE")),
            "PIXVERSE_PROVIDER_MODE": _normalize_provider_mode(os.getenv("PIXVERSE_PROVIDER_MODE")),
            "LLM_PROVIDER": llm_provider,
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
            "API_HOST": os.getenv("API_HOST", ""),
            "API_PORT": os.getenv("API_PORT", ""),
            "endpoint_overrides": endpoint_overrides,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# ============================================
# Prop CRUD Endpoints
# ============================================

class CreatePropRequest(BaseModel):
    name: str
    description: str = ""

@app.post("/projects/{script_id}/props")
async def create_prop(script_id: str, request: CreatePropRequest):
    """Creates a new prop in the project."""
    script = pipeline.get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Project not found")

    import uuid
    from .models import Prop, GenerationStatus

    new_prop = Prop(
        id=f"prop_{uuid.uuid4().hex[:8]}",
        name=request.name,
        description=request.description,
        status=GenerationStatus.PENDING
    )

    script.props.append(new_prop)
    script.updated_at = time.time()
    pipeline._save_data()

    return signed_response(script)


@app.delete("/projects/{script_id}/props/{prop_id}")
async def delete_prop(script_id: str, prop_id: str):
    """Deletes a prop from the project."""
    script = pipeline.get_script(script_id)
    if not script:
        raise HTTPException(status_code=404, detail="Project not found")

    original_count = len(script.props)
    script.props = [p for p in script.props if p.id != prop_id]

    if len(script.props) == original_count:
        raise HTTPException(status_code=404, detail="Prop not found")

    # Remove prop references from frames
    for frame in script.frames:
        if prop_id in frame.prop_ids:
            frame.prop_ids.remove(prop_id)

    script.updated_at = time.time()
    pipeline._save_data()

    return signed_response(script)
