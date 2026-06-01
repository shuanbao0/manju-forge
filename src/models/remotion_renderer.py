"""
RemotionRenderClient — Facade over the Node Remotion render microservice.

The actual rendering (headless Chrome, ``@remotion/renderer``, pre-bundled
composition) lives in a long-lived Node service (see ``remotion/server.mjs``).
This client hides the HTTP transport and the backend↔renderer path mapping
behind one method: ``render(spec, output_path) -> seconds``.

Path contract
=============
Backend and renderer share the same ``output/`` directory (a mounted volume in
Docker, the same folder locally). To stay correct across the container
boundary we never send absolute paths — we send a path **relative to the
output root**. The renderer joins it against its own output root to decide
where to write the mp4, and serves that same root statically so image layers
(``KenBurnsImage.src``) can be loaded by relative path.
"""
from __future__ import annotations

import logging
import os
from typing import Union

import requests

from .remotion_spec import VideoSpec

logger = logging.getLogger(__name__)

DEFAULT_RENDER_URL = "http://localhost:3001"
DEFAULT_TIMEOUT = 600


def _default_output_root() -> str:
    return os.path.abspath(os.environ.get("REMOTION_OUTPUT_ROOT") or "output")


class RemotionRenderError(RuntimeError):
    """Raised when the render service reports a failure or is unreachable."""


class RemotionRenderClient:
    """Talks to the Remotion render microservice over HTTP."""

    def __init__(
        self,
        render_url: str | None = None,
        output_root: str | None = None,
        timeout: int | None = None,
    ):
        self.render_url = (
            render_url or os.environ.get("REMOTION_RENDER_URL") or DEFAULT_RENDER_URL
        ).rstrip("/")
        self.output_root = os.path.abspath(output_root or _default_output_root())
        self.timeout = timeout or int(os.environ.get("REMOTION_RENDER_TIMEOUT") or DEFAULT_TIMEOUT)

    # ── public API ───────────────────────────────────────────────────────

    def health(self) -> bool:
        """Return True if the render service answers ``/health`` with 200."""
        try:
            resp = requests.get(f"{self.render_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.debug("Remotion render service health check failed: %s", e)
            return False

    def render(self, spec: Union[VideoSpec, dict], output_path: str) -> float:
        """Render ``spec`` to ``output_path``; return generation seconds.

        ``output_path`` is an absolute/relative path on the *backend* side; we
        translate it to a path relative to the shared output root so the
        renderer writes to the matching location on its side.
        """
        payload_spec = spec.to_payload() if isinstance(spec, VideoSpec) else spec
        output_rel = self._to_output_rel(output_path)

        body = {"spec": payload_spec, "outputRel": output_rel}
        try:
            resp = requests.post(
                f"{self.render_url}/render", json=body, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise RemotionRenderError(
                f"Remotion render service unreachable at {self.render_url}: {e}"
            ) from e

        if resp.status_code != 200:
            raise RemotionRenderError(
                f"Remotion render failed ({resp.status_code}): {resp.text[:500]}"
            )

        data = resp.json()
        if not data.get("ok"):
            raise RemotionRenderError(f"Remotion render failed: {data.get('error')}")

        # Self-verify the cross-process path contract: the renderer reported
        # success, but if its REMOTION_OUTPUT_ROOT differs from ours the MP4
        # landed somewhere we will never be able to serve. Turn that silent
        # broken-<video> into an actionable error here.
        abs_out = os.path.abspath(output_path)
        if not os.path.isfile(abs_out):
            raise RemotionRenderError(
                f"渲染服务报告成功,但成片未出现在 {abs_out}。通常是渲染服务的 "
                f"REMOTION_OUTPUT_ROOT({self.output_root})与后端 output 根不一致 —— "
                f"请让两端指向同一目录。"
            )
        return float(data.get("seconds", 0.0))

    # ── helpers ──────────────────────────────────────────────────────────

    def _to_output_rel(self, output_path: str) -> str:
        """Path of ``output_path`` relative to the shared output root.

        Uses POSIX separators on the wire so the renderer (which may run on a
        different OS in dev) resolves it consistently.
        """
        rel = os.path.relpath(os.path.abspath(output_path), self.output_root)
        return rel.replace(os.sep, "/")


_DEFAULT_CLIENT: RemotionRenderClient | None = None


def get_render_client() -> RemotionRenderClient:
    """Process-wide singleton — cheap to reuse, holds no open connections."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = RemotionRenderClient()
    return _DEFAULT_CLIENT


__all__ = ["RemotionRenderClient", "RemotionRenderError", "get_render_client"]
