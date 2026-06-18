"""
Generator stage — thin client over a headless ComfyUI server.

Calque's worker does not load diffusion models itself; it drives ComfyUI via
its HTTP API. This module handles the mechanics that every workflow shares:

  * upload an input image (and the precision/strength map) to ComfyUI
  * submit a workflow (API-format JSON) to /prompt
  * track progress over the WebSocket (/ws)
  * pull the finished image back from /history + /view

Because a render can take minutes-to-hours (hi-res tiled upscaling), generation
is a *job*: submit returns a job id, and callers poll status. The worker's HTTP
layer (server.py) exposes that as POST /generate -> {job_id}, GET /generate/{id}.

Workflow JSON templates live in calque/workflows/*.json with placeholder tokens
the worker fills in (input image name, prompts, seed, the precision map, etc.).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
WORKFLOWS_DIR = Path(__file__).parent / "workflows"


@dataclass
class GenResult:
    images: list[bytes] = field(default_factory=list)
    prompt_id: str = ""


class ComfyClient:
    def __init__(self, base_url: str = DEFAULT_COMFY_URL, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        # A stable client id lets us filter our own WS messages.
        self.client_id = uuid.uuid4().hex

    # ---- health ---------------------------------------------------------
    def is_up(self) -> bool:
        try:
            r = requests.get(f"{self.base}/system_stats", timeout=self.timeout)
            return r.ok
        except requests.RequestException:
            return False

    # ---- inputs ---------------------------------------------------------
    def upload_image(self, data: bytes, filename: str, overwrite: bool = True) -> str:
        """Upload bytes to ComfyUI's input folder; returns the stored name."""
        files = {"image": (filename, data, "image/png")}
        payload = {"overwrite": "true" if overwrite else "false"}
        r = requests.post(
            f"{self.base}/upload/image", files=files, data=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["name"]

    # ---- workflow templates --------------------------------------------
    @staticmethod
    def load_workflow(name: str, replacements: dict[str, Any]) -> dict:
        """
        Load a workflow template and substitute @TOKEN@ placeholders.

        Tokens are replaced in the raw JSON text so values keep their JSON type
        (numbers stay numbers, strings get quoted by being written as "@TOK@").
        """
        text = (WORKFLOWS_DIR / f"{name}.json").read_text(encoding="utf-8")
        for token, value in replacements.items():
            text = text.replace(f"@{token}@", json.dumps(value).strip('"')
                                if isinstance(value, str) else json.dumps(value))
        return json.loads(text)

    # ---- execution ------------------------------------------------------
    def submit(self, workflow: dict) -> str:
        """Queue a workflow; returns prompt_id."""
        r = requests.post(
            f"{self.base}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait(
        self,
        prompt_id: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        poll_interval: float = 1.5,
        max_seconds: float = 6 * 3600,
    ) -> GenResult:
        """Poll /history until the prompt completes, then fetch its images."""
        deadline = time.time() + max_seconds
        while time.time() < deadline:
            hist = requests.get(
                f"{self.base}/history/{prompt_id}", timeout=self.timeout
            ).json()
            if prompt_id in hist:
                return self._collect(prompt_id, hist[prompt_id])
            time.sleep(poll_interval)
        raise TimeoutError(f"ComfyUI job {prompt_id} exceeded {max_seconds}s")

    def _collect(self, prompt_id: str, entry: dict) -> GenResult:
        result = GenResult(prompt_id=prompt_id)
        outputs = entry.get("outputs", {})
        for node_out in outputs.values():
            for img in node_out.get("images", []):
                params = {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                }
                r = requests.get(f"{self.base}/view", params=params, timeout=self.timeout)
                r.raise_for_status()
                result.images.append(r.content)
        return result

    # ---- one-shot convenience ------------------------------------------
    def run(self, workflow: dict, **wait_kwargs) -> GenResult:
        return self.wait(self.submit(workflow), **wait_kwargs)
