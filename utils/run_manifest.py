"""Run manifest helpers for reproducible processing outputs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


_REDACTED_KEYS = ("token", "password", "secret", "api_key", "access_key")
_VERSION_PACKAGES = [
    "torch",
    "torchvision",
    "lightglue",
    "numpy",
    "Pillow",
    "opencv-python",
    "pandas",
    "requests",
    "transformers",
]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _redacted_args(args: Any) -> dict[str, Any]:
    values = vars(args).copy()
    redacted: dict[str, Any] = {}
    for key, value in values.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _REDACTED_KEYS) and value:
            redacted[key] = "***redacted***"
        else:
            redacted[key] = _json_safe(value)
    return redacted


def _should_redact_name(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    return any(marker in lowered for marker in _REDACTED_KEYS)


def _redacted_command(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            redacted.append("***redacted***")
            redact_next = False
            continue
        if value.startswith("--") and "=" in value:
            flag, raw = value.split("=", 1)
            redacted.append(f"{flag}=***redacted***" if _should_redact_name(flag) and raw else value)
            continue
        redacted.append(value)
        if value.startswith("--") and _should_redact_name(value):
            redact_next = True
    return redacted


def _git_value(project_root: Path, *git_args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *git_args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _VERSION_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write_run_manifest(
    *,
    path: str | Path,
    project_root: str | Path,
    script_name: str,
    args: Any,
    started_at: datetime,
    status: str = "completed",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON sidecar describing how a processing run was produced."""
    manifest_path = Path(path)
    if manifest_path.parent != Path("."):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    root = Path(project_root)
    status_short = _git_value(root, "status", "--short")
    manifest = {
        "schema_version": 1,
        "status": status,
        "script": script_name,
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": _redacted_command(sys.argv),
        "cwd": str(Path.cwd()),
        "project_root": str(root),
        "git": {
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "branch": _git_value(root, "branch", "--show-current"),
            "dirty": bool(status_short),
            "status_short": status_short or "",
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": _dependency_versions(),
        "settings": _redacted_args(args),
        "extra": _json_safe(extra or {}),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path
