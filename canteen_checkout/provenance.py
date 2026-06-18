from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def git_sha(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_run_provenance(
    output_path: Path,
    *,
    project_root: Path,
    model_key: str,
    model_path: Path,
    dataset_archive: Path,
    dataset_manifest: Path,
    hyperparameters: dict[str, Any],
    training_seconds: float,
) -> dict[str, Any]:
    lock_path = project_root / "dvc.lock"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_key": model_key,
        "git_sha": git_sha(project_root),
        "dvc_lock": artifact_record(lock_path),
        "dataset": {
            "archive": artifact_record(dataset_archive),
            "manifest": artifact_record(dataset_manifest),
        },
        "hyperparameters": hyperparameters,
        "model": artifact_record(model_path),
        "training_seconds": round(float(training_seconds), 3),
    }
    write_json(output_path, payload)
    return json_safe(payload)
