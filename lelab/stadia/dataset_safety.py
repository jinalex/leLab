"""Durable, dependency-neutral upload eligibility for Stadia datasets."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_SAFETY_SCHEMA_VERSION = 1
DATASET_SAFETY_DIRECTORY = ".lelab-stadia-safety"
_MANIFEST_KEYS = {
    "schema_version",
    "dataset_repo_id",
    "session_id",
    "dataset_safe",
    "dataset_finalized",
    "dataset_uploaded",
    "saved_episodes",
    "error",
    "updated_at_utc",
}


def default_dataset_home() -> Path:
    configured = os.getenv("HF_LEROBOT_HOME")
    if configured:
        return Path(configured).expanduser()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot"
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    cache = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache / "huggingface" / "lerobot"


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class DatasetSafetyManifest:
    dataset_repo_id: str
    session_id: str
    dataset_safe: bool
    dataset_finalized: bool
    dataset_uploaded: bool
    saved_episodes: int
    error: str | None = None
    updated_at_utc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_repo_id", _nonempty_text(self.dataset_repo_id, "dataset_repo_id"))
        object.__setattr__(self, "session_id", _nonempty_text(self.session_id, "session_id"))
        for field_name in ("dataset_safe", "dataset_finalized", "dataset_uploaded"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        if (
            isinstance(self.saved_episodes, bool)
            or not isinstance(self.saved_episodes, int)
            or self.saved_episodes < 0
        ):
            raise ValueError("saved_episodes must be a non-negative integer")
        if self.error is not None:
            object.__setattr__(self, "error", _nonempty_text(self.error, "error"))
        timestamp = self.updated_at_utc or datetime.now(UTC).isoformat()
        _nonempty_text(timestamp, "updated_at_utc")
        object.__setattr__(self, "updated_at_utc", timestamp)
        if self.dataset_uploaded and not (self.dataset_safe and self.dataset_finalized):
            raise ValueError("an uploaded dataset must be safe and finalized")

    @property
    def upload_allowed(self) -> bool:
        return (
            self.dataset_safe
            and self.dataset_finalized
            and not self.dataset_uploaded
            and self.saved_episodes > 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_SAFETY_SCHEMA_VERSION,
            "dataset_repo_id": self.dataset_repo_id,
            "session_id": self.session_id,
            "dataset_safe": self.dataset_safe,
            "dataset_finalized": self.dataset_finalized,
            "dataset_uploaded": self.dataset_uploaded,
            "saved_episodes": self.saved_episodes,
            "error": self.error,
            "updated_at_utc": self.updated_at_utc,
        }


def manifest_path(dataset_repo_id: str, *, dataset_home: Path | None = None) -> Path:
    identity = _nonempty_text(dataset_repo_id, "dataset_repo_id")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (dataset_home or default_dataset_home()) / DATASET_SAFETY_DIRECTORY / f"{digest}.json"


def write_dataset_safety_manifest(
    manifest: DatasetSafetyManifest,
    *,
    dataset_home: Path | None = None,
) -> Path:
    if not isinstance(manifest, DatasetSafetyManifest):
        raise TypeError("manifest must be DatasetSafetyManifest")
    path = manifest_path(manifest.dataset_repo_id, dataset_home=dataset_home)
    directory = path.parent
    if directory.is_symlink():
        raise OSError(f"dataset safety directory must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise OSError(f"dataset safety directory is not a real directory: {directory}")
    if path.is_symlink():
        raise OSError("dataset safety manifest target must not be a symlink")
    payload = json.dumps(manifest.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    return path


def read_dataset_safety_manifest(
    dataset_repo_id: str,
    *,
    dataset_home: Path | None = None,
) -> DatasetSafetyManifest | None:
    path = manifest_path(dataset_repo_id, dataset_home=dataset_home)
    if path.is_symlink():
        raise ValueError("dataset safety manifest must be a regular non-symlink file")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("dataset safety manifest must be a regular non-symlink file")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
        raise ValueError("dataset safety manifest has an invalid schema")
    if raw["schema_version"] != DATASET_SAFETY_SCHEMA_VERSION:
        raise ValueError("dataset safety manifest version is unsupported")
    if raw["dataset_repo_id"] != dataset_repo_id:
        raise ValueError("dataset safety manifest identity does not match its lookup key")
    # Reject JSON's non-finite numeric extensions even though this schema has
    # no floats; it keeps future extensions from silently accepting them.
    if any(isinstance(value, float) and not math.isfinite(value) for value in raw.values()):
        raise ValueError("dataset safety manifest contains a non-finite value")
    return DatasetSafetyManifest(
        dataset_repo_id=raw["dataset_repo_id"],
        session_id=raw["session_id"],
        dataset_safe=raw["dataset_safe"],
        dataset_finalized=raw["dataset_finalized"],
        dataset_uploaded=raw["dataset_uploaded"],
        saved_episodes=raw["saved_episodes"],
        error=raw["error"],
        updated_at_utc=raw["updated_at_utc"],
    )


def delete_dataset_safety_manifest(
    dataset_repo_id: str,
    *,
    dataset_home: Path | None = None,
) -> bool:
    path = manifest_path(dataset_repo_id, dataset_home=dataset_home)
    if path.is_symlink():
        raise ValueError("dataset safety manifest must be a regular non-symlink file")
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError("dataset safety manifest must be a regular non-symlink file")
    path.unlink()
    return True


__all__ = [
    "DATASET_SAFETY_DIRECTORY",
    "DATASET_SAFETY_SCHEMA_VERSION",
    "DatasetSafetyManifest",
    "default_dataset_home",
    "delete_dataset_safety_manifest",
    "manifest_path",
    "read_dataset_safety_manifest",
    "write_dataset_safety_manifest",
]
