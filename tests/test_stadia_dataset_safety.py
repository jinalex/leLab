"""Filesystem-only durability tests for Stadia upload eligibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lelab.stadia.dataset_safety import (
    DatasetSafetyManifest,
    delete_dataset_safety_manifest,
    manifest_path,
    read_dataset_safety_manifest,
    write_dataset_safety_manifest,
)


def test_manifest_round_trip_is_atomic_and_identity_bound(tmp_path: Path) -> None:
    manifest = DatasetSafetyManifest(
        dataset_repo_id="alex/demo_20260902_040000",
        session_id="session-1",
        dataset_safe=True,
        dataset_finalized=True,
        dataset_uploaded=False,
        saved_episodes=2,
    )

    path = write_dataset_safety_manifest(manifest, dataset_home=tmp_path)
    loaded = read_dataset_safety_manifest(manifest.dataset_repo_id, dataset_home=tmp_path)

    assert loaded == manifest
    assert loaded is not None and loaded.upload_allowed
    assert path == manifest_path(manifest.dataset_repo_id, dataset_home=tmp_path)
    assert list(path.parent.glob("*.tmp")) == []
    assert delete_dataset_safety_manifest(manifest.dataset_repo_id, dataset_home=tmp_path)
    assert read_dataset_safety_manifest(manifest.dataset_repo_id, dataset_home=tmp_path) is None


def test_malformed_or_mismatched_manifest_fails_closed(tmp_path: Path) -> None:
    repo_id = "alex/demo_20260902_040000"
    path = manifest_path(repo_id, dataset_home=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid schema"):
        read_dataset_safety_manifest(repo_id, dataset_home=tmp_path)

    valid = DatasetSafetyManifest(
        dataset_repo_id=repo_id,
        session_id="session-1",
        dataset_safe=False,
        dataset_finalized=False,
        dataset_uploaded=False,
        saved_episodes=0,
        error="rollback unproven",
    ).as_dict()
    valid["dataset_repo_id"] = "alex/other"
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_dataset_safety_manifest(repo_id, dataset_home=tmp_path)


def test_uploaded_manifest_requires_safe_finalized_dataset() -> None:
    with pytest.raises(ValueError, match="uploaded dataset"):
        DatasetSafetyManifest(
            dataset_repo_id="alex/demo",
            session_id="session-1",
            dataset_safe=False,
            dataset_finalized=False,
            dataset_uploaded=True,
            saved_episodes=1,
        )
