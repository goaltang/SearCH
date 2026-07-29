"""Unit tests for photofinder.index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from photofinder.index import FaceIndex


class FakeEngine:
    def __init__(self):
        self.calls: list[Path] = []

    def process_file(self, path: Path):
        self.calls.append(path)
        return [
            {"bbox": [0, 0, 30, 30, 0.9],
             "det_score": 0.9,
             "embedding": np.ones(512, dtype=np.float32) * 0.5}
        ]


def test_index_loads_existing_data(tmp_path: Path):
    embeddings = np.array([[1.0] * 512, [2.0] * 512], dtype=np.float32)
    faces = [
        {"photo_id": 1, "bbox": [0, 0, 10, 10], "det_score": 0.9},
        {"photo_id": 2, "bbox": [0, 0, 20, 20], "det_score": 0.8},
    ]
    np.savez_compressed(tmp_path / "faces.npz", embeddings=embeddings)
    (tmp_path / "faces.json").write_text(json.dumps(faces), encoding="utf-8")
    (tmp_path / "done.json").write_text(json.dumps([1, 2]), encoding="utf-8")

    index = FaceIndex(tmp_path)
    assert len(index.faces) == 2
    assert index.done_ids == {1, 2}
    np.testing.assert_array_equal(index.embeddings, embeddings)


def test_index_saves_data(tmp_path: Path):
    index = FaceIndex(tmp_path)
    index.embeddings = np.array([[3.0] * 512], dtype=np.float32)
    index.faces = [{"photo_id": 3, "bbox": [0, 0, 30, 30], "det_score": 0.9}]
    index.done_ids = {3}
    index.save()

    loaded = FaceIndex(tmp_path)
    assert loaded.done_ids == {3}
    assert len(loaded.faces) == 1
    np.testing.assert_array_equal(loaded.embeddings, index.embeddings)


def test_index_skips_missing_paths(tmp_path: Path, monkeypatch, caplog):
    index = FaceIndex(tmp_path)
    engine = FakeEngine()
    missing = tmp_path / "missing.jpg"
    valid = tmp_path / "valid.jpg"
    valid.write_bytes(b"thumb")

    thumbs = {1: missing, 2: valid}
    with caplog.at_level("WARNING", logger="photofinder.index"):
        processed = index.build(engine, thumbs, workers=1)

    assert processed == 1
    assert valid in engine.calls
    assert missing not in engine.calls
    assert any("missing/invalid thumbs" in rec.message for rec in caplog.records)


def test_index_search_returns_best_score_per_photo(tmp_path: Path):
    index = FaceIndex(tmp_path)
    # two faces on the same photo, one with higher similarity
    embs = np.array([
        [1.0] + [0.0] * 511,
        [0.5] + [0.0] * 511,
        [0.0] + [1.0] * 511,
    ], dtype=np.float32)
    index.embeddings = embs
    index.faces = [
        {"photo_id": 1, "bbox": [0, 0, 10, 10], "det_score": 0.9},
        {"photo_id": 1, "bbox": [0, 0, 20, 20], "det_score": 0.8},
        {"photo_id": 2, "bbox": [0, 0, 30, 30], "det_score": 0.9},
    ]

    ref = np.array([1.0] + [0.0] * 511, dtype=np.float32)
    results = index.search([ref], threshold=0.0)

    assert len(results) == 2
    by_id = {r["photo_id"]: r["score"] for r in results}
    # photo 1 best face has cosine 1.0 with ref
    assert by_id[1] == pytest.approx(1.0, abs=1e-5)
    # photo 2 best face is orthogonal -> 0.0
    assert by_id[2] == pytest.approx(0.0, abs=1e-5)


def test_index_search_returns_empty_for_no_embeddings(tmp_path: Path):
    index = FaceIndex(tmp_path)
    assert index.search([np.ones(512, dtype=np.float32)]) == []


def test_index_reset_clears_data(tmp_path: Path):
    index = FaceIndex(tmp_path)
    index.embeddings = np.ones((1, 512), dtype=np.float32)
    index.faces = [{"photo_id": 1}]
    index.done_ids = {1}
    index.save()
    assert (tmp_path / "faces.npz").exists()

    index.reset()
    assert len(index.faces) == 0
    assert len(index.embeddings) == 0
    assert index.done_ids == set()
    assert not (tmp_path / "faces.npz").exists()


def test_index_recovers_from_corrupt_data(tmp_path: Path, caplog):
    embeddings = np.ones((1, 512), dtype=np.float32)
    np.savez_compressed(tmp_path / "faces.npz", embeddings=embeddings)
    (tmp_path / "faces.json").write_text(
        json.dumps([{"photo_id": 1}, {"photo_id": 2}]), encoding="utf-8")

    with caplog.at_level("ERROR", logger="photofinder.index"):
        index = FaceIndex(tmp_path)

    assert len(index.faces) == 0
    assert len(index.embeddings) == 0
    assert not (tmp_path / "faces.npz").exists()
    assert any("Corrupt index" in rec.message for rec in caplog.records)
