"""Unit tests for photofinder.pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from photofinder import pipeline as pipeline_mod
from photofinder.crawler import Photo
from photofinder.pipeline import MatchResult, PhotoFinder


# --------------------------------------------------------------------------- fakes

class FakeEngine:
    def __init__(self):
        self.process_calls = []
        self.process_file_calls = []

    def process(self, img):
        self.process_calls.append(img)
        if img is None:
            return []
        # return two faces; the largest is used as reference
        return [
            {"bbox": [0, 0, 10, 10, 0.95], "embedding": np.ones(512, dtype=np.float32) * 0.5},
            {"bbox": [0, 0, 5, 5, 0.70], "embedding": np.ones(512, dtype=np.float32) * 0.2},
        ]

    def process_file(self, path: Path):
        self.process_file_calls.append(path)
        return [
            {"bbox": [0, 0, 30, 30, 0.90], "embedding": np.ones(512, dtype=np.float32) * 0.8}
        ]


class FakeCrawler:
    def __init__(self, order_id: str, cache_root, pwd=None, tag_id=None):
        self.order_id = order_id
        self.cache_root = Path(cache_root)
        self.pwd = pwd
        self.tag_id = tag_id
        self.dir = self.cache_root / order_id
        self._photos = []
        self._thumbs = {}

    def set_data(self, photos, thumbs):
        self._photos = photos
        self._thumbs = thumbs

    def get_metadata(self, max_photos=None, refresh=False, progress_cb=None):
        if progress_cb:
            progress_cb("meta", len(self._photos), len(self._photos))
        if max_photos:
            return self._photos[:max_photos]
        return self._photos

    def download_thumbs(self, photos, progress_cb=None):
        if progress_cb:
            progress_cb("download", len(photos), len(photos))
        return {p.photo_id: self._thumbs[p.photo_id] for p in photos}


class FakeFaceIndex:
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self._hits = []

    def set_hits(self, hits):
        self._hits = hits

    def build(self, engine, thumbs, workers=4, min_face=24.0, progress_cb=None):
        if progress_cb:
            progress_cb("index", len(thumbs), len(thumbs))

    def search(self, ref_embeddings, threshold=0.35, top_k=None):
        return self._hits


# --------------------------------------------------------------------------- tests

def test_extract_reference_returns_largest_face_embedding():
    finder = PhotoFinder(cache_root=Path("/tmp"))
    finder._engine = FakeEngine()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    embs, qualities = finder.extract_reference(img)
    assert len(embs) == 1
    assert len(qualities) == 1
    assert qualities[0].used is True
    np.testing.assert_array_equal(embs[0], np.ones(512, dtype=np.float32) * 0.5)


def test_extract_reference_from_file_path(tmp_path: Path):
    ref_path = tmp_path / "ref.jpg"
    cv2.imwrite(str(ref_path), np.zeros((100, 100, 3), dtype=np.uint8))

    finder = PhotoFinder(cache_root=tmp_path)
    finder._engine = FakeEngine()
    embs, qualities = finder.extract_reference(ref_path)
    assert len(embs) == 1


def test_extract_reference_accepts_multiple_images():
    finder = PhotoFinder(cache_root=Path("/tmp"))
    finder._engine = FakeEngine()
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8)

    embs, qualities = finder.extract_reference([img1, img2])
    assert len(embs) == 2
    assert len(qualities) == 2
    # each image contributes the largest face embedding
    np.testing.assert_array_equal(embs[0], np.ones(512, dtype=np.float32) * 0.5)


def test_extract_reference_raises_when_image_unreadable():
    finder = PhotoFinder(cache_root=Path("/tmp"))
    finder._engine = FakeEngine()
    with pytest.raises(ValueError, match="no reference image provided"):
        finder.extract_reference(None)


def test_extract_reference_raises_when_no_face_detected():
    finder = PhotoFinder(cache_root=Path("/tmp"))
    engine = FakeEngine()
    engine.process = lambda img: []
    finder._engine = engine

    with pytest.raises(ValueError, match="no valid reference face found"):
        finder.extract_reference(np.zeros((100, 100, 3), dtype=np.uint8))


def test_run_returns_matches(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "AlbumCrawler", FakeCrawler)
    monkeypatch.setattr(pipeline_mod, "FaceIndex", FakeFaceIndex)

    finder = PhotoFinder(cache_root=tmp_path)
    finder._engine = FakeEngine()

    photos = [
        Photo(photo_id=1, fname="a.jpg", width=100, height=100,
              detect_url="http://d/1", preview_url="http://p/1", full_url="http://f/1"),
        Photo(photo_id=2, fname="b.jpg", width=100, height=100,
              detect_url="http://d/2", preview_url="http://p/2", full_url="http://f/2"),
    ]
    thumbs = {
        1: tmp_path / "1.jpg",
        2: tmp_path / "2.jpg",
    }
    for p in thumbs.values():
        p.write_bytes(b"thumb")

    crawler = FakeCrawler("999", tmp_path)
    crawler.set_data(photos, thumbs)

    # override the constructor to return our configured instance
    monkeypatch.setattr(pipeline_mod, "AlbumCrawler", lambda *args, **kwargs: crawler)

    index = FakeFaceIndex(crawler.dir)
    index.set_hits([{"photo_id": 2, "score": 0.78}])
    monkeypatch.setattr(pipeline_mod, "FaceIndex", lambda *args, **kwargs: index)

    results = finder.run("https://example.com/?orderId=999", np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, MatchResult)
    assert result.photo_id == 2
    assert result.fname == "b.jpg"
    assert result.score == pytest.approx(0.78, rel=1e-4)
    assert "orderId=999" in result.album_url
    assert result.preview_url == "http://p/2"
    assert result.full_url == "http://f/2"
    assert result.thumb_path == str(thumbs[2])


def test_run_respects_max_photos(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "AlbumCrawler", FakeCrawler)
    monkeypatch.setattr(pipeline_mod, "FaceIndex", FakeFaceIndex)

    finder = PhotoFinder(cache_root=tmp_path)
    finder._engine = FakeEngine()

    photos = [Photo(photo_id=i, fname=f"{i}.jpg", width=100, height=100,
                    detect_url=f"d{i}", preview_url=f"p{i}", full_url=f"f{i}")
              for i in range(1, 5)]
    thumbs = {p.photo_id: tmp_path / f"{p.photo_id}.jpg" for p in photos}
    for p in thumbs.values():
        p.write_bytes(b"thumb")

    crawler = FakeCrawler("111", tmp_path)
    crawler.set_data(photos, thumbs)
    monkeypatch.setattr(pipeline_mod, "AlbumCrawler", lambda *args, **kwargs: crawler)

    index = FakeFaceIndex(crawler.dir)
    index.set_hits([])
    monkeypatch.setattr(pipeline_mod, "FaceIndex", lambda *args, **kwargs: index)

    results = finder.run("111", np.zeros((100, 100, 3), dtype=np.uint8), max_photos=2)
    assert len(results) == 0
    # metadata was truncated
    assert len(crawler.get_metadata(max_photos=2)) == 2


def test_run_reports_progress(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "AlbumCrawler", FakeCrawler)
    monkeypatch.setattr(pipeline_mod, "FaceIndex", FakeFaceIndex)

    finder = PhotoFinder(cache_root=tmp_path)
    finder._engine = FakeEngine()

    photos = [Photo(photo_id=1, fname="a.jpg", width=100, height=100,
                    detect_url="d", preview_url="p", full_url="f")]
    thumbs = {1: tmp_path / "1.jpg"}
    thumbs[1].write_bytes(b"thumb")

    crawler = FakeCrawler("222", tmp_path)
    crawler.set_data(photos, thumbs)
    monkeypatch.setattr(pipeline_mod, "AlbumCrawler", lambda *args, **kwargs: crawler)

    index = FakeFaceIndex(crawler.dir)
    index.set_hits([{"photo_id": 1, "score": 0.88}])
    monkeypatch.setattr(pipeline_mod, "FaceIndex", lambda *args, **kwargs: index)

    progress = []
    finder.run("222", np.zeros((100, 100, 3), dtype=np.uint8), progress_cb=lambda stage, done, total: progress.append(stage))
    assert "meta" in progress
    assert "download" in progress
    assert "index" in progress
