"""Unit tests for photofinder.cli (--prepare mode and argument validation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from photofinder.cli import main


class FakeEngine:
    def process_file(self, path):
        return [{"bbox": [0, 0, 30, 30], "det_score": 0.9,
                 "embedding": np.ones(512, dtype=np.float32)}]

    def process(self, img):
        return [{"bbox": [0, 0, 30, 30], "det_score": 0.9,
                 "embedding": np.ones(512, dtype=np.float32)}]


class FakePhoto:
    def __init__(self, photo_id):
        self.photo_id = photo_id
        self.fname = f"{photo_id}.jpg"


class FakeCrawler:
    def __init__(self, order_id, cache_root, pwd=None, tag_id=None):
        self.order_id = order_id
        self.dir = Path(cache_root) / order_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def get_metadata(self, max_photos=None, refresh=False, progress_cb=None):
        return [FakePhoto(i) for i in range(1, 4)]

    def download_thumbs(self, photos, progress_cb=None):
        result = {}
        for p in photos:
            thumb = self.dir / f"{p.photo_id}.jpg"
            thumb.write_bytes(b"fake")
            result[p.photo_id] = thumb
        return result


class FakeFaceIndex:
    def __init__(self, index_dir):
        self.dir = Path(index_dir)
        self.faces = [{"photo_id": 1}, {"photo_id": 2}]
        self.done_ids = {1, 2, 3}

    def build(self, engine, thumbs, workers=4, min_face=24.0, progress_cb=None):
        return len(thumbs)


def test_prepare_mode_runs_without_ref(tmp_path, monkeypatch):
    monkeypatch.setattr("photofinder.pipeline.PhotoFinder",
                        lambda **kw: MagicMock(engine=FakeEngine()))
    monkeypatch.setattr("photofinder.crawler.AlbumCrawler", FakeCrawler)
    monkeypatch.setattr("photofinder.index.FaceIndex", FakeFaceIndex)

    ret = main(["--url", "999", "--prepare",
                "--cache", str(tmp_path), "--models", str(tmp_path)])
    assert ret == 0


def test_search_mode_requires_ref(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["--url", "999", "--cache", str(tmp_path),
              "--models", str(tmp_path)])
    assert exc_info.value.code == 2
