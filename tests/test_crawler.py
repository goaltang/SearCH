"""Unit tests for photofinder.crawler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from photofinder.crawler import (
    API_BASE,
    AlbumCrawler,
    HEADERS,
    Photo,
    album_url,
    parse_albums,
    parse_order_id,
)


def _photo_item(pid: int = 1, **overrides) -> dict:
    defaults = {
        "photoId": pid,
        "fname": f"{pid}.jpg",
        "width": 1920,
        "height": 1080,
        "img": {
            "primary": "https://example.com/",
            "path": f"/{pid}/",
            "sign": "?x-oss-signature=full",
            "s1080": "?x-oss-signature=s1080",
            "s375": "?x-oss-signature=s375",
        },
    }
    defaults.update(overrides)
    return defaults


class _FakeResponse:
    def __init__(self, payload: dict | bytes, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        if isinstance(self._payload, bytes):
            raise ValueError("not json")
        return self._payload

    @property
    def content(self):
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


def test_photo_from_api_builds_urls():
    photo = Photo.from_api(_photo_item(42))
    assert photo.photo_id == 42
    assert photo.fname == "42.jpg"
    assert photo.width == 1920
    assert photo.height == 1080
    assert photo.full_url.endswith("full")
    assert photo.detect_url.endswith("s1080")
    assert photo.preview_url.endswith("s375")


def test_photo_from_api_fallback_to_sign_when_variant_missing():
    item = _photo_item(1)
    item["img"].pop("s1080")
    item["img"].pop("s375")
    photo = Photo.from_api(item)
    assert photo.detect_url == photo.full_url
    assert photo.preview_url == photo.full_url


def test_parse_order_id_accepts_raw_id():
    assert parse_order_id("12345") == "12345"


def test_parse_order_id_extracts_from_url():
    url = "https://www.yipai360.com/photolivepc/?orderId=98765"
    assert parse_order_id(url) == "98765"


def test_parse_order_id_strips_whitespace():
    assert parse_order_id("  12345  ") == "12345"


def test_parse_order_id_raises_on_invalid_input():
    with pytest.raises(ValueError):
        parse_order_id("not-a-url-and-no-order-id")


def test_album_url():
    assert album_url("123") == "https://www.yipai360.com/photolivepc/?orderId=123"


def test_crawler_creates_cache_directories(tmp_path: Path):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    assert crawler.dir.exists()
    assert crawler.thumb_dir.exists()
    assert crawler.meta_path == crawler.dir / "photos.json"


def test_crawler_uses_custom_headers(tmp_path: Path):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    for key, value in HEADERS.items():
        assert crawler.session.headers.get(key) == value


def test_fetch_metadata_pages_through_api(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)

    page1 = {
        "data": {
            "pagination": {"page": 1, "pageSize": 500, "count": 3, "totalPage": 2},
            "photos": [_photo_item(1), _photo_item(2)],
        }
    }
    page2 = {
        "data": {
            "pagination": {"page": 2, "pageSize": 500, "count": 3, "totalPage": 2},
            "photos": [_photo_item(3)],
        }
    }
    responses = {1: page1, 2: page2}

    def fake_get(url, params=None, **kwargs):
        assert url.startswith(API_BASE)
        page = params.get("page", 1) if params else 1
        return _FakeResponse(responses[page])

    monkeypatch.setattr(crawler.session, "get", fake_get)
    photos = crawler.fetch_metadata()

    assert len(photos) == 3
    assert [p.photo_id for p in photos] == [1, 2, 3]
    assert crawler.meta_path.exists()
    saved = json.loads(crawler.meta_path.read_text(encoding="utf-8"))
    assert saved["complete"] is True
    assert len(saved["photos"]) == 3


def test_fetch_metadata_respects_max_photos(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)

    payload = {
        "data": {
            "pagination": {"page": 1, "pageSize": 500, "count": 10, "totalPage": 1},
            "photos": [_photo_item(i) for i in range(1, 6)],
        }
    }
    monkeypatch.setattr(crawler.session, "get", lambda *args, **kwargs: _FakeResponse(payload))

    photos = crawler.fetch_metadata(max_photos=2)
    assert len(photos) == 2
    saved = json.loads(crawler.meta_path.read_text(encoding="utf-8"))
    assert saved["complete"] is False


def test_load_metadata_reads_cache(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    monkeypatch.setattr(crawler.session, "get", lambda *args, **kwargs: _FakeResponse({
        "data": {
            "pagination": {"page": 1, "pageSize": 500, "count": 1, "totalPage": 1},
            "photos": [_photo_item(7)],
        }
    }))
    crawler.fetch_metadata()

    loaded, complete = crawler.load_metadata()
    assert complete is True
    assert len(loaded) == 1
    assert loaded[0].photo_id == 7


def test_load_metadata_handles_legacy_list_format(tmp_path: Path):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    photo = Photo.from_api(_photo_item(9))
    crawler.meta_path.write_text(
        json.dumps([photo.__dict__], ensure_ascii=False), encoding="utf-8"
    )
    loaded, complete = crawler.load_metadata()
    assert complete is False
    assert loaded[0].photo_id == 9


def test_get_metadata_uses_cache_when_complete(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    # seed cache
    crawler._save_metadata([Photo.from_api(_photo_item(1))], complete=True)
    # session.get should not be called
    spy = []
    monkeypatch.setattr(crawler.session, "get", lambda *args, **kwargs: spy.append(1) or _FakeResponse({}))

    photos = crawler.get_metadata()
    assert len(photos) == 1
    assert photos[0].photo_id == 1
    assert not spy


def test_get_metadata_refreshes_when_requested(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    crawler._save_metadata([Photo.from_api(_photo_item(1))], complete=True)

    payload = {
        "data": {
            "pagination": {"page": 1, "pageSize": 500, "count": 1, "totalPage": 1},
            "photos": [_photo_item(2)],
        }
    }
    monkeypatch.setattr(crawler.session, "get", lambda *args, **kwargs: _FakeResponse(payload))

    photos = crawler.get_metadata(refresh=True)
    assert photos[0].photo_id == 2


def test_download_thumbs_skips_existing_files(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    photo = Photo.from_api(_photo_item(1))
    existing = crawler.thumb_path(photo)
    existing.write_bytes(b"cached")

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(b"newdata")

    monkeypatch.setattr(crawler.session, "get", fake_get)
    thumbs = crawler.download_thumbs([photo])

    assert thumbs[photo.photo_id] == existing
    assert existing.read_bytes() == b"cached"
    assert not calls


def test_download_thumbs_writes_new_files(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    photos = [Photo.from_api(_photo_item(i)) for i in (1, 2)]

    monkeypatch.setattr(crawler.session, "get", lambda *args, **kwargs: _FakeResponse(b"jpgbytes"))
    thumbs = crawler.download_thumbs(photos)

    for photo in photos:
        assert thumbs[photo.photo_id].exists()
        assert thumbs[photo.photo_id].read_bytes() == b"jpgbytes"


def test_download_thumbs_reports_progress(tmp_path: Path, monkeypatch):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    photos = [Photo.from_api(_photo_item(i)) for i in (1, 2)]
    monkeypatch.setattr(crawler.session, "get", lambda *args, **kwargs: _FakeResponse(b"x"))

    progress = []
    crawler.download_thumbs(photos, progress_cb=lambda stage, done, total: progress.append((stage, done, total)))

    assert progress
    assert progress[0] == ("download", 0, 2)
    assert progress[-1] == ("download", 2, 2)


def test_download_thumbs_omits_failed_downloads(tmp_path: Path, monkeypatch, caplog):
    crawler = AlbumCrawler("123", cache_root=tmp_path)
    photos = [Photo.from_api(_photo_item(1)), Photo.from_api(_photo_item(2))]

    def fake_get(url, **kwargs):
        if "/1/" in url:
            raise Exception("network down")
        return _FakeResponse(b"ok")

    monkeypatch.setattr(crawler.session, "get", fake_get)
    with caplog.at_level("WARNING", logger="photofinder.crawler"):
        thumbs = crawler.download_thumbs(photos)

    assert 1 not in thumbs
    assert 2 in thumbs
    assert any("Thumb download failed" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- parse_albums

def test_parse_albums_single_url_with_label():
    text = "省赛 https://www.yipai360.com/photolivepc/?orderId=111&channel=h5"
    albums = parse_albums(text)
    assert len(albums) == 1
    assert albums[0]["order_id"] == "111"
    assert albums[0]["label"] == "省赛"
    assert "orderId=111" in albums[0]["url"]


def test_parse_albums_multiple_lines():
    text = ("省赛·毕节 https://x/?orderId=111\n"
            "国赛·上海 https://x/?orderId=222")
    albums = parse_albums(text)
    assert [a["order_id"] for a in albums] == ["111", "222"]
    assert [a["label"] for a in albums] == ["省赛·毕节", "国赛·上海"]


def test_parse_albums_default_label_when_omitted():
    albums = parse_albums("https://x/?orderId=111\nhttps://x/?orderId=222")
    assert [a["label"] for a in albums] == ["相册 1", "相册 2"]


def test_parse_albums_accepts_bare_order_id():
    albums = parse_albums("国赛 20260727190944809942")
    assert albums[0]["order_id"] == "20260727190944809942"
    assert albums[0]["label"] == "国赛"


def test_parse_albums_strips_label_separators():
    albums = parse_albums("国赛: https://x/?orderId=222")
    assert albums[0]["label"] == "国赛"


def test_parse_albums_ignores_blank_and_comment_lines():
    text = "\n# comment\n省赛 https://x/?orderId=111\n\n"
    albums = parse_albums(text)
    assert len(albums) == 1
    assert albums[0]["order_id"] == "111"


def test_parse_albums_dedupes_order_ids_first_label_wins():
    text = "A https://x/?orderId=111\nB https://x/?orderId=111"
    albums = parse_albums(text)
    assert len(albums) == 1
    assert albums[0]["label"] == "A"


def test_parse_albums_raises_when_nothing_parseable():
    with pytest.raises(ValueError):
        parse_albums("no links here\n# just a comment")
