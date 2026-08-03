"""Crawler for yipai360 (一拍即传) live-photo albums.

The website is a React SPA; all photos are served by a JSON API:

    GET /api/v1/yipai/order/{orderId}/audience/photos
        ?page=N&pageSize=M[&pwd=..][&tagId=..][&sortType=..]
    Headers: appName: yipai, appAccess: yipai

Response:
    data.pagination: {page, pageSize, count, totalPage}
    data.photos[]:   {photoId, fname, width, height,
                      img: {primary, failover, path, sign, s1920, s1080, s375, ...}}

Image URLs are pre-signed OSS URLs:  primary + path + variant_query
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

from .logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

API_BASE = "https://www.yipai360.com"
HEADERS = {"appName": "yipai", "appAccess": "yipai"}
PAGE_SIZE = 500

# size variant used for face detection (720px wide, good speed/accuracy trade-off)
DETECT_VARIANT = "s1080"
# small variant used for result previews in the UI
PREVIEW_VARIANT = "s375"


@dataclass
class Photo:
    photo_id: int
    fname: str
    width: int
    height: int
    detect_url: str   # 720px, for face detection
    preview_url: str  # 375px, for UI thumbnails
    full_url: str     # original resolution (signed OSS link)

    @staticmethod
    def from_api(item: dict) -> "Photo":
        img = item["img"]
        base = img["primary"] + img["path"]
        return Photo(
            photo_id=int(item["photoId"]),
            fname=item.get("fname", ""),
            width=int(item.get("width", 0) or 0),
            height=int(item.get("height", 0) or 0),
            detect_url=base + img.get("s1080", img["sign"]),
            preview_url=base + img.get("s375", img["sign"]),
            full_url=base + img["sign"],
        )


def parse_order_id(url_or_id: str) -> str:
    """Extract orderId from an album URL (or return the raw id)."""
    url_or_id = url_or_id.strip()
    if url_or_id.isdigit():
        return url_or_id
    qs = parse_qs(urlparse(url_or_id).query)
    if "orderId" in qs and qs["orderId"]:
        return qs["orderId"][0]
    m = re.search(r"orderId=(\d+)", url_or_id)
    if m:
        return m.group(1)
    raise ValueError(f"cannot parse orderId from: {url_or_id!r}")


def album_url(order_id: str) -> str:
    return f"https://www.yipai360.com/photolivepc/?orderId={order_id}"


def parse_albums(text: str) -> list[dict]:
    """Parse a multi-line album specification into a list of albums.

    Each non-empty line describes one album: ``[标签] 链接或orderId``.
    The label is optional — any text on the line that is not part of the
    URL/orderId becomes the label (surrounding separators stripped). When
    omitted, a default ``相册 N`` is assigned. Blank lines and lines starting
    with ``#`` are ignored; duplicate orderIds are de-duplicated (first wins).

    Returns [{"order_id", "label", "url"}] with ``url`` the canonical album
    page. Raises ValueError when no album can be parsed.
    """
    albums: list[dict] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        url_m = re.search(r"https?://\S+", line)
        try:
            if url_m:
                order_id = parse_order_id(url_m.group(0))
                label_part = line[:url_m.start()] + line[url_m.end():]
            else:
                num_m = re.search(r"(?<!\d)(\d{6,})(?!\d)", line)
                if not num_m:
                    continue
                order_id = num_m.group(1)
                label_part = line[:num_m.start()] + line[num_m.end():]
        except ValueError:
            logger.warning("Skipping unparseable album line: %r", line)
            continue
        if order_id in seen:
            continue
        seen.add(order_id)
        label = label_part.strip().strip(":：,，、 \t") or f"相册 {len(albums) + 1}"
        albums.append({"order_id": order_id, "label": label,
                       "url": album_url(order_id)})
    if not albums:
        raise ValueError("未能从输入中解析出任何相册链接")
    return albums


class AlbumCrawler:
    """Fetches photo metadata and downloads thumbnails with disk caching."""

    def __init__(self, order_id: str, cache_root: str | Path = "cache",
                 pwd: str | None = None, tag_id: str | None = None):
        self.order_id = order_id
        self.dir = Path(cache_root) / order_id
        self.thumb_dir = self.dir / "thumbs"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "photos.json"
        self.pwd = pwd
        self.tag_id = tag_id
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        if pwd:
            self.session.headers["pwd"] = pwd

    # ------------------------------------------------------------------ meta
    def fetch_metadata(self, max_photos: int | None = None,
                       progress_cb=None) -> list[Photo]:
        """Page through the API and cache photos.json. Returns Photo list."""
        logger.info("Fetching metadata for order %s", self.order_id)
        photos: list[Photo] = []
        page = 1
        total = None
        while True:
            params = {"page": page, "pageSize": PAGE_SIZE}
            if self.tag_id:
                params["tagId"] = self.tag_id
            for attempt in range(4):
                try:
                    r = self.session.get(
                        f"{API_BASE}/api/v1/yipai/order/{self.order_id}/audience/photos",
                        params=params, timeout=30)
                    r.raise_for_status()
                    data = r.json()["data"]
                    break
                except Exception as exc:
                    logger.warning(
                        "Metadata fetch failed (attempt %d/4) for order %s page %d: %s",
                        attempt + 1, self.order_id, page, exc)
                    if attempt == 3:
                        logger.error(
                            "Giving up on metadata for order %s page %d",
                            self.order_id, page)
                        raise
                    time.sleep(1.5 * (attempt + 1))
            batch = [Photo.from_api(p) for p in data.get("photos", [])]
            photos.extend(batch)
            pg = data.get("pagination", {})
            total = pg.get("count", total)
            if progress_cb:
                progress_cb("meta", len(photos), total)
            total_page = pg.get("totalPage", page)
            if not batch or page >= total_page:
                break
            if max_photos and len(photos) >= max_photos:
                break
            page += 1
        complete = not (max_photos and len(photos) > max_photos)
        if max_photos:
            photos = photos[:max_photos]
        logger.info("Fetched %d photos for order %s (complete=%s)",
                    len(photos), self.order_id, complete)
        self._save_metadata(photos, complete=complete)
        return photos

    def load_metadata(self) -> tuple[list[Photo], bool] | None:
        """Returns (photos, is_complete) or None if no cache."""
        if not self.meta_path.exists():
            return None
        raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):  # legacy format: treat as incomplete
            return [Photo(**p) for p in raw], False
        photos = [Photo(**p) for p in raw["photos"]]
        return photos, bool(raw.get("complete", False))

    def get_metadata(self, max_photos: int | None = None,
                     refresh: bool = False, progress_cb=None) -> list[Photo]:
        cached = None if refresh else self.load_metadata()
        if cached is not None:
            photos, complete = cached
            if max_photos is not None and len(photos) >= max_photos:
                return photos[:max_photos]
            if max_photos is None and complete:
                return photos
        return self.fetch_metadata(max_photos=max_photos, progress_cb=progress_cb)

    def _save_metadata(self, photos: list[Photo], complete: bool) -> None:
        logger.debug("Saving metadata for order %s: %d photos", self.order_id, len(photos))
        self.meta_path.write_text(
            json.dumps({"complete": complete,
                        "photos": [asdict(p) for p in photos]},
                       ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------------- thumbs
    def thumb_path(self, photo: Photo) -> Path:
        return self.thumb_dir / f"{photo.photo_id}.jpg"

    def download_thumbs(self, photos: list[Photo], workers: int = 12,
                        progress_cb=None) -> dict[int, Path]:
        """Download detection thumbnails concurrently; skip cached files.

        Returns only successfully downloaded or already-cached paths.
        Failed downloads are logged and omitted so the pipeline can continue.
        """
        todo = [p for p in photos if not self.thumb_path(p).exists()]
        done = {p.photo_id: self.thumb_path(p) for p in photos
                if self.thumb_path(p).exists()}
        total = len(photos)
        logger.info("Downloading %d thumbs for order %s (%d cached)",
                    len(todo), self.order_id, len(done))
        if progress_cb:
            progress_cb("download", len(done), total)

        def _dl(p: Photo):
            path = self.thumb_path(p)
            tmp = path.with_suffix(".part")
            for attempt in range(3):
                try:
                    r = self.session.get(p.detect_url, timeout=60)
                    r.raise_for_status()
                    tmp.write_bytes(r.content)
                    tmp.rename(path)
                    return p.photo_id, path
                except Exception as exc:
                    logger.warning(
                        "Thumb download failed for photo %d (attempt %d/3): %s",
                        p.photo_id, attempt + 1, exc)
                    if attempt == 2:
                        return p.photo_id, None
                    time.sleep(1 + attempt)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_dl, p) for p in todo]
            for fut in as_completed(futs):
                pid, path = fut.result()
                if path is not None:
                    done[pid] = path
                if progress_cb:
                    progress_cb("download", len(done), total)
        if len(done) < total:
            logger.warning("Downloaded %d/%d thumbs for order %s",
                           len(done), total, self.order_id)
        return done

    # ----------------------------------------------------------- incremental
    def fetch_incremental(self, progress_cb=None) -> list[Photo]:
        """Fetch only photos added since the last cached snapshot.

        Returns the list of *new* photos (already appended to cache).
        """
        cached = self.load_metadata()
        if cached is None:
            return self.fetch_metadata(progress_cb=progress_cb)
        old_photos, _complete = cached
        known_ids = {p.photo_id for p in old_photos}

        new_photos: list[Photo] = []
        page = 1
        while True:
            params = {"page": page, "pageSize": PAGE_SIZE}
            if self.tag_id:
                params["tagId"] = self.tag_id
            try:
                r = self.session.get(
                    f"{API_BASE}/api/v1/yipai/order/{self.order_id}/audience/photos",
                    params=params, timeout=30)
                r.raise_for_status()
                data = r.json()["data"]
            except Exception as exc:
                logger.warning("Incremental fetch failed page %d: %s", page, exc)
                break
            batch = [Photo.from_api(p) for p in data.get("photos", [])]
            fresh = [p for p in batch if p.photo_id not in known_ids]
            new_photos.extend(fresh)
            if progress_cb:
                progress_cb("meta", len(new_photos), None)
            pg = data.get("pagination", {})
            if not batch or page >= pg.get("totalPage", page):
                break
            # NOTE: photo_id is NOT monotonic with upload time (new photos
            # interleave with old ids), so scan every page; an id-based early
            # stop silently drops newly uploaded photos.
            page += 1

        if new_photos:
            merged = old_photos + new_photos
            self._save_metadata(merged, complete=True)
            logger.info("Incremental: +%d new photos (total %d)",
                        len(new_photos), len(merged))
        return new_photos

    # --------------------------------------------------------- batch download
    def download_full_images(self, urls: dict[str, str],
                             out_dir: str | Path,
                             workers: int = 6) -> dict[str, Path]:
        """Download full-resolution images.

        urls: {filename: url}.  Returns {filename: local_path} for successes.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}

        def _dl(fname: str, url: str):
            dest = out / fname
            if dest.exists():
                return fname, dest
            try:
                r = self.session.get(url, timeout=120)
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(dest)
                return fname, dest
            except Exception as exc:
                logger.warning("Full-image download failed %s: %s", fname, exc)
                return fname, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_dl, f, u) for f, u in urls.items()]
            for fut in as_completed(futs):
                fname, path = fut.result()
                if path is not None:
                    results[fname] = path
        return results
