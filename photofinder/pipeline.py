"""End-to-end pipeline: album URL + reference face photo -> matched photos."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .crawler import AlbumCrawler, Photo, album_url, parse_order_id
from .face_engine import FaceEngine
from .index import FaceIndex
from .logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# ArcFace(w600k_r50) cosine threshold: same person typically >= 0.6,
# different people < 0.3. 0.45 is a conservative default for recall.
DEFAULT_THRESHOLD = 0.45

# How many album indexes stay hot in memory. Each face costs ~2 KB
# (512 float32), so a 50k-face album is ~100 MB.
INDEX_CACHE_SIZE = int(os.environ.get("PHOTOFINDER_INDEX_CACHE", "4"))


class SearchCancelled(Exception):
    """Raised when the user cancels an in-progress search."""


@dataclass
class MatchResult:
    photo_id: int
    fname: str
    score: float
    album_url: str    # original web page of the album
    preview_url: str  # 375px signed OSS url
    full_url: str     # original-resolution signed OSS url
    thumb_path: str   # local cached 720px thumb
    detect_url: str = ""  # 1080px signed OSS url (reliable; full_url often 403s)
    bbox: list[float] = field(default_factory=list)  # [x1,y1,x2,y2] on thumb
    label: str = ""     # album display label (e.g. 省赛/国赛); "" for single-album run()
    order_id: str = ""  # album orderId this match came from


@dataclass
class RefQuality:
    """Quality feedback for a single reference image."""
    faces_found: int
    used: bool
    reason: str = ""


class PhotoFinder:
    def __init__(self, cache_root: str | Path = "cache",
                 models_dir: str | Path = "models"):
        self.cache_root = Path(cache_root)
        self.models_dir = Path(models_dir)
        self._engine: FaceEngine | None = None
        self._engine_lock = threading.Lock()
        # Per-album FaceIndex instances kept hot in memory (LRU) so repeated
        # searches don't re-read/re-parse faces.npz + faces.json from disk.
        self._index_cache: OrderedDict[str, FaceIndex] = OrderedDict()
        # Per-album build locks: two concurrent searches on the same album
        # must not build the index simultaneously (wasted work + double
        # downloads); different albums build independently.
        self._album_locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()  # guards the two dicts above

    @property
    def engine(self) -> FaceEngine:
        if self._engine is None:
            with self._engine_lock:  # avoid loading models twice concurrently
                if self._engine is None:
                    self._engine = FaceEngine(
                        self.models_dir / "det_10g.onnx",
                        self.models_dir / "w600k_r50.onnx")
        return self._engine

    def _get_album_lock(self, order_id: str) -> threading.Lock:
        with self._meta_lock:
            return self._album_locks.setdefault(order_id, threading.Lock())

    def _get_index(self, order_id: str, index_dir: Path) -> FaceIndex:
        with self._meta_lock:
            index = self._index_cache.get(order_id)
            if index is None:
                index = FaceIndex(index_dir)
                self._index_cache[order_id] = index
                while len(self._index_cache) > INDEX_CACHE_SIZE:
                    evicted, _ = self._index_cache.popitem(last=False)
                    logger.info("Evicted index of album %s from memory cache",
                                evicted)
            else:
                self._index_cache.move_to_end(order_id)
            return index

    @staticmethod
    def _read_ref_image(ref_image):
        """Return a cv2 image from a path or an already-loaded image."""
        if isinstance(ref_image, (str, Path)):
            return cv2.imdecode(np.fromfile(str(ref_image), dtype=np.uint8),
                                cv2.IMREAD_COLOR)
        return ref_image

    def extract_reference(self, ref_images) -> tuple[list[np.ndarray], list[RefQuality]]:
        """Embeddings of the target person from one or more reference images.

        Each image contributes only its largest (most prominent) face: extra
        background faces would pollute the search with unintended people.

        Returns (embeddings, quality_report).
        """
        if ref_images is None:
            logger.error("No reference image provided")
            raise ValueError("no reference image provided")

        # Accept a single image or a sequence of images.
        if isinstance(ref_images, (str, Path)) or (
                isinstance(ref_images, np.ndarray) and ref_images.ndim == 3):
            items = [ref_images]
        else:
            items = list(ref_images)

        embeddings: list[np.ndarray] = []
        qualities: list[RefQuality] = []
        for item in items:
            img = self._read_ref_image(item)
            if img is None:
                qualities.append(RefQuality(0, False, "无法读取图片"))
                continue
            faces = self.engine.process(img)
            if not faces:
                qualities.append(RefQuality(0, False, "未检测到人脸"))
                continue
            embeddings.append(faces[0]["embedding"])
            qualities.append(RefQuality(len(faces), True))

        if not embeddings:
            logger.error("No valid reference face found in any reference image")
            raise ValueError("no valid reference face found in any reference image")
        return embeddings, qualities

    def run(self, url: str, ref_image, max_photos: int | None = None,
            threshold: float = DEFAULT_THRESHOLD, refresh: bool = False,
            pwd: str | None = None, workers: int = 4, min_face: float = 24.0,
            progress_cb=None, cancel_event: threading.Event | None = None,
            excluded_ids: list[int] | None = None,
            incremental: bool = False) -> list[MatchResult]:
        """Search a single album.

        Thin wrapper over _search_one_album kept for the CLI and backward
        compatibility; see run_multi() for searching several albums at once.
        """
        order_id = parse_order_id(url)
        logger.info("Pipeline start: order=%s max_photos=%s threshold=%.3f",
                    order_id, max_photos, threshold)
        ref_embs, _qualities = self.extract_reference(ref_image)
        results = self._search_one_album(
            order_id, "", ref_embs, max_photos=max_photos, threshold=threshold,
            refresh=refresh, pwd=pwd, workers=workers, min_face=min_face,
            progress_cb=progress_cb, cancel_event=cancel_event,
            excluded_ids=excluded_ids, incremental=incremental)
        logger.info("Pipeline finish: returning %d results", len(results))
        return results

    @staticmethod
    def _normalize_albums(albums) -> list[dict]:
        """Coerce album specs (str URL/orderId or dict) into
        {"order_id", "label", "url"} dicts."""
        norm: list[dict] = []
        for i, a in enumerate(albums):
            if isinstance(a, str):
                oid = parse_order_id(a)
                norm.append({"order_id": oid, "label": f"相册 {i + 1}",
                             "url": album_url(oid)})
            elif isinstance(a, dict):
                oid = a.get("order_id") or parse_order_id(a.get("url", ""))
                norm.append({"order_id": oid,
                             "label": a.get("label") or f"相册 {i + 1}",
                             "url": a.get("url") or album_url(oid)})
            else:
                raise ValueError(f"unsupported album spec: {a!r}")
        if not norm:
            raise ValueError("no albums provided")
        return norm

    def run_multi(self, albums, ref_image, max_photos: int | None = None,
                  threshold: float = DEFAULT_THRESHOLD, refresh: bool = False,
                  pwd: str | None = None, workers: int = 4,
                  min_face: float = 24.0, progress_cb=None,
                  cancel_event: threading.Event | None = None,
                  excluded_ids: list[int] | None = None,
                  incremental: bool = False) -> list[MatchResult]:
        """Search several albums with one reference; merge results by score.

        albums: list of {"order_id", "label", "url"} dicts (see
        crawler.parse_albums) or plain URLs/orderIds. The reference face is
        extracted once and reused for every album. A failing album is logged
        and skipped so one bad album doesn't abort the whole search;
        SearchCancelled always propagates.
        """
        norm = self._normalize_albums(albums)
        logger.info("Pipeline multi-start: %d albums (%s)", len(norm),
                    ", ".join(a["order_id"] for a in norm))
        ref_embs, _qualities = self.extract_reference(ref_image)

        all_results: list[MatchResult] = []
        for a in norm:
            try:
                all_results.extend(self._search_one_album(
                    a["order_id"], a["label"], ref_embs,
                    max_photos=max_photos, threshold=threshold, refresh=refresh,
                    pwd=pwd, workers=workers, min_face=min_face,
                    progress_cb=progress_cb, cancel_event=cancel_event,
                    excluded_ids=excluded_ids, incremental=incremental))
            except SearchCancelled:
                raise
            except Exception as exc:
                logger.exception("Album %s (%s) failed, skipping: %s",
                                 a["order_id"], a["label"], exc)
        all_results.sort(key=lambda r: r.score, reverse=True)
        logger.info("Pipeline multi-finish: %d total results", len(all_results))
        return all_results

    def _search_one_album(self, order_id: str, label: str,
                          ref_embs: list[np.ndarray], *,
                          max_photos: int | None = None,
                          threshold: float = DEFAULT_THRESHOLD,
                          refresh: bool = False, pwd: str | None = None,
                          workers: int = 4, min_face: float = 24.0,
                          progress_cb=None,
                          cancel_event: threading.Event | None = None,
                          excluded_ids: list[int] | None = None,
                          incremental: bool = False) -> list[MatchResult]:
        """Build/refresh one album's index and search it with pre-computed
        reference embeddings. Returns this album's matches (labelled)."""
        crawler = AlbumCrawler(order_id, self.cache_root, pwd=pwd)

        def _check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise SearchCancelled("搜索已取消")

        def _cb(stage, done, total):
            _check_cancel()
            if progress_cb:
                progress_cb(stage, done, total)

        # 1) metadata
        if incremental:
            photos = crawler.fetch_incremental(progress_cb=_cb)
            cached = crawler.load_metadata()
            if cached:
                photos = cached[0]
        else:
            photos = crawler.get_metadata(max_photos=max_photos,
                                          refresh=refresh, progress_cb=_cb)
        if max_photos:
            photos = photos[:max_photos]
        logger.info("Album %s metadata ready: %d photos", order_id, len(photos))
        _check_cancel()

        # 2) thumbnails (continue even if some fail)
        thumbs = crawler.download_thumbs(photos, progress_cb=_cb)
        available = {pid: path for pid, path in thumbs.items()
                     if path is not None and path.exists()}
        if len(available) < len(thumbs):
            logger.warning("Album %s: only %d/%d thumbs available",
                           order_id, len(available), len(thumbs))
        if not available:
            logger.error("Album %s: no usable thumbnails", order_id)
            raise ValueError("no usable thumbnails available")
        _check_cancel()

        # 3) face index (incremental). Serialized per album: if another
        # search is already building this album's index, wait for it instead
        # of duplicating the work; the build below then finds nothing new
        # and returns immediately.
        with self._get_album_lock(order_id):
            index = self._get_index(order_id, crawler.dir)
            if excluded_ids:
                index.exclude(excluded_ids)
            index.build(self.engine, available, workers=workers,
                        min_face=min_face, progress_cb=_cb)
        _check_cancel()

        # 4) search with the shared reference embeddings
        hits = index.search(ref_embs, threshold=threshold)
        logger.info("Album %s search found %d hits", order_id, len(hits))

        by_id = {p.photo_id: p for p in photos}
        results = []
        for h in hits:
            p: Photo | None = by_id.get(h["photo_id"])
            if p is None:
                continue
            results.append(MatchResult(
                photo_id=p.photo_id,
                fname=p.fname,
                score=round(h["score"], 4),
                album_url=album_url(order_id),
                preview_url=p.preview_url,
                full_url=p.full_url,
                thumb_path=str(available[p.photo_id]),
                detect_url=p.detect_url,
                bbox=h.get("bbox", []),
                label=label,
                order_id=order_id,
            ))
        return results
