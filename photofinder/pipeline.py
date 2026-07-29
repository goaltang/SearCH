"""End-to-end pipeline: album URL + reference face photo -> matched photos."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class MatchResult:
    photo_id: int
    fname: str
    score: float
    album_url: str    # original web page of the album
    preview_url: str  # 375px signed OSS url
    full_url: str     # original-resolution signed OSS url
    thumb_path: str   # local cached 720px thumb


class PhotoFinder:
    def __init__(self, cache_root: str | Path = "cache",
                 models_dir: str | Path = "models"):
        self.cache_root = Path(cache_root)
        self.models_dir = Path(models_dir)
        self._engine: FaceEngine | None = None

    @property
    def engine(self) -> FaceEngine:
        if self._engine is None:
            self._engine = FaceEngine(
                self.models_dir / "det_10g.onnx",
                self.models_dir / "w600k_r50.onnx")
        return self._engine

    @staticmethod
    def _read_ref_image(ref_image):
        """Return a cv2 image from a path or an already-loaded image."""
        if isinstance(ref_image, (str, Path)):
            return cv2.imdecode(np.fromfile(str(ref_image), dtype=np.uint8),
                                cv2.IMREAD_COLOR)
        return ref_image

    def extract_reference(self, ref_images) -> list[np.ndarray]:
        """Embeddings of the target person from one or more reference images.

        Each image contributes only its largest (most prominent) face: extra
        background faces would pollute the search with unintended people.
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
        for item in items:
            img = self._read_ref_image(item)
            if img is None:
                logger.warning("Cannot read one of the reference images, skipping")
                continue
            faces = self.engine.process(img)
            if not faces:
                logger.warning("No face detected in one of the reference images, skipping")
                continue
            logger.info("Reference image: %d face(s), using the largest one", len(faces))
            embeddings.append(faces[0]["embedding"])

        if not embeddings:
            logger.error("No valid reference face found in any reference image")
            raise ValueError("no valid reference face found in any reference image")
        return embeddings

    def run(self, url: str, ref_image, max_photos: int | None = None,
            threshold: float = DEFAULT_THRESHOLD, refresh: bool = False,
            pwd: str | None = None, workers: int = 4, min_face: float = 24.0,
            progress_cb=None) -> list[MatchResult]:
        order_id = parse_order_id(url)
        logger.info("Pipeline start: order=%s max_photos=%s threshold=%.3f",
                    order_id, max_photos, threshold)
        crawler = AlbumCrawler(order_id, self.cache_root, pwd=pwd)

        # 1) metadata
        photos = crawler.get_metadata(max_photos=max_photos, refresh=refresh,
                                      progress_cb=progress_cb)
        if max_photos:
            photos = photos[:max_photos]
        logger.info("Metadata ready: %d photos", len(photos))

        # 2) thumbnails (continue even if some fail)
        thumbs = crawler.download_thumbs(photos, progress_cb=progress_cb)
        available = {pid: path for pid, path in thumbs.items()
                     if path is not None and path.exists()}
        if len(available) < len(thumbs):
            logger.warning("Only %d/%d thumbs available", len(available), len(thumbs))
        if not available:
            logger.error("No usable thumbnails; aborting")
            raise ValueError("no usable thumbnails available")

        # 3) face index (incremental)
        index = FaceIndex(crawler.dir)
        index.build(self.engine, available, workers=workers, min_face=min_face,
                    progress_cb=progress_cb)

        # 4) reference embedding + search
        ref_embs = self.extract_reference(ref_image)
        hits = index.search(ref_embs, threshold=threshold)
        logger.info("Search found %d hits", len(hits))

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
            ))
        logger.info("Pipeline finish: returning %d results", len(results))
        return results
