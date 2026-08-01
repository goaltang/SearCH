"""Face embedding index: build incrementally from cached thumbs, search by cosine similarity.

Persistence per album (cache/{orderId}/):
    faces.npz      - float32 (N, 512) embedding matrix, aligned with faces.json
    faces.json     - [{photo_id, bbox, det_score}] one row per detected face
    done.json      - [photo_id] all photos already processed (incl. no-face ones)
    excluded.json  - [photo_id] user-marked false positives to skip in search
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .face_engine import FaceEngine
from .logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via tmp-file + rename so concurrent readers never see a
    half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FaceIndex:
    def __init__(self, index_dir: str | Path):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # Guards every read/write of embeddings/faces/done_ids/excluded_ids so
        # the index can be shared across concurrent searches (see
        # PhotoFinder._get_index). RLock: save() is called with the lock held.
        self._lock = threading.RLock()
        self.embeddings: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        self.faces: list[dict] = []
        self.done_ids: set[int] = set()
        self.excluded_ids: set[int] = set()
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        emb_p, faces_p, done_p, excl_p = (
            self.dir / "faces.npz", self.dir / "faces.json",
            self.dir / "done.json", self.dir / "excluded.json")
        try:
            if emb_p.exists() and faces_p.exists():
                self.embeddings = np.load(emb_p)["embeddings"]
                self.faces = json.loads(faces_p.read_text(encoding="utf-8"))
                if len(self.faces) != len(self.embeddings):
                    raise ValueError(
                        f"corrupt index: faces.json ({len(self.faces)}) / "
                        f"faces.npz ({len(self.embeddings)}) mismatch")
            if done_p.exists():
                self.done_ids = set(json.loads(done_p.read_text(encoding="utf-8")))
            if excl_p.exists():
                self.excluded_ids = set(
                    json.loads(excl_p.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.error("Corrupt index in %s, resetting: %s", self.dir, exc)
            self.reset()

    def save(self) -> None:
        with self._lock:
            logger.debug("Saving index: %d faces, %d done, %d excluded",
                         len(self.faces), len(self.done_ids),
                         len(self.excluded_ids))
            npz_tmp = self.dir / "faces.npz.tmp"
            # open() handle: np.savez* appends ".npz" to plain file names
            with open(npz_tmp, "wb") as f:
                np.savez_compressed(
                    f, embeddings=self.embeddings.astype(np.float32))
            os.replace(npz_tmp, self.dir / "faces.npz")
            _atomic_write_text(self.dir / "faces.json",
                               json.dumps(self.faces, ensure_ascii=False))
            _atomic_write_text(self.dir / "done.json",
                               json.dumps(sorted(self.done_ids)))
            _atomic_write_text(self.dir / "excluded.json",
                               json.dumps(sorted(self.excluded_ids)))

    def reset(self) -> None:
        """Drop all indexed data (keeps downloaded thumbs)."""
        logger.info("Resetting face index in %s", self.dir)
        with self._lock:
            self.embeddings = np.zeros((0, 512), dtype=np.float32)
            self.faces = []
            self.done_ids = set()
            self.excluded_ids = set()
            for name in ("faces.npz", "faces.json", "done.json",
                         "excluded.json"):
                p = self.dir / name
                if p.exists():
                    p.unlink()

    # -------------------------------------------------------------- exclusion
    def exclude(self, photo_ids: list[int]) -> None:
        """Mark photos as false positives; they will be skipped in search."""
        with self._lock:
            self.excluded_ids.update(photo_ids)
            self.save()

    def unexclude(self, photo_ids: list[int]) -> None:
        with self._lock:
            self.excluded_ids.difference_update(photo_ids)
            self.save()

    # ------------------------------------------------------------------ build
    def build(self, engine: FaceEngine, thumbs: dict[int, Path],
              workers: int = 4, min_face: float = 24.0,
              progress_cb=None) -> int:
        """Detect faces in all unprocessed thumbs; append to index.

        thumbs: {photo_id} -> image_path.
        min_face: minimum face bbox width in pixels; smaller detections are
        dropped (they are mostly false positives on texture-less regions and
        are too small for reliable recognition anyway).
        Returns # of newly processed photos.
        """
        # Drop missing/invalid paths so one broken thumb doesn't break the batch.
        valid = {pid: p for pid, p in thumbs.items()
                 if p is not None and Path(p).exists()}
        skipped = set(thumbs) - set(valid)
        if skipped:
            logger.warning("Skipping %d missing/invalid thumbs", len(skipped))

        with self._lock:
            todo = {pid: p for pid, p in valid.items()
                    if pid not in self.done_ids}
        if not todo:
            logger.info("No new photos to index in %s", self.dir)
            return 0

        new_embs: list[np.ndarray] = []
        new_faces: list[dict] = []
        finished: list[int] = []
        total = len(todo)
        logger.info("Indexing %d new photos in %s", total, self.dir)

        def _work(item):
            pid, path = item
            try:
                faces = engine.process_file(path)
            except Exception as exc:
                logger.warning("Face processing failed for photo %d: %s", pid, exc)
                return pid, []
            return pid, [f for f in faces
                         if (f["bbox"][2] - f["bbox"][0]) >= min_face]

        n_faces_total = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, item) for item in todo.items()]
            for n, fut in enumerate(as_completed(futs), 1):
                pid, faces = fut.result()
                n_faces_total += len(faces)
                for f in faces:
                    new_embs.append(f["embedding"].astype(np.float32))
                    new_faces.append({"photo_id": pid, "bbox": f["bbox"],
                                      "det_score": f["det_score"]})
                finished.append(pid)
                if progress_cb and (n % 10 == 0 or n == total):
                    progress_cb("index", n, total)
                # checkpoint every 500 photos so interruptions don't lose work
                if n % 500 == 0:
                    logger.info("Checkpoint after %d/%d photos", n, total)
                    self._absorb(new_embs, new_faces, finished)
                    new_embs, new_faces, finished = [], [], []

        self._absorb(new_embs, new_faces, finished)
        if total > 0 and n_faces_total == 0:
            logger.warning(
                "Indexed %d photos but found 0 faces! Recognition is likely "
                "broken in this environment (on some onnxruntime builds set "
                "PHOTOFINDER_FACE_BATCH=0). Do NOT trust an empty index.",
                total)
        logger.info("Indexed %d new photos (%d faces) in %s",
                    total, n_faces_total, self.dir)
        return total

    def _absorb(self, new_embs, new_faces, finished) -> None:
        with self._lock:
            if new_embs:
                block = np.stack(new_embs).astype(np.float32)
                self.embeddings = np.vstack([self.embeddings, block]) \
                    if len(self.embeddings) else block
                self.faces.extend(new_faces)
            self.done_ids.update(finished)
            self.save()

    # ----------------------------------------------------------------- search
    def search(self, ref_embeddings: list[np.ndarray],
               threshold: float = 0.35, top_k: int | None = None) -> list[dict]:
        """Aggregate max cosine similarity per photo. Returns sorted matches.

        Each result dict: {photo_id, score, bbox} where bbox belongs to the
        highest-scoring face of that photo.
        """
        # Snapshot under the lock: _absorb() swaps in a new embeddings array
        # but mutates faces/excluded_ids in place, so copy those. The heavy
        # math below then runs lock-free on a consistent view.
        with self._lock:
            embeddings = self.embeddings
            faces = list(self.faces)
            excluded = set(self.excluded_ids)
        if len(embeddings) == 0 or not ref_embeddings:
            return []
        refs = np.stack([e / np.linalg.norm(e) for e in ref_embeddings])

        if _HAS_FAISS and len(embeddings) > 500:
            idx = faiss.IndexFlatIP(512)
            idx.add(np.ascontiguousarray(embeddings, dtype=np.float32))
            sims, inds = idx.search(
                np.ascontiguousarray(refs, dtype=np.float32), idx.ntotal)
            # sims/inds are sorted per query; scatter back to original order
            face_best = np.full(len(embeddings), -np.inf, dtype=np.float32)
            for i in range(len(refs)):
                np.maximum.at(face_best, inds[i], sims[i])
        else:
            face_best = (embeddings @ refs.T).max(axis=1)

        per_photo: dict[int, tuple[float, list]] = {}
        for m, s in zip(faces, face_best):
            pid = m["photo_id"]
            if pid in excluded:
                continue
            prev = per_photo.get(pid)
            if prev is None or s > prev[0]:
                per_photo[pid] = (float(s), m.get("bbox", []))
        results = [{"photo_id": pid, "score": sc, "bbox": bbox}
                   for pid, (sc, bbox) in per_photo.items() if sc >= threshold]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k] if top_k else results
