"""Face embedding index: build incrementally from cached thumbs, search by cosine similarity.

Persistence per album (cache/{orderId}/):
    faces.npz      - float32 (N, 512) embedding matrix, aligned with faces.json
    faces.json     - [{photo_id, bbox, det_score}] one row per detected face
    done.json      - [photo_id] all photos already processed (incl. no-face ones)
    excluded.json  - [photo_id] user-marked false positives to skip in search
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .face_engine import FaceEngine
from .logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FaceIndex:
    def __init__(self, index_dir: str | Path):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
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
        logger.debug("Saving index: %d faces, %d done, %d excluded",
                     len(self.faces), len(self.done_ids), len(self.excluded_ids))
        np.savez_compressed(self.dir / "faces.npz",
                            embeddings=self.embeddings.astype(np.float32))
        (self.dir / "faces.json").write_text(
            json.dumps(self.faces, ensure_ascii=False), encoding="utf-8")
        (self.dir / "done.json").write_text(
            json.dumps(sorted(self.done_ids)), encoding="utf-8")
        (self.dir / "excluded.json").write_text(
            json.dumps(sorted(self.excluded_ids)), encoding="utf-8")

    def reset(self) -> None:
        """Drop all indexed data (keeps downloaded thumbs)."""
        logger.info("Resetting face index in %s", self.dir)
        self.embeddings = np.zeros((0, 512), dtype=np.float32)
        self.faces = []
        self.done_ids = set()
        self.excluded_ids = set()
        for name in ("faces.npz", "faces.json", "done.json", "excluded.json"):
            p = self.dir / name
            if p.exists():
                p.unlink()

    # -------------------------------------------------------------- exclusion
    def exclude(self, photo_ids: list[int]) -> None:
        """Mark photos as false positives; they will be skipped in search."""
        self.excluded_ids.update(photo_ids)
        self.save()

    def unexclude(self, photo_ids: list[int]) -> None:
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

        todo = {pid: p for pid, p in valid.items() if pid not in self.done_ids}
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

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, item) for item in todo.items()]
            for n, fut in enumerate(as_completed(futs), 1):
                pid, faces = fut.result()
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
        logger.info("Indexed %d new photos (%d faces) in %s",
                    total, len(new_faces), self.dir)
        return total

    def _absorb(self, new_embs, new_faces, finished) -> None:
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
        if len(self.embeddings) == 0 or not ref_embeddings:
            return []
        refs = np.stack([e / np.linalg.norm(e) for e in ref_embeddings])

        if _HAS_FAISS and len(self.embeddings) > 500:
            idx = faiss.IndexFlatIP(512)
            idx.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
            sims, _ = idx.search(
                np.ascontiguousarray(refs, dtype=np.float32), idx.ntotal)
            face_best = sims.max(axis=0)
        else:
            face_best = (self.embeddings @ refs.T).max(axis=1)

        per_photo: dict[int, tuple[float, list]] = {}
        for m, s in zip(self.faces, face_best):
            pid = m["photo_id"]
            if pid in self.excluded_ids:
                continue
            prev = per_photo.get(pid)
            if prev is None or s > prev[0]:
                per_photo[pid] = (float(s), m.get("bbox", []))
        results = [{"photo_id": pid, "score": sc, "bbox": bbox}
                   for pid, (sc, bbox) in per_photo.items() if sc >= threshold]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k] if top_k else results
