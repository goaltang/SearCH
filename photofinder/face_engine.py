"""Face detection (SCRFD) + recognition (ArcFace) using InsightFace ONNX models.

Models (buffalo_l pack, https://github.com/deepinsight/insightface):
    det_10g.onnx    - SCRFD face detector, 640x640 input, strides 8/16/32
    w600k_r50.onnx  - ArcFace R50, 112x112 aligned input, 512-d embedding
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# ONNX Runtime intra-op threads. Default: at most 4 so that several
# concurrent searches don't oversubscribe the CPU (ORT's own default is
# "all cores per session", which thrashes under concurrency).
# Override with PHOTOFINDER_ORT_THREADS.
_ORT_THREADS = int(os.environ.get(
    "PHOTOFINDER_ORT_THREADS",
    str(min(4, os.cpu_count() or 4))))

# Batched ArcFace inference is faster, but some onnxruntime builds crash
# (native, uncatchable) on batch>1 — observed with onnxruntime 1.28 on
# Windows. Set PHOTOFINDER_FACE_BATCH=0 to force safe per-face recognition
# for offline index builds. Live search embeds a single reference face, so
# it is unaffected either way.
_FACE_BATCH = os.environ.get("PHOTOFINDER_FACE_BATCH", "1").strip() != "0"


def _session_options() -> "ort.SessionOptions":
    so = ort.SessionOptions()
    so.intra_op_num_threads = _ORT_THREADS
    so.inter_op_num_threads = 1
    return so


def _default_providers() -> list[str]:
    """Auto-detect the best available ONNX execution provider."""
    try:
        available = ort.get_available_providers()
    except AttributeError:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

# ArcFace reference landmarks for 112x112 alignment
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = (dets[:, i] for i in range(5))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class SCRFD:
    def __init__(self, model_path: str | Path, input_size=(640, 640),
                 det_thresh=0.5, nms_thresh=0.4, providers=None):
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_size = tuple(input_size)  # (w, h)
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.session = ort.InferenceSession(
            str(model_path), sess_options=_session_options(),
            providers=providers or _default_providers())
        self.input_name = self.session.get_inputs()[0].name
        out_names = [o.name for o in self.session.get_outputs()]
        self.output_names = out_names  # score_8/16/32, bbox_*, kps_*
        self._anchor_cache: dict[tuple[int, int], list[np.ndarray]] = {}

    # insightface scrfd anchor generation
    def _anchors(self, height: int, width: int) -> list[np.ndarray]:
        key = (height, width)
        if key in self._anchor_cache:
            return self._anchor_cache[key]
        anchors = []
        for stride in self._feat_stride_fpn:
            fh, fw = height // stride, width // stride
            ax = np.arange(fw, dtype=np.float32)
            ay = np.arange(fh, dtype=np.float32)
            xv, yv = np.meshgrid(ax, ay)
            anchor = np.stack((xv, yv), axis=-1).reshape(-1, 2)
            # position-outer / anchor-inner interleaving (matches the
            # flattened SCRFD onnx outputs: [p0a0, p0a1, p1a0, p1a1, ...])
            anchor = np.stack(
                [anchor] * self._num_anchors, axis=1).reshape(-1, 2) * stride
            anchors.append(anchor)
        self._anchor_cache[key] = anchors
        return anchors

    def detect(self, img: np.ndarray, max_num: int = 0, metric: str = "default"):
        input_w, input_h = self.input_size
        im_h, im_w = img.shape[:2]
        im_ratio = im_h / im_w
        model_ratio = input_h / input_w
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / im_h
        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 127.5, self.input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names,
                                    {self.input_name: blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        anchors = self._anchors(input_h, input_w)
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            kps_preds = net_outs[idx + self.fmc * 2] * stride
            anchor = anchors[idx]
            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes_list.append(_distance2bbox(anchor[pos_inds], bbox_preds[pos_inds]))
            scores_list.append(scores[pos_inds])
            kpss_list.append(_distance2kps(anchor[pos_inds], kps_preds[pos_inds]))

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32)
        keep = _nms(pre_det, self.nms_thresh)
        det = pre_det[keep, :]
        kpss = kpss[keep]
        if max_num > 0 and det.shape[0] > max_num:
            if metric == "max":
                area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                center = (img.shape[1] // 2, img.shape[0] // 2)
                offsets = np.vstack([
                    (det[:, 0] + det[:, 2]) / 2 - center[0],
                    (det[:, 3] + det[:, 1]) / 2 - center[1]])
                offset_dist = np.sum(offsets ** 2, axis=0)
                values = area - offset_dist * 2.0
            else:
                values = det[:, 4]
            order = values.argsort()[::-1][:max_num]
            det = det[order, :]
            kpss = kpss[order]
        return det, kpss


class ArcFace:
    def __init__(self, model_path: str | Path, providers=None):
        self.session = ort.InferenceSession(
            str(model_path), sess_options=_session_options(),
            providers=providers or _default_providers())
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = tuple(inp.shape[2:4][::-1])  # (112, 112)

    def _align(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        kps = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        M, _ = cv2.estimateAffinePartial2D(
            kps, ARCFACE_DST, method=cv2.LMEDS)
        return cv2.warpAffine(img, M, self.input_size, borderValue=0.0)

    def get(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        aligned = self._align(img, kps)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, self.input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        return emb / np.linalg.norm(emb)

    def get_batch(self, img: np.ndarray,
                  kpss_list: list[np.ndarray]) -> list[np.ndarray]:
        """Compute normalized embeddings for multiple faces in one ONNX call."""
        if not kpss_list:
            return []
        aligned = [self._align(img, kps) for kps in kpss_list]
        blob = cv2.dnn.blobFromImages(aligned, 1.0 / 127.5, self.input_size,
                                      (127.5, 127.5, 127.5), swapRB=True)
        embs = self.session.run(None, {self.input_name: blob})[0]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return list(embs / norms)


class FaceEngine:
    """Detect faces in an image and compute normalized 512-d embeddings."""

    def __init__(self, det_model: str | Path, rec_model: str | Path,
                 det_thresh: float = 0.5, det_size=(640, 640)):
        self.det = SCRFD(det_model, input_size=det_size, det_thresh=det_thresh)
        self.rec = ArcFace(rec_model)

    def process(self, img: np.ndarray) -> list[dict]:
        """Returns [{bbox(4), det_score, embedding(512)}], largest face first."""
        if img is None:
            return []
        try:
            bboxes, kpss = self.det.detect(img)
        except Exception as exc:
            logger.warning("Face detection failed: %s", exc)
            return []
        if len(bboxes) == 0:
            return []
        embs = None
        if _FACE_BATCH:
            try:
                embs = self.rec.get_batch(img, list(kpss))
            except Exception:
                embs = None  # fall through to per-face recognition
        if embs is None:
            # Per-face fallback: some ONNX exports are fixed batch=1, and some
            # onnxruntime builds crash on batch>1 (see _FACE_BATCH note).
            embs = []
            for kps in kpss:
                try:
                    embs.append(self.rec.get(img, kps))
                except Exception as exc:
                    logger.warning("Face recognition failed: %s", exc)
                    embs.append(None)
        faces = []
        for box, emb in zip(bboxes, embs):
            if emb is None:
                continue
            faces.append({
                "bbox": box[:4].astype(float).tolist(),
                "det_score": float(box[4]),
                "embedding": emb,
            })
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0])
                   * (f["bbox"][3] - f["bbox"][1]), reverse=True)
        return faces

    def process_file(self, path: str | Path) -> list[dict]:
        try:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.warning("Failed to read image %s: %s", path, exc)
            return []
        return self.process(img)

    def reference_embedding(self, img: np.ndarray) -> np.ndarray | None:
        """Embedding of the most prominent face in a reference photo."""
        faces = self.process(img)
        if not faces:
            logger.warning("No face detected in reference image")
            return None
        return faces[0]["embedding"]
