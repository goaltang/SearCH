"""Unit tests for photofinder.face_engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from photofinder import face_engine as fe


# --------------------------------------------------------------------------- helpers

class FakeInferenceSession:
    """Base fake onnxruntime InferenceSession for SCRFD/ArcFace."""

    def __init__(self, model_path: str | Path, providers=None):
        self.model_path = str(model_path)

    def get_inputs(self):
        raise NotImplementedError

    def get_outputs(self):
        raise NotImplementedError

    def run(self, output_names, feed_dict):
        raise NotImplementedError


class FakeSCRFDSession(FakeInferenceSession):
    def get_inputs(self):
        return [type("Inp", (), {"name": "input"})()]

    def get_outputs(self):
        names = [
            "score_8", "score_16", "score_32",
            "bbox_8", "bbox_16", "bbox_32",
            "kps_8", "kps_16", "kps_32",
        ]
        return [type("Out", (), {"name": n})() for n in names]


class FakeArcFaceSession(FakeInferenceSession):
    def __init__(self, model_path, providers=None):
        super().__init__(model_path, providers)
        self._embedding = np.random.rand(1, 512).astype(np.float32)

    def get_inputs(self):
        return [type("Inp", (), {"name": "data", "shape": [1, 3, 112, 112]})()]

    def get_outputs(self):
        return [type("Out", (), {"name": "fc1"})()]

    def run(self, output_names, feed_dict):
        return [self._embedding]


class FakeCombinedSession(FakeInferenceSession):
    """Pretends to be SCRFD for detection models and ArcFace for recognition models."""

    def __init__(self, model_path, providers=None):
        super().__init__(model_path, providers)
        self._net_outs: list | None = None

    def get_inputs(self):
        if "det" in self.model_path:
            return [type("Inp", (), {"name": "input"})()]
        return [type("Inp", (), {"name": "data", "shape": [1, 3, 112, 112]})()]

    def get_outputs(self):
        if "det" in self.model_path:
            names = [
                "score_8", "score_16", "score_32",
                "bbox_8", "bbox_16", "bbox_32",
                "kps_8", "kps_16", "kps_32",
            ]
            return [type("Out", (), {"name": n})() for n in names]
        return [type("Out", (), {"name": "fc1"})()]

    def run(self, output_names, feed_dict):
        if "det" in self.model_path:
            return self._net_outs if self._net_outs is not None else []
        return [np.random.rand(1, 512).astype(np.float32)]


@pytest.fixture
def no_ort(monkeypatch):
    """Replace onnxruntime with a minimal stub."""
    monkeypatch.setattr(fe, "ort", type("ort", (), {"InferenceSession": FakeInferenceSession}))


# --------------------------------------------------------------------------- geometry helpers

def test_distance2bbox():
    points = np.array([[2.0, 2.0]], dtype=np.float32)
    distance = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    boxes = fe._distance2bbox(points, distance)
    np.testing.assert_array_equal(boxes, [[1.0, 1.0, 3.0, 3.0]])


def test_distance2kps():
    points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    distance = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [-1.0, -2.0, -3.0, -4.0],
    ], dtype=np.float32)
    kps = fe._distance2kps(points, distance)
    expected = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.0, -1.0, -2.0, -3.0],
    ], dtype=np.float32)
    np.testing.assert_array_equal(kps, expected)


def test_nms_suppresses_overlapping_boxes():
    dets = np.array([
        [0, 0, 10, 10, 0.9],
        [1, 1, 11, 11, 0.8],
        [20, 20, 30, 30, 0.7],
    ], dtype=np.float32)
    keep = fe._nms(dets, thresh=0.5)
    assert keep == [0, 2]


# --------------------------------------------------------------------------- SCRFD

def test_scrfd_anchor_shapes(no_ort, monkeypatch):
    monkeypatch.setattr(fe.ort, "InferenceSession", FakeSCRFDSession)
    det = fe.SCRFD("dummy.onnx", input_size=(160, 160))
    anchors = det._anchors(160, 160)
    counts = [a.shape[0] for a in anchors]
    # 20x20x2, 10x10x2, 5x5x2
    assert counts == [800, 200, 50]
    for a in anchors:
        assert a.shape[1] == 2
        assert a.dtype == np.float32


def test_scrfd_detect_with_synthetic_output(no_ort, monkeypatch):
    monkeypatch.setattr(fe.ort, "InferenceSession", FakeSCRFDSession)
    det = fe.SCRFD("dummy.onnx", input_size=(160, 160), det_thresh=0.5)

    # 160x160 input -> anchor counts [800, 200, 50]
    def _zeros(count):
        return np.zeros((count, 1), dtype=np.float32)

    def _bbox(count):
        return np.zeros((count, 4), dtype=np.float32)

    def _kps(count):
        return np.zeros((count, 10), dtype=np.float32)

    scores8, scores16, scores32 = _zeros(800), _zeros(200), _zeros(50)
    bbox8, bbox16, bbox32 = _bbox(800), _bbox(200), _bbox(50)
    kps8, kps16, kps32 = _kps(800), _kps(200), _kps(50)

    # place a single face at stride-16 grid position (5, 5)
    # 160/16 = 10, so index for (x=5, y=5), anchor 0 is ((5*10)+5)*2 = 110
    center_idx = 110
    scores16[center_idx] = 0.99
    # bbox and kps predictions are multiplied by stride (16) inside detect()
    bbox16[center_idx] = np.array([0.625, 0.625, 0.625, 0.625], dtype=np.float32)
    kps16[center_idx] = np.array(
        [-0.3125, -0.3125, 0.3125, -0.3125, 0, 0, -0.3125, 0.3125, 0.3125, 0.3125], dtype=np.float32
    )

    net_outs = [scores8, scores16, scores32, bbox8, bbox16, bbox32, kps8, kps16, kps32]
    monkeypatch.setattr(det.session, "run", lambda names, feed: net_outs)

    # original image is 100x100; model input 160x160 => det_scale = 1.6
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes, kpss = det.detect(img)

    assert bboxes.shape[0] == 1
    assert kpss.shape[0] == 1
    # center anchor is at (80, 80) in 160x160 space -> (50, 50) original
    # box is center +/- 10 in 160x160 -> /1.6 = +/- 6.25 in original
    np.testing.assert_allclose(bboxes[0, :4], [43.75, 43.75, 56.25, 56.25], atol=1e-4)
    assert bboxes[0, 4] == pytest.approx(0.99, rel=1e-4)
    # keypoints are also scaled back to original image coordinates
    np.testing.assert_allclose(
        kpss[0],
        [46.875, 46.875, 53.125, 46.875, 50.0, 50.0, 46.875, 53.125, 53.125, 53.125],
        atol=1e-4,
    )


# --------------------------------------------------------------------------- ArcFace

def test_arcface_get_normalizes_embedding(no_ort, monkeypatch):
    monkeypatch.setattr(fe.ort, "InferenceSession", FakeArcFaceSession)
    rec = fe.ArcFace("dummy.onnx")
    raw = rec.session._embedding[0].copy()

    img = np.zeros((112, 112, 3), dtype=np.uint8)
    kps = np.array([[38, 51], [73, 51], [56, 71], [41, 92], [70, 92]], dtype=np.float32)
    emb = rec.get(img, kps)

    assert emb.shape == (512,)
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-6)
    np.testing.assert_allclose(emb, raw / np.linalg.norm(raw), atol=1e-6)


# --------------------------------------------------------------------------- FaceEngine

def test_face_engine_process_sorts_by_face_area(no_ort, monkeypatch):
    monkeypatch.setattr(fe.ort, "InferenceSession", FakeCombinedSession)
    engine = fe.FaceEngine("det.onnx", "rec.onnx")

    def fake_detect(img, max_num=0, metric="default"):
        bboxes = np.array([
            [0, 0, 5, 5, 0.7],
            [0, 0, 10, 10, 0.9],
        ], dtype=np.float32)
        kps = np.zeros((2, 10), dtype=np.float32)
        return bboxes, kps

    def fake_get(img, kps):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(engine.det, "detect", fake_detect)
    monkeypatch.setattr(engine.rec, "get", fake_get)

    faces = engine.process(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(faces) == 2
    # largest face first
    assert faces[0]["bbox"] == [0, 0, 10, 10]
    assert faces[1]["bbox"] == [0, 0, 5, 5]


def test_face_engine_process_returns_empty_for_none():
    engine = fe.FaceEngine.__new__(fe.FaceEngine)
    assert engine.process(None) == []


def test_face_engine_reference_embedding_returns_largest_face(no_ort, monkeypatch):
    monkeypatch.setattr(fe.ort, "InferenceSession", FakeCombinedSession)
    engine = fe.FaceEngine("det.onnx", "rec.onnx")

    embeddings = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]

    def fake_detect(img, max_num=0, metric="default"):
        # largest face first so embeddings line up with popping order
        bboxes = np.array([
            [0, 0, 10, 10, 0.9],
            [0, 0, 5, 5, 0.7],
        ], dtype=np.float32)
        kps = np.zeros((2, 10), dtype=np.float32)
        return bboxes, kps

    def fake_get(img, kps):
        return embeddings.pop(0)

    monkeypatch.setattr(engine.det, "detect", fake_detect)
    monkeypatch.setattr(engine.rec, "get", fake_get)

    ref = engine.reference_embedding(np.zeros((100, 100, 3), dtype=np.uint8))
    np.testing.assert_array_equal(ref, np.array([1.0, 0.0], dtype=np.float32))


def test_face_engine_reference_embedding_returns_none_when_no_faces(no_ort, monkeypatch):
    monkeypatch.setattr(fe.ort, "InferenceSession", FakeCombinedSession)
    engine = fe.FaceEngine("det.onnx", "rec.onnx")
    monkeypatch.setattr(engine.det, "detect", lambda *args, **kwargs: (np.zeros((0, 5)), np.zeros((0, 10))))
    assert engine.reference_embedding(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_face_engine_process_file_returns_empty_on_error(no_ort, monkeypatch, caplog, tmp_path: Path):
    monkeypatch.setattr(fe.ort, "InferenceSession", FakeCombinedSession)
    engine = fe.FaceEngine("det.onnx", "rec.onnx")
    missing = tmp_path / "missing.jpg"
    with caplog.at_level("WARNING", logger="photofinder.face_engine"):
        faces = engine.process_file(missing)
    assert faces == []
    assert any("Failed to read image" in rec.message for rec in caplog.records)
