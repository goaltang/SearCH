<div align="center">

# 🔍 SearCH

### Find yourself in thousands of event photos — in seconds.

Paste an album link. Upload a selfie. Get every photo you appear in.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Live%20Demo-ff9d00)](http://47.76.47.229:7860/)

[English](README.md) · [中文](README_CN.md)

</div>

---

## ✨ What is this?

Event photographers upload hundreds (sometimes thousands) of photos to sharing platforms. Finding *yourself* in that sea of images is painful.

**SearCH** does it for you:

1. Paste the album URL
2. Upload one (or more) reference photos of your face
3. Click search → get every photo you appear in, ranked by similarity

All face detection and recognition runs **locally on your machine** (or your server). No photos are uploaded to any third-party service.

### Supported Platforms

| Platform | Status |
|----------|--------|
| [一拍即传 (Yipai360)](https://www.yipai360.com/) | ✅ Fully supported — auto-paginates lazy-loaded albums, downloads 720px thumbnails |

Using a different photo-sharing platform? The crawler layer is pluggable — see [Adapting to Other Platforms](#-roadmap--adapting-to-other-platforms) below, or open an issue.

<div align="center">

<img src="https://github.com/user-attachments/assets/3c72d231-b0fc-4856-8aaa-cbef84cef125" alt="Main interface — paste album URL and upload reference photo" width="700" />

*Paste an album link, upload a selfie, hit search.*

<br/>

<img src="https://github.com/user-attachments/assets/34319a31-21b2-4dc3-9b2f-04d5238e78be" alt="Search results — face matches ranked by similarity" width="700" />

*Every photo you appear in — ranked by similarity, with annotated face boxes.*

<br/>

<img src="https://github.com/user-attachments/assets/e3104cae-f46f-4feb-bc68-4fca90947a70" alt="Batch download all matches as zip" width="420" />

*One click to pack all matches into a zip.*

</div>

## 🧠 How it works

```
Album URL ──► Parse API ──► Paginate all photo metadata (handles lazy-load/SPA)
                                    │
Reference photo ─► SCRFD detect ─► ArcFace 512-d embedding (batch inference) ─┐
                                                                              ├─► Cosine similarity ─► Per-photo max aggregation
Download 720px thumbs (concurrent + resume) ─► Detect faces ─► Incremental face index (disk cache)
                                                                              │
Results: annotated preview + similarity score + full-res link + album link ───┘
```

| Component | Implementation |
|-----------|---------------|
| Face detection | SCRFD `det_10g` (InsightFace buffalo_l, ONNX) |
| Face recognition | ArcFace `w600k_r50` (512-d, cosine similarity) |
| Inference | ONNX Runtime — auto-detects CUDA → DirectML → CPU |
| Vector search | NumPy (≤500 faces) / FAISS (>500 faces, auto-switch) |
| Web UI | Gradio with custom design, mobile-responsive |
| Deployment | Docker, 2-core/4GB server is enough |

**Key design decisions:**

- **No insightface Python package** — raw ONNX inference with hand-written pre/post-processing (anchor decoding, NMS, affine alignment). Lighter dependency, full control.
- **Incremental indexing** — processed photos are never recomputed. Interrupted builds resume from checkpoint. New photos added mid-event are picked up automatically.
- **Concurrency-safe** — per-album locks, atomic file writes, configurable ORT threads, LRU index cache. Multiple users can search simultaneously without corrupting the index.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- (Windows) [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Install

```bash
git clone https://github.com/goaltang/SearCH.git
cd SearCH
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python download_models.py --models-dir models       # ~300MB, one-time
```

### Run

```bash
python -m photofinder.webui
# Open http://127.0.0.1:7860
```

### CLI

```bash
photofinder --url "https://www.yipai360.com/photolivepc/?orderId=YOUR_ID" --ref selfie.jpg
```

## 🌐 Self-Host for Events

Deploy it so event attendees can search their own photos from their phones:

```bash
docker compose up -d --build
# Share http://YOUR_SERVER_IP:7860 in the event group chat
```

Pre-build the face index once → users get results in **2-3 seconds**.
A 2-core/4GB server handles dozens of concurrent users. Full guide: [DEPLOY.md](DEPLOY.md)

## 📂 Project Structure

```
photofinder/
├── crawler.py      # Album API reverse-engineering + concurrent downloads
├── face_engine.py  # SCRFD + ArcFace, raw ONNX (no insightface pkg)
├── index.py        # Incremental face index, FAISS-accelerated search
├── pipeline.py     # End-to-end orchestration, concurrency control
├── cli.py          # Command-line interface
├── webui.py        # Gradio Web UI (mobile-responsive)
└── logger.py       # Structured logging
tests/              # 1094 lines of unit tests (pytest)
```

## ⚙️ Configuration

| Env Variable | Default | Description |
|---|---|---|
| `PHOTOFINDER_ACCESS_CODE` | (empty) | Access code gate, empty = open |
| `PHOTOFINDER_MAX_CONCURRENT` | `3` | Max simultaneous searches |
| `PHOTOFINDER_ORT_THREADS` | `min(4, cores)` | ONNX intra-op threads |
| `PHOTOFINDER_DOWNLOAD_MAX` | `300` | Max photos per zip download |
| `PHOTOFINDER_INDEX_CACHE` | `4` | Hot album indexes in memory |

## 🗺️ Roadmap / Adapting to Other Platforms

The crawler layer (`crawler.py`) is the only platform-specific piece. To support a different photo-sharing platform:

1. Implement `fetch_metadata()` → return a list of `Photo` dataclasses
2. Implement `download_thumbs()` → save images to disk
3. Everything downstream (detection, indexing, search, UI) works unchanged

PRs for new platform adapters are welcome.

## 🔒 Privacy

- All computation is local. No photos or embeddings leave your machine/server.
- Reference photos uploaded by users are used for the current search only — never persisted.
- You are responsible for obtaining consent from people being searched.

## 📄 License

MIT — do whatever you want with it.
The InsightFace buffalo_l models are subject to [their own license](https://github.com/deepinsight/insightface).

---

<div align="center">

**If this saved you from manually scrolling through 3000 event photos, consider a ⭐**

</div>
