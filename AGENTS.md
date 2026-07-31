# AGENTS.md — Instructions for AI Coding Agents

> This file is for AI agents (Claude Code, Codex, Cursor, etc.) that need to
> install, run, test, or extend this project. Humans: read README.md instead.

## What This Project Is

SearCH (photofinder) is an event-photo face-search tool. Given an album URL
and a reference selfie, it finds every photo containing that person.
Stack: Python 3.10+, ONNX Runtime, SCRFD + ArcFace models, Gradio Web UI.

## Install (step by step)

```bash
# 1. Clone
git clone https://github.com/goaltang/SearCH.git
cd SearCH

# 2. Create virtualenv (REQUIRED — do not install into system Python)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install package + dependencies
pip install -e .

# 4. Download ONNX models (~300MB, one-time)
python download_models.py --models-dir models
```

### Install pitfalls

- Python must be 3.10+. Check with `python --version` first.
- On Windows, the Visual C++ Redistributable is required for onnxruntime.
  Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- If `pip install onnxruntime` fails on ARM/Apple Silicon, try
  `pip install onnxruntime-silicon` or `pip install onnxruntime-openvino`.
- The model download script needs internet access. Models go into `models/`.
  If download fails, manual URLs are in `download_models.py`.
- Do NOT `pip install insightface` — this project intentionally does NOT use
  the insightface Python package. It loads raw ONNX files directly.

## Run

### Web UI (primary interface)
```bash
python -m photofinder.webui
# Serves on http://127.0.0.1:7860
```

### CLI
```bash
photofinder --url "https://www.yipai360.com/photolivepc/?orderId=XXXX" --ref selfie.jpg
```

### Docker (production)
```bash
docker compose up -d --build
# Serves on http://0.0.0.0:7860
```

## Verify Installation

```bash
# Run the test suite (no GPU or network required for unit tests)
pip install -e ".[dev]"
pytest tests/ -v

# Quick smoke test: import the engine (will fail if models are missing)
python -c "from photofinder.face_engine import FaceEngine; print('OK')"
```

## Project Structure

```
photofinder/
├── crawler.py      # Album API client (platform-specific)
├── face_engine.py  # SCRFD detection + ArcFace embedding (raw ONNX)
├── index.py        # Incremental face index + similarity search
├── pipeline.py     # End-to-end orchestration + concurrency control
├── cli.py          # CLI entry point
├── webui.py        # Gradio Web UI
└── logger.py       # Logging
tests/              # pytest suite
models/             # ONNX model files (gitignored, ~300MB)
```

## Key Architecture Rules (for agents modifying code)

- `face_engine.py` does raw ONNX inference. Do NOT add `import insightface`.
- `crawler.py` is the ONLY platform-specific module. All other modules are
  platform-agnostic. To support a new platform, subclass/replace crawler only.
- `index.py` uses NumPy for ≤500 faces, auto-switches to FAISS above that.
- All file writes are atomic (write tmp → rename). Preserve this pattern.
- Concurrency is controlled by `PHOTOFINDER_MAX_CONCURRENT` env var (default 3).
- The pipeline uses per-album locks. Do not remove them.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PHOTOFINDER_ACCESS_CODE` | (empty) | Access gate for Web UI |
| `PHOTOFINDER_MAX_CONCURRENT` | `3` | Max parallel searches |
| `PHOTOFINDER_ORT_THREADS` | `min(4, cores)` | ONNX Runtime intra-op threads |
| `PHOTOFINDER_DOWNLOAD_MAX` | `300` | Max photos per zip download |
| `PHOTOFINDER_INDEX_CACHE` | `4` | LRU cache size for hot album indexes |
| `PHOTOFINDER_DATA_DIR` | `cache/` | Where indexes and thumbnails are stored |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v          # All tests
pytest tests/ -k engine   # Face engine tests only
pytest tests/ -x          # Stop at first failure
```

Tests do NOT require GPU, network, or model files (they use mocks).
Exception: `test_face_engine.py` integration tests need models in `models/`.

## Common Agent Tasks

### "Add support for a new photo platform"
1. Create `photofinder/crawlers/newplatform.py`
2. Implement `fetch_metadata(album_url) -> list[Photo]`
3. Implement `download_thumbs(photos, output_dir, max_workers=4)`
4. Wire it into `pipeline.py` platform detection (URL-based routing)
5. Add tests in `tests/test_crawler_newplatform.py`

### "Optimize for low-memory server"
- Set `PHOTOFINDER_ORT_THREADS=1`
- Set `PHOTOFINDER_INDEX_CACHE=1`
- The index uses disk-backed storage; only embeddings are in RAM

### "Add batch face registration (search multiple people)"
- `webui.py` already accepts multiple reference images
- `face_engine.py:extract_embeddings()` handles batch input
- Results are aggregated per-photo with max-similarity across all references

## Do NOT

- Do not install the `insightface` pip package
- Do not commit anything in `models/` (gitignored, 300MB+)
- Do not commit `cache/` or `logs/`
- Do not change the atomic-write pattern in `index.py`
- Do not remove concurrency locks in `pipeline.py`
- Do not hardcode platform URLs outside `crawler.py`
