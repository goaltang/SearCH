# AGENTS.md — Instructions for AI Coding Agents

> This file is for AI agents (Claude Code, Codex, Cursor, etc.) that need to
> install, run, test, or extend this project. Humans: read README.md instead.

## What This Project Is

SearCH (photofinder) is an event-photo face-search tool. Given an album URL
and a reference selfie, it finds every photo containing that person.
Stack: Python 3.10+, ONNX Runtime, SCRFD + ArcFace models, Gradio Web UI.

> **Helping with deployment, index building, or "updating photos"? Read
> [OPS.md](OPS.md) and the "Operations & Index-Building Gotchas" section below
> FIRST.** These tasks have non-obvious failure modes (e.g. a silently empty
> index) that are easy to get wrong.

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
| `PHOTOFINDER_FACE_BATCH` | `1` | Batched ArcFace recognition. **Set `0` for local index builds on Windows** (onnxruntime 1.28 batched inference crashes natively → silent 0-face index). Server/Docker keeps `1`. |

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

### "Update an album's photos" / "Add a new album" / "Deploy a change"
These are OPERATIONS tasks, not code changes — follow **[OPS.md](OPS.md)** and the
"Operations & Index-Building Gotchas" section below. In short: build the index
**locally with `PHOTOFINDER_FACE_BATCH=0`**, upload the index-only files into
`cache/<orderId>/` on the server (**never `rm -rf cache`** when >1 album exists),
then `docker restart photofinder` (index-only update) or
`git pull && docker compose up -d --build` (code change).

## Operations & Index-Building Gotchas (READ BEFORE any deploy/index task)

Full runbook: **[OPS.md](OPS.md)**. The failure modes below are non-obvious and
have actually bitten us — an agent that skips these will silently produce a
broken deployment.

- **Local index builds on Windows MUST set `PHOTOFINDER_FACE_BATCH=0`.**
  onnxruntime 1.28 crashes natively (uncatchable) on batched ArcFace inference,
  so a plain `--prepare` silently builds a **0-face index** (log just says
  `Indexed N new photos (0 faces)`, no error). Always:
  `set PHOTOFINDER_FACE_BATCH=0&& .venv\Scripts\python -m photofinder.cli --url <orderId> --prepare`.
  `index.py` now warns loudly on a 0-face build — if you see that warning, the
  index is empty and must NOT be uploaded. Server/Docker batches fine (leave default).
- **Do NOT build an index via `docker exec` inside the running webui container**
  — it gets OOM-killed (exit 137): the CLI loads a second set of models on top
  of the running app and blows the 3G container limit. Build locally (preferred)
  or `docker stop photofinder` → temp `docker run --rm ... --prepare` → `docker start`.
- **Multiple albums live under `cache/<orderId>/`.** The live deployment serves
  several albums (e.g. 省赛·毕节 `20260720172647201236` + 国赛·上海
  `20260727190944809942`). **Never `rm -rf cache`** when >1 album exists — it
  wipes the others. Add/update one album by writing only its `<orderId>/` subdir.
- **Workflow = build locally, upload index-only.** Zip the index files
  (photos.json, faces.npz, faces.json, done.json — NOT thumbs/) → `scp` from the
  LOCAL machine → unzip into `~/SearCH/cache/<orderId>/` on the server. `scp` run
  on the server fails (Host key verification) — it must run locally.
- **`thumbs/` is only for the green bbox overlay**, not for search. Skip it for a
  fast upload (results fall back to online preview URLs); upload it if the user
  wants face boxes drawn.
- **Index-only update → `docker restart photofinder`. Code change → `git pull` +
  `docker compose up -d --build`.** Don't rebuild the image for a pure index update.
- **Server facts:** project dir is `~/SearCH` (NOT `~/photofinder`); container
  name `photofinder`; the `photofinder` CLI exists only inside the image (a bare
  `photofinder` on the host = "command not found").

## Do NOT

- Do not install the `insightface` pip package
- Do not commit anything in `models/` (gitignored, 300MB+)
- Do not commit `cache/` or `logs/`
- Do not change the atomic-write pattern in `index.py`
- Do not remove concurrency locks in `pipeline.py`
- Do not hardcode platform URLs outside `crawler.py`
- Do not build a face index locally on Windows without `PHOTOFINDER_FACE_BATCH=0`
  (silent 0-face index — see Operations gotchas above)
- Do not build an index via `docker exec` in the running webui container (OOM, exit 137)
- Do not `rm -rf cache` when multiple albums exist (wipes the other albums' indexes)
