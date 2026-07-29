r"""Download InsightFace buffalo_l models into models/.

This avoids committing large ONNX files into the repository.
Run manually:
    python download_models.py
Or:
    .\download_models.ps1
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
REQUIRED_FILES = {
    "det_10g.onnx",
    "w600k_r50.onnx",
    "genderage.onnx",
    "2d106det.onnx",
    "1k3d68.onnx",
}


def download(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with a progress bar and resume support."""
    headers = {}
    mode = "wb"
    if dest.exists():
        headers["Range"] = f"bytes={dest.stat().st_size}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if "content-range" in r.headers:
            total = int(r.headers["content-range"].split("/")[-1])
        initial = dest.stat().st_size if dest.exists() and mode == "ab" else 0

        with open(dest, mode) as f, tqdm(
            total=total,
            initial=initial,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def verify(models_dir: Path) -> set[str]:
    """Return the set of missing required files."""
    existing = {p.name for p in models_dir.iterdir() if p.is_file()}
    return REQUIRED_FILES - existing


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download InsightFace buffalo_l model files")
    ap.add_argument(
        "--models-dir", default="models",
        help="Directory to place ONNX models (default: models)")
    ap.add_argument(
        "--url", default=DEFAULT_URL,
        help="URL of the buffalo_l.zip release")
    ap.add_argument(
        "--force", action="store_true",
        help="Re-download even if required files already exist")
    args = ap.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    missing = verify(models_dir)
    if not missing and not args.force:
        print(f"所有必需模型已存在于 {models_dir}，跳过下载。")
        return 0

    print(f"将下载模型到: {models_dir}")
    zip_path = models_dir / "buffalo_l.zip"

    # Clean up stale partial download if force requested
    if args.force and zip_path.exists():
        zip_path.unlink()

    try:
        download(args.url, zip_path)
    except requests.exceptions.RequestException as e:
        print(f"\n下载失败: {e}", file=sys.stderr)
        print("请手动下载 buffalo_l.zip 并解压到 models/ 目录:", file=sys.stderr)
        print(f"  {args.url}", file=sys.stderr)
        return 1

    print("\n解压模型...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(models_dir)

    # Remove the zip only if all files are present
    missing = verify(models_dir)
    if missing:
        print(f"\n警告: 解压后仍缺少文件: {missing}", file=sys.stderr)
        return 1

    zip_path.unlink()
    print(f"模型下载完成: {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
