"""CLI entry:  python -m photofinder.cli --url <album_url> --ref <face.jpg>"""

from __future__ import annotations

import argparse
import json as _json
import sys
import time

from .logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _progress(stage: str, done: int, total: int | None):
    label = {"meta": "获取照片列表", "download": "下载照片",
             "index": "建立人脸索引"}.get(stage, stage)
    total_s = "?" if total is None else str(total)
    print(f"\r[{label}] {done}/{total_s}", end="", flush=True)


def main(argv=None):
    from .pipeline import DEFAULT_THRESHOLD, PhotoFinder

    ap = argparse.ArgumentParser(
        prog="photofinder",
        description="在一拍即传(yipai360)相册中按人脸查找目标人物的全部照片")
    ap.add_argument("--url", required=True, help="相册链接或 orderId")
    ap.add_argument("--ref", nargs='+', default=None,
                    help="参考人脸照片路径，可传多张（--prepare 模式下可省略）")
    ap.add_argument("--prepare", action="store_true",
                    help="仅下载照片并建立人脸索引，不进行搜索（无需 --ref）")
    ap.add_argument("--max-photos", type=int, default=None,
                    help="只处理前 N 张照片(调试用)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="人脸相似度阈值 (默认 %(default)s)")
    ap.add_argument("--refresh", action="store_true", help="重新拉取照片列表")
    ap.add_argument("--incremental", action="store_true",
                    help="仅拉取上次之后新增的照片")
    ap.add_argument("--pwd", default=None, help="相册密码(如有)")
    ap.add_argument("--workers", type=int, default=4, help="索引进程并发数")
    ap.add_argument("--min-face", type=float, default=24.0,
                    help="最小人脸宽度像素, 过滤误检 (默认 24)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="清空人脸索引并重建 (不重新下载图片)")
    ap.add_argument("--exclude", type=int, nargs='*', default=None,
                    help="排除指定 photoId (误命中)")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="以 JSON 格式输出结果")
    ap.add_argument("--cache", default="cache", help="缓存目录")
    ap.add_argument("--models", default="models", help="模型目录")
    args = ap.parse_args(argv)

    finder = PhotoFinder(cache_root=args.cache, models_dir=args.models)

    # ── prepare 模式：仅下载 + 建索引 ──
    if args.prepare:
        from .crawler import AlbumCrawler, parse_order_id
        from .index import FaceIndex
        order_id = parse_order_id(args.url)
        crawler = AlbumCrawler(order_id, args.cache, pwd=args.pwd)
        print(f"[准备模式] 相册 {order_id}")
        photos = crawler.get_metadata(max_photos=args.max_photos,
                                      refresh=args.refresh,
                                      progress_cb=_progress)
        print(f"\n照片列表: {len(photos)} 张")
        thumbs = crawler.download_thumbs(photos, progress_cb=_progress)
        available = {pid: p for pid, p in thumbs.items()
                     if p is not None and p.exists()}
        print(f"\n缩略图就绪: {len(available)} 张")
        index = FaceIndex(crawler.dir)
        index.build(finder.engine, available, workers=args.workers,
                    min_face=args.min_face, progress_cb=_progress)
        print(f"\n索引完成: {len(index.faces)} 张人脸, "
              f"{len(index.done_ids)} 张照片已处理")
        return 0

    if not args.ref:
        ap.error("搜索模式下必须提供 --ref 参考照片（或使用 --prepare 仅建索引）")

    if args.rebuild_index:
        from .crawler import parse_order_id
        from .index import FaceIndex
        FaceIndex(finder.cache_root / parse_order_id(args.url)).reset()
        print("索引已清空, 将重建")
    t0 = time.time()
    print(f"\r[提取参考人脸特征] {args.ref}", flush=True)
    logger.info("CLI start: url=%s refs=%d", args.url, len(args.ref))
    try:
        results = finder.run(
            args.url, args.ref, max_photos=args.max_photos,
            threshold=args.threshold, refresh=args.refresh, pwd=args.pwd,
            workers=args.workers, min_face=args.min_face,
            progress_cb=_progress, excluded_ids=args.exclude,
            incremental=args.incremental)
    except ValueError as e:
        logger.error("Pipeline error: %s", e)
        print(f"\n错误: {e}", file=sys.stderr)
        return 2
    except Exception as exc:
        logger.exception("Unexpected error during search")
        print(f"\n发生错误: {exc}", file=sys.stderr)
        return 3

    elapsed = time.time() - t0
    logger.info("CLI finish: %d hits in %.1fs", len(results), elapsed)

    if args.as_json:
        out = [{"photo_id": r.photo_id, "fname": r.fname, "score": r.score,
                "full_url": r.full_url, "preview_url": r.preview_url,
                "bbox": r.bbox} for r in results]
        print(_json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"\n\n完成, 耗时 {elapsed:.1f}s, "
              f"命中 {len(results)} 张 (阈值 {args.threshold})\n")
        print(f"{'score':>6}  {'文件名':<20}  原图链接")
        for r in results:
            print(f"{r.score:>6.3f}  {r.fname:<20}  {r.full_url}")
        if results:
            print(f"\n相册原网页: {results[0].album_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
