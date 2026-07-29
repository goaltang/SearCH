"""Gradio Web UI:  python -m photofinder.webui  ->  http://127.0.0.1:7860"""

from __future__ import annotations

import base64
import html
import os
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .logger import get_logger, setup_logging
from .pipeline import DEFAULT_THRESHOLD, PhotoFinder, SearchCancelled

setup_logging()
logger = get_logger(__name__)

FINDER = PhotoFinder()

# ── 部署配置（环境变量） ─────────────────────────────────────────
# PHOTOFINDER_ACCESS_CODE  访问码，留空则无需登录
# PHOTOFINDER_MAX_CONCURRENT  最大同时搜索数（默认 3）
# PHOTOFINDER_HOST  监听地址（默认 127.0.0.1，部署时设为 0.0.0.0）
# PHOTOFINDER_DOWNLOAD_MAX  单次打包下载的最大照片数（默认 300，防止 OOM）
ACCESS_CODE = os.environ.get("PHOTOFINDER_ACCESS_CODE", "")
MAX_CONCURRENT = int(os.environ.get("PHOTOFINDER_MAX_CONCURRENT", "3"))
SERVER_HOST = os.environ.get("PHOTOFINDER_HOST", "127.0.0.1")
DOWNLOAD_MAX = int(os.environ.get("PHOTOFINDER_DOWNLOAD_MAX", "300"))

STAGE_LABELS = {"meta": "获取照片列表", "download": "下载照片",
                "index": "建立人脸索引"}

# 常用活动相册链接（预填充，可修改）
DEFAULT_ALBUM_URL = ("https://www.yipai360.com/photolivepc/"
                     "?orderId=20260720172647201236&channel=h5&origin=qrcode")

def _force_light_theme(theme: gr.themes.Base) -> gr.themes.Base:
    """UI 仅按浅色设计：把主题的全部 dark 变量钉为对应的浅色值，
    避免系统深色模式下 Gradio 自动切换暗色导致界面混乱/黑边。"""
    overrides = {}
    for attr, val in list(vars(theme).items()):
        if attr.startswith("_") or not attr.endswith("_dark"):
            continue
        light_val = vars(theme).get(attr[:-5])
        if light_val is not None:
            overrides[attr] = light_val
    return theme.set(**overrides)


THEME = _force_light_theme(gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
))

CSS = """
:root {
  --pf-primary: #6366f1;
  --pf-primary-dark: #4f46e5;
  --pf-border: #e6e8f0;
  --pf-text: #1e293b;
  --pf-muted: #64748b;
}

body, .gradio-container {
  background: linear-gradient(180deg, #eceeff 0%, #f6f7ff 320px, #f8fafc 100%) !important;
}
.gradio-container {
  max-width: 1240px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 24px !important;
}

/* ---------- Hero ---------- */
.pf-hero { padding: 4px 6px 18px; }
.pf-hero-row { display: flex; align-items: center; gap: 16px; }
.pf-logo {
  width: 56px; height: 56px; border-radius: 16px; flex: none;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 8px 20px rgba(99, 102, 241, .35);
}
.pf-title {
  margin: 0; font-size: 30px; font-weight: 800; letter-spacing: .5px;
  background: linear-gradient(90deg, #312e81, #6d28d9);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.pf-sub { margin: 4px 0 0; color: var(--pf-muted); font-size: 14px; line-height: 1.7; }
.pf-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.pf-badge {
  font-size: 12px; padding: 4px 12px; border-radius: 999px;
  background: #eef2ff; color: #4338ca; border: 1px solid #e0e7ff;
}
.pf-badge--green { background: #ecfdf5; color: #047857; border-color: #d1fae5; }

/* ---------- Panels ---------- */
.pf-panel {
  background: #fff; border: 1px solid var(--pf-border); border-radius: 16px;
  padding: 20px; box-shadow: 0 4px 16px rgba(15, 23, 42, .05); height: 100%;
}
.pf-panel-title {
  margin: 0 0 14px; font-size: 15px; font-weight: 700; color: var(--pf-text);
  display: flex; align-items: center; gap: 8px;
}
.pf-panel-title .pf-step-no {
  width: 22px; height: 22px; border-radius: 8px; flex: none;
  background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff;
  font-size: 12px; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center;
}

.pf-hint {
  margin: 4px 0 8px; font-size: 12px; color: #94a3b8; line-height: 1.6;
}

/* ---------- Run button ---------- */
.pf-run-btn button {
  background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
  border: none !important; border-radius: 12px !important;
  font-size: 16px !important; font-weight: 700 !important; letter-spacing: 2px;
  box-shadow: 0 6px 16px rgba(99, 102, 241, .35);
  transition: transform .15s ease, box-shadow .15s ease;
}
.pf-run-btn button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(99, 102, 241, .45);
}

/* ---------- Welcome / empty states ---------- */
.pf-welcome, .pf-empty {
  text-align: center; padding: 48px 24px;
  background: #fbfcff; border: 1px dashed #d9dfea; border-radius: 14px;
}
.pf-welcome h3, .pf-empty h3 { margin: 14px 0 6px; color: var(--pf-text); font-size: 17px; }
.pf-welcome p, .pf-empty p { margin: 0; color: var(--pf-muted); font-size: 13px; }
.pf-steps {
  display: flex; flex-direction: column; gap: 10px;
  max-width: 320px; margin: 22px auto 0; text-align: left;
}
.pf-step {
  display: flex; align-items: center; gap: 10px;
  background: #f6f7ff; border: 1px solid #eef2ff; border-radius: 10px;
  padding: 10px 14px; font-size: 13px; color: #475569;
}
.pf-step span {
  width: 22px; height: 22px; border-radius: 50%; flex: none;
  background: var(--pf-primary); color: #fff;
  font-size: 12px; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center;
}

/* ---------- Result summary ---------- */
.pf-summary {
  display: flex; align-items: center; gap: 12px;
  padding: 2px 2px 16px; font-size: 14px; color: #475569;
}
.pf-count {
  background: linear-gradient(90deg, #6366f1, #8b5cf6); color: #fff;
  font-weight: 700; font-size: 13px; border-radius: 999px; padding: 4px 14px;
}
.pf-open-album {
  margin-left: auto; color: var(--pf-primary-dark); font-weight: 600;
  text-decoration: none; font-size: 13px;
}
.pf-open-album:hover { text-decoration: underline; }

/* ---------- Result cards ---------- */
.pf-grid { display: flex; flex-wrap: wrap; gap: 14px; }
.pf-card {
  width: 212px; background: #fff; border: 1px solid var(--pf-border);
  border-radius: 14px; overflow: hidden;
  box-shadow: 0 2px 8px rgba(15, 23, 42, .05);
  transition: transform .18s ease, box-shadow .18s ease;
}
.pf-card:hover { transform: translateY(-4px); box-shadow: 0 14px 28px rgba(15, 23, 42, .13); }
.pf-thumb { position: relative; display: block; }
.pf-thumb img { width: 100%; height: 142px; object-fit: cover; display: block; }
.pf-score {
  position: absolute; top: 8px; right: 8px;
  font-size: 12px; font-weight: 700; color: #fff;
  padding: 3px 9px; border-radius: 999px;
}
.pf-score--high { background: rgba(16, 185, 129, .94); }
.pf-score--mid { background: rgba(59, 130, 246, .94); }
.pf-score--low { background: rgba(148, 163, 184, .94); }
.pf-meta { padding: 10px 12px 11px; font-size: 12px; }
.pf-fname {
  font-weight: 600; color: var(--pf-text); font-size: 13px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pf-meta-row { display: flex; align-items: center; margin-top: 6px; color: #94a3b8; }
.pf-links { margin-left: auto; display: flex; gap: 10px; }
.pf-links a { color: var(--pf-primary-dark); text-decoration: none; font-weight: 600; }
.pf-links a:hover { text-decoration: underline; }
"""

_SVG_SEARCH = ("<svg viewBox='0 0 24 24' width='28' height='28' fill='none' "
               "stroke='#fff' stroke-width='2.2' stroke-linecap='round'>"
               "<circle cx='11' cy='11' r='7'/><path d='m20 20-3.5-3.5'/></svg>")

_SVG_PHOTO = ("<svg viewBox='0 0 24 24' width='52' height='52' fill='none' "
              "stroke='#a5b4fc' stroke-width='1.6' stroke-linecap='round' "
              "stroke-linejoin='round'><rect x='3' y='3' width='18' height='18' "
              "rx='3'/><circle cx='9' cy='9' r='2'/><path d='m21 15-4.5-4.5L6 21'/></svg>")

_SVG_NO_RESULT = ("<svg viewBox='0 0 24 24' width='52' height='52' fill='none' "
                  "stroke='#94a3b8' stroke-width='1.6' stroke-linecap='round'>"
                  "<circle cx='11' cy='11' r='7'/><path d='m20 20-3.5-3.5'/>"
                  "<path d='M8.5 8.5l5 5M13.5 8.5l-5 5'/></svg>")

HEADER_HTML = f"""
<div class="pf-hero">
  <div class="pf-hero-row">
    <div class="pf-logo">{_SVG_SEARCH}</div>
    <div>
      <h1 class="pf-title">活动照片找人</h1>
      <p class="pf-sub">粘贴 <b>一拍即传 (yipai360)</b> 活动相册链接，上传一张参考人脸照片，
        自动遍历相册全部照片（含懒加载），找出目标人物出现的所有瞬间。</p>
    </div>
  </div>
  <div class="pf-badges">
    <span class="pf-badge">SCRFD 人脸检测</span>
    <span class="pf-badge">ArcFace 512 维特征</span>
    <span class="pf-badge">余弦相似度检索</span>
    <span class="pf-badge pf-badge--green">AI 驱动 · 隐私保护</span>
  </div>
</div>
"""

WELCOME_HTML = f"""
<div class="pf-welcome">
  <div>{_SVG_PHOTO}</div>
  <h3>查找结果将在这里展示</h3>
  <p>命中照片会以卡片墙形式呈现，包含预览图、相似度和原图链接。</p>
  <div class="pf-steps">
    <div class="pf-step"><span>1</span>粘贴活动相册链接</div>
    <div class="pf-step"><span>2</span>用摄像头或上传添加参考照片</div>
    <div class="pf-step"><span>3</span>点击「开始查找」</div>
  </div>
</div>
"""

EMPTY_HTML = f"""
<div class="pf-empty">
  <div>{_SVG_NO_RESULT}</div>
  <h3>未找到匹配照片</h3>
  <p>试试降低相似度阈值，或换一张更清晰的正脸参考照。</p>
</div>
"""


def _score_class(score: float) -> str:
    if score >= 0.6:
        return "pf-score--high"
    if score >= 0.45:
        return "pf-score--mid"
    return "pf-score--low"


def _thumb_with_bbox(thumb_path: str, bbox: list[float],
                     max_width: int = 420) -> str | None:
    """Draw bbox on local thumbnail, return base64 data-URI (or None)."""
    if not bbox or len(bbox) < 4:
        return None
    try:
        img = cv2.imdecode(np.fromfile(thumb_path, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_width:
            s = max_width / w
            img = cv2.resize(img, (max_width, int(h * s)))
            bbox = [v * s for v in bbox]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 80), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    except Exception:
        return None


def _render_results(results, album_url: str) -> str:
    if not results:
        return EMPTY_HTML
    cards = []
    for r in results:
        fname = html.escape(r.fname)
        img_src = _thumb_with_bbox(r.thumb_path, r.bbox) or r.preview_url
        cards.append(f"""
<div class="pf-card">
  <a class="pf-thumb" href="{r.full_url}" target="_blank" title="查看原图">
    <img src="{img_src}" loading="lazy" alt="{fname}"/>
    <span class="pf-score {_score_class(r.score)}">{r.score:.3f}</span>
  </a>
  <div class="pf-meta">
    <div class="pf-fname" title="{fname}">{fname}</div>
    <div class="pf-meta-row">
      <span>ID {r.photo_id}</span>
      <span class="pf-links">
        <a href="{r.full_url}" target="_blank">原图</a>
        <a href="{album_url}" target="_blank">活动网页</a>
      </span>
    </div>
  </div>
</div>""")
    return (f"<div class='pf-summary'><span class='pf-count'>命中 "
            f"{len(results)} 张</span><span>按相似度从高到低排列</span>"
            f"<a class='pf-open-album' href='{album_url}' target='_blank'>"
            f"打开活动原网页 →</a></div>"
            f"<div class='pf-grid'>" + "".join(cards) + "</div>")


# ── quality feedback ──────────────────────────────────────────────
def check_quality(gallery_value):
    """Instant face-detection feedback for the current reference photos."""
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for img, _c in (gallery_value or [])]
    if not imgs:
        return ""
    parts = []
    for i, img in enumerate(imgs, 1):
        faces = FINDER.engine.process(img)
        if not faces:
            parts.append(f"第{i}张：<b style='color:#dc2626'>未检测到人脸</b>，"
                         f"请换一张清晰正脸")
        elif len(faces) > 1:
            parts.append(f"第{i}张：检测到 {len(faces)} 张人脸，"
                         f"将使用最大的一张")
        else:
            parts.append(f"第{i}张：<b style='color:#16a34a'>✓ 人脸清晰</b>")
    return "<p class='pf-hint'>" + "；".join(parts) + "</p>"


# ── search ────────────────────────────────────────────────────────
def run_search(url, gallery_value, threshold, max_photos, pwd,
               exclude_text, incremental, cancel_state,
               progress=gr.Progress()):
    if not url or not url.strip():
        raise gr.Error("请输入活动相册链接")
    ref_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for img, _cap in (gallery_value or [])]
    if not ref_imgs:
        raise gr.Error("请先添加至少一张参考人脸照片")

    if cancel_state is None:
        cancel_state = threading.Event()
    cancel_state.clear()
    max_n = int(max_photos) if max_photos and max_photos > 0 else None
    excl = [int(x) for x in (exclude_text or "").replace("，", ",").split(",")
            if x.strip().isdigit()] if exclude_text else None

    def cb(stage, done, total):
        if cancel_state.is_set():
            raise SearchCancelled("搜索已取消")
        label = STAGE_LABELS.get(stage, stage)
        frac = done / total if total else 0
        progress(frac, desc=f"{label} {done}/{total or '?'}")

    progress(0, desc="启动")
    logger.info("WebUI search: url=%s refs=%d", url.strip(), len(ref_imgs))
    try:
        results = FINDER.run(
            url.strip(), ref_imgs, max_photos=max_n,
            threshold=float(threshold), pwd=pwd or None,
            progress_cb=cb, cancel_event=cancel_state,
            excluded_ids=excl, incremental=bool(incremental))
    except SearchCancelled:
        logger.info("WebUI search cancelled by user")
        return ("<div class='pf-empty'><h3>搜索已取消</h3>"
                "<p>你可以随时重新开始查找。</p></div>"), [], cancel_state
    except ValueError as e:
        logger.error("WebUI search error: %s", e)
        raise gr.Error(str(e))
    except Exception as exc:
        logger.exception("Unexpected WebUI search error")
        raise gr.Error(f"搜索出错: {exc}")
    progress(1, desc="完成")
    from .crawler import album_url, parse_order_id
    logger.info("WebUI search finished: %d hits", len(results))
    return (_render_results(results, album_url(parse_order_id(url.strip()))),
            results, cancel_state)


def cancel_search(cancel_state):
    if cancel_state is not None:
        cancel_state.set()


# ── batch download ────────────────────────────────────────────────
# Zip archives are written to a dedicated temp dir (streamed to disk, never
# fully held in memory) and cleaned up periodically.
ZIP_DIR = Path(tempfile.gettempdir()) / "photofinder_zips"
ZIP_MAX_AGE = 3600  # seconds; archives older than this are deleted


def _cleanup_zips() -> None:
    try:
        ZIP_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for p in ZIP_DIR.glob("photofinder_*.zip"):
            try:
                if now - p.stat().st_mtime > ZIP_MAX_AGE:
                    p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def download_all(results):
    """Download all hit photos concurrently and pack them into a zip.

    Photos are fetched in small batches (peak memory = one batch, not the
    whole album) and streamed straight to an on-disk zip file.
    """
    if not results:
        raise gr.Error("没有可下载的结果，请先搜索")
    if len(results) > DOWNLOAD_MAX:
        raise gr.Error(
            f"命中 {len(results)} 张，超过单次打包上限 {DOWNLOAD_MAX} 张。"
            f"请提高相似度阈值缩小范围，或分批用「原图」链接下载")
    import requests as _req

    _cleanup_zips()
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ZIP_DIR / f"photofinder_{int(time.time())}_{os.getpid()}.zip"

    def _fetch(r):
        resp = _req.get(r.full_url, timeout=120)
        resp.raise_for_status()
        return r, resp.content

    ok, fail = 0, 0
    batch_size = 12  # bounds peak memory: ~12 x one photo's bytes
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED,
                             allowZip64=True) as zf, \
                ThreadPoolExecutor(max_workers=6) as ex:
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                futs = [ex.submit(_fetch, r) for r in batch]
                for fut in as_completed(futs):
                    try:
                        r, content = fut.result()
                        zf.writestr(r.fname or f"{r.photo_id}.jpg", content)
                        ok += 1
                    except Exception as exc:
                        fail += 1
                        logger.warning("Download failed: %s", exc)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if ok == 0:
        tmp.unlink(missing_ok=True)
        raise gr.Error(f"全部 {fail} 张下载失败，请检查网络或链接是否过期")
    logger.info("Batch download: %d ok, %d failed -> %s", ok, fail, tmp)
    return gr.update(value=str(tmp), visible=True)


def _append_snapshot(gallery_value, snapshot):
    """摄像头拍照/上传后自动追加到参考照片列表（不清空取景框，
    避免编程式重置干扰 Gradio 前端的摄像头状态）。"""
    if snapshot is None:
        return gallery_value
    items = list(gallery_value or [])
    items.append((snapshot, None))
    return items


def _undo_last(gallery_value):
    items = list(gallery_value or [])
    if items:
        items.pop()
    return items


def _clear_gallery():
    return []


def build_app() -> gr.Blocks:
    with gr.Blocks(title="活动照片找人") as app:
        gr.HTML(HEADER_HTML)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes=["pf-panel"]):
                gr.HTML("<p class='pf-panel-title'>"
                        "<span class='pf-step-no'>1</span>设置查找条件</p>")
                url = gr.Textbox(
                    label="活动相册链接",
                    value=DEFAULT_ALBUM_URL,
                    info="已预填常用活动链接，可直接修改",
                    lines=1, max_lines=1,
                    placeholder="https://www.yipai360.com/…")
                gr.HTML("<p class='pf-hint'>在下方拍照或上传，照片自动加入列表，"
                        "可反复添加多张；查找时取每张中最清晰的人脸</p>")
                ref_gallery = gr.Gallery(
                    label="已添加的参考照片",
                    type="numpy", columns=4, object_fit="cover",
                    height=160, interactive=False)
                quality_html = gr.HTML("")
                ref_input = gr.Image(
                    label="添加参考照片（拍照 / 上传）",
                    sources=["webcam", "upload"], type="numpy", height=220)
                with gr.Row():
                    undo_btn = gr.Button("↩ 撤销最后一张", size="sm")
                    clear_btn = gr.Button("✕ 清空全部", size="sm")
                with gr.Accordion("高级选项", open=False):
                    threshold = gr.Slider(
                        0.3, 0.8, value=DEFAULT_THRESHOLD, step=0.01,
                        label="相似度阈值 (越高越严格)")
                    max_photos = gr.Number(
                        value=0, precision=0,
                        label="最多处理照片数 (0 = 全部)")
                    pwd = gr.Textbox(label="相册密码 (如有)")
                    exclude_text = gr.Textbox(
                        label="排除的照片 ID (逗号分隔，误命中时填写)")
                    incremental = gr.Checkbox(
                        label="仅拉取新增照片 (活动进行中时勾选)", value=False)
                with gr.Row():
                    btn = gr.Button("开始查找", variant="primary", size="lg",
                                    elem_classes=["pf-run-btn"], scale=3)
                    cancel_btn = gr.Button("✕ 取消", size="lg", scale=1)
            with gr.Column(scale=8, elem_classes=["pf-panel"]):
                gr.HTML("<p class='pf-panel-title'>"
                        "<span class='pf-step-no'>2</span>查找结果</p>")
                out = gr.HTML(WELCOME_HTML)
                with gr.Row():
                    download_btn = gr.Button("📦 打包下载全部命中照片",
                                             size="sm")
                download_file = gr.File(label="打包结果", visible=False)
        results_state = gr.State(value=[])
        cancel_state = gr.State(value=None)

        # ── events ──
        ref_input.change(_append_snapshot,
                         [ref_gallery, ref_input], [ref_gallery])
        ref_gallery.change(check_quality, [ref_gallery], [quality_html])
        undo_btn.click(_undo_last, [ref_gallery], [ref_gallery])
        clear_btn.click(_clear_gallery, [], [ref_gallery])
        btn.click(run_search,
                  [url, ref_gallery, threshold, max_photos, pwd,
                   exclude_text, incremental, cancel_state],
                  [out, results_state, cancel_state],
                  concurrency_limit=MAX_CONCURRENT)
        cancel_btn.click(cancel_search, [cancel_state], [cancel_state])
        download_btn.click(download_all, [results_state], [download_file],
                           concurrency_limit=2)
    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2)

    auth = None
    if ACCESS_CODE:
        def auth(username, password):
            return password == ACCESS_CODE
        logger.info("Access code protection enabled")

    app.launch(server_name=SERVER_HOST, server_port=7860,
               theme=THEME, css=CSS, auth=auth)
