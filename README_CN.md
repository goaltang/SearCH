<div align="center">

# 🔍 SearCH

### 在数千张活动照片中找到自己 — 只需几秒。

粘贴相册链接，上传一张自拍，获取所有包含你的照片。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![在线体验](https://img.shields.io/badge/🤗-在线体验-ff9d00)](http://47.76.47.229:7860/)

[English](README.md) · [中文](README_CN.md)

</div>

---

## ✨ 这是什么？

活动摄影师会把几百甚至上千张照片上传到相册分享平台。在照片海洋里翻找自己的脸，非常痛苦。

**SearCH** 帮你搞定：

1. 粘贴相册链接
2. 上传一张（或多张）你的照片
3. 点击搜索 → 获取所有包含你的照片，按相似度排序

所有人脸检测和识别均**在本地运行**（你的电脑或你的服务器）。照片不会上传到任何第三方服务。

### 已适配平台

| 平台 | 状态 |
|------|------|
| [一拍即传 (Yipai360)](https://www.yipai360.com/) | ✅ 完整支持 — 自动分页懒加载相册，下载 720px 缩略图 |

使用其他相册平台？爬虫层是可插拔的 — 参见下方[适配其他平台](#️-路线图--适配其他平台)，或提交 issue。

<div align="center">

<img src="https://github.com/user-attachments/assets/3c72d231-b0fc-4856-8aaa-cbef84cef125" alt="主界面 — 粘贴相册链接并上传参考照片" width="700" />

*粘贴相册链接，上传自拍，点击搜索。*

<br/>

<img src="https://github.com/user-attachments/assets/34319a31-21b2-4dc3-9b2f-04d5238e78be" alt="搜索结果 — 人脸匹配按相似度排序" width="700" />

*所有包含你的照片 — 按相似度排序，标注人脸框。*

<br/>

<img src="https://github.com/user-attachments/assets/e3104cae-f46f-4feb-bc68-4fca90947a70" alt="一键打包下载所有匹配照片" width="420" />

*一键打包所有匹配照片为 zip 下载。*

</div>

## 🧠 工作原理

```
相册链接 ──► 解析 API ──► 分页拉取全部照片元数据（处理懒加载/SPA）
                                 │
参考照片 ─► SCRFD 检测 ─► ArcFace 512 维特征向量（批量推理）──┐
                                                                    ├─► 余弦相似度 ─► 单照片取最大值聚合
并发下载 720px 缩略图（支持断点续传）──► 人脸检测 ─► 增量人脸索引（磁盘缓存）
                                                                    │
结果：标注预览图 + 相似度分数 + 原图链接 + 相册链接 ──────────────┘
```

| 组件 | 实现 |
|------|------|
| 人脸检测 | SCRFD `det_10g`（InsightFace buffalo_l，ONNX 格式） |
| 人脸识别 | ArcFace `w600k_r50`（512 维，余弦相似度） |
| 推理引擎 | ONNX Runtime — 自动检测 CUDA → DirectML → CPU |
| 向量检索 | NumPy（≤500 人脸）/ FAISS（>500 人脸，自动切换） |
| Web 界面 | Gradio 自定义设计，移动端自适应 |
| 部署 | Docker，2 核 4G 服务器即可运行 |

**关键设计决策：**

- **不依赖 insightface Python 包** — 直接加载 ONNX 模型，手写前/后处理（anchor 解码、NMS、仿射对齐）。依赖更轻，完全可控。
- **增量索引** — 已处理的照片不会重复计算。中断的构建从断点恢复。活动进行中新增的照片自动纳入索引。
- **并发安全** — 相册级锁、原子文件写入、可配置 ORT 线程数、LRU 索引缓存。多用户同时搜索不会损坏索引。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- (Windows) [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### 安装

```bash
git clone https://github.com/goaltang/SearCH.git
cd SearCH
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python download_models.py --models-dir models       # ~300MB，仅需下载一次
```

### 运行

```bash
python -m photofinder.webui
# 打开 http://127.0.0.1:7860
```

### 命令行

```bash
photofinder --url "https://www.yipai360.com/photolivepc/?orderId=YOUR_ID" --ref selfie.jpg
```

## 🌐 活动场景自部署

部署到服务器，让活动参与者用手机搜索自己的照片：

```bash
docker compose up -d --build
# 在活动群里分享 http://你的服务器IP:7860
```

预建人脸索引一次 → 用户搜索 **2-3 秒**出结果。
2 核 4G 服务器可支撑数十人同时使用。完整部署指南：[DEPLOY.md](DEPLOY.md)

## 📂 项目结构

```
photofinder/
├── crawler.py      # 相册 API 逆向 + 并发下载
├── face_engine.py  # SCRFD + ArcFace，裸 ONNX（不用 insightface 包）
├── index.py        # 增量人脸索引，FAISS 加速检索
├── pipeline.py     # 端到端编排，并发控制
├── cli.py          # 命令行接口
├── webui.py        # Gradio Web 界面（移动端适配）
└── logger.py       # 结构化日志
tests/              # 1094 行单元测试（pytest）
```

## ⚙️ 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PHOTOFINDER_ACCESS_CODE` | （空） | 访问码，为空则开放 |
| `PHOTOFINDER_MAX_CONCURRENT` | `3` | 最大同时搜索数 |
| `PHOTOFINDER_ORT_THREADS` | `min(4, 核心数)` | ONNX 推理线程数 |
| `PHOTOFINDER_DOWNLOAD_MAX` | `300` | 单次 zip 下载最大照片数 |
| `PHOTOFINDER_INDEX_CACHE` | `4` | 内存中缓存的热门相册索引数 |

## 🗺️ 路线图 / 适配其他平台

爬虫层（`crawler.py`）是唯一与平台相关的部分。要支持新的相册分享平台：

1. 实现 `fetch_metadata()` → 返回 `Photo` 数据类列表
2. 实现 `download_thumbs()` → 将图片保存到磁盘
3. 下游所有环节（检测、索引、搜索、界面）无需改动

欢迎提交新平台适配的 PR。

## 🔒 隐私

- 所有计算在本地完成。照片和特征向量不会离开你的机器/服务器。
- 用户上传的参考照片仅用于当前搜索 — 不会持久化存储。
- 请确保获得被搜索者的同意。

## 📄 许可证

MIT — 随意使用。
InsightFace buffalo_l 模型受[其自身许可证](https://github.com/deepinsight/insightface)约束。

---

<div align="center">

**如果这个项目帮你省去了在 3000 张活动照片里人肉翻找的麻烦，请考虑给个 ⭐**

</div>
