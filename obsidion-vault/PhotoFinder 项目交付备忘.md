---
title: PhotoFinder 项目交付备忘
date: 2026-07-29
tags:
  - photofinder
  - project
  - delivery
  - face-recognition
---

# PhotoFinder 项目交付备忘

> 项目路径：`D:/SearCH`
> 作用：在一拍即传（yipai360）活动相册中，通过人脸相似度找出目标人物出现的所有照片。

## 1. 项目核心原理

```text
相册 URL ──► 解析 orderId ──► 调官网 JSON API 分页拉取全部照片元数据
                                (无头破解 SPA, 天然覆盖懒加载)
参考照片 ──► SCRFD 检测 ──► ArcFace 512 维特征(批量推理) ───┐
                                                  ├──► 余弦相似度检索 ──► 按照片聚合(max)
下载 720px 缩略图(并发+断点续传) ──► 逐张检测人脸 ──► 增量人脸索引(磁盘缓存)
                                                  │
结果: 预览图(标注人脸框) + 相似度 + 相册原网页链接 + OSS 原图直链
```

- **检测模型**：SCRFD `det_10g`
- **识别模型**：ArcFace `w600k_r50`（512-d 余弦相似度，支持批量推理）
- **GPU 加速**：自动探测 CUDA → DirectML → CPU，无需手动配置
- **向量检索**：可选 FAISS 加速（>500 条人脸时自动启用）
- **默认阈值**：`0.45`（同一人通常 ≥ 0.6，不同人 < 0.3）
- **误检过滤**：小于 24px 的人脸不进入索引

## 2. 交付环节

| 序号 | 环节 | 状态 | 说明 |
|------|------|------|------|
| 1 | `.gitignore` | ✅ | 排除 `.venv/`、`cache/`、`models/*.onnx`、打包产物等 |
| 2 | 统一 `threshold` 默认值 | ✅ | CLI/WebUI/Pipeline 统一为 `0.45` |
| 3 | `pyproject.toml` + 一键脚本 | ✅ | 可安装为 Python 包，支持 `photofinder` 命令 |
| 4 | 模型下载脚本 | ✅ | `download_models.py` / `download_models.ps1` |
| 5 | 完善 README 文档 | ✅ | 安装、分发、隐私声明、故障排查 |
| 6 | 打包成 `.exe` | ⏭️ | 后续考虑 PyInstaller |
| 7 | `LICENSE` 文件 | ✅ | MIT 许可证 + 第三方模型声明 |
| 8 | GPU 自动探测 | ✅ | CUDA → DirectML → CPU 自动选择 |
| 9 | ArcFace 批量推理 | ✅ | 同一张图多张人脸一次 ONNX 调用 |
| 10 | FAISS 可选加速 | ✅ | `pip install -e ".[fast]"`，>500 人脸自动启用 |
| 11 | 结果标注人脸框 | ✅ | 绿框标注命中人脸位置 |
| 12 | 打包下载 zip | ✅ | Web UI 一键打包全部命中照片 |
| 13 | 搜索可取消 | ✅ | 协作式取消，随时中断 |
| 14 | 参考照片质量反馈 | ✅ | 添加后即时检测人脸并反馈 |
| 15 | 误命中排除 | ✅ | 按 photoId 排除，持久化到 excluded.json |
| 16 | 增量拉取 | ✅ | 活动进行中只拉取新增照片 |
| 17 | CLI JSON 输出 | ✅ | `--json` 结构化输出 |
| 18 | 密码安全传递 | ✅ | 从 URL query 改到 HTTP header |
| 19 | 使用手册 | ✅ | [[PhotoFinder 使用手册]] |

## 3. 关键文件清单

```text
SearCH/
├── photofinder/          # 核心代码包
│   ├── __init__.py
│   ├── cli.py            # 命令行入口
│   ├── webui.py          # Gradio Web UI
│   ├── pipeline.py       # 端到端流程
│   ├── crawler.py        # yipai360 相册抓取
│   ├── face_engine.py    # SCRFD + ArcFace 推理
│   └── index.py          # 人脸索引与检索
├── models/               # ONNX 模型（*.onnx 被 gitignore 忽略）
│   └── .gitkeep
├── README.md             # 完整文档
├── LICENSE               # MIT 许可证
├── requirements.txt      # 依赖
├── pyproject.toml        # 现代 Python 包配置
├── install.ps1           # 一键安装
├── start-webui.ps1       # 启动 Web UI
├── start-webui.bat       # 双击启动 Web UI
├── download_models.py    # 模型下载脚本
└── download_models.ps1   # 一键下载模型
```

## 4. 分发给他人的正确方式

### 不要打包的大文件

| 目录/文件 | 典型大小 | 说明 |
|-----------|---------|------|
| `.venv/` | ~400 MB | Python 虚拟环境 |
| `cache/` | 随使用增长 | 运行时相册缓存、人脸索引 |
| `models/*.onnx` | ~600 MB | InsightFace 模型 |

### 应该打包的内容

```text
photofinder/
README.md
requirements.txt
.gitignore
pyproject.toml
install.ps1
start-webui.ps1
start-webui.bat
download_models.py
download_models.ps1
models/.gitkeep
LICENSE
```

打包后仅约 **100 KB**。

### 接收方使用步骤

1. 解压到任意目录。
2. 双击或在 PowerShell 执行：
   ```powershell
   .\install.ps1
   ```
   会自动创建 `.venv`、安装依赖、下载模型。
3. 运行：
   ```powershell
   .\start-webui.bat
   ```
   浏览器打开 http://127.0.0.1:7860 即可使用。

## 5. 前置要求

- **Python 3.10+**
- **Windows 用户**：安装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)，否则 ONNX Runtime 可能报 DLL 缺失。

## 6. 使用方式

详细操作指南见 [[PhotoFinder 使用手册]]。

### Web UI

```powershell
.\start-webui.ps1
# 或双击 start-webui.bat
```

拍照/上传参考照片 → 开始查找 → 查看结果（绿框标注人脸）→ 打包下载 zip。

### CLI

```powershell
photofinder `
  --url "https://www.yipai360.com/photolivepc/?orderId=YOUR_ORDER_ID" `
  --ref face.jpg `
  --json
```

## 7. 隐私与合规

- **本地计算**：人脸检测、特征提取、相似度检索均在本地运行，不上传照片或人脸特征到第三方服务器。
- **数据责任**：使用者应确保获得被搜索人物的授权，并遵守目标平台（一拍即传/yipai360）的使用条款。
- **模型许可**：代码采用 MIT 许可证；InsightFace `buffalo_l` 模型受该项目及其数据提供方许可条款约束。

## 8. 模型下载失败怎么办

如果 `install.ps1` 自动下载模型失败，可手动下载：

1. 下载 `buffalo_l.zip`：
   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
2. 解压到 `models/` 目录，确保包含：
   - `det_10g.onnx`
   - `w600k_r50.onnx`
   - `genderage.onnx`
   - `2d106det.onnx`
   - `1k3d68.onnx`
3. 重新运行 Web UI 或 CLI。

## 9. 缓存结构

```text
cache/{orderId}/
├── photos.json     # 全部照片元数据
├── thumbs/         # 720px 检测用图
├── faces.npz       # 人脸 embedding 矩阵
├── faces.json      # 人脸对应的 photoId/bbox
├── done.json       # 已索引 photoId 清单
└── excluded.json   # 用户标记的误命中 photoId
```

删除 `cache/{orderId}` 可彻底清理该活动数据。

## 10. 后续待办

- [ ] 考虑 PyInstaller 打包成 `.exe`，方便完全不懂 Python 的用户
- [x] 添加单元测试（crawler / face_engine / pipeline / index）
- [x] 增加日志文件和错误恢复机制
- [x] GPU 自动探测 + ArcFace 批量推理
- [x] FAISS 可选加速向量检索
- [x] 结果标注人脸框 + 打包下载 zip
- [x] 搜索可取消 + 参考照片质量反馈
- [x] 误命中排除 + 增量拉取
- [x] CLI JSON 输出 + 密码安全传递
- [x] 使用手册
- [ ] 评估是否支持其他相册平台

---

*创建于 2026-07-29 · 更新于 2026-07-29*
