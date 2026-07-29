# PhotoFinder — 活动照片找人 Agent

输入**一拍即传(yipai360)** 活动相册链接 + 一张参考人脸照片，Agent 自动遍历相册中的
全部照片（含瀑布流/懒加载内容），缓存图片并建立临时人脸索引，通过人脸相似度检索
找出目标人物出现的所有照片，以**图片预览 + 原网页链接**形式返回结果。

## 原理

```
相册 URL ──► 解析 orderId ──► 调官网 JSON API 分页拉取全部照片元数据
                                (无头破解 SPA, 天然覆盖懒加载)
参考照片 ─► SCRFD 检测 ─► ArcFace 512 维特征(批量推理) ─┐
                                              ├─► 余弦相似度检索 ─► 按照片聚合(max)
下载 720px 缩略图(并发+断点续传) ─► 逐张检测人脸 ─► 增量人脸索引(磁盘缓存)
                                              │
结果: 预览图(标注人脸框) + 相似度 + 相册原网页链接 + OSS 原图直链
```

- **检测**: SCRFD `det_10g` (InsightFace buffalo_l, ONNX)
- **识别**: ArcFace `w600k_r50` (512-d embedding, 余弦相似度, 支持批量推理)
- **加速**: 自动探测 GPU（CUDA / DirectML），未安装则回退 CPU；可选 FAISS 加速向量检索
- 同一人典型得分 ≥ 0.6，不同人 < 0.3，默认阈值 0.45
- 误检抑制：小于 24px 的"人脸"不参与索引（活动大合照的远小人脸对识别也不可靠）

## 前置要求

- **Python 3.10 或更高版本**（类型注解 `str | Path` 需要 Python ≥ 3.10）
- **Windows 用户**：ONNX Runtime 依赖 Microsoft Visual C++ Redistributable。
  如果启动时报 `VCRUNTIME140.dll` 等错误，请先安装：
  https://aka.ms/vs/17/release/vc_redist.x64.exe

## 安装

### 一键安装（推荐）

在 Windows PowerShell 中进入项目目录，执行：

```powershell
.\install.ps1
```

脚本会自动：
1. 创建 `.venv` 虚拟环境
2. 安装 Python 依赖
3. 下载 InsightFace `buffalo_l` 模型到 `models/`（首次约 300MB+）

### 手动安装

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
python download_models.py --models-dir models
```

> 注意：模型文件（`det_10g.onnx`、`w600k_r50.onnx` 等）未包含在代码仓库中，
> 需要通过 `download_models.py` 或 `download_models.ps1` 自动下载。

## 使用

### Web UI（推荐）

双击 `start-webui.bat`，或在 PowerShell 中执行：

```powershell
.\start-webui.ps1
```

然后打开 http://127.0.0.1:7860。

> 已适配移动端：手机浏览器打开即可使用，上传参考照片时可直接拍照。

**操作流程：**

1. 粘贴相册链接（已预填常用链接）
2. 在"添加参考照片"区域上传照片（手机浏览器上传时可直接拍照；可反复添加多张正脸/侧脸）
   - 添加后自动检测人脸并反馈质量（是否清晰、是否有多张人脸）
   - 支持"撤销最后一张"和"清空全部"
3. 点击"开始查找"，搜索过程中可随时点"✕ 取消"
4. 结果以卡片墙展示：缩略图预览（**绿框标注命中人脸**）、相似度、文件名
5. 点击"📦 打包下载全部命中照片"可一键导出 zip

**高级选项：**

- 相似度阈值：调低提高召回、调高提高精度
- 最多处理照片数：调试用，0 = 全部
- 相册密码：如有
- 排除的照片 ID：误命中时填入对应 ID（逗号分隔），下次搜索自动跳过
- 仅拉取新增照片：活动进行中时勾选，只下载和索引新照片

### CLI

```powershell
photofinder `
  --url "https://www.yipai360.com/photolivepc/?orderId=20260720172647201236&channel=h5&origin=qrcode" `
  --ref face.jpg
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ref` | (必填) | 参考人脸照片路径，可传多张 |
| `--threshold` | 0.45 | 相似度阈值，调低提高召回、调高提高精度 |
| `--max-photos N` | 全部 | 只处理前 N 张（调试用） |
| `--pwd` | 无 | 相册密码 |
| `--refresh` | 关 | 重新拉取照片列表 |
| `--incremental` | 关 | 仅拉取上次之后新增的照片 |
| `--rebuild-index` | 关 | 清空人脸索引重建（不重新下载图片） |
| `--exclude ID...` | 无 | 排除指定 photoId（误命中） |
| `--json` | 关 | 以 JSON 格式输出结果（便于脚本对接） |
| `--min-face` | 24 | 最小人脸宽度(px)，过滤误检 |
| `--workers` | 4 | 索引并发数 |
| `--cache` | cache | 缓存目录 |
| `--models` | models | 模型目录 |

## 部署为在线服务

想把工具交给活动参与者自助使用？参考 [DEPLOY.md](DEPLOY.md)：

- Docker 一键部署（`docker compose up -d --build`，自带 CPU/内存资源限制）
- 服务器上预建人脸索引，用户搜索秒级返回
- 并发上限、访问码、打包下载上限等多用户保护
- 已适配移动端，手机打开链接上传自拍即可查询
- 可直接用 `http://服务器IP:7860` 访问，也可选加 Nginx + HTTPS + 域名

## 缓存结构

```
cache/{orderId}/
├── photos.json     # 全部照片元数据(含签名 URL)
├── thumbs/         # 720px 检测用图, 增量下载
├── faces.npz       # 人脸 embedding 矩阵
├── faces.json      # 每个人脸对应的 photoId/bbox
├── done.json       # 已索引 photoId 清单(断点续跑)
└── excluded.json   # 用户标记的误命中 photoId
```

二次运行同一相册只会增量处理新照片；删除 `cache/{orderId}` 即彻底清理该活动数据。

## 性能优化

- **GPU 加速**：自动探测 CUDA → DirectML → CPU，安装 `onnxruntime-gpu` 即可启用 NVIDIA GPU 加速
- **批量推理**：ArcFace 对同一张图中的多张人脸一次 ONNX 调用完成，减少推理开销
- **FAISS 检索**：`pip install faiss-cpu`（或 `pip install -e ".[fast]"`），>500 条人脸时自动启用 FAISS 加速向量检索
- **增量索引**：已处理的照片不会重复计算，中断后断点续跑

## 分发给其他用户

不要把 `.venv/`、`cache/`、`models/*.onnx` 一起打包，这些文件体积很大：

| 目录 | 典型大小 |
|---|---|
| `.venv/` | ~400 MB |
| `cache/` | 随使用增长 |
| `models/` | ~600 MB |
| `photofinder/` 代码 | < 1 MB |

正确做法：
1. 仅打包代码：`photofinder/`、`README.md`、`requirements.txt`、`.gitignore`、
   `pyproject.toml`、`*.ps1`、`*.bat`、`download_models.py`、`models/.gitkeep`
2. 接收方解压后运行 `install.ps1`，会自动创建环境并下载模型。

## 隐私与免责声明

- **本地计算**：所有人脸检测、特征提取和相似度计算都在本地运行，不会把照片或人脸特征上传到任何第三方服务器。
- **数据责任**：使用者应确保获得被搜索人物的授权，并遵守目标平台（一拍即传/yipai360）的使用条款。本项目仅供个人合法场景使用。
- **许可**：本项目代码采用 MIT 许可证（见 `LICENSE` 文件）。使用的 InsightFace `buffalo_l` 模型受该项目及其数据提供方的许可条款约束。

## 说明与限制

- 目标站点无独立"单张照片页"，结果中的"活动网页"即相册页链接，配合文件名/photoId 定位。
- OSS 链接为签名 URL，有过期时间（通常较长），过期后重新运行即可刷新。
- 参考照片请使用清晰正脸照；侧脸/低头/遮挡会显著降低召回。支持多张参考照（正脸+侧脸），取每张中最大人脸。
- 支持带密码的相册（密码通过 HTTP header 传递，不会出现在 URL 或日志中）。
- 其他类似平台可参照 `crawler.py` 适配 API。

## 模型下载失败怎么办？

如果自动下载因网络问题失败，请手动下载：

1. 下载 `buffalo_l.zip`：
   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
2. 解压到 `models/` 目录，确保包含以下文件：
   - `det_10g.onnx`
   - `w600k_r50.onnx`
   - `genderage.onnx`
   - `2d106det.onnx`
   - `1k3d68.onnx`
3. 重新运行 `start-webui.ps1` 或 `photofinder`。
