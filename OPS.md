# 找找小禾 · 运维手册（OPS）

> 本手册讲**日常运维**：系统怎么运作、如何更新照片/新增相册/更新代码、踩坑记录、故障排查。
> **首次部署**请看 [DEPLOY.md](DEPLOY.md)；面向 AI agent 的架构约定看 [AGENTS.md](AGENTS.md)。

---

## 1. 系统是如何运作的

### 1.1 整体架构

```
用户浏览器(手机/电脑)
      │  上传一张自拍
      ▼
服务器公网 IP:7860  ──►  Docker 容器 photofinder (Gradio Web UI)
                              │
                  ┌───────────┴───────────┐
                  ▼                        ▼
        预建人脸索引 cache/          ONNX 模型 models/
        (每个相册一个目录)        det_10g(检测) + w600k_r50(特征)
```

**核心思路**：运营者**提前**把相册全部照片下载、检测人脸、算出 512 维特征向量并建好索引（一次性、离线）。
用户访问时只需上传自拍 → 提取特征 → 和索引里所有人脸向量做余弦相似度比对 → 返回命中照片。**秒级完成**，服务器只做轻量向量检索。

### 1.2 代码模块（`photofinder/`）

| 模块 | 职责 |
|------|------|
| `crawler.py` | 平台相关：调 yipai360 API 拉照片列表、下载缩略图。`parse_albums()` 解析多行「标签 链接」配置 |
| `face_engine.py` | SCRFD 人脸检测 + ArcFace 特征提取（原生 ONNX，不依赖 insightface 包） |
| `index.py` | 人脸向量索引：增量构建、磁盘持久化、余弦相似度搜索（≤500 用 numpy，>500 自动切 FAISS） |
| `pipeline.py` | 端到端编排。`run()` 单相册；`run_multi()` 多相册联合搜索（参考人脸只提取一次，结果按分数合并，坏相册自动跳过） |
| `webui.py` | Gradio 界面：多相册预填、参考照上传、结果按相册分组渲染、打包下载 |
| `cli.py` | 命令行：`--prepare` 建索引、搜索、`--incremental` 增量 |

### 1.3 一次搜索的完整链路

1. 用户上传 1~N 张参考自拍，每张取**最大的一张人脸**算特征（避免背景人脸污染）。
2. `run_multi()` 遍历配置的每个相册（orderId）：
   - 该相册索引不在内存就从 `cache/<orderId>/faces.npz` 读入（LRU 缓存，`PHOTOFINDER_INDEX_CACHE` 控制热索引数）。
   - 用参考特征在该相册索引里搜，超过阈值的算命中。
3. 所有相册的命中**按相似度全局排序**合并，每张结果带上来源相册标签。
4. 前端按相册分组展示；有本地缩略图就画**绿框**标注人脸，否则用在线预览图。

### 1.4 数据布局（`cache/<orderId>/`）

```
cache/
├── 20260720172647201236/   ← 省赛·毕节
│   ├── photos.json         照片元数据（photoId、各级尺寸 URL）
│   ├── faces.npz           人脸特征矩阵 (N, 512) float32
│   ├── faces.json          每张人脸的 {photo_id, bbox, det_score}
│   ├── done.json           已处理过的 photoId（增量靠它跳过）
│   ├── excluded.json       用户标记的误命中（搜索时排除）
│   └── thumbs/             缩略图 <photoId>.jpg（绿框标注需要；搜索本身不需要）
└── 20260727190944809942/   ← 国赛·上海（结构同上）
```

> **索引文件** = photos.json + faces.npz + faces.json + done.json（+ excluded.json）。
> **thumbs/ 可选**：搜索不依赖它，但结果页的绿框人脸标注需要它。

### 1.5 当前线上配置

| 项 | 值 |
|----|----|
| 服务器 | 阿里云 ECS（香港，2核4G），公网 IP `47.76.47.229`，端口 `7860` |
| 访问地址 | `http://47.76.47.229:7860`（无访问码，链接只在活动群分享） |
| 项目目录 | **`/root/SearCH`**（注意：不是 DEPLOY.md 示例里的 `~/photofinder`） |
| 容器名 | `photofinder`（Docker Compose，限 1.5 核 / 3G 内存） |
| 挂载卷 | `/root/SearCH/models` → `/app/models`，`/root/SearCH/cache` → `/app/cache` |
| 相册 | 省赛·毕节 `20260720172647201236`（不再更新）；国赛·上海 `20260727190944809942` |

---

## 2. 日常运维操作

> **总原则**：索引在**本地 Windows（D:\SearCH）建好**，只把索引文件传到服务器；服务器只做检索，不在上面建索引（慢、易 OOM，见 §3）。

### 2.1 给相册补新照片（增量更新）

活动进行中摄影师追加了新照片时：

**① 本地增量建索引**（PowerShell，`D:\SearCH`）——**务必带 `PHOTOFINDER_FACE_BATCH=0`**（原因见 §3.1）：
```powershell
cd D:\SearCH
set PHOTOFINDER_FACE_BATCH=0&& .venv\Scripts\python -m photofinder.cli --url "相册链接" --prepare --incremental
```
> `--incremental` 只拉新增照片并续建索引（靠 done.json 跳过已处理的）。结尾应显示 `索引完成: N 张人脸`，**N 必须 > 0**。

**② 打包索引文件**（不含 thumbs，约几十 MB）：
```powershell
Compress-Archive -Path 'cache\<orderId>\photos.json','cache\<orderId>\faces.npz','cache\<orderId>\faces.json','cache\<orderId>\done.json' -DestinationPath index-only.zip -Force
```

**③ 上传**（在**本地** Windows 终端，不是服务器 SSH 窗口）：
```powershell
scp D:\SearCH\index-only.zip root@47.76.47.229:~/index-only.zip
```

**④ 服务器解压 + 重启**（SSH）：
```bash
cd ~/SearCH/cache/<orderId>
unzip -o ~/index-only.zip
rm ~/index-only.zip
docker restart photofinder     # 索引是 volume 挂载，重启即加载新索引，无需重建镜像
```

### 2.2 新增一个相册

1. **本地建索引**（同 §2.1 ①，去掉 `--incremental`）：
   ```powershell
   set PHOTOFINDER_FACE_BATCH=0&& .venv\Scripts\python -m photofinder.cli --url "新相册链接" --prepare
   ```
2. **上传索引**（同 §2.1 ②③④，解压到 `~/SearCH/cache/<新orderId>/`）。
   - 想要结果带**绿框**，再把 `thumbs/` 也打包上传解压到同一目录（见 §2.4）。
3. **把它加进 Web UI 预填**：编辑 `photofinder/webui.py` 的 `DEFAULT_ALBUM_URL`，加一行 `标签 链接`。
4. **上线代码**（同 §2.3）。

### 2.3 更新代码后部署

本地改了代码（如新增相册配置、功能改动）：
```powershell
# 本地
git add <改动的文件>
git commit -m "..."
git push origin master
```
```bash
# 服务器
cd ~/SearCH
git pull
docker compose up -d --build   # 代码变了必须重建镜像（区别于只更新索引的 docker restart）
```

> **记住区别**：
> - 只更新**索引** → `docker restart photofinder`（快，不重建）。
> - 更新了**代码** → `git pull` + `docker compose up -d --build`（重建镜像）。

### 2.4 绿框人脸标注

- 绿框画在**服务器本地的缩略图**上（`cache/<orderId>/thumbs/<photoId>.jpg`）。
- 服务器上有该相册的 `thumbs/` → 结果带绿框；没有 → 回退用在线预览图（搜索和命中照常，只是没框）。
- 要绿框就把 `thumbs/` 打包上传（上海相册约 78.6 MB / 1500 张）：
  ```powershell
  # 本地
  Compress-Archive -Path 'cache\<orderId>\thumbs' -DestinationPath thumbs.zip -Force
  scp D:\SearCH\thumbs.zip root@47.76.47.229:~/thumbs.zip
  ```
  ```bash
  # 服务器
  cd ~/SearCH/cache/<orderId>
  unzip -o ~/thumbs.zip
  rm ~/thumbs.zip
  # 无需重启：缩略图每次搜索实时读盘
  ```

---

## 3. 踩坑记录（重要）

### 3.1 本地建索引必须设 `PHOTOFINDER_FACE_BATCH=0`
本地 Windows + onnxruntime 1.28 跑 ArcFace **批量**识别（一次多张人脸）会**原生崩溃**，且 Python 接不住，导致**静默建出 0 人脸的空索引**（日志只显示 `Indexed N new photos (0 faces)`，无报错）。检测是好的，单人脸识别也是好的。
- **对策**：本地建索引一律 `set PHOTOFINDER_FACE_BATCH=0&& ...`，强制逐人脸识别。服务器（Docker，Linux）批量正常，不受影响（该开关默认开批量）。
- 代码已加保险：建索引若 0 人脸会**大声告警**（commit `1145f3b`），空索引不会再悄悄蒙混过关。
- **cmd 坑**：`set VAR=0 && ...` 会让值变成 `"0 "`（尾随空格）；代码已 `.strip()` 兼容，但最好写 `set VAR=0&& ...`（`&&` 前不留空格）。

### 3.2 别在运行中的容器里建索引（OOM，exit 137）
`docker exec photofinder ... --prepare` 会被 OOM 杀掉：webui 已加载一套模型+索引，exec 的 CLI 又加载第二套，两者挤爆容器 3G 内存上限（机器共 4G）。
- **对策**（如确需在服务器建索引）：先 `docker stop photofinder`，再用临时容器建，最后 `docker start photofinder`：
  ```bash
  docker stop photofinder
  docker run --rm -v /root/SearCH/models:/app/models -v /root/SearCH/cache:/app/cache \
    photofinder python -m photofinder.cli --url "链接" --prepare
  docker start photofinder
  ```
- 建索引可断点续传（每 500 张 checkpoint，缩略图复用），中断了重跑损失很小。
- 索引时的 `onnxruntime VerifyOutputSizes ... {1,512} vs {N,512}` 警告是动态 batch 的正常提示，**不是错误**。

### 3.3 多相册后，解压别 `rm -rf ~/SearCH/cache`
只有一个相册时的旧习惯是 `rm -rf cache` 再重建。现在有毕节+上海两个相册，`rm -rf cache` 会**把另一个相册的索引也删掉**。新增/更新某个相册时，只 `mkdir -p` 并解压进**对应的 `<orderId>` 子目录**。

### 3.4 服务器路径与命令
- 项目目录是 **`~/SearCH`**，`cd ~/photofinder` 会失败（DEPLOY.md 的示例路径不适用本机）。
- `photofinder` 命令**只存在于容器内**，宿主机直接敲会 `command not found`；要在宿主机用就走 Docker（见 §3.2）。

### 3.5 FAISS / numpy 搜索路径一致性
服务器 Docker 装了 faiss（`.[fast]`），>500 人脸走 FAISS 路径；本地无 faiss 走 numpy 路径。曾因 FAISS 路径把分数错配到别的照片（已修复，commit `7b63565`）。
- **对策**：若「服务器结果不对、本地正确」，先怀疑这条路径；保持 `tests/test_index.py::test_faiss_path_matches_numpy` 通过。

---

## 4. 常用命令速查

```bash
# —— 服务器（SSH，项目目录 ~/SearCH）——
docker logs -f photofinder                 # 实时日志
docker restart photofinder                 # 重启（更新索引后用）
cd ~/SearCH && git pull && docker compose up -d --build   # 更新代码后重建
docker compose ps                          # 看容器状态
ls ~/SearCH/cache                          # 看有哪些相册索引
```

```powershell
# —— 本地（PowerShell，D:\SearCH）——
# 全量建索引
set PHOTOFINDER_FACE_BATCH=0&& .venv\Scripts\python -m photofinder.cli --url "链接" --prepare
# 增量更新
set PHOTOFINDER_FACE_BATCH=0&& .venv\Scripts\python -m photofinder.cli --url "链接" --prepare --incremental
# 跑测试
.venv\Scripts\python -m pytest tests/ -q
```

---

## 5. 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 搜索 0 结果 / 索引 0 人脸 | 本地建索引忘了 `PHOTOFINDER_FACE_BATCH=0`（空索引） | 重设环境变量重建索引，确认 `索引完成: N 张人脸` 的 N>0 |
| 某个相册没出现在结果 | 服务器 `cache/<orderId>/` 缺失，或 `DEFAULT_ALBUM_URL` 没这行 | 检查服务器该目录是否存在；检查 webui 配置并重新部署 |
| 结果没有绿框 | 服务器该相册没有 `thumbs/` | 按 §2.4 上传 thumbs（不影响搜索，只影响标注） |
| 服务器结果和本地不一致 | FAISS/numpy 路径分歧（§3.5） | 跑 `test_faiss_path_matches_numpy`，检查 `index.py` |
| 建索引进程被杀（exit 137） | 在运行容器里 exec 建索引导致 OOM（§3.2） | 停服务后用临时容器建 |
| 容器起不来 / 搜索报错 | 模型或缓存挂载缺失、镜像构建失败 | `docker logs photofinder` 看报错；确认 models/cache 挂载 |
| `scp` 报 Host key verification failed | 在**服务器**上敲了 scp（应本地执行） | 在**本地** Windows 终端执行 scp |

---

## 6. 安全与隐私

- 访问码（若启用）只在活动群内分享，不要公开。
- 用户上传的参考照仅用于本次特征提取，**不落盘**。
- 服务器缓存来自公开相册；活动结束后可清理：`docker stop photofinder && rm -rf ~/SearCH/cache/*`。
- 在群内说明：本工具仅供查找本人照片，请勿用于搜索他人。
