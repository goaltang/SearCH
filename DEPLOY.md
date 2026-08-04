# 部署指南：活动照片找人 Web 服务

将 PhotoFinder 部署为在线服务，让活动参与者通过浏览器自助搜索自己的照片。

> 本文讲**首次部署**。上线后的**日常运维**（更新照片、新增相册、更新代码、踩坑与故障排查）请看 [OPS.md](OPS.md)。

## 架构概览

```
用户浏览器(电脑/手机) ──HTTP──► 服务器公网 IP:7860 ──► Docker 容器 (Gradio App)
                                                          │
                                                预建好的人脸索引 (cache/)
                                                ONNX 模型 (models/)
```

> 直接用 `http://服务器IP:7860` 访问即可（当前线上方案，零额外配置）。
> 如果申请了域名、需要 HTTPS，再按第 6 节在前面加 Nginx 反向代理。

**核心思路**：你提前在服务器上把相册的全部照片下载并建好人脸索引（一次性），
之后用户访问时只需上传自拍 → 向量比对 → 返回结果，**几秒钟完成**。

---

## 1. 服务器选择

| 方案 | 配置 | 月费参考 | 适用场景 |
|------|------|----------|----------|
| **2 核 4G CPU 云服务器（推荐）** | Ubuntu 22.04，40G SSD | ¥0（阿里云免费试用）~ ¥100 | 索引已预建，几百次搜索（当前线上配置） |
| 4 核 8G CPU 云服务器 | 同上 | ¥100-300 | 需要更高并发 |
| GPU 云服务器 | T4/A10，16G+ | ¥2000-5000 | 需要现场实时建索引 |

> 索引预建好后，每次搜索只做一次人脸检测（~1-2s CPU）+ 向量比对（<0.1s），
> **普通 CPU 服务器完全够用**，2 核 4G 即可。

推荐：阿里云/腾讯云轻量应用服务器，2 核 4G 起步；阿里云经济型 e 有免费试用，
短期活动基本零成本。注意国内服务器需备案，可选**香港/海外地域**免备案。

## 2. 服务器初始化

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 重新登录使 docker 组生效

# 创建项目目录
mkdir -p ~/photofinder/{models,cache}
cd ~/photofinder
```

## 3. 上传代码和模型

```bash
# 方式 A：从 Git 拉取
git clone <你的仓库地址> .

# 方式 B：本地打包上传（仅代码，不含 .venv/cache/models）
# 本地执行：
#   tar czf photofinder-code.tar.gz photofinder/ requirements.txt pyproject.toml Dockerfile .dockerignore
#   scp photofinder-code.tar.gz user@server:~/photofinder/
# 服务器执行：
#   cd ~/photofinder && tar xzf photofinder-code.tar.gz

# 下载模型（约 300MB）
python download_models.py --models-dir models
# 或手动下载 buffalo_l.zip 解压到 models/：
# https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
```

## 4. 预建人脸索引（关键步骤）

在服务器上运行，把活动相册的全部照片下载并建好索引：

```bash
# 先构建 Docker 镜像
docker build -t photofinder .

# 运行 prepare 模式（替换为你的相册链接）
docker run --rm \
  -v ~/photofinder/models:/app/models \
  -v ~/photofinder/cache:/app/cache \
  photofinder \
  python -m photofinder.cli \
    --url "https://www.yipai360.com/photolivepc/?orderId=你的orderId" \
    --prepare

# 7000 张照片在 4 核 CPU 上大约需要 20-40 分钟
# 完成后 cache/{orderId}/ 下会有 faces.npz、faces.json 等索引文件
```

> 如果活动进行中还有新照片，可以加 `--incremental` 增量更新。

## 5. 启动 Web 服务

**方式 A：docker compose（推荐，自带资源限制）**

```bash
PHOTOFINDER_ACCESS_CODE="你的访问码" docker compose up -d --build
# 不需要访问码就不带这个变量，用户打开页面直接进入
# compose 配置限制了容器最多 1.5 核 / 3G 内存（按 2核4G 服务器调校），防止拖垮整台服务器
```

**方式 B：docker run**

```bash
docker run -d \
  --name photofinder \
  --restart unless-stopped \
  --memory 3g --cpus 1.5 \
  -p 7860:7860 \
  -v ~/photofinder/models:/app/models \
  -v ~/photofinder/cache:/app/cache \
  -e PHOTOFINDER_ACCESS_CODE="你的访问码" \
  -e PHOTOFINDER_MAX_CONCURRENT=3 \
  -e PHOTOFINDER_ORT_THREADS=2 \
  photofinder
```

> `-p 7860:7860` 监听公网，用户可直接通过 IP 访问。如果前面挂了 Nginx 反向代理，
> 改回 `-p 127.0.0.1:7860:7860` 只允许本机访问更安全。

环境变量说明：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PHOTOFINDER_ACCESS_CODE` | (空) | 访问码，留空则无需登录 |
| `PHOTOFINDER_MAX_CONCURRENT` | `3` | 最大同时搜索数，超出自动排队 |
| `PHOTOFINDER_HOST` | `127.0.0.1` | 监听地址，Docker 内已设为 `0.0.0.0` |
| `PHOTOFINDER_DOWNLOAD_MAX` | `300` | 单次打包下载的最大照片数（防止内存爆掉） |
| `PHOTOFINDER_ORT_THREADS` | `min(4, 核数)` | 每次 ONNX 推理占用的 CPU 线程数；并发场景建议 `2` |
| `PHOTOFINDER_INDEX_CACHE` | `4` | 内存中保留几个相册的人脸索引（避免每次搜索重新读盘） |

验证：`curl http://127.0.0.1:7860` 应返回 HTML。

### 并发容量参考

索引已预建的前提下，一次搜索 ≈ 1 次人脸检测（~1-2s CPU）+ 向量比对（<0.5s）：

| 服务器 | 推荐配置 | 可承载能力 |
|--------|----------|------------|
| 2 核 4G（当前线上） | `MAX_CONCURRENT=3`（默认）, `ORT_THREADS=2` | 同时 3 人搜索（容器限 1.5 核，满载时每人稍慢），几十人轮流用无压力 |
| 4 核 8G | `MAX_CONCURRENT=3`, `ORT_THREADS=2` | 同时 3 人搜索，其余排队；10-20 人轮流使用无压力 |
| 8 核 16G | `MAX_CONCURRENT=5`, `ORT_THREADS=2` | 同时 5 人搜索 |

注意：**「并发搜索数」不等于「在线人数」**。超出并发的用户只是在队列里多等几秒，
体验略慢但服务不会挂。真正危险的是无限制并发导致 CPU/内存耗尽，上面的限制就是防这个。

## 6. （可选）Nginx 反向代理 + HTTPS

直接用 IP:7860 访问已经可用；只有在申请了域名、需要 HTTPS 时才需要这一节。

```bash
sudo apt install -y nginx certbot python3-certbot-nginx

# 使用仓库自带的配置（已含每 IP 限流、连接数限制、SSE 支持）
sudo cp deploy/nginx.conf /etc/nginx/sites-available/photofinder
sudo nano /etc/nginx/sites-available/photofinder   # 把 server_name 改成你的域名
sudo ln -s /etc/nginx/sites-available/photofinder /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 申请免费 HTTPS 证书（需要域名已解析到本服务器）
sudo certbot --nginx -d photo.你的域名.com
```

该配置在默认反代之外增加了：
- **限流**：`/queue/join`（搜索提交）每 IP 30 次/分钟、每 IP 最多 20 并发连接，防脚本刷接口
- **SSE 直通**：`/queue/data` 关闭缓冲，进度条实时推送
- **gzip**：减小页面体积

## 7. 域名

1. 在域名注册商（阿里云/腾讯云/Namesilo）购买域名，约 ¥50-100/年
2. 添加 A 记录：`photo.你的域名.com` → 服务器公网 IP
3. 国内服务器需完成 ICP 备案（约 1-2 周），**或选择境外服务器免备案**

> 如果活动紧急来不及备案，可以：
> - 用境外服务器（香港/新加坡），延迟稍高但免备案
> - 或直接用 IP 访问：`http://服务器IP:7860`（无 HTTPS，不推荐）

## 8. 活动当天运维

```bash
# 查看日志
docker logs -f photofinder

# 重启服务（更新索引后用）
docker restart photofinder

# 停止服务
docker stop photofinder && docker rm photofinder
```

**摄影师追加了新照片时，不要在运行中的容器里 `docker exec ... --prepare`**——webui 已占用一套模型+索引的内存，exec 的 CLI 会再加载第二套，挤爆容器 3G 内存上限被 OOM 杀掉（exit 137）。正确做法：

1. **推荐**：本地跑 `--prepare --incremental` 增量重建索引，只把索引文件传到服务器，然后 `docker restart photofinder`。完整流程见 OPS.md §2.1
2. **应急**（必须在服务器上建时）：先停服务，用临时容器建，再启动：

```bash
docker stop photofinder
docker run --rm -v ~/photofinder/models:/app/models -v ~/photofinder/cache:/app/cache \
  photofinder python -m photofinder.cli --url "你的相册链接" --prepare --incremental
docker start photofinder
```

> 踩坑详情见 OPS.md §3.2（OOM）；本地建索引还需注意 OPS.md §3.1（`PHOTOFINDER_FACE_BATCH=0`）。

## 9. 用户体验流程

1. 你把链接（`http://服务器IP:7860`，有域名则用域名）发到活动群；设了访问码就一并告知
2. 用户打开链接（设了访问码则先输入，用户名随意）；**手机/电脑均可，已适配移动端**
3. 相册链接已预填，用户只需**上传一张正脸自拍**（手机上传时可直接拍照）
4. 点击「开始查找」→ 几秒后看到结果
5. 点击「打包下载」获取 zip

## 10. 安全与隐私注意事项

- **访问码不要公开发布**，只在活动群内分享
- 用户上传的参考照片仅用于本次搜索的特征提取，**不会持久化存储**
- 服务器上的缓存（缩略图、人脸索引）来自公开相册，活动结束后建议清理**本活动相册**的数据（多相册共存时只删对应 `<orderId>` 子目录，别整个 `cache/` 全删，见 OPS.md §3.3）：
  ```bash
  # orderId 是相册链接里的那串数字；实际服务器目录见 OPS.md（~/SearCH）
  rm -rf ~/photofinder/cache/<orderId>/
  docker restart photofinder
  ```
  整个服务彻底下线时才全清：`rm -rf ~/photofinder/cache/*`
- 建议在活动群说明：本工具仅供查找本人照片，请勿用于搜索他人

## 费用总结

| 项目 | 费用 |
|------|------|
| 域名 | ¥50-100/年 |
| 云服务器（4核8G，用几天） | ¥50-200（按量）或 ¥100-300/月 |
| HTTPS 证书 | 免费（Let's Encrypt） |
| **合计** | **¥100-400** |
