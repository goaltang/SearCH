FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY photofinder/ photofinder/
RUN pip install --no-cache-dir -e ".[fast]"

# 模型文件通过 volume 挂载或预先下载到 /app/models/
# 缓存目录通过 volume 挂载到 /app/cache/
RUN mkdir -p /app/models /app/cache

ENV PHOTOFINDER_HOST=0.0.0.0
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "-m", "photofinder.webui"]
