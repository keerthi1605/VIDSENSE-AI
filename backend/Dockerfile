# VidSense AI backend: FastAPI + the CPU-only model stack.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# ffmpeg: audio extraction and yt-dlp merging. libgl1/libglib2.0-0: needed by opencv-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so this layer is cached until requirements.txt changes.
# torch comes from the CPU-only wheel index (no CUDA runtime, far smaller).
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app ./app

# settings.base_dir resolves to /app, so storage/ and chroma_db/ live at
# /app/storage and /app/chroma_db -- mount volumes there to keep data.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker on purpose: the models are loaded once per process and held in memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
