FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEMATCH_MODEL_DIR=/models \
    DEEPFACE_HOME=/models \
    HF_HOME=/models/hf \
    TF_CPP_MIN_LOG_LEVEL=2

# tesseract: OCR engine · libgl1/libglib: OpenCV runtime · (PyAV and PyMuPDF ship their own native libs)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ml,liveness,voice,mongo,api]"

COPY app ./app

# Model weights (ArcFace, YuNet, MediaPipe, Whisper) download on first use into /models: mount a volume.
RUN useradd -m appuser && mkdir -p /models && chown appuser /models
USER appuser
VOLUME /models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "videokyc.api:app", "--host", "0.0.0.0", "--port", "8000"]
