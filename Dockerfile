FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEMATCH_MODEL_DIR=/models \
    DEEPFACE_HOME=/models

# libgl/glib: required by OpenCV wheels that DeepFace pulls in
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ml,api]"

COPY app ./app

RUN useradd -m appuser && mkdir -p /models && chown appuser /models
USER appuser
VOLUME /models

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "facematch.api:app", "--host", "0.0.0.0", "--port", "8000"]
