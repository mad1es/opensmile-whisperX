# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=python:3.10-slim
ARG OPENSMILE_VERSION=v3.0.1
ARG TORCH_CHANNEL=cpu
FROM ${BASE_IMAGE} AS runtime

ARG OPENSMILE_VERSION
ARG TORCH_CHANNEL

ENV OPENSMILE_VERSION=${OPENSMILE_VERSION} \
    TORCH_CHANNEL=${TORCH_CHANNEL} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libasound2-dev \
    libpulse-dev \
    libfftw3-dev \
    libsndfile1-dev \
    libsamplerate0-dev \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY docker/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    if [ "${TORCH_CHANNEL}" = "cpu" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            torch==2.4.1 torchaudio==2.4.1; \
    else \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL} \
            torch==2.4.1 torchaudio==2.4.1; \
    fi && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip

COPY . /app

RUN git clone --depth 1 --branch ${OPENSMILE_VERSION} https://github.com/audeering/opensmile.git /tmp/opensmile && \
    cmake -S /tmp/opensmile -B /tmp/opensmile/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/opensmile/build --target SMILExtract --config Release -- -j "$(nproc)" && \
    install -m 755 /tmp/opensmile/build/progsrc/smilextract/SMILExtract /app/opensmile/bin/SMILExtract && \
    rm -rf /tmp/opensmile

ENV OPENSMILE_BIN=/app/opensmile/bin/SMILExtract \
    PYTHONPATH=/app

ENV PATH="$PATH:/app/opensmile/bin"

CMD ["bash"]


