ARG USE_GPU=false

FROM python:3.11-slim AS cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    faiss-cpu \
    sentence-transformers \
    scikit-learn \
    flask

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    faiss-gpu \
    sentence-transformers \
    scikit-learn \
    flask

FROM ${USE_GPU:+gpu} AS final

WORKDIR /app

COPY --from=cpu /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=gpu /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY . .

CMD ["python3", "main.py"]
