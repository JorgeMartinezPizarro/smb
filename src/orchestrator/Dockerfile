FROM python:3.11-slim

ARG USE_GPU=false
ENV USE_GPU=${USE_GPU}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalación común
RUN pip install --no-cache-dir \
    sentence-transformers \
    scikit-learn \
    flask

# Instalación condicional según GPU o CPU
RUN if [ "$USE_GPU" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-11-8 cuda-cublas-dev-11-8 && \
      pip install --no-cache-dir faiss-gpu; \
    else \
      pip install --no-cache-dir faiss-cpu; \
    fi && \
    rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "main.py"]