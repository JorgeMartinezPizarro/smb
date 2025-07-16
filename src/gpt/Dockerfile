FROM python:3.9-slim

ARG USE_GPU=false
ENV USE_GPU=${USE_GPU}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Compila llama-cpp-python desde fuente con o sin cuBLAS
RUN if [ "$USE_GPU" = "true" ]; then \
      pip install llama-cpp-python[cuda] --force-reinstall --no-cache-dir; \
    else \
      pip install llama-cpp-python --force-reinstall --no-cache-dir; \
    fi

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
