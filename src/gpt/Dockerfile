# ========================
# Etapa 1: Build llama.cpp con CUDA (builder)
# ========================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        ca-certificates \
        libopenblas-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        ninja-build \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /llama.cpp

RUN git clone https://github.com/ggerganov/llama.cpp.git . && \
    mkdir build && \
    cmake -S . -B build -DGGML_CUDA=ON && \
    cmake --build build -j$(nproc)

RUN pip install --upgrade pip setuptools wheel && \
    pip install . --prefix=/llama-pkg

# ========================
# Etapa 2: Imagen final
# ========================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libopenblas-dev libcurl4-openssl-dev \
    build-essential cmake git curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

COPY --from=builder /llama-pkg /usr/local

WORKDIR /app

COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
