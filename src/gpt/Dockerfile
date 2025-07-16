# Imagen base con CUDA toolkit (sin driver, lo pondrá el host)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG USE_GPU=false
ENV USE_GPU=${USE_GPU}

# Instalación de paquetes necesarios
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3-pip python3-dev \
    build-essential cmake curl git libopenblas-dev libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# IMPORTANTE: usar stub para linkar libcuda en build-time (evita linker errors)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH

# Clonar y compilar llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    cd /llama.cpp && mkdir -p build && cd build && \
    if [ "$USE_GPU" = "true" ]; then \
      cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && \
      make -j$(nproc); \
    else \
      cmake .. && \
      make -j$(nproc); \
    fi

# Instalar bindings Python
RUN pip3 install --force-reinstall --no-cache-dir /llama.cpp/python-package

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
