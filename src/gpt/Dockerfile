# ========================
# ARG comunes
# ========================
ARG USE_CUDA=cpu
ARG GGML_CUDA=0

# ========================
# Base images
# ========================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base-builder-cuda
ARG USE_CUDA=cuda

FROM ubuntu:22.04 AS base-builder-cpu
ARG USE_CUDA=cpu

# ========================
# Builder único para llama.cpp + wheel
# ========================
FROM base-builder-${USE_CUDA} AS builder
ARG USE_CUDA
ARG GGML_CUDA

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Stubs si usamos CUDA
RUN if [ "$USE_CUDA" = "cuda" ]; then \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 ; \
fi

# Dependencias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl ca-certificates \
    libopenblas-dev libssl-dev libcurl4-openssl-dev \
    ninja-build python3.10 python3.10-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git .

RUN mkdir build && cd build && \
    cmake .. -DGGML_CUDA=${GGML_CUDA} -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES="50;61;70;75;80;86;89;90" && \
    cmake --build . -j$(nproc)

# Verificación rápida para debug
RUN echo "Verificando símbolos CUDA en libllama.so" && \
    nm -D build/bin/libllama.so | grep cuda || echo "No se encontraron símbolos CUDA"

# Build wheel llama-cpp-python
WORKDIR /llama-cpp-python
RUN git clone https://github.com/abetlen/llama-cpp-python.git . && \
    git submodule update --init --recursive

RUN pip install --upgrade pip setuptools wheel cmake ninja scikit-build-core

# Copiamos la librería CUDA build
RUN cp /llama.cpp/build/bin/libllama.so /llama-cpp-python/libllama.so

# Variables para forzar uso de esa lib
ENV LLAMA_CPP_LIB=/llama-cpp-python/libllama.so
ENV FORCE_CMAKE=1
ENV GGML_CUDA=${GGML_CUDA}

# Construir wheel usando esa lib sin recompilar
RUN FORCE_CMAKE=1 GGML_CUDA=${GGML_CUDA} \
    LLAMA_CPP_LIB=/llama-cpp-python/libllama.so \
    pip wheel . -w wheelhouse --no-deps

# ========================
# Runtime con CUDA
# ========================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime
ARG USE_CUDA
ARG GGML_CUDA

ENV USE_CUDA=${USE_CUDA}
ENV GGML_CUDA=${GGML_CUDA}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip \
    libopenblas-dev libcurl4-openssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

# Copiamos wheel y la librería
COPY --from=builder /llama-cpp-python/wheelhouse /tmp/wheelhouse
COPY --from=builder /llama.cpp /llama.cpp

RUN pip install /tmp/wheelhouse/llama_cpp_python-*.whl

COPY --from=builder /llama.cpp/build/bin/libllama.so /usr/local/lib/python3.10/dist-packages/llama_cpp/lib/libllama.so

# Dependencias del proyecto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY load_model.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]
