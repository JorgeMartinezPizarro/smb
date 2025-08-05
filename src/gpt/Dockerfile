# ========================
# ARG comunes
# ========================
ARG USE_CUDA=cpu
ARG GGML_CUDA=0

# ========================
# Etapas de build
# ========================

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base-builder-cuda
ARG USE_CUDA=cuda

FROM ubuntu:22.04 AS base-builder-cpu
ARG USE_CUDA=cpu

FROM base-builder-${USE_CUDA} AS llama-builder
ARG USE_CUDA
ARG GGML_CUDA

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Stubs si usamos CUDA
RUN if [ "$USE_CUDA" = "cuda" ]; then \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 ; \
fi

# Dependencias básicas de build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl ca-certificates \
        libopenblas-dev libssl-dev libcurl4-openssl-dev \
        ninja-build python3.10 python3.10-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git . && \
    mkdir build && \
    cd build && \
    cmake .. -DGGML_CUDA=${GGML_CUDA} -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES="50;61;70;75;80;86;89;90" && \
    cmake --build . -j$(nproc)

# ------------------------------
# Build wheel llama-cpp-python
# ------------------------------
FROM llama-builder AS wheel-builder

WORKDIR /llama-cpp-python
RUN git clone https://github.com/abetlen/llama-cpp-python.git . && \
    git submodule update --init --recursive

RUN pip install --upgrade pip setuptools wheel cmake ninja scikit-build-core

# Copiamos libllama.so a la ruta esperada por el binding
RUN cp /llama.cpp/build/bin/libllama.so /llama-cpp-python/libllama.so

ENV LLAMA_CPP_LIB=/llama-cpp-python/libllama.so
ENV FORCE_CMAKE=1

RUN pip wheel . -w wheelhouse --no-deps

# ========================
# Etapas de runtime
# ========================

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime-cuda
ARG USE_CUDA=cuda

FROM ubuntu:22.04 AS runtime-cpu
ARG USE_CUDA=cpu

FROM runtime-${USE_CUDA} AS final
ARG USE_CUDA
ARG GGML_CUDA

ENV USE_CUDA=${USE_CUDA}
ENV GGML_CUDA=${GGML_CUDA}

# Dependencias para ejecutar
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip \
    libopenblas-dev libcurl4-openssl-dev \
    build-essential cmake git curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# ---------------------
# Archivos desde builder
# ---------------------
COPY --from=wheel-builder /llama-cpp-python/wheelhouse /tmp/wheelhouse
COPY --from=llama-builder /llama.cpp /llama.cpp
RUN pip install /tmp/wheelhouse/llama_cpp_python-*.whl

# ---------------------
# Archivos del proyecto
# ---------------------
WORKDIR /app

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY load_model.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
