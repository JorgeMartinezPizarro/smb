# ========================
# ARG antes de FROM
# ========================
ARG USE_CUDA=cpu

# ========================
# Etapas builder
# ========================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder-cuda
ARG USE_CUDA=cuda

FROM ubuntu:22.04 AS builder-cpu
ARG USE_CUDA=cpu

# Etapa común del builder según USE_CUDA
FROM builder-${USE_CUDA} AS builder

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN if [ "$USE_CUDA" = "cuda" ]; then \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 ; \
fi

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
        python3.10 \
        python3.10-distutils \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git . || echo "Repo already cloned"
RUN mkdir build && \
    if [ "$USE_CUDA" = "cuda" ]; then \
        cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF ; \
    else \
        cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF ; \
    fi && \
    cmake --build build -j$(nproc)

WORKDIR /llama-cpp-python
RUN git clone https://github.com/abetlen/llama-cpp-python.git . && git submodule update --init --recursive
RUN pip install --upgrade pip setuptools wheel cmake ninja scikit-build-core

RUN ls -la /llama.cpp/build/ && ls -la /llama.cpp/build/bin

RUN cp /llama.cpp/build/bin/libllama.so /llama-cpp-python/libllama.so

ARG USE_CUDA=cpu
ENV USE_CUDA=${USE_CUDA}
ARG GGML_CUDA=0
ENV GGML_CUDA=${GGML_CUDA}
ENV CMAKE_ARGS="-DGGML_CUDA=${GGML_CUDA}"
ENV FORCE_CMAKE=1

RUN LLAMA_CPP_LIB=/llama-cpp-python/libllama.so pip wheel . -w wheelhouse --no-deps

# ========================
# Etapas runtime
# ========================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime-cuda
ARG USE_CUDA=cuda

FROM ubuntu:22.04 AS runtime-cpu
ARG USE_CUDA=cpu

FROM runtime-${USE_CUDA} AS final

ARG USE_CUDA=cpu
ENV USE_CUDA=${USE_CUDA}
ARG GGML_CUDA=0
ENV GGML_CUDA=${GGML_CUDA}
ENV LLAMA_CPP_LIB=/llama-cpp-python/libllama.so

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip \
    libopenblas-dev libcurl4-openssl-dev \
    build-essential cmake git curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

# Copiar wheel y llama.cpp desde builder
COPY --from=builder /llama-cpp-python/wheelhouse /tmp/wheelhouse
COPY --from=builder /llama.cpp /llama.cpp

RUN echo "Contenido de /tmp/wheelhouse:" && ls -l /tmp/wheelhouse

# Instalar wheel del binding llama_cpp
RUN pip install /tmp/wheelhouse/llama_cpp_python-*.whl

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh load_model.py ./
RUN chmod +x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
