FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG USE_GPU=false
ENV USE_GPU=${USE_GPU}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3-pip python3-dev \
    build-essential cmake curl git libopenblas-dev libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.9 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /llama.cpp

RUN git clone https://github.com/ggerganov/llama.cpp.git . && \
    mkdir build && cd build && \
    if [ "$USE_GPU" = "true" ]; then \
      cmake .. -DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=86 && make -j$(nproc); \
    else \
      cmake .. && make -j$(nproc); \
    fi

WORKDIR /llama.cpp

RUN pip install --upgrade pip setuptools wheel

RUN CMAKE_ARGS="-DLLAMA_CUBLAS=${USE_GPU}" FORCE_CMAKE=1 pip install .

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]