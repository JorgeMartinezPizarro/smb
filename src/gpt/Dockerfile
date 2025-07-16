FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG USE_GPU=false
ENV USE_GPU=${USE_GPU}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3-pip python3-dev \
    build-essential cmake curl git libopenblas-dev libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH

RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    mkdir -p /llama.cpp/build && cd /llama.cpp/build && \
    if [ "$USE_GPU" = "true" ]; then \
      cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && make -j$(nproc); \
    else \
      cmake .. && make -j$(nproc); \
    fi

WORKDIR /app

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]
