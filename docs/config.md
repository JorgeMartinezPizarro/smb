## Benchmarking

The service has been tested with the following hardware and setups:

### i5 13500 64GB RAM

**Config**

```shell
COMPOSE_PROFILES=cpu
GPT_SERVICE=gpt-cpu
BATCH_SIZE=4096 
NUM_THREADS=16
USE_GPU=false
MAX_PROMPT_LENGTH=2048
MAX_TOKENS=256
GPU_LAYERS=0
```

**Results**

- 26ms per read token

- 130ms per generated token

### RTX 3050 8GB VRAM

**Config**

```shell
COMPOSE_PROFILES=gpu
GPT_SERVICE=gpt-gpu
BATCH_SIZE=4096 
NUM_THREADS=6
USE_GPU=true
MAX_PROMPT_LENGTH=2048
MAX_TOKENS=512
GPU_LAYERS=33
```

**Results**

- 1.3ms per read token

- 30.7ms per generated token
