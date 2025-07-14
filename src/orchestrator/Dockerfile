FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias sistema para compilar y BLAS/OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar las librerías Python necesarias
RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    scikit-learn \
    flask

# Copiar el código
COPY . .

CMD ["python", "main.py"]