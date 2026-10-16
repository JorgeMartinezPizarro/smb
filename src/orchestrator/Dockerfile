FROM python:3.11-slim

WORKDIR /app

RUN apt-get update

RUN apt-get install -y --no-install-recommends curl sqlite3 libsqlite3-dev

# CPU-only torch first, so sentence-transformers doesn't pull the multi-GB CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "sentence-transformers>=4" \
    flask \
    faiss-cpu \
	numpy \
    markdown

RUN pip install --no-cache-dir wikipedia-api requests

COPY *.py ./

CMD ["python", "main.py"]
