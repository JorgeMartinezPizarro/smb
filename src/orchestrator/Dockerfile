FROM python:3.11-slim

WORKDIR /app

RUN apt-get update

RUN apt-get install -y --no-install-recommends curl sqlite3 libsqlite3-dev

RUN pip install --no-cache-dir \
    sentence-transformers \
    scikit-learn \
    flask \
    faiss-cpu \
    markdown

COPY faq_ingest.py main.py retriever.py .

CMD ["python", "main.py"]