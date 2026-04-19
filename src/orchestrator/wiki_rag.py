import os
import re
import time
import logging
import numpy as np
import requests
import faiss
import unicodedata

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import wikipediaapi

# ----------------------------
# Models (isolated RAG module)
# ----------------------------

model_embed = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

# ----------------------------
# Config
# ----------------------------

CACHE_DIR = "/app/cache/wikipedia"
os.makedirs(CACHE_DIR, exist_ok=True)

WIKIPEDIA_TOP_K = int(os.getenv("WIKIPEDIA_TOP_K", "5"))
WIKIPEDIA_MAX_CONTENT = int(os.getenv("WIKIPEDIA_MAX_CONTENT", "12000"))
WIKIPEDIA_LANG = os.getenv("WIKIPEDIA_LANG", "en")
WIKIPEDIA_MIN_SECTION = int(os.getenv("WIKIPEDIA_MIN_SECTION", "200"))
WIKIPEDIA_BATCH_SIZE = int(os.getenv("WIKIPEDIA_BATCH_SIZE", "32"))

# ----------------------------
# Public API
# ----------------------------

def get_wikipedia_context(subject: str, query: str) -> str:
    chunks, embeddings = get_cached_wikipedia(subject)

    if not chunks:
        return f"No Wikipedia information found about '{subject}'"

    query_embed = model_embed.encode([query], convert_to_numpy=True)[0]
    query_embed /= (np.linalg.norm(query_embed) + 1e-10)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    k = min(80, len(chunks))
    distances, indices = index.search(query_embed.reshape(1, -1), k)

    candidates = [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

    cross_scores = reranker.predict([(query, c[0]) for c in candidates], batch_size=16)

    scored = sorted(
        zip((c[0] for c in candidates), cross_scores),
        key=lambda x: x[1],
        reverse=True
    )

    filtered = filter_redundant_chunks([c for c, _ in scored])

    selected = []
    total = 0

    for c in filtered[:WIKIPEDIA_TOP_K]:
        if total + len(c) > WIKIPEDIA_MAX_CONTENT:
            break
        selected.append(c)
        total += len(c)

    return "\n\n".join(selected)


# ----------------------------
# Cache layer
# ----------------------------

def get_cached_wikipedia(subject: str):
    safe = subject.replace("/", "_").replace(" ", "_")
    cache_file = os.path.join(CACHE_DIR, f"{safe}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        return data["chunks"].tolist(), data["embeddings"]

    page = get_wikipedia_page(subject)
    if not page or not page.exists():
        return [], np.array([])

    chunks = chunk_page(page)

    embeddings = []
    for i in range(0, len(chunks), WIKIPEDIA_BATCH_SIZE):
        batch = chunks[i:i + WIKIPEDIA_BATCH_SIZE]
        emb = model_embed.encode(batch, convert_to_numpy=True)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    np.savez_compressed(cache_file, chunks=np.array(chunks), embeddings=embeddings)

    return chunks, embeddings


# ----------------------------
# Wikipedia fetching
# ----------------------------

def get_wikipedia_page(subject):
    wiki = wikipediaapi.Wikipedia(
        user_agent="knowledge-bot/1.0",
        language=WIKIPEDIA_LANG
    )

    page = wiki.page(subject)
    if page.exists():
        return page

    try:
        url = f"https://{WIKIPEDIA_LANG}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": subject,
            "format": "json"
        }

        r = requests.get(url, params=params).json()
        title = r.get("query", {}).get("search", [{}])[0].get("title")

        if title:
            return wiki.page(title)

    except Exception as e:
        logging.warning(f"Wikipedia fallback failed: {e}")

    return None


# ----------------------------
# Chunking (semantic)
# ----------------------------

def chunk_page(page):
    chunks = []

    if page.summary:
        chunks.append(f"summary\n{clean(page.summary)}")

    def walk(section, path=""):
        title = f"{path} > {section.title}" if path else section.title
        text = clean(section.text)

        if len(text) > 50:
            for p in text.split("\n\n"):
                p = p.strip()
                if len(p) > 50:
                    chunks.append(f"{title}\n{p}")

        for s in section.sections:
            walk(s, title)

    walk(page)
    return chunks


# ----------------------------
# Utils
# ----------------------------

def clean(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def filter_redundant_chunks(chunks, threshold=0.85):
    if not chunks:
        return []

    embeddings = model_embed.encode(chunks, convert_to_numpy=True)

    filtered = []
    kept = []

    for i, emb in enumerate(embeddings):
        if not kept:
            filtered.append(chunks[i])
            kept.append(emb)
            continue

        sims = cosine_similarity([emb], kept)[0]
        if max(sims) < threshold:
            filtered.append(chunks[i])
            kept.append(emb)

    return filtered