"""
Retrieval for the prompt placeholders {wikipedia} and {context} (FAQ).

Both sources share one pipeline, wrapped by `Corpus`:
chunk -> embed -> cache -> multi-query FAISS -> rerank -> threshold -> dedupe -> budgeted selection.
"""

import hashlib, logging, os, re, threading, time, unicodedata
from dataclasses import dataclass
from urllib.parse import quote

import faiss
import numpy as np
import requests
import wikipediaapi


def env(name, default):
	"""Read an env var; unset or empty (compose passes `VAR=${VAR}` as "") falls back to default."""
	value = os.getenv(name)
	return value if value else default


# --- Configuration ---
# Models (multilingual: users write in Spanish, sources are Spanish FAQ + Wikipedia in any language)
EMBED_MODEL = env("EMBED_MODEL", "BAAI/bge-m3")
EMBED_MAX_LENGTH = int(env("EMBED_MAX_LENGTH", "512"))                    # Max tokens per embedded text
EMBED_BATCH_SIZE = int(env("EMBED_BATCH_SIZE", "32"))
RERANK_MODEL = env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(env("RERANK_MAX_LENGTH", "512"))                  # Query + ~1000-char chunk fit untruncated
DEDUPE_THRESHOLD = 0.9                                                    # Cosine above which chunks are duplicates

# Wikipedia settings
WIKIPEDIA_CHUNK_SIZE = int(env("WIKIPEDIA_CHUNK_SIZE", "1000"))           # Max chars per content chunk
WIKIPEDIA_TOP_K = int(env("WIKIPEDIA_TOP_K", "5"))                        # Number of chunks to retrieve
WIKIPEDIA_MAX_CONTENT = int(env("WIKIPEDIA_MAX_CONTENT", "12000"))        # Max total chars for prompt
WIKIPEDIA_LANG = env("WIKIPEDIA_LANG", "en")                              # Default language
WIKIPEDIA_MIN_SECTION = int(env("WIKIPEDIA_MIN_SECTION", "200"))          # Min section length to consider
WIKIPEDIA_THRESHOLD = float(env("WIKIPEDIA_THRESHOLD", "0.1"))            # Min reranker relevance (0-1)
WIKIPEDIA_CACHE_TTL = int(env("WIKIPEDIA_CACHE_TTL", "86400"))            # Cache TTL in seconds (default: 1 day)
WIKIPEDIA_FALLBACK_LANGS = env("WIKIPEDIA_FALLBACK_LANGS", "es,en")       # Comma-separated fallback langs
WIKIPEDIA_MAX_SEARCH_RESULTS = int(env("WIKIPEDIA_MAX_SEARCH_RESULTS", "5"))  # Max search candidates
# Wikimedia throttles generic agents: name the bot and a contact URL
WIKIPEDIA_USER_AGENT = "smb-knowledge-bot/1.0 (https://github.com/JorgeMartinezPizarro/smb)"

# FAQ settings
FAQ_FILE = "assets/" + env("FAQ_FILE", "faq") + ".txt"
FAQ_TOP_K = int(env("FAQ_TOP_K", "3"))                                    # Number of Q/A pairs to retrieve
FAQ_THRESHOLD = float(env("FAQ_THRESHOLD", "0.1"))                        # Min reranker relevance (0-1)

CACHE_DIR = env("CACHE_DIR", "/app/cache")
WIKIPEDIA_CACHE_DIR = os.path.join(CACHE_DIR, "wikipedia")
FAQ_CACHE_DIR = os.path.join(CACHE_DIR, "faq")

# e5 models are trained with "query: " / "passage: " prefixes; bge-m3 needs none.
_E5 = "e5" in EMBED_MODEL.lower()
QUERY_PREFIX = "query: " if _E5 else ""
PASSAGE_PREFIX = "passage: " if _E5 else ""
# Cache files carry the embedder name so switching EMBED_MODEL invalidates them.
MODEL_TAG = re.sub(r"[^\w.\-]", "-", EMBED_MODEL.split("/")[-1])

_models = None  # (embedder, reranker)
_models_lock = threading.Lock()


def load_models():
	"""
	Embedder and reranker, loaded on first use: an instance whose prompt has no {wikipedia}
	or {context} never imports torch nor loads them (~5GB of RAM).
	"""
	global _models
	with _models_lock:
		if _models is None:
			from sentence_transformers import SentenceTransformer, CrossEncoder
			logging.info(f"Loading embedder '{EMBED_MODEL}' and reranker '{RERANK_MODEL}'")
			start = time.time()
			embedder = SentenceTransformer(EMBED_MODEL)
			embedder.max_seq_length = EMBED_MAX_LENGTH
			_models = (embedder, CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH))
			logging.info(f"Models loaded in {time.time() - start:.1f}s")
	return _models


def embed(texts, prefix=""):
	"""L2-normalised embeddings, so inner product == cosine similarity."""
	embedder, _ = load_models()
	return embedder.encode(
		[prefix + t for t in texts], batch_size=EMBED_BATCH_SIZE,
		convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
	).astype(np.float32)


def rerank(queries, texts):
	"""
	Cross-encoder relevance in [0, 1] of each text: the best over queries, so every question of a
	multi-question email counts. Sigmoid of the raw logit, whatever the model's default activation.
	"""
	import torch
	_, reranker = load_models()
	logits = reranker.predict(
		[(q, t) for q in queries for t in texts], batch_size=16,
		activation_fn=torch.nn.Identity(), convert_to_numpy=True, show_progress_bar=False,
	)
	logits = np.asarray(logits, dtype=np.float32).reshape(len(queries), len(texts)).max(axis=0)
	return 1.0 / (1.0 + np.exp(-logits))


# --- Corpus abstraction ---

@dataclass
class Hit:
	text: str
	score: float  # reranker relevance in [0, 1]


def cache_path(cache_dir, key):
	return os.path.join(cache_dir, re.sub(r"[^\w\-]", "_", key) + f"__{MODEL_TAG}.npz")


class Corpus:
	"""A set of chunks with their embeddings and a FAISS index over them."""

	def __init__(self, name, chunks, embeddings):
		self.name = name
		self.chunks = list(chunks)
		self.embeddings = embeddings
		self.index = faiss.IndexFlatIP(embeddings.shape[1])
		self.index.add(embeddings)

	@classmethod
	def load(cls, name, cache_file, build_chunks, ttl=None):
		"""
		Load chunks + embeddings from cache_file, or call build_chunks() and embed them.
		The cache is ignored once older than ttl seconds. Returns None if there is nothing to index.
		"""
		fresh = os.path.exists(cache_file) and (ttl is None or time.time() - os.path.getmtime(cache_file) < ttl)
		if fresh:
			try:
				data = np.load(cache_file)
				if str(data["model"]) == EMBED_MODEL:
					logging.info(f"{name}: cache hit ({len(data['chunks'])} chunks)")
					return cls(name, data["chunks"].tolist(), data["embeddings"])
			except Exception as e:
				logging.warning(f"{name}: cache load failed ({e}), rebuilding")

		chunks = build_chunks()
		if not chunks:
			return None
		start = time.time()
		corpus = cls(name, chunks, embed(chunks, PASSAGE_PREFIX))
		logging.info(f"{name}: embedded {len(chunks)} chunks in {time.time() - start:.1f}s")
		try:
			os.makedirs(os.path.dirname(cache_file), exist_ok=True)
			np.savez_compressed(cache_file, chunks=np.array(chunks), embeddings=corpus.embeddings, model=np.array(EMBED_MODEL))
		except Exception as e:
			logging.warning(f"{name}: cache save failed: {e}")
		return corpus

	def search(self, queries, rerank_queries, top_k, threshold=0.0, max_chars=None):
		"""
		Recall candidates with every query variant (best FAISS score per chunk), rerank them
		against rerank_queries, drop those below threshold and near-duplicates, and return up to
		top_k hits within max_chars, best first.
		"""
		if not queries:
			return []
		k = min(top_k * 4, len(self.chunks))
		sims, ids = self.index.search(embed(queries, QUERY_PREFIX), k)
		best = {}
		for row_sims, row_ids in zip(sims, ids):
			for sim, i in zip(row_sims, row_ids):
				if i >= 0:
					best[int(i)] = max(best.get(int(i), -1.0), float(sim))
		candidates = sorted(best, key=best.get, reverse=True)[:k]

		try:
			scores = rerank(rerank_queries, [self.chunks[i] for i in candidates])
		except Exception as e:
			logging.warning(f"{self.name}: reranker failed, using embedding similarity: {e}")
			scores = [best[i] for i in candidates]
		ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

		hits, kept, total = [], [], 0
		for i, score in ranked:
			if score < threshold or len(hits) >= top_k:
				break
			if kept and float(np.max(self.embeddings[kept] @ self.embeddings[i])) >= DEDUPE_THRESHOLD:
				continue
			if max_chars and total + len(self.chunks[i]) > max_chars:
				continue
			hits.append(Hit(self.chunks[i], float(score)))
			kept.append(i)
			total += len(self.chunks[i])

		logging.info(
			f"{self.name}: {len(hits)}/{len(candidates)} candidates kept (threshold {threshold}), "
			f"scores {[round(h.score, 3) for h in hits]}, top rejected "
			f"{[round(float(s), 3) for _, s in ranked[len(hits):len(hits) + 3]]}"
		)
		return hits


# --- Chunking & queries ---

def chunk_text(text, max_chars, overlap=None):
	"""
	Pack paragraphs into chunks of at most max_chars, splitting oversized paragraphs by sentence.
	Each chunk starts with ~20% of the previous one as overlap.
	"""
	overlap = max_chars // 5 if overlap is None else overlap
	pieces = []
	for p in (p.strip() for p in text.split("\n\n")):
		if len(p) <= max_chars:
			pieces += [p] if p else []
			continue
		for sentence in re.split(r"(?<=[.!?])\s+", p):
			pieces += [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)]

	chunks, current = [], ""
	for piece in pieces:
		if current and len(current) + len(piece) + 2 > max_chars:
			chunks.append(current)
			tail = current[-overlap:] if overlap else ""
			tail = tail[tail.find(" ") + 1:] if " " in tail else tail
			current = tail if len(tail) + len(piece) + 2 <= max_chars else ""
		current = f"{current}\n\n{piece}" if current else piece
	if current:
		chunks.append(current)
	return chunks


def chunk_faq(text):
	"""One chunk per question/answer pair; a leading line without '?' is a category header and is dropped."""
	chunks = []
	for block in re.split(r"\n\s*\n", text):
		lines = [l.strip() for l in block.splitlines() if l.strip()]
		if len(lines) > 2 and "?" not in lines[0]:
			lines = lines[1:]
		if lines:
			chunks.append("\n".join(lines))
	return chunks


def questions(body, limit=3):
	"""The first question sentences of the message."""
	return [q.strip() for q in re.findall(r"[^.!?\n]*\?", body) if len(q.strip()) > 10][:limit]


def expand_queries(body, subject=""):
	"""Query variants for recall: body, subject, subject+body, each question, capitalised key phrases."""
	queries = [body]
	if subject:
		queries += [subject, f"{subject}: {body[:200]}"]
	queries += questions(body)
	key_phrases = re.findall(r"\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*\b", body)
	if key_phrases:
		queries.append(" ".join(key_phrases[:5]))
	return list(dict.fromkeys(q for q in queries if q.strip()))


# --- FAQ ---

_faq = (None, None)  # (file hash, Corpus)


def faq_hits(body):
	"""Top FAQ question/answer pairs for the message. The index is rebuilt when the file changes."""
	global _faq
	try:
		with open(FAQ_FILE, encoding="utf-8") as f:
			text = f.read()
	except FileNotFoundError:
		logging.warning(f"FAQ file not found: {FAQ_FILE}")
		return []
	digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
	if _faq[0] != digest:
		key = f"{os.path.basename(FAQ_FILE)}_{digest}"
		_faq = (digest, Corpus.load("FAQ", cache_path(FAQ_CACHE_DIR, key), lambda: chunk_faq(text)))
	corpus = _faq[1]
	if not corpus:
		return []
	return corpus.search(expand_queries(body), [body] + questions(body), FAQ_TOP_K, FAQ_THRESHOLD)


# --- Wikipedia ---

def strip_reply_prefixes(subject):
	"""Remove repeated reply/forward prefixes: 'Re: RE: Fwd: X' -> 'X'"""
	return re.sub(r"^(\s*(re|fwd?|rv|res)\s*:\s*)+", "", subject, flags=re.IGNORECASE).strip()


def normalize_subject(subject):
	"""
	Normalize a subject string for more reliable Wikipedia lookups.
	Strips common filler words and normalizes unicode. The case is kept: titles are case-sensitive
	after the first letter ("Club de Fútbol", "Revolución francesa") and MediaWiki capitalizes the first one.
	"""
	subject = unicodedata.normalize("NFKC", subject)
	subject = strip_reply_prefixes(subject)
	subject = re.sub(r"^(asunto|pregunta|duda sobre|consulta sobre)[:\s]+", "", subject, flags=re.IGNORECASE)
	subject = re.sub(r"[^\w\s\-áéíóúñüÁÉÍÓÚÑÜ]", " ", subject)
	return re.sub(r"\s+", " ", subject).strip()


def search_wikipedia_candidates(subject, lang, max_results=WIKIPEDIA_MAX_SEARCH_RESULTS):
	"""
	Search Wikipedia for candidate page titles.
	Returns a list of (title, snippet) tuples ranked by search relevance.
	"""
	params = {
		"action": "query",
		"list": "search",
		"srsearch": subject,
		"srlimit": max_results,
		"srprop": "snippet|titlesnippet",
		"format": "json",
	}
	try:
		resp = requests.get(
			f"https://{lang}.wikipedia.org/w/api.php", params=params, timeout=10,
			headers={"User-Agent": WIKIPEDIA_USER_AGENT},
		)
		if resp.status_code == 200:
			return [(r["title"], r.get("snippet", "")) for r in resp.json().get("query", {}).get("search", [])]
	except Exception as e:
		logging.warning(f"Wikipedia candidate search failed for '{subject}' in '{lang}': {e}")
	return []


def wikipedia_langs(lang=WIKIPEDIA_LANG):
	"""lang first, then WIKIPEDIA_FALLBACK_LANGS"""
	return [lang] + [l.strip() for l in WIKIPEDIA_FALLBACK_LANGS.split(",") if l.strip() and l.strip() != lang]


def get_wikipedia_page(subject, lang=WIKIPEDIA_LANG):
	"""
	Retrieve the best Wikipedia page for a subject.

	Strategy:
	  1. Direct lookup by exact title.
	  2. API search for up to WIKIPEDIA_MAX_SEARCH_RESULTS candidates; pick the
	     one whose title or snippet best matches the subject (simple token overlap).
	  3. If configured, fall back to alternative languages.
	"""
	def score_candidate(title, snippet, query):
		query_tokens = set(re.findall(r"\w+", query.lower()))
		target_tokens = set(re.findall(r"\w+", (title + " " + snippet).lower()))
		return len(query_tokens & target_tokens) / len(query_tokens) if query_tokens else 0.0

	for current_lang in wikipedia_langs(lang):
		wiki = wikipediaapi.Wikipedia(
			user_agent=WIKIPEDIA_USER_AGENT,
			language=current_lang,
			extract_format=wikipediaapi.ExtractFormat.WIKI,
		)

		page = wiki.page(subject)
		if page.exists():
			logging.info(f"Wikipedia direct hit: '{subject}' in '{current_lang}'")
			return page

		candidates = search_wikipedia_candidates(subject, current_lang)
		if candidates:
			# An exact title match goes first; ties keep Wikipedia's own relevance order (stable sort)
			scored = sorted(
				((score_candidate(t, s, subject) + (t.lower() == subject.lower()), t) for t, s in candidates),
				key=lambda c: c[0], reverse=True,
			)
			logging.info(
				f"Wikipedia candidates for '{subject}' in '{current_lang}': "
				+ ", ".join(f"{t} ({score:.2f})" for score, t in scored[:3])
			)
			for score, title in scored:
				if score == 0:
					break  # unrelated results (e.g. a Spanish subject on en.wikipedia): try the next language
				page = wiki.page(title)
				if page.exists():
					logging.info(f"Wikipedia search hit: '{title}' in '{current_lang}'")
					return page

	logging.warning(f"No Wikipedia page found for '{subject}' in any language.")
	return None


def clean_wikipedia_text(text):
	"""Normalize and clean Wikipedia text"""
	if not text:
		return ""
	text = unicodedata.normalize("NFKC", text)
	text = "".join(ch for ch in text if (ch == "\n" or unicodedata.category(ch)[0] != "C"))
	text = re.sub(r"[\t\r]+", " ", text)
	text = re.sub(r" *\n+ *", "\n\n", text)
	return re.sub(r"[ ]{2,}", " ", text).strip()


def extract_sections(page):
	"""(title, text) for the intro and every section of at least WIKIPEDIA_MIN_SECTION chars."""
	sections = []
	intro = clean_wikipedia_text(page.summary)
	if len(intro) >= WIKIPEDIA_MIN_SECTION:
		sections.append((page.title, intro))

	def process_section(section, parent_title):
		full_title = f"{parent_title} > {section.title}"
		text = clean_wikipedia_text(section.text)
		if len(text) >= WIKIPEDIA_MIN_SECTION:
			sections.append((full_title, text))
		for subsection in section.sections:
			process_section(subsection, full_title)

	for section in page.sections:
		process_section(section, page.title)
	return sections


def wikipedia_corpus(subject, lang=WIKIPEDIA_LANG):
	def build():
		page = get_wikipedia_page(subject, lang)
		if not page:
			return []
		return [
			f"{title}\n{chunk}"
			for title, text in extract_sections(page)
			for chunk in chunk_text(text, WIKIPEDIA_CHUNK_SIZE)
		]
	return Corpus.load(
		f"Wikipedia '{subject}'", cache_path(WIKIPEDIA_CACHE_DIR, f"{lang}_{subject}"), build, ttl=WIKIPEDIA_CACHE_TTL
	)


class ArticleNotFound(Exception):
	"""No Wikipedia article for the subject; carries Wikipedia's suggestions for the reply."""

	def __init__(self, subject, suggestions):
		super().__init__(f"No Wikipedia article for '{subject}'")
		self.subject = subject
		self.suggestions = suggestions  # [(title, url)]


def wikipedia_suggestions(subject, max_results=5):
	"""
	[(title, url)] that Wikipedia proposes for a subject without article: the search results of the
	first language that has any, retrying with Wikipedia's spelling suggestion ("¿quisiste decir…?").
	"""
	for lang in wikipedia_langs():
		query = subject
		for _ in range(2):  # the subject, then Wikipedia's spelling suggestion
			try:
				data = requests.get(
					f"https://{lang}.wikipedia.org/w/api.php", timeout=10, headers={"User-Agent": WIKIPEDIA_USER_AGENT},
					params={"action": "query", "list": "search", "srsearch": query, "srlimit": max_results,
						"srinfo": "suggestion", "srprop": "", "format": "json"},
				).json().get("query", {})
			except Exception as e:
				logging.warning(f"Wikipedia suggestions failed for '{query}' in '{lang}': {e}")
				break
			titles = [r["title"] for r in data.get("search", [])]
			if titles:
				return [(t, f"https://{lang}.wikipedia.org/wiki/{quote(t.replace(' ', '_'))}") for t in titles]
			query = data.get("searchinfo", {}).get("suggestion")
			if not query:
				break
	return []


def wikipedia_hits(subject, body):
	"""
	Most relevant Wikipedia chunks about the email subject for the message ([] if none is relevant).
	Raises ArticleNotFound when the subject is empty or names no article.
	"""
	normalized = normalize_subject(subject)
	if not normalized:
		raise ArticleNotFound("", [])
	corpus = wikipedia_corpus(normalized)
	if not corpus and normalized != subject:
		logging.info(f"Retrying Wikipedia fetch with raw subject: '{subject}'")
		corpus = wikipedia_corpus(subject)
	if not corpus:
		raise ArticleNotFound(normalized, wikipedia_suggestions(normalized))
	# The subject names the topic; the body alone ("¿cuándo nació?") may not.
	prefix = "" if normalized.lower() in body.lower() else f"{normalized}: "
	rerank_queries = [prefix + q for q in [body] + questions(body)]
	return corpus.search(
		expand_queries(body, normalized), rerank_queries, WIKIPEDIA_TOP_K, WIKIPEDIA_THRESHOLD, WIKIPEDIA_MAX_CONTENT
	)
