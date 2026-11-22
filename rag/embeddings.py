"""
Text embeddings for policy retrieval.

Two implementations of LangChain's `Embeddings` interface:

* **Gemini embeddings** (`GoogleGenerativeAIEmbeddings`) when GOOGLE_API_KEY is set.
* **LocalHashingEmbeddings** in DEMO MODE: a deterministic, offline
  "hashed bag of words" vector. It is not a neural model and is labelled as
  such everywhere (`embedding_id = "local-hashing-v1-512"`). It needs no
  download and no network, so the demo works on any laptop.

Beginner note — how hashing embeddings work:
    Each token (word, word pair, word fragment) is hashed to one of `dim`
    positions and added there (with a +/- sign from the hash). Texts that share
    words end up with similar vectors, so cosine similarity measures overlap in
    vocabulary. It captures *lexical* similarity only — that is why the RAG
    agent applies a minimum score and reports uncertainty instead of guessing.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import get_logger, settings

log = get_logger("embeddings")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to under was were will with "
    "any all such which who may shall not no than then there these those their per into been being".split()
)
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stop words, light stemming."""
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


class LocalHashingEmbeddings(Embeddings):
    """Deterministic offline embeddings (DEMO MODE). Not a neural model."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    @property
    def embedding_id(self) -> str:
        return f"local-hashing-v1-{self.dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        features: dict[str, float] = {}
        for tok in tokens:
            features[f"w:{tok}"] = features.get(f"w:{tok}", 0.0) + 1.0
            padded = f"#{tok}#"
            for i in range(len(padded) - 3):  # character 4-grams: robust to word forms and OCR noise
                features[f"c:{padded[i:i + 4]}"] = features.get(f"c:{padded[i:i + 4]}", 0.0) + 0.1
        for a, b in zip(tokens, tokens[1:]):
            features[f"b:{a}_{b}"] = features.get(f"b:{a}_{b}", 0.0) + 0.5
        for feature, weight in features.items():
            idx, sign = self._bucket(feature)
            vec[idx] += sign * math.sqrt(weight)  # sub-linear: repeated words count less
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@dataclass
class EmbeddingBackend:
    model: Embeddings
    embedding_id: str
    is_llm: bool  # True only when a Gemini API model produces the vectors
    note: str = ""


def get_embeddings(prefer_gemini: bool | None = None) -> EmbeddingBackend:
    """Gemini embeddings when a key is configured, otherwise local hashing embeddings."""
    use_gemini = (not settings.demo_mode) if prefer_gemini is None else prefer_gemini
    if use_gemini and not settings.demo_mode:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            model = GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model,
                                                 google_api_key=settings.google_api_key)
            slug = settings.gemini_embedding_model.split("/")[-1]
            return EmbeddingBackend(model, f"gemini-{slug}", True)
        except Exception as exc:  # package/config problem -> honest fallback
            log.warning("Gemini embeddings unavailable (%s); using local hashing embeddings", exc)
            local = LocalHashingEmbeddings()
            return EmbeddingBackend(local, local.embedding_id, False, f"Gemini embeddings unavailable: {exc}")
    local = LocalHashingEmbeddings()
    return EmbeddingBackend(local, local.embedding_id, False, "Demo mode: deterministic local hashing embeddings")
