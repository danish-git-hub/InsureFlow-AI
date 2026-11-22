"""
FAISS vector index over the policy clause chunks.

    python -m rag.retriever                 # build (or load) the index and sync clauses to the DB
    python -m rag.retriever --rebuild
    python -m rag.retriever --query "cosmetic surgery" --plan IHS

Layout on disk (one folder per embedding model, so Gemini and local vectors
are never mixed):

    data/vectorstore/<embedding_id>/index.faiss    FAISS IndexFlatIP
    data/vectorstore/<embedding_id>/chunks.json    clause text + citation metadata
    data/vectorstore/<embedding_id>/manifest.json  embedding id, dim, source-file hashes

Vectors are L2-normalised, so the inner product equals cosine similarity
(1.0 = identical direction, 0 = unrelated). The index is rebuilt automatically
when a policy PDF is added, removed or changed (hashes in the manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy.orm import Session

from app.config import get_logger, settings
from rag.document_loader import ClauseChunk, load_policy_directory
from rag.embeddings import EmbeddingBackend, LocalHashingEmbeddings, get_embeddings

log = get_logger("retriever")


@dataclass
class SearchHit:
    chunk: ClauseChunk
    score: float  # cosine similarity


def _embed_text(chunk: ClauseChunk) -> str:
    """What gets embedded: the clause heading adds strong topical words to the body."""
    head = " ".join(x for x in (chunk.clause_code, chunk.title) if x)
    return f"{head}. {chunk.text}" if head else chunk.text


def _file_hashes(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(directory).glob("*.pdf"))}


def _as_matrix(vectors: list[list[float]]) -> np.ndarray:
    mat = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(mat)
    return mat


class PolicyIndex:
    """Clause-level FAISS index with metadata filtering (plan, category)."""

    def __init__(self, backend: EmbeddingBackend, index: faiss.Index, chunks: list[ClauseChunk],
                 sources: dict[str, str]):
        self.backend = backend
        self.index = index
        self.chunks = chunks
        self.sources = sources

    # ----------------------------------------------------------------- build
    @classmethod
    def build(cls, policies_dir: Path | None = None, backend: EmbeddingBackend | None = None) -> PolicyIndex:
        policies_dir = Path(policies_dir or settings.policies_dir)
        backend = backend or get_embeddings()
        chunks = load_policy_directory(policies_dir)
        if not chunks:
            raise FileNotFoundError(f"No policy PDFs with text found in {policies_dir}")
        try:
            vectors = backend.model.embed_documents([_embed_text(c) for c in chunks])
        except Exception as exc:
            if not backend.is_llm:
                raise
            # Gemini quota/network failure: index locally and say so (never silently).
            log.warning("Gemini embedding failed (%s); falling back to local hashing embeddings", exc)
            local = LocalHashingEmbeddings()
            backend = EmbeddingBackend(local, local.embedding_id, False, f"Gemini embedding failed: {exc}")
            vectors = backend.model.embed_documents([_embed_text(c) for c in chunks])
        mat = _as_matrix(vectors)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        log.info("Indexed %s clause chunks from %s documents with %s", len(chunks),
                 len({c.source_document for c in chunks}), backend.embedding_id)
        return cls(backend, index, chunks, _file_hashes(policies_dir))

    # ------------------------------------------------------------- persist
    @staticmethod
    def folder(embedding_id: str, base: Path | None = None) -> Path:
        return Path(base or settings.vectorstore_dir) / embedding_id

    def save(self, base: Path | None = None) -> Path:
        folder = self.folder(self.backend.embedding_id, base)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        (folder / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8")
        manifest = {"embedding_id": self.backend.embedding_id, "dimension": self.index.d,
                    "chunks": len(self.chunks), "sources": self.sources}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, backend: EmbeddingBackend, base: Path | None = None) -> PolicyIndex | None:
        folder = cls.folder(backend.embedding_id, base)
        if not (folder / "index.faiss").exists():
            return None
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        chunks = [ClauseChunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        return cls(backend, faiss.read_index(str(folder / "index.faiss")), chunks, manifest.get("sources", {}))

    @classmethod
    def load_or_build(cls, policies_dir: Path | None = None, base: Path | None = None,
                      backend: EmbeddingBackend | None = None, rebuild: bool = False) -> PolicyIndex:
        policies_dir = Path(policies_dir or settings.policies_dir)
        backend = backend or get_embeddings()
        if not rebuild:
            existing = cls.load(backend, base)
            if existing and existing.sources == _file_hashes(policies_dir):
                return existing
        index = cls.build(policies_dir, backend)
        index.save(base)
        return index

    # -------------------------------------------------------------- search
    @property
    def embedding_id(self) -> str:
        return self.backend.embedding_id

    def plans(self) -> set[str]:
        return {c.plan_code for c in self.chunks if c.plan_code}

    def search(self, query: str, k: int = 3, plan_code: str | None = None,
               categories: tuple[str, ...] | None = None, min_score: float = 0.0) -> list[SearchHit]:
        """Top-k clauses for `query`, optionally restricted to one plan and some categories.

        The index is small (hundreds of chunks), so we score every chunk and
        filter afterwards; this keeps metadata filtering exact.
        """
        if not query.strip() or self.index.ntotal == 0:
            return []
        q = _as_matrix([self.backend.model.embed_query(query)])
        scores, ids = self.index.search(q, self.index.ntotal)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if plan_code and chunk.plan_code != plan_code:
                continue
            if categories and chunk.category not in categories:
                continue
            if score < min_score:
                break  # results are sorted by score
            hits.append(SearchHit(chunk, float(score)))
            if len(hits) == k:
                break
        return hits


def sync_clauses_to_db(session: Session, index: PolicyIndex) -> int:
    """Mirror the indexed clauses into the PolicyClause table (dashboard + audit)."""
    from database.repository import PolicyClauseRepository

    repo = PolicyClauseRepository(session)
    total = 0
    by_doc: dict[str, list[ClauseChunk]] = {}
    for c in index.chunks:
        by_doc.setdefault(c.source_document, []).append(c)
    for source, chunks in by_doc.items():
        total += repo.replace_for_document(source, [c.db_fields() for c in chunks])
    return total


def ingest_policies(session: Session | None = None, rebuild: bool = False) -> PolicyIndex:
    """Generate wording PDFs if missing, build/load the index, and sync clauses to the DB."""
    from rag.policy_documents import generate_policy_documents

    generate_policy_documents()
    index = PolicyIndex.load_or_build(rebuild=rebuild)
    if session is not None:
        sync_clauses_to_db(session, index)
    return index


@lru_cache(maxsize=1)
def get_policy_index() -> PolicyIndex:
    """Process-wide index used by the RAG agent (built once, then loaded from disk)."""
    from rag.policy_documents import generate_policy_documents

    generate_policy_documents()
    return PolicyIndex.load_or_build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the policy FAISS index")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--plan")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    from database.connection import SessionLocal, session_scope

    with session_scope(SessionLocal) as session:
        index = ingest_policies(session, rebuild=args.rebuild)
    print(f"Index: {len(index.chunks)} clauses, plans {sorted(index.plans())}, embeddings {index.embedding_id}")
    if args.query:
        for hit in index.search(args.query, k=args.k, plan_code=args.plan):
            c = hit.chunk
            print(f"  {hit.score:.3f}  {c.plan_code} {c.clause_code} {c.title}  ({c.source_document} p.{c.page_number})")


if __name__ == "__main__":
    main()
