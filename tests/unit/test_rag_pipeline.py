"""Phase 5 — policy wording PDFs, clause chunking, embeddings and the FAISS index."""

import shutil

import numpy as np
import pymupdf
import pytest

from database.catalog import PLANS
from database.repository import PolicyClauseRepository
from rag.document_loader import load_policy_directory, load_policy_document
from rag.embeddings import EmbeddingBackend, LocalHashingEmbeddings, get_embeddings, stem, tokenize
from rag.retriever import PolicyIndex, sync_clauses_to_db


# --------------------------------------------------------------- documents
def test_one_wording_pdf_per_plan(policy_dir):
    names = sorted(p.name for p in policy_dir.glob("*.pdf"))
    assert names == sorted(plan.document_filename for plan in PLANS.values())
    with pymupdf.open(policy_dir / "IHS_policy_wording.pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    assert "COV-01  In-patient hospitalisation" in text and "Fictional policy wording" in text


# ------------------------------------------------------------------ loader
def test_chunks_carry_citation_metadata(policy_dir):
    chunks = load_policy_document(policy_dir / "IHS_policy_wording.pdf")
    codes = [c.clause_code for c in chunks]
    assert len(codes) == len(set(codes))  # one chunk per clause
    assert {"DEF-01", "COV-01", "COV-02", "WP-01", "EXC-01", "EXC-07", "CON-01"} <= set(codes)
    for c in chunks:
        assert c.plan_code == "IHS" and c.source_document == "IHS_policy_wording.pdf"
        assert c.page_number >= 1 and c.title and c.text
        assert "Fictional policy wording generated" not in c.text  # page furniture removed


def test_categories_follow_sections(policy_dir):
    by_code = {c.clause_code: c.category for c in load_policy_document(policy_dir / "PAG_policy_wording.pdf")}
    assert by_code["EXC-PA"] == "EXCLUSION" and by_code["WP-01"] == "WAITING_PERIOD"
    assert by_code["COV-04"] == "COVERAGE" and by_code["DEF-03"] == "DEFINITION" and by_code["CON-02"] == "CONDITION"


def test_plan_specific_clauses(policy_dir):
    chunks = load_policy_directory(policy_dir)
    pa = {c.plan_code for c in chunks if c.clause_code == "EXC-PA"}
    copay = {c.plan_code for c in chunks if c.clause_code == "COV-03"}
    assert pa == {"PAG"} and copay == {"SCP"}
    wp = next(c for c in chunks if c.plan_code == "SCP" and c.clause_code == "WP-01")
    assert "90 days" in wp.text


def test_unstructured_pdf_falls_back_to_general_chunks(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), "Some policy text without clause headings. " * 60)
    path = tmp_path / "plain.pdf"
    doc.save(path)
    chunks = load_policy_document(path)
    assert len(chunks) >= 2
    assert all(c.category == "GENERAL" and c.clause_code is None for c in chunks)
    assert all(len(c.text) <= 900 for c in chunks)


# -------------------------------------------------------------- embeddings
def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The hospitalisation of the patients") == ["hospitalis", "patient"]
    assert stem("excluded") == "exclud" and stem("is") == "is"


def test_local_embeddings_are_deterministic_and_normalised():
    emb = LocalHashingEmbeddings(dim=256)
    a, b = emb.embed_query("cosmetic surgery"), emb.embed_query("cosmetic surgery")
    assert a == b and len(a) == 256
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert emb.embed_query("") == [0.0] * 256


def test_local_embeddings_capture_lexical_similarity():
    emb = LocalHashingEmbeddings()
    q = np.array(emb.embed_query("waiting period after policy start"))
    near = np.array(emb.embed_query("initial waiting period from the policy start date"))
    far = np.array(emb.embed_query("dental implant root canal"))
    assert q @ near > q @ far + 0.2


def test_demo_mode_uses_local_embeddings(monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings.__class__, "demo_mode", property(lambda self: True))
    backend = get_embeddings()
    assert not backend.is_llm and backend.embedding_id.startswith("local-hashing")
    assert "Demo mode" in backend.note


# ------------------------------------------------------------------- index
def test_search_respects_plan_and_category_filters(policy_index):
    hits = policy_index.search("cosmetic rhinoplasty", k=3, plan_code="FFS", categories=("EXCLUSION",))
    assert hits and all(h.chunk.plan_code == "FFS" and h.chunk.category == "EXCLUSION" for h in hits)
    assert hits[0].chunk.clause_code == "EXC-01"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_finds_expected_clause_for_topic(policy_index):
    assert policy_index.search("initial waiting period", k=1, plan_code="CIP")[0].chunk.clause_code == "WP-01"
    assert policy_index.search("sum insured limit of cover", k=1, plan_code="IHS",
                               categories=("COVERAGE",))[0].chunk.clause_code == "COV-02"


def test_min_score_and_empty_query(policy_index):
    assert policy_index.search("", k=3) == []
    assert policy_index.search("quantum spaceship telescope", k=3, min_score=0.2) == []


def test_index_save_load_and_rebuild_on_change(policy_dir, tmp_path):
    local = LocalHashingEmbeddings()
    backend = EmbeddingBackend(local, local.embedding_id, False)
    docs = tmp_path / "policies"
    shutil.copytree(policy_dir, docs)
    store = tmp_path / "store"

    built = PolicyIndex.load_or_build(docs, store, backend)
    assert (store / local.embedding_id / "index.faiss").exists()
    loaded = PolicyIndex.load(backend, store)
    assert loaded.index.ntotal == built.index.ntotal and loaded.sources == built.sources

    (docs / "CIP_policy_wording.pdf").unlink()  # a policy document was removed -> rebuild
    rebuilt = PolicyIndex.load_or_build(docs, store, backend)
    assert "CIP" not in rebuilt.plans() and rebuilt.index.ntotal < built.index.ntotal


def test_build_fails_without_documents(tmp_path):
    with pytest.raises(FileNotFoundError):
        PolicyIndex.build(tmp_path, EmbeddingBackend(LocalHashingEmbeddings(), "x", False))


def test_clauses_synced_to_database_idempotently(session, policy_index):
    assert sync_clauses_to_db(session, policy_index) == len(policy_index.chunks)
    assert sync_clauses_to_db(session, policy_index) == len(policy_index.chunks)
    repo = PolicyClauseRepository(session)
    assert repo.count() == len(policy_index.chunks)
    assert {c.clause_code for c in repo.list(plan_code="PAG", category="EXCLUSION")} == {"EXC-03", "EXC-04", "EXC-PA"}
