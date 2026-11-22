"""Shared pytest fixtures: an isolated, throw-away database per test."""

from datetime import date

import pytest
from sqlalchemy.orm import sessionmaker

from database.connection import make_engine
from database.init_db import create_tables, seed
from database.synthetic import generate

REFERENCE_DATE = date(2026, 6, 15)  # fixed so tests never depend on "today"


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_tables(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        yield s
        s.rollback()


@pytest.fixture(scope="session")
def synthetic_data():
    return generate(reference_date=REFERENCE_DATE)


@pytest.fixture
def seeded_session(session, synthetic_data):
    seed(session, synthetic_data)
    session.commit()
    return session


@pytest.fixture(scope="session")
def policy_dir(tmp_path_factory):
    """Policy wording PDFs generated into a temp folder (never touches data/)."""
    from rag.policy_documents import generate_policy_documents

    folder = tmp_path_factory.mktemp("policies")
    generate_policy_documents(folder)
    return folder


@pytest.fixture(scope="session")
def policy_index(policy_dir):
    """FAISS index over the generated wording, using offline local embeddings."""
    from rag.embeddings import LocalHashingEmbeddings, EmbeddingBackend
    from rag.retriever import PolicyIndex

    local = LocalHashingEmbeddings()
    return PolicyIndex.build(policy_dir, EmbeddingBackend(local, local.embedding_id, False))
