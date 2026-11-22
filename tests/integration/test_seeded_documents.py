"""Integration: seeding generates PDFs, registers them, and the agent reads them back."""

from pathlib import Path

from agents.document_agent import DocumentExtractionAgent
from app.config import PROJECT_ROOT
from database.init_db import attach_documents
from database.repository import ClaimRepository, DocumentRepository


def test_seed_attaches_documents_and_extraction_matches(seeded_session, synthetic_data, tmp_path):
    out = tmp_path / "claims"
    added = attach_documents(seeded_session, synthetic_data, out_dir=out)
    assert added == sum(len(c["documents"]) for c in synthetic_data["claims"])
    assert attach_documents(seeded_session, synthetic_data, out_dir=out) == 0  # idempotent

    claims, docs = ClaimRepository(seeded_session), DocumentRepository(seeded_session)
    agent = DocumentExtractionAgent(use_llm=False)
    for row in synthetic_data["claims"]:
        stored = docs.for_claim(claims.get_by_number(row["claim_number"]).id)
        assert {d.document_type for d in stored} == set(row["documents"])
        assert all(len(d.sha256) == 64 for d in stored)
        result = agent.run([PROJECT_ROOT / Path(d.file_path) for d in stored])
        assert result.fields.policy_number == row["policy_number"]
        assert result.fields.claim_amount == row["claim_amount"]
        assert result.fields.invoice_number == row["invoice_number"]
