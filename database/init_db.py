"""
Create tables and load synthetic data.

    python -m database.init_db            # create tables + seed if empty
    python -m database.init_db --reset    # drop everything, regenerate data, reseed

Seeding is idempotent: running it twice does not duplicate records.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import PROJECT_ROOT, get_logger, settings
from database.connection import engine as default_engine
from database.models import Base, ProcessingStatus
from database.repository import (
    ClaimRepository,
    CustomerRepository,
    DocumentRepository,
    LogRepository,
    PolicyRepository,
)
from database.synthetic import generate, write_json

log = get_logger("init_db")


def create_tables(engine: Engine = default_engine) -> None:
    Base.metadata.create_all(engine)


def drop_tables(engine: Engine = default_engine) -> None:
    Base.metadata.drop_all(engine)


def _d(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def seed(session: Session, data: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Insert customers, policies and claims that are not already present."""
    customers, policies, claims = CustomerRepository(session), PolicyRepository(session), ClaimRepository(session)
    added = {"customers": 0, "policies": 0, "claims": 0}

    by_code = {}
    for row in data["customers"]:
        existing = customers.get_by_code(row["customer_code"])
        if existing:
            by_code[row["customer_code"]] = existing
            continue
        by_code[row["customer_code"]] = customers.create(**{**row, "date_of_birth": _d(row["date_of_birth"])})
        added["customers"] += 1

    for row in data["policies"]:
        if policies.get_by_number(row["policy_number"]):
            continue
        fields = {k: v for k, v in row.items() if k != "customer_code"}
        fields.update(
            customer_id=by_code[row["customer_code"]].id,
            start_date=_d(row["start_date"]),
            end_date=_d(row["end_date"]),
        )
        policies.create(**fields)
        added["policies"] += 1

    for row in data["claims"]:
        if claims.get_by_number(row["claim_number"]):
            continue
        policy = policies.get_by_number(row["policy_number"])  # None for unknown policy numbers
        claims.create(
            claim_number=row["claim_number"],
            customer_id=by_code[row["customer_code"]].id,
            policy_id=policy.id if policy else None,
            customer_name_submitted=row["customer_name"],
            policy_number_submitted=row["policy_number"],
            insurance_type=row["insurance_type"],
            claim_amount=row["claim_amount"],
            hospital_name=row["hospital_name"],
            admission_date=_d(row["admission_date"]),
            discharge_date=_d(row["discharge_date"]),
            diagnosis=row["diagnosis"],
            description=row["description"],
            invoice_number=row["invoice_number"],
            billed_amount=row["billed_amount"],
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            processing_status=ProcessingStatus.PENDING.value,
            scenario=row["scenario"],
        )
        added["claims"] += 1

    if any(added.values()):
        LogRepository(session).log("init_db", "DATABASE_SEEDED", f"Seeded synthetic data: {added}", details=added)
    return added


def scanned_types_for(claim: dict[str, Any], index: int) -> set[str]:
    """Pick a few documents to produce as image-only scans, to exercise OCR:
    Rahul Sharma's diagnostic report, and every 6th valid claim's discharge summary."""
    if claim["customer_name"] == "Rahul Sharma":
        return {"DIAGNOSTIC_REPORT"}
    if claim["scenario"] == "VALID" and index % 6 == 0:
        return {"DISCHARGE_SUMMARY"}
    return set()


def _project_relative(path: Path) -> str:
    """Store paths relative to the project root (portable between machines)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def attach_documents(session: Session, data: dict[str, list[dict[str, Any]]], out_dir: Path | None = None) -> int:
    """Generate claim PDFs (if missing) and register them as ClaimDocument rows."""
    from documents.pdf_generator import generate_claim_documents
    from documents.storage import sha256_bytes

    out_dir = out_dir or settings.claims_dir
    claims, docs = ClaimRepository(session), DocumentRepository(session)
    added = 0
    for index, row in enumerate(data["claims"]):
        claim = claims.get_by_number(row["claim_number"])
        if claim is None or docs.for_claim(claim.id):
            continue
        folder = out_dir / row["claim_number"]
        for path in generate_claim_documents(row, folder, scanned_types=scanned_types_for(row, index)):
            content = path.read_bytes()
            doc_type = next(t for t in row["documents"] if t.lower() in path.name)
            docs.add(
                claim_id=claim.id,
                document_type=doc_type,
                original_filename=path.name,
                stored_filename=path.name,
                file_path=_project_relative(path),
                file_size_bytes=len(content),
                sha256=sha256_bytes(content),
            )
            added += 1
    return added


def index_policies(session: Session) -> int:
    """Generate the policy wording PDFs, build the FAISS index and mirror clauses into the DB."""
    from rag.retriever import PolicyIndex, sync_clauses_to_db
    from rag.policy_documents import generate_policy_documents

    generate_policy_documents()
    return sync_clauses_to_db(session, PolicyIndex.load_or_build())


def load_json(directory: Path) -> dict[str, list[dict[str, Any]]]:
    return {n: json.loads((directory / f"{n}.json").read_text(encoding="utf-8"))
            for n in ("customers", "policies", "claims")}


def init_database(reset: bool = False, engine: Engine = default_engine) -> dict[str, int]:
    if reset:
        log.info("Dropping all tables and generated claim PDFs")
        drop_tables(engine)
        import shutil

        for folder in settings.claims_dir.glob("CLM-*"):
            shutil.rmtree(folder, ignore_errors=True)
    create_tables(engine)

    synthetic = settings.synthetic_dir
    if reset or not (synthetic / "claims.json").exists():
        log.info("Generating synthetic data relative to today")
        write_json(generate())

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        try:
            data = load_json(synthetic)
            added = seed(session, data)
            added["documents"] = attach_documents(session, data)
            added["policy_clauses"] = index_policies(session)
            session.commit()
        except Exception:
            session.rollback()
            raise
    log.info("Seed complete: %s", added)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the InsureFlow AI database")
    parser.add_argument("--reset", action="store_true", help="drop tables, regenerate synthetic data and reseed")
    args = parser.parse_args()
    added = init_database(reset=args.reset)
    print(f"Database ready at {settings.database_url}  (added: {added})")


if __name__ == "__main__":
    main()
