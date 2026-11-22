"""Phase 9 — FastAPI endpoints against an isolated, seeded database."""

import dataclasses
import shutil

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.main import app
from database.connection import get_db
from database.init_db import seed
from documents.pdf_generator import build_document


@pytest.fixture
def client(engine, synthetic_data, tmp_path, monkeypatch, policy_dir):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        seed(s, synthetic_data)
        s.commit()

    def override():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Keep uploads, policy PDFs and the vector index inside tmp_path.
    import documents.storage
    import rag.retriever
    import services.claim_service
    from app.config import settings

    policies = tmp_path / "policies"
    shutil.copytree(policy_dir, policies)
    patched = dataclasses.replace(settings, uploads_dir=tmp_path / "uploads", policies_dir=policies,
                                  vectorstore_dir=tmp_path / "vectorstore")
    for module in (documents.storage, rag.retriever, services.claim_service):
        monkeypatch.setattr(module, "settings", patched)
    rag.retriever.get_policy_index.cache_clear()
    monkeypatch.setattr(rag.retriever, "generate_policy_documents", lambda *a, **k: [], raising=False)

    app.dependency_overrides[get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()
    rag.retriever.get_policy_index.cache_clear()


def claim_files(form: dict, types=("CLAIM_FORM", "HOSPITAL_BILL", "DISCHARGE_SUMMARY")):
    amount = float(form["claim_amount"])
    data = {**form, "claim_amount": amount, "customer_code": form["customer_id"], "invoice_number": "INV-API-2026-00001",
            "billed_amount": amount, "submitted_at": "2026-06-10", "claim_number": "NEW",
            "bill_items": [{"description": "Hospitalisation charges", "amount": amount}]}
    return [("files", (f"{t.lower()}.pdf", build_document(t, data), "application/pdf")) for t in types]


@pytest.fixture
def rahul_form(client):
    policy = client.get("/api/v1/policies", params={"search": "Rahul"}).json()[0]
    return {"customer_name": "Rahul Sharma", "customer_id": policy["customer_code"],
            "policy_number": policy["policy_number"], "insurance_type": policy["insurance_type"],
            "claim_amount": "42000", "hospital_name": "Lotus Care Hospital, Bengaluru",
            "admission_date": "2026-05-02", "discharge_date": "2026-05-05", "diagnosis": "Typhoid fever"}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["demo_mode"] in (True, False)


def test_openapi_documents_all_groups(client):
    spec = client.get("/openapi.json").json()
    paths = set(spec["paths"])
    for p in ("/api/v1/claims", "/api/v1/claims/{claim_number}/process", "/api/v1/claims/{claim_number}/reviews",
              "/api/v1/claims/{claim_number}/executions", "/api/v1/policies/documents", "/api/v1/customers",
              "/api/v1/analytics/kpis", "/api/v1/reviews/queue"):
        assert p in paths


def test_customers_and_policies(client):
    customers = client.get("/api/v1/customers", params={"limit": 500}).json()
    assert len(customers) == 55
    assert client.get(f"/api/v1/customers/{customers[0]['customer_code']}").status_code == 200
    created = client.post("/api/v1/customers", json={"full_name": "Asha Verma", "date_of_birth": "1990-04-01",
                                                      "email": "asha@example.com", "city": "Pune"})
    assert created.status_code == 201 and created.json()["customer_code"] == "CUST-0056"
    policy = client.post("/api/v1/policies", json={"customer_id": "CUST-0056", "plan_code": "IHS",
                                                   "sum_insured": 500000, "start_date": "2026-01-01"})
    assert policy.status_code == 201 and policy.json()["waiting_period_days"] == 30
    number = policy.json()["policy_number"]
    upd = client.patch(f"/api/v1/policies/{number}/status", json={"status": "CANCELLED"})
    assert upd.json()["status"] == "CANCELLED"
    assert client.patch(f"/api/v1/policies/{number}/status", json={"status": "NOPE"}).status_code == 422


def test_structured_errors(client):
    r = client.get("/api/v1/claims/CLM-NOPE")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    r = client.post("/api/v1/customers", json={"full_name": "X"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert {d["field"] for d in r.json()["error"]["details"]} >= {"full_name", "date_of_birth"}


def test_submit_process_review_flow(client, rahul_form):
    r = client.post("/api/v1/claims", data={**rahul_form, "process_now": "true",
                                            "document_types": "CLAIM_FORM,HOSPITAL_BILL,DISCHARGE_SUMMARY"},
                    files=claim_files(rahul_form))
    assert r.status_code == 201, r.text
    body = r.json()
    number = body["claim_number"]
    assert body["documents"] == 3 and body["processing"]["processing_status"] == "COMPLETED"
    assert body["processing"]["path"][0] == "intake" and body["processing"]["path"][-1] == "human_review_queue"

    detail = client.get(f"/api/v1/claims/{number}").json()
    assert detail["decision_support"]["human_decision_required"] is True
    assert detail["extracted_data"]["fields"]["invoice_number"] == "INV-API-2026-00001"
    assert detail["invoice_number"] == "INV-API-2026-00001"  # filled from the documents
    status = client.get(f"/api/v1/claims/{number}/status").json()
    assert status["ai_recommendation"] == detail["decision_support"]["recommendation"]
    history = client.get(f"/api/v1/claims/{number}/executions").json()
    assert [h["agent_name"] for h in history][:2] == ["intake", "document_agent"]
    assert all(h["used_llm"] is False for h in history)

    # Role separation: an investigator cannot approve
    denied = client.post(f"/api/v1/claims/{number}/reviews",
                         json={"reviewer_name": "Ivy", "reviewer_role": "INVESTIGATOR", "decision": "APPROVED"})
    assert denied.status_code == 403
    # Rejection needs a comment
    assert client.post(f"/api/v1/claims/{number}/reviews",
                       json={"reviewer_name": "Ravi", "decision": "REJECTED"}).status_code == 422
    ok = client.post(f"/api/v1/claims/{number}/reviews",
                     json={"reviewer_name": "Ravi Menon", "reviewer_role": "CLAIMS_REVIEWER", "decision": "APPROVED",
                           "comments": "Documents verified."})
    assert ok.status_code == 201 and ok.json()["claim_status"] == "HUMAN_APPROVED"
    assert ok.json()["ai_recommendation_snapshot"] == detail["decision_support"]["recommendation"]
    reviews = client.get(f"/api/v1/claims/{number}/reviews").json()
    assert reviews[0]["decision"] == "APPROVED"
    # A final human decision cannot be overwritten or re-processed
    assert client.post(f"/api/v1/claims/{number}/process").status_code == 409
    assert client.post(f"/api/v1/claims/{number}/reviews",
                       json={"reviewer_name": "X Y", "decision": "APPROVED"}).status_code == 409


def test_missing_documents_then_upload_more(client, rahul_form):
    r = client.post("/api/v1/claims", data={**rahul_form, "process_now": "true", "document_types": "CLAIM_FORM"},
                    files=claim_files(rahul_form, ("CLAIM_FORM",)))
    number = r.json()["claim_number"]
    assert r.json()["processing"]["recommendation"] == "ADDITIONAL_INFORMATION_REQUIRED"
    add = client.post(f"/api/v1/claims/{number}/documents", data={"document_types": "HOSPITAL_BILL,DISCHARGE_SUMMARY"},
                      files=claim_files(rahul_form, ("HOSPITAL_BILL", "DISCHARGE_SUMMARY")))
    assert add.status_code == 201 and len(add.json()) == 2
    again = client.post(f"/api/v1/claims/{number}/process").json()
    assert again["recommendation"] != "ADDITIONAL_INFORMATION_REQUIRED"


def test_upload_validation(client, rahul_form):
    bad = [("files", ("bill.pdf", b"not a pdf", "application/pdf"))]
    r = client.post("/api/v1/claims", data=rahul_form, files=bad)
    assert r.status_code == 422 and "not a valid PDF" in str(r.json()["error"]["details"])
    r = client.post("/api/v1/claims", data={**rahul_form, "customer_id": "CUST-9999"}, files=claim_files(rahul_form))
    assert r.status_code == 404
    r = client.post("/api/v1/claims", data={**rahul_form, "discharge_date": "2026-04-01"}, files=claim_files(rahul_form))
    assert r.status_code == 422
    r = client.post("/api/v1/claims", data={**rahul_form, "document_types": "PASSPORT"}, files=claim_files(rahul_form))
    assert r.status_code == 422


def test_unknown_policy_is_accepted_and_flagged(client, rahul_form):
    r = client.post("/api/v1/claims", data={**rahul_form, "policy_number": "IHS-2099-123456", "process_now": "true"},
                    files=claim_files({**rahul_form, "policy_number": "IHS-2099-123456"}))
    assert r.status_code == 201
    assert any("not on record" in w for w in r.json()["warnings"])
    assert r.json()["processing"]["recommendation"] == "POLICY_NOT_FOUND"
    assert "rag_agent" in r.json()["processing"]["skipped_agents"]


def test_review_requires_processing(client):
    claims = client.get("/api/v1/claims").json()
    r = client.post(f"/api/v1/claims/{claims[0]['claim_number']}/reviews",
                    json={"reviewer_name": "Ravi Menon", "decision": "APPROVED"})
    assert r.status_code == 409


def test_policy_search_and_document_upload(client):
    hits = client.get("/api/v1/policies/search", params={"q": "cosmetic surgery", "plan_code": "IHS"}).json()
    assert hits[0]["clause"]["clause_code"] == "EXC-01"
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Section 2 - What is covered\nCOV-01  Dental cover\nDental care is covered.")
    r = client.post("/api/v1/policies/documents", files={"file": ("DEN_policy_wording.pdf", doc.tobytes(), "application/pdf")})
    assert r.status_code == 201, r.text
    assert r.json()["chunks"] >= 1 and r.json()["filename"] == "DEN_policy_wording.pdf"
    assert any(c["source_document"] == "DEN_policy_wording.pdf" for c in client.get("/api/v1/policies/clauses").json())


def test_analytics_and_workflow(client):
    k = client.get("/api/v1/analytics/kpis").json()
    assert k["total_claims"] == 33
    summary = client.get("/api/v1/analytics/summary").json()
    assert sum(summary["status_distribution"].values()) == 33
    agents = client.get("/api/v1/agents").json()
    assert [a["key"] for a in agents] == ["document_agent", "policy_agent", "rag_agent", "validation_agent",
                                          "fraud_agent", "decision_agent"]
    assert "human_review_queue" in client.get("/api/v1/workflow/graph").text
    assert client.get("/api/v1/reviews/queue").json() == []  # nothing processed in this database yet
