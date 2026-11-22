"""Unit tests for Agent 1 — Document Extraction Agent (PDF text, OCR, merging, Gemini gap-fill)."""

from datetime import date

import pymupdf
import pytest

from agents import document_agent as da
from agents.document_agent import DocumentExtractionAgent, DocumentInput, classify_document, parse_date, parse_field
from agents.schemas import ClaimFields
from documents import ocr
from documents.pdf_generator import build_document, generate_claim_documents


@pytest.fixture
def rahul(synthetic_data):
    claim = next(c for c in synthetic_data["claims"] if c["customer_name"] == "Rahul Sharma")
    return {**claim, "documents": ["CLAIM_FORM", "HOSPITAL_BILL", "DISCHARGE_SUMMARY"]}


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ------------------------------- parsers -------------------------------- #
@pytest.mark.parametrize("raw", ["27/10/2026", "27-Oct-2026", "2026-10-27", "27 October 2026", "27.10.2026",
                                 "27/10/2026 (10:30 AM)"])
def test_parse_date_formats(raw):
    assert parse_date(raw) == date(2026, 10, 27)


def test_parse_date_invalid():
    assert parse_date("not a date") is None


@pytest.mark.parametrize("raw,expected", [("INR 85,000.00", 85000.0), ("Rs. 1,23,456", 123456.0),
                                          ("₹ 999.5", 999.5), ("85000", 85000.0)])
def test_parse_amount(raw, expected):
    assert parse_field("claim_amount", raw) == expected


def test_policy_number_tolerates_ocr_spacing():
    assert parse_field("policy_number", "ihs - 2026 - 100001  extra") == "IHS-2026-100001"


@pytest.mark.parametrize("text,expected", [
    ("HEALTH INSURANCE CLAIM FORM", "CLAIM_FORM"), ("FINAL HOSPITAL BILL / TAX INVOICE", "HOSPITAL_BILL"),
    ("DISCHARGE SUMMARY", "DISCHARGE_SUMMARY"), ("DIAGNOSTIC REPORT", "DIAGNOSTIC_REPORT"),
    ("random letter", "OTHER"),
])
def test_classify_document(text, expected):
    assert classify_document(text) == expected


# --------------------------- end-to-end extraction --------------------------- #
def test_extracts_all_fields_from_text_pdfs(tmp_path, rahul):
    paths = generate_claim_documents(rahul, tmp_path)
    r = DocumentExtractionAgent(use_llm=False).run(paths)
    f = r.fields
    assert f.customer_name == "Rahul Sharma"
    assert f.policy_number == rahul["policy_number"]
    assert f.claim_amount == 85000 and f.billed_amount == 85000
    assert f.invoice_number == rahul["invoice_number"]
    assert f.admission_date == date.fromisoformat(rahul["admission_date"])
    assert f.discharge_date == date.fromisoformat(rahul["discharge_date"])
    assert f.diagnosis == rahul["diagnosis"]
    assert f.hospital_name.startswith("Sunrise Multispeciality Hospital")
    assert r.missing_fields == [] and r.conflicts == []
    assert r.extraction_mode == "RULE_BASED" and r.used_llm is False
    assert sorted(r.document_types_found) == ["CLAIM_FORM", "DISCHARGE_SUMMARY", "HOSPITAL_BILL"]
    assert sum(i.amount for i in r.bill_items) == 85000
    assert r.field_sources["policy_number"] == "rules"


def test_missing_documents_reported_as_missing_fields(tmp_path, rahul):
    paths = generate_claim_documents({**rahul, "documents": ["CLAIM_FORM"]}, tmp_path)
    r = DocumentExtractionAgent(use_llm=False).run(paths)
    assert r.missing_fields == ["billed_amount"]
    assert r.document_types_found == ["CLAIM_FORM"]


def test_no_documents_everything_missing():
    r = DocumentExtractionAgent(use_llm=False).run([])
    assert set(r.missing_fields) == set(da.REQUIRED_CLAIM_FIELDS)


def test_conflicting_documents_recorded(tmp_path, rahul):
    form = _write(tmp_path, "form.pdf", build_document("CLAIM_FORM", rahul))
    bill = _write(tmp_path, "bill.pdf", build_document("HOSPITAL_BILL", {**rahul, "policy_number": "IHS-2026-999999"}))
    r = DocumentExtractionAgent(use_llm=False).run([form, bill])
    conflict = next(c for c in r.conflicts if c.field == "policy_number")
    assert conflict.values == {"form.pdf": rahul["policy_number"], "bill.pdf": "IHS-2026-999999"}
    assert r.fields.policy_number == rahul["policy_number"]  # claim form has priority


def test_corrupt_file_does_not_crash(tmp_path, rahul):
    bad = _write(tmp_path, "broken.pdf", b"%PDF-1.4 broken")
    good = _write(tmp_path, "form.pdf", build_document("CLAIM_FORM", rahul))
    r = DocumentExtractionAgent(use_llm=False).run([bad, good])
    assert r.fields.customer_name == "Rahul Sharma"
    assert any("Could not open" in w for w in r.warnings)


def test_declared_type_mismatch_warns(tmp_path, rahul):
    path = _write(tmp_path, "x.pdf", build_document("HOSPITAL_BILL", rahul))
    r = DocumentExtractionAgent(use_llm=False).run([DocumentInput(path, declared_type="DISCHARGE_SUMMARY")])
    assert r.documents[0].document_type == "HOSPITAL_BILL"
    assert any("looks like HOSPITAL_BILL" in w for w in r.warnings)


# ------------------------------------ OCR ------------------------------------ #
@pytest.mark.skipif(not ocr.ocr_available(), reason="OCR engine not installed")
def test_scanned_bill_is_read_with_ocr(tmp_path, rahul):
    path = _write(tmp_path, "scan.pdf", build_document("HOSPITAL_BILL", rahul, scanned=True))
    with pymupdf.open(path) as doc:
        assert doc[0].get_text().strip() == ""  # really has no text layer
    r = DocumentExtractionAgent(use_llm=False).run([path])
    doc = r.documents[0]
    assert doc.extraction_method == "OCR" and doc.ocr_confidence > 0.8 and r.ocr_used
    assert r.fields.invoice_number == rahul["invoice_number"]
    assert r.fields.billed_amount == rahul["billed_amount"]
    assert r.fields.policy_number == rahul["policy_number"]


def test_scanned_pdf_without_ocr_engine_degrades_gracefully(tmp_path, rahul, monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    path = _write(tmp_path, "scan.pdf", build_document("CLAIM_FORM", rahul, scanned=True))
    r = DocumentExtractionAgent(use_llm=False).run([path])
    assert r.documents[0].extraction_method == "NONE"
    assert any("OCR is not installed" in w for w in r.warnings)
    assert "customer_name" in r.missing_fields


# ---------------------------- Gemini gap-filling ---------------------------- #
class _FakeStructured:
    def __init__(self, result):
        self.result = result

    def invoke(self, prompt):
        if isinstance(self.result, Exception):
            raise self.result
        self.prompt = prompt
        return self.result


class FakeLLM:
    def __init__(self, result):
        self.runner = _FakeStructured(result)

    def with_structured_output(self, schema):
        return self.runner


def test_gemini_only_fills_gaps_and_is_labelled(tmp_path, rahul):
    paths = generate_claim_documents({**rahul, "documents": ["CLAIM_FORM"]}, tmp_path)
    fake = FakeLLM(ClaimFields(billed_amount=85000, customer_name="SOMEONE ELSE"))
    r = DocumentExtractionAgent(llm_model=fake).run(paths)
    assert r.used_llm and r.extraction_mode == "RULE_BASED+GEMINI"
    assert r.fields.billed_amount == 85000 and r.field_sources["billed_amount"] == "gemini"
    assert r.fields.customer_name == "Rahul Sharma" and r.field_sources["customer_name"] == "rules"
    assert r.missing_fields == []
    assert "Use ONLY values explicitly written" in fake.runner.prompt


def test_gemini_failure_falls_back_to_rules(tmp_path, rahul, monkeypatch):
    monkeypatch.setattr("agents.llm.time.sleep", lambda s: None)
    paths = generate_claim_documents({**rahul, "documents": ["CLAIM_FORM"]}, tmp_path)
    r = DocumentExtractionAgent(llm_model=FakeLLM(RuntimeError("quota exceeded"))).run(paths)
    assert r.used_llm is False and r.extraction_mode == "RULE_BASED"
    assert r.missing_fields == ["billed_amount"]
    assert any("Gemini extraction unavailable" in w for w in r.warnings)


def test_demo_mode_never_calls_llm(tmp_path, rahul, monkeypatch):
    monkeypatch.setattr(da, "llm_available", lambda: False)

    def boom(*a, **k):
        raise AssertionError("LLM must not be called in demo mode")

    monkeypatch.setattr(da, "invoke_structured", boom)
    paths = generate_claim_documents({**rahul, "documents": ["CLAIM_FORM"]}, tmp_path)
    r = DocumentExtractionAgent().run(paths)
    assert r.used_llm is False
