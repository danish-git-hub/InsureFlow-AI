"""Unit tests for upload validation and safe storage."""

import pymupdf
import pytest

from documents.storage import (
    DocumentValidationError,
    safe_filename,
    store_claim_document,
    validate_pdf_bytes,
)


def _pdf_bytes(text="hello") -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd.pdf", "passwd.pdf"),
    ("C:\\Users\\x\\My Bill (1).PDF", "My_Bill_1.pdf"),
    ("   .pdf", "document.pdf"),
    ("rahul's bill#2.pdf", "rahul_s_bill_2.pdf"),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_valid_pdf_returns_page_count():
    assert validate_pdf_bytes(_pdf_bytes(), "a.pdf") == 1


@pytest.mark.parametrize("data,name,msg", [
    (b"%PDF-1.7 ...", "a.exe", "only PDF"),
    (b"", "a.pdf", "empty"),
    (b"MZ\x90\x00 not a pdf", "a.pdf", "signature"),
    (b"%PDF-1.4 garbage that is not really a pdf", "a.pdf", "could not be read"),
])
def test_invalid_uploads_rejected(data, name, msg):
    with pytest.raises(DocumentValidationError, match=msg):
        validate_pdf_bytes(data, name)


def test_oversized_upload_rejected():
    with pytest.raises(DocumentValidationError, match="limit"):
        validate_pdf_bytes(_pdf_bytes(), "a.pdf", max_bytes=10)


def test_encrypted_pdf_rejected():
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(DocumentValidationError, match="password"):
        validate_pdf_bytes(data, "locked.pdf")


def test_store_keeps_file_inside_claim_folder(tmp_path):
    stored = store_claim_document("CLM-1/../../x", "../../evil.pdf", _pdf_bytes(), base_dir=tmp_path)
    assert stored.file_path.resolve().is_relative_to(tmp_path.resolve())
    assert stored.stored_filename.endswith("_evil.pdf")
    assert stored.file_path.read_bytes().startswith(b"%PDF-")
    assert len(stored.sha256) == 64 and stored.page_count == 1
