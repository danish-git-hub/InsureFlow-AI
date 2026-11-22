"""
OCR fallback for scanned (image-only) PDF pages.

Engine: RapidOCR (PaddleOCR models running on ONNX Runtime). It installs with
pip — no separate Tesseract install — so it works identically on Windows,
macOS and Linux. If the package is missing, OCR is reported as unavailable
and the extraction agent degrades gracefully instead of crashing.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pymupdf

from app.config import get_logger

log = get_logger("ocr")
OCR_DPI = 200


@lru_cache(maxsize=1)
def _engine():
    try:
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        from rapidocr import RapidOCR

        return RapidOCR()
    except Exception as exc:  # package missing or model load failure
        log.warning("OCR engine unavailable: %s", exc.__class__.__name__)
        return None


def ocr_available() -> bool:
    return _engine() is not None


def _group_into_lines(boxes, texts, scores) -> tuple[list[str], float]:
    """OCR returns word/phrase boxes; group boxes sharing a baseline into lines."""
    items = []
    for box, text, score in zip(boxes, texts, scores):
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), text, float(score)))
    items.sort(key=lambda i: (i[0], i[2]))

    lines: list[list[tuple]] = []
    for item in items:
        if lines and abs(item[0] - lines[-1][0][0]) < max(item[1], lines[-1][0][1]) * 0.6:
            lines[-1].append(item)
        else:
            lines.append([item])
    text_lines = ["  ".join(i[3] for i in sorted(line, key=lambda i: i[2])) for line in lines]
    confidence = sum(i[4] for i in items) / len(items) if items else 0.0
    return text_lines, confidence


def ocr_page(page: pymupdf.Page, dpi: int = OCR_DPI) -> tuple[list[str], float]:
    """Render a page to an image and OCR it. Returns (lines, mean confidence)."""
    engine = _engine()
    if engine is None:
        raise RuntimeError("OCR engine is not installed (pip install rapidocr onnxruntime)")
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    result = engine(png)
    if result is None or result.txts is None or len(result.txts) == 0:
        return [], 0.0
    return _group_into_lines(result.boxes, result.txts, result.scores)
