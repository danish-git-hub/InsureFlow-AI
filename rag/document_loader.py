"""
Load policy wording PDFs and split them into clause-level chunks.

Why clause-level chunking?
    A reviewer needs a citation such as "EXC-01 Cosmetic or aesthetic treatment,
    IHS_policy_wording.pdf, page 2". Splitting at clause headings keeps each
    chunk a complete, citable unit of policy wording. Clauses longer than
    `MAX_CHARS` are split further with LangChain's RecursiveCharacterTextSplitter
    (each part keeps the clause metadata).

Unstructured PDFs (no clause headings found) fall back to plain recursive
splitting with category GENERAL, so any uploaded policy can still be searched.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.models import ClauseCategory

MAX_CHARS = 900
CLAUSE_HEADING = re.compile(r"^(?P<code>[A-Z]{2,3}-(?:\d{2}|[A-Z]{2}))\s{1,4}(?P<title>[A-Z][^\n]{2,80})$")
SECTION_HEADING = re.compile(r"^Section\s+\d+\s*-\s*(?P<name>.+)$", re.IGNORECASE)
PLAN_LINE = re.compile(r"Plan code:\s*(?P<code>[A-Z]{2,5})")
# Page furniture repeated on every page: not policy wording.
NOISE = (
    re.compile(r"^InsureFlow Insurance"),
    re.compile(r"^Policy wording - "),
    re.compile(r"^Fictional policy wording generated"),
    re.compile(r".+\(\w+\) - continued$"),
    re.compile(r".+ - Policy Wording$"),
    re.compile(r"^Plan code:"),
)

SECTION_CATEGORY = {
    "definition": ClauseCategory.DEFINITION,
    "covered": ClauseCategory.COVERAGE,
    "coverage": ClauseCategory.COVERAGE,
    "benefit": ClauseCategory.COVERAGE,
    "waiting": ClauseCategory.WAITING_PERIOD,
    "exclusion": ClauseCategory.EXCLUSION,
    "condition": ClauseCategory.CONDITION,
}
CODE_CATEGORY = {
    "DEF": ClauseCategory.DEFINITION, "COV": ClauseCategory.COVERAGE, "WP": ClauseCategory.WAITING_PERIOD,
    "EXC": ClauseCategory.EXCLUSION, "CON": ClauseCategory.CONDITION,
}


@dataclass
class ClauseChunk:
    """One searchable piece of policy wording, with everything needed to cite it."""

    chunk_id: str
    source_document: str
    plan_code: str | None
    clause_code: str | None
    title: str | None
    category: str
    page_number: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    def db_fields(self) -> dict:
        """Columns for the PolicyClause table (source_document is passed separately)."""
        return {"plan_code": self.plan_code, "clause_code": self.clause_code, "title": self.title,
                "category": self.category, "text": self.text, "page_number": self.page_number}


def _category(section: str | None, code: str | None) -> str:
    if code and code.split("-")[0] in CODE_CATEGORY:
        return CODE_CATEGORY[code.split("-")[0]].value
    low = (section or "").lower()
    for key, cat in SECTION_CATEGORY.items():
        if key in low:
            return cat.value
    return ClauseCategory.GENERAL.value


def _plan_from_filename(path: Path) -> str | None:
    m = re.match(r"^([A-Z]{2,5})_policy_wording", path.name)
    return m.group(1) if m else None


def read_lines(path: Path) -> tuple[list[tuple[int, str]], str | None]:
    """Return (page_number, line) pairs without page furniture, plus the plan code if printed."""
    lines: list[tuple[int, str]] = []
    plan_code = None
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for raw in page.get_text().splitlines():
                line = raw.strip()
                if not line:
                    continue
                if plan_code is None and (m := PLAN_LINE.search(line)):
                    plan_code = m.group("code")
                if any(p.search(line) for p in NOISE):
                    continue
                lines.append((page_no, line))
    return lines, plan_code


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=120, separators=["\n\n", ". ", " ", ""])


def load_policy_document(path: Path) -> list[ClauseChunk]:
    """Split one policy PDF into clause chunks."""
    path = Path(path)
    lines, printed_plan = read_lines(path)
    plan_code = printed_plan or _plan_from_filename(path)
    source = path.name

    # Group lines into clauses: a clause runs from its heading to the next heading.
    clauses: list[dict] = []
    section: str | None = None
    for page_no, line in lines:
        if m := SECTION_HEADING.match(line):
            section = m.group("name")
            continue
        if m := CLAUSE_HEADING.match(line):
            clauses.append({"code": m.group("code"), "title": m.group("title").strip(), "section": section,
                            "page": page_no, "body": []})
            continue
        if clauses:
            clauses[-1]["body"].append(line)

    chunks: list[ClauseChunk] = []
    if not clauses:  # unstructured document: plain recursive splitting
        text = "\n".join(line for _, line in lines)
        first_page = lines[0][0] if lines else 1
        for i, part in enumerate(_splitter().split_text(text), start=1):
            chunks.append(ClauseChunk(f"{source}#chunk{i}", source, plan_code, None, None,
                                      ClauseCategory.GENERAL.value, first_page, part))
        return chunks

    splitter = _splitter()
    for c in clauses:
        body = " ".join(c["body"]).strip()
        parts = splitter.split_text(body) if len(body) > MAX_CHARS else [body]
        for i, part in enumerate(parts, start=1):
            suffix = f"-{i}" if len(parts) > 1 else ""
            chunks.append(ClauseChunk(
                chunk_id=f"{source}#{c['code']}{suffix}",
                source_document=source,
                plan_code=plan_code,
                clause_code=c["code"],
                title=c["title"],
                category=_category(c["section"], c["code"]),
                page_number=c["page"],
                text=part,
            ))
    return chunks


def load_policy_directory(directory: Path) -> list[ClauseChunk]:
    chunks: list[ClauseChunk] = []
    for path in sorted(Path(directory).glob("*.pdf")):
        chunks.extend(load_policy_document(path))
    return chunks
