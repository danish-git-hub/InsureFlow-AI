"""System Status page — live environment and dependency checks.

Everything shown here is measured at render time (no hardcoded values):
Python version, installed library versions, AI mode, folder layout and
whether the FastAPI backend is reachable.
"""

import importlib.metadata as md
import platform
import sys

import httpx
import pandas as pd
import streamlit as st

from app.components.ui import badge, kpi_card, page_header
from app.config import settings

page_header("System Status", "Live environment, dependency and service health checks.")

# --- Backend health -------------------------------------------------------
try:
    resp = httpx.get(f"{settings.api_base_url}/health", timeout=2.0)
    api_ok = resp.status_code == 200
    api_detail = resp.json().get("status", "unknown") if api_ok else f"HTTP {resp.status_code}"
except httpx.HTTPError:
    api_ok, api_detail = False, "not running"

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Python", platform.python_version(), sys.implementation.name)
with c2:
    kpi_card("AI Mode", "Demo" if settings.demo_mode else "Gemini", settings.mode_label)
with c3:
    kpi_card("FastAPI Backend", "Online" if api_ok else "Offline", settings.api_base_url)
with c4:
    db_name = settings.database_url.rsplit("/", 1)[-1]
    kpi_card("Database", db_name, "SQLite via SQLAlchemy")

from documents.ocr import ocr_available  # noqa: E402

st.caption(
    "OCR fallback for scanned PDFs: "
    + ("**RapidOCR available**" if ocr_available() else "**not installed** (`pip install rapidocr onnxruntime`)")
)
if not api_ok:
    st.caption("Start the backend with `python run.py api` to bring it online.")

st.markdown("### Installed core libraries")
packages = [
    "streamlit", "plotly", "pandas", "fastapi", "uvicorn", "pydantic", "langchain", "langchain-core",
    "langgraph", "langchain-google-genai", "langchain-community", "faiss-cpu", "pymupdf",
    "SQLAlchemy", "python-dotenv", "pytest",
]
rows = []
for name in packages:
    try:
        rows.append({"Package": name, "Version": md.version(name), "Status": "Installed"})
    except md.PackageNotFoundError:
        rows.append({"Package": name, "Version": "—", "Status": "Missing"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("### Database tables")
try:
    from database.connection import session_scope
    from database.repository import AnalyticsRepository

    with session_scope() as session:
        counts = AnalyticsRepository(session).table_counts()
    st.dataframe(
        pd.DataFrame([{"Table": t, "Rows": n} for t, n in counts.items()]),
        hide_index=True, width="stretch",
    )
    if counts.get("claims", 0) == 0:
        st.warning("Database is empty. Run `python -m database.init_db` to load synthetic data.")
except Exception as exc:  # database file missing or schema not created yet
    st.error(f"Database not initialised: {exc.__class__.__name__}. Run `python -m database.init_db`.")

st.markdown("### Data folders")
folder_rows = []
for label, path in [
    ("Policy PDFs", settings.policies_dir),
    ("Claim documents", settings.claims_dir),
    ("Synthetic data", settings.synthetic_dir),
    ("Uploads", settings.uploads_dir),
    ("Vector store", settings.vectorstore_dir),
]:
    files = [p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")] if path.exists() else []
    folder_rows.append({"Folder": label, "Path": str(path.relative_to(settings.data_dir.parent)), "Files": len(files)})
st.dataframe(pd.DataFrame(folder_rows), hide_index=True, width="stretch")

st.markdown(
    "AI governance: " + badge("Human-in-the-loop decisions", "info") + " "
    + badge("No automated approvals/rejections", "success"),
    unsafe_allow_html=True,
)
