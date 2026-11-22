"""
InsureFlow AI — Streamlit dashboard entry point.

Run from the project root:
    streamlit run app/main.py        (or: python run.py dashboard)

Uses Streamlit's `st.navigation` API so the sidebar menu, page titles and icons
are defined in one place below.
"""
import secrets
import sys
from pathlib import Path

# Make the project root importable (so `from app.config import settings`,
# `from agents...`, `from database...` work when Streamlit runs this file).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.ui import load_theme, sidebar_brand  # noqa: E402
from app.config import settings  # noqa: E402

st.set_page_config(
    page_title=settings.app_name,
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Dashboard password protection
if settings.app_access_password:
    if not st.session_state.get("authenticated", False):
        st.title("InsureFlow AI")
        st.subheader("Secure Demo Login")

        with st.form("login_form"):
            password = st.text_input("Enter demo password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if secrets.compare_digest(password, settings.app_access_password):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

        st.stop()
load_theme()
sidebar_brand()

PAGES = Path(__file__).resolve().parent / "pages"

navigation = st.navigation(
    {
        "Overview": [
            st.Page(PAGES / "dashboard.py", title="Executive Dashboard", icon=":material/dashboard:", default=True),
            st.Page(PAGES / "system_status.py", title="System Status", icon=":material/monitor_heart:"),
        ],
        "Claims": [
            st.Page(PAGES / "new_claim.py", title="New Claim", icon=":material/note_add:"),
            st.Page(PAGES / "processing_center.py", title="AI Processing Center", icon=":material/hub:"),
            st.Page(PAGES / "claim_management.py", title="Claim Management", icon=":material/table_view:"),
            st.Page(PAGES / "claim_details.py", title="Claim Details", icon=":material/description:"),
        ],
        "Operations": [
            st.Page(PAGES / "policy_management.py", title="Policy Management", icon=":material/policy:"),
            st.Page(PAGES / "human_review.py", title="Human Review Center", icon=":material/fact_check:"),
            st.Page(PAGES / "reports.py", title="Reports & Analytics", icon=":material/analytics:"),
        ],
    }
)
navigation.run()
