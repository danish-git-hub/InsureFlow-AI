"""
Convenience launcher (works the same in Windows PowerShell and macOS/Linux).

    python run.py dashboard   # Streamlit UI on http://localhost:8501
    python run.py api         # FastAPI on http://127.0.0.1:8000 (docs at /docs)
    python run.py all         # both together
    python run.py test        # full pytest suite
    python run.py setup       # (re)create the database, synthetic data, PDFs, FAISS index,
                              # then run the AI workflow on every claim
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

COMMANDS = {
    "dashboard": [PY, "-m", "streamlit", "run", str(ROOT / "app" / "main.py")],
    "api": [PY, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
    "test": [PY, "-m", "pytest", "-q"],
}
SETUP = [
    [PY, "-m", "database.init_db", "--reset"],   # tables, synthetic data, claim PDFs, policy index
    [PY, "-m", "workflows.graph", "--all"],      # six-agent workflow on every claim
]


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
    if target == "setup":
        for cmd in SETUP:
            if code := subprocess.call(cmd, cwd=ROOT):
                return code
        return 0
    if target == "all":
        api = subprocess.Popen(COMMANDS["api"], cwd=ROOT)
        try:
            return subprocess.call(COMMANDS["dashboard"], cwd=ROOT)
        finally:
            api.terminate()
    if target not in COMMANDS:
        print(__doc__)
        return 1
    return subprocess.call(COMMANDS[target], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
