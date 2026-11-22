"""Smoke test: every core dependency imports in this environment."""

import importlib

import pytest

MODULES = ["streamlit", "plotly", "pandas", "fastapi", "pydantic", "langchain", "langgraph.graph",
           "langchain_google_genai", "faiss", "pymupdf", "sqlalchemy", "dotenv"]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_langgraph_stategraph_compiles():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        n: int

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    assert g.compile().invoke({"n": 1}) == {"n": 2}
