"""Phase 5 — Policy RAG Agent: grounded retrieval, uncertainty and LLM safety."""

import pytest

from agents.rag_agent import PolicyRAGAgent, diagnosis_terms_in
from agents.schemas import PolicyRAGInput


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
        self.schema = schema
        return self.runner


@pytest.fixture
def agent(policy_index):
    return PolicyRAGAgent(index=policy_index, use_llm=False)


def run(agent, plan, diagnosis, **flags):
    return agent.run(PolicyRAGInput(plan_code=plan, diagnosis=diagnosis, policy_flags=flags))


def test_covered_claim_finds_cited_evidence(agent, policy_index):
    r = run(agent, "IHS", "Acute appendicitis - laparoscopic appendectomy")
    assert r.status == "EVIDENCE_FOUND" and not r.requires_human_review
    assert r.possible_exclusions == [] and not r.flags["exclusion_wording_found"]
    topics = {c.topic for c in r.clauses}
    assert {"coverage", "waiting_period", "limits", "policy_in_force", "documents"} <= topics
    assert r.used_llm is False and r.explanation_source == "template"
    assert r.embedding_model.startswith("local-hashing") and r.retrieval_mode == "FAISS"


def test_clause_text_is_copied_verbatim_with_citation(agent, policy_index):
    indexed = {(c.source_document, c.clause_code): c for c in policy_index.chunks}
    r = run(agent, "FFS", "Cosmetic rhinoplasty (aesthetic nasal reshaping)")
    for c in r.clauses:
        source = indexed[(c.source_document, c.clause_code)]
        assert c.text == source.text and c.page_number == source.page_number  # never generated
        assert c.source_document in c.citation and f"p.{c.page_number}" in c.citation


def test_cosmetic_exclusion_wording_is_found(agent):
    r = run(agent, "FFS", "Cosmetic rhinoplasty (aesthetic nasal reshaping)", possible_exclusion=True)
    assert r.possible_exclusions == ["EXC-01"]
    exc = next(c for c in r.clauses if c.topic == "exclusions")
    assert set(exc.matched_terms) >= {"cosmetic", "rhinoplasty"}
    assert r.requires_human_review and r.status == "EVIDENCE_FOUND"
    assert any("Possible exclusion" in f for f in r.findings)


def test_dental_exclusion_without_false_positives(agent):
    r = run(agent, "IHS", "Dental implant - non-accidental tooth loss")
    assert r.possible_exclusions == ["EXC-02"]  # "loss" must not drag in the weight-loss clause


def test_personal_accident_plan_flags_illness(agent):
    r = run(agent, "PAG", "Acute appendicitis - laparoscopic appendectomy")
    assert r.possible_exclusions == ["EXC-PA"]


def test_personal_accident_plan_accepts_injury(agent):
    r = run(agent, "PAG", "Fracture of right radius - open reduction internal fixation")
    assert r.status == "EVIDENCE_FOUND" and r.possible_exclusions == []
    assert "COV-04" in {c.clause_code for c in r.clauses}  # accident-only cover clause


def test_generic_words_do_not_ground_an_exclusion():
    dental = "Dental treatment or surgery is excluded unless necessitated by an accident."
    assert diagnosis_terms_in("Road traffic accident - ligament tear", dental) == []
    assert diagnosis_terms_in(None, dental) == []
    assert diagnosis_terms_in("Dental implant", dental) == ["dental"]


def test_unknown_policy_returns_uncertainty_not_guesses(agent):
    r = run(agent, None, "Typhoid fever", policy_not_found=True)
    assert r.status == "POLICY_CONTEXT_UNAVAILABLE" and r.clauses == []
    assert r.requires_human_review and r.flags["insufficient_evidence"] and r.uncertainty_reasons


def test_plan_without_wording_is_insufficient_evidence(agent):
    r = run(agent, "XYZ", "Typhoid fever")
    assert r.status == "INSUFFICIENT_EVIDENCE" and r.clauses == [] and r.requires_human_review


def test_missing_diagnosis_is_insufficient_evidence(agent):
    r = run(agent, "IHS", None)
    assert r.status == "INSUFFICIENT_EVIDENCE"
    assert any("diagnosis is missing" in u for u in r.uncertainty_reasons)
    assert "exclusions" not in {q.topic for q in r.queries}


def test_unconfirmed_rule_flag_requires_review(agent):
    r = run(agent, "IHS", "Typhoid fever", possible_exclusion=True)
    assert r.status == "INSUFFICIENT_EVIDENCE" and r.requires_human_review
    assert any("rule-based check flagged" in u for u in r.uncertainty_reasons)


def test_flag_related_findings(agent):
    r = run(agent, "SCP", "Typhoid fever", waiting_period_incomplete=True, exceeds_sum_insured=True)
    text = " ".join(r.findings)
    assert "SCP WP-01" in text and "SCP COV-02" in text
    assert "not a claim decision" in r.explanation


# --------------------------------------------------------------- Gemini
def test_gemini_explanation_used_when_citations_are_valid(policy_index):
    from agents.rag_agent import _Explanation

    fake = FakeLLM(_Explanation(explanation="COV-01 describes in-patient cover; EXC-01 should be read by the reviewer.",
                                cited_clause_codes=["COV-01", "EXC-01"]))
    r = PolicyRAGAgent(index=policy_index, llm_model=fake).run(
        PolicyRAGInput(plan_code="IHS", diagnosis="Cosmetic rhinoplasty"))
    assert r.explanation_source == "gemini" and r.used_llm
    assert r.explanation.startswith("COV-01")
    assert "Use ONLY the clauses below" in fake.runner.prompt and "EXC-01" in fake.runner.prompt


def test_gemini_answer_citing_unretrieved_clause_is_discarded(policy_index):
    from agents.rag_agent import _Explanation

    fake = FakeLLM(_Explanation(explanation="Clause EXC-99 excludes this claim.", cited_clause_codes=["EXC-99"]))
    r = PolicyRAGAgent(index=policy_index, llm_model=fake).run(
        PolicyRAGInput(plan_code="IHS", diagnosis="Typhoid fever"))
    assert r.explanation_source == "template" and "EXC-99" not in r.explanation
    assert any("discarded" in w and "EXC-99" in w for w in r.warnings)


def test_gemini_failure_falls_back_to_template(policy_index, monkeypatch):
    monkeypatch.setattr("agents.llm.time.sleep", lambda s: None)
    r = PolicyRAGAgent(index=policy_index, llm_model=FakeLLM(RuntimeError("quota exceeded"))).run(
        PolicyRAGInput(plan_code="IHS", diagnosis="Typhoid fever"))
    assert r.explanation_source == "template" and not r.used_llm
    assert any("quota exceeded" in w for w in r.warnings)


def test_demo_mode_never_calls_llm(policy_index, monkeypatch):
    monkeypatch.setattr("agents.rag_agent.llm_available", lambda: False)
    agent = PolicyRAGAgent(index=policy_index)
    assert agent.use_llm is False
    assert agent.run(PolicyRAGInput(plan_code="IHS", diagnosis="Typhoid fever")).used_llm is False
