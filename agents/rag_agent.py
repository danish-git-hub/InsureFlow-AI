"""
Agent 3 — Policy RAG Agent.

Purpose:
    Find the *actual policy wording* relevant to a claim (coverage, exclusions,
    waiting period, limits, conditions) and present it to the reviewer with a
    citation: plan, clause code, title, PDF file and page.

Retrieval-Augmented Generation (RAG) in three steps:
    1. Retrieve — topic queries built from the claim are searched in the FAISS
       index, filtered to the claim's plan and to the relevant clause category.
    2. Ground  — a clause only counts as evidence if its similarity score clears
       a minimum, and an exclusion only counts as "possibly applicable" if words
       from the diagnosis literally appear in the clause wording.
    3. Generate — an explanation is written from the retrieved clauses only:
       by Gemini when a key is set, otherwise by a deterministic template. A
       Gemini answer that cites any clause not retrieved is discarded.

Safety rules:
    * Clause text is copied from the indexed PDF — never generated or paraphrased
      into new obligations.
    * Not enough evidence -> status INSUFFICIENT_EVIDENCE plus reasons, and the
      claim is routed to human review. The agent never guesses.
    * The agent explains wording; it does not approve or reject anything.

    python -m agents.rag_agent CLM-...      # run Agents 2 + 3 on a stored claim
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from agents.llm import invoke_structured, llm_available
from agents.schemas import (
    PolicyRAGInput,
    PolicyRAGResult,
    PolicyVerificationInput,
    PolicyVerificationResult,
    RAGQuery,
    RetrievedClause,
)
from app.config import get_logger, settings
from rag.embeddings import stem, tokenize
from rag.retriever import PolicyIndex, SearchHit, get_policy_index

log = get_logger("rag_agent")

# Minimum cosine similarity for a clause to count as evidence. Score scales
# differ between embedding models, so the threshold depends on the backend.
MIN_SCORE = {"local": 0.10, "gemini": 0.45}
# An exclusion needs lexical grounding but is allowed a lower score, because a
# one-word diagnosis matches only a small part of a long examples list.
MIN_EXCLUSION_SCORE = {"local": 0.03, "gemini": 0.35}

# Diagnosis words too generic to tie a diagnosis to a specific exclusion.
GENERIC_TERMS = frozenset(stem(w) for w in (
    "acute", "chronic", "severe", "accident", "accidental", "injury", "injuries", "treatment", "surgery", "surgical",
    "illness", "disease", "condition", "hospital", "hospitalisation", "medical", "procedure", "non", "due", "care",
    "with", "right", "left", "open", "internal", "laparoscopic", "reduction", "loss", "pain",
))
CLAUSE_CODE = re.compile(r"\b[A-Z]{2,3}-(?:\d{2}|[A-Z]{2})\b")


@dataclass(frozen=True)
class Topic:
    key: str
    template: str
    categories: tuple[str, ...]
    k: int
    required: bool = False  # missing evidence for a required topic -> INSUFFICIENT_EVIDENCE


TOPICS: tuple[Topic, ...] = (
    Topic("coverage", "{diagnosis} in-patient hospitalisation cover admissible expenses", ("COVERAGE",), 2, True),
    Topic("exclusions", "{diagnosis}", ("EXCLUSION",), 3),
    Topic("waiting_period", "initial waiting period days from policy start date accident", ("WAITING_PERIOD",), 1),
    Topic("limits", "sum insured limit of cover claimed amount co-payment", ("COVERAGE", "DEFINITION"), 2),
    Topic("policy_in_force", "policy in force on date of admission lapsed cancelled end date", ("CONDITION",), 1),
    Topic("documents", "documents required claim form hospital bill discharge summary", ("CONDITION",), 1),
)


class _Explanation(BaseModel):
    explanation: str = Field(description="3-6 sentences for a claims reviewer, using only the clauses provided")
    cited_clause_codes: list[str] = Field(description="Clause codes from the provided list that support the text")


def build_rag_input(verification: PolicyVerificationResult, facts: PolicyVerificationInput) -> PolicyRAGInput:
    """Connect Agent 2's output to Agent 3: the verified plan decides which wording is searched."""
    policy = verification.policy
    return PolicyRAGInput(
        plan_code=policy.plan_code if policy else None,
        policy_number=policy.policy_number if policy else facts.policy_number,
        insurance_type=policy.insurance_type if policy else facts.insurance_type,
        diagnosis=facts.diagnosis,
        policy_flags=dict(verification.flags),
    )


def diagnosis_terms_in(diagnosis: str | None, clause_text: str) -> list[str]:
    """Specific diagnosis words that literally occur (after stemming) in the clause text."""
    if not diagnosis:
        return []
    clause_stems = set(tokenize(clause_text))
    matched = []
    for word in re.findall(r"[a-z0-9]+", diagnosis.lower()):
        s = stem(word)
        if len(word) >= 4 and s not in GENERIC_TERMS and s in clause_stems and word not in matched:
            matched.append(word)
    return matched


class PolicyRAGAgent:
    key = "rag_agent"
    name = "Policy RAG Agent"
    purpose = "Retrieves cited policy wording relevant to the claim and flags insufficient evidence."

    def __init__(self, index: PolicyIndex | None = None, use_llm: bool | None = None, llm_model=None):
        self._index = index
        self.llm_model = llm_model
        self.use_llm = (llm_model is not None or llm_available()) if use_llm is None else use_llm

    @property
    def index(self) -> PolicyIndex:
        if self._index is None:
            self._index = get_policy_index()
        return self._index

    def _thresholds(self) -> tuple[float, float]:
        kind = "gemini" if self.index.backend.is_llm else "local"
        return MIN_SCORE[kind], MIN_EXCLUSION_SCORE[kind]

    # ------------------------------------------------------------------ run
    def run(self, data: PolicyRAGInput) -> PolicyRAGResult:
        min_score, min_exclusion = self._thresholds()
        base = dict(plan_code=data.plan_code, embedding_model=self.index.embedding_id, min_score=min_score)

        if not data.plan_code:
            reason = ("The policy could not be identified, so there is no policy wording to search. "
                      "No clauses are cited rather than guessing which product applies.")
            return PolicyRAGResult(
                status="POLICY_CONTEXT_UNAVAILABLE", uncertainty_reasons=[reason], requires_human_review=True,
                flags={"policy_context_unavailable": True, "insufficient_evidence": True,
                       "exclusion_wording_found": False},
                explanation=reason, **base)
        if data.plan_code not in self.index.plans():
            reason = f"No indexed policy wording exists for plan {data.plan_code}."
            return PolicyRAGResult(
                status="INSUFFICIENT_EVIDENCE", uncertainty_reasons=[reason], requires_human_review=True,
                flags={"policy_context_unavailable": False, "insufficient_evidence": True,
                       "exclusion_wording_found": False},
                explanation=reason, **base)

        diagnosis = (data.diagnosis or "").strip()
        queries: list[RAGQuery] = []
        clauses: list[RetrievedClause] = []
        uncertainty: list[str] = []
        seen: set[str] = set()

        for topic in TOPICS:
            if topic.key == "exclusions" and not diagnosis:
                uncertainty.append("The diagnosis is missing, so exclusions could not be checked against the wording.")
                continue
            query = topic.template.format(diagnosis=diagnosis).strip()
            floor = min_exclusion if topic.key == "exclusions" else min_score
            hits = self.index.search(query, k=topic.k, plan_code=data.plan_code, categories=topic.categories,
                                     min_score=floor)
            kept = 0
            for hit in hits:
                clause = self._to_clause(topic.key, hit, diagnosis)
                if topic.key == "exclusions" and not clause.matched_terms:
                    continue  # similar-sounding but not grounded in the diagnosis: not evidence
                key = f"{hit.chunk.chunk_id}"
                if key in seen:
                    continue
                seen.add(key)
                clauses.append(clause)
                kept += 1
            queries.append(RAGQuery(topic=topic.key, query=query, categories=list(topic.categories), hits=kept))
            if topic.required and kept == 0:
                uncertainty.append(f"No {topic.key} wording scored above the minimum relevance of {floor:.2f}.")

        possible = [c.clause_code for c in clauses if c.topic == "exclusions" and c.clause_code]
        if data.policy_flags.get("possible_exclusion") and not possible:
            uncertainty.append("The rule-based check flagged a possible exclusion, but no exclusion wording matching "
                               "the diagnosis was retrieved. A reviewer must confirm.")

        status = "INSUFFICIENT_EVIDENCE" if uncertainty else "EVIDENCE_FOUND"
        findings = self._findings(clauses, data)
        result = PolicyRAGResult(
            status=status, queries=queries, clauses=clauses, possible_exclusions=possible, findings=findings,
            uncertainty_reasons=uncertainty, requires_human_review=bool(uncertainty or possible),
            flags={"policy_context_unavailable": False, "insufficient_evidence": bool(uncertainty),
                   "exclusion_wording_found": bool(possible)},
            **base,
        )
        result.explanation = self._template_explanation(result, data)
        if self.use_llm and clauses:
            self._llm_explanation(result, data)
        return result

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _to_clause(topic: str, hit: SearchHit, diagnosis: str) -> RetrievedClause:
        c = hit.chunk
        return RetrievedClause(
            topic=topic, clause_code=c.clause_code, title=c.title, category=c.category, plan_code=c.plan_code,
            source_document=c.source_document, page_number=c.page_number, score=round(hit.score, 4), text=c.text,
            matched_terms=diagnosis_terms_in(diagnosis, c.text) if topic == "exclusions" else [],
        )

    @staticmethod
    def _findings(clauses: list[RetrievedClause], data: PolicyRAGInput) -> list[str]:
        out = []
        for c in clauses:
            if c.topic == "exclusions":
                out.append(f"Possible exclusion: the diagnosis mentions {', '.join(repr(t) for t in c.matched_terms)}, "
                           f"which appears in {c.citation}.")
            elif c.topic == "coverage":
                out.append(f"Coverage wording: {c.citation}.")
            elif c.topic == "waiting_period" and data.policy_flags.get("waiting_period_incomplete"):
                out.append(f"Waiting period wording relevant to the rule-based flag: {c.citation}.")
            elif c.topic == "limits" and data.policy_flags.get("exceeds_sum_insured"):
                out.append(f"Sum insured wording relevant to the rule-based flag: {c.citation}.")
            elif c.topic == "policy_in_force" and data.policy_flags.get("policy_inactive"):
                out.append(f"Policy-in-force wording relevant to the rule-based flag: {c.citation}.")
        return out

    @staticmethod
    def _template_explanation(result: PolicyRAGResult, data: PolicyRAGInput) -> str:
        parts = [f"{len(result.clauses)} clause(s) of the {data.plan_code} policy wording were retrieved for this claim."]
        cov = [c for c in result.clauses if c.topic == "coverage"]
        if cov:
            parts.append(f"Coverage is described in {', '.join(c.clause_code for c in cov)}.")
        if result.possible_exclusions:
            parts.append(f"Exclusion wording that mentions the diagnosis was found in "
                         f"{', '.join(result.possible_exclusions)}; a reviewer should read it before assessing.")
        elif data.diagnosis:
            parts.append("No exclusion clause in the wording mentions the diagnosis.")
        if result.uncertainty_reasons:
            parts.append("Evidence is incomplete: " + " ".join(result.uncertainty_reasons))
        parts.append("This is a summary of policy wording, not a claim decision.")
        return " ".join(parts)

    def _llm_explanation(self, result: PolicyRAGResult, data: PolicyRAGInput) -> None:
        """Ask Gemini for a reviewer-friendly explanation; keep it only if every citation was retrieved."""
        allowed = {c.clause_code for c in result.clauses if c.clause_code}
        context = "\n\n".join(f"[{c.clause_code}] {c.title} (category {c.category}, page {c.page_number}):\n{c.text}"
                              for c in result.clauses)
        flags = ", ".join(k for k, v in data.policy_flags.items() if v) or "none"
        prompt = (
            "You help an insurance claims reviewer read policy wording. Use ONLY the clauses below. "
            "Do not invent clauses, numbers or conditions. Do not approve, reject or recommend payment; "
            "do not accuse anyone. If the clauses do not settle a question, say that a reviewer must decide.\n\n"
            f"Plan: {data.plan_code}\nDiagnosis: {data.diagnosis or 'not provided'}\n"
            f"Rule-based flags from policy verification: {flags}\n\nClauses:\n{context}"
        )
        answer, call = invoke_structured(prompt, _Explanation, model=self.llm_model)
        if answer is None:
            result.warnings.append(f"Gemini explanation unavailable ({call.error}); template explanation used.")
            return
        cited = set(answer.cited_clause_codes) | set(CLAUSE_CODE.findall(answer.explanation))
        invented = sorted(cited - allowed)
        if invented or not cited:
            result.warnings.append(
                "Gemini explanation discarded: it cited clauses that were not retrieved "
                f"({', '.join(invented) or 'no citations'}); template explanation used.")
            result.used_llm, result.llm_model = True, call.model or settings.gemini_model
            return
        result.explanation = answer.explanation.strip()
        result.explanation_source = "gemini"
        result.used_llm, result.llm_model = True, call.model or settings.gemini_model


def main() -> None:
    import argparse

    from agents.policy_agent import PolicyVerificationAgent, build_input
    from database.connection import SessionLocal
    from database.repository import ClaimRepository

    parser = argparse.ArgumentParser(description="Run policy verification + policy RAG on a stored claim")
    parser.add_argument("claim_number", nargs="?")
    args = parser.parse_args()
    with SessionLocal() as session:
        repo = ClaimRepository(session)
        claim = repo.get_by_number(args.claim_number) if args.claim_number else next(
            c for c in repo.list() if c.customer_name_submitted == "Rahul Sharma")
        facts = build_input(claim)
        verification = PolicyVerificationAgent(session).run(facts)
        result = PolicyRAGAgent().run(build_rag_input(verification, facts))
    print(f"{claim.claim_number}  scenario={claim.scenario}  policy={verification.status}  rag={result.status}")
    print(f"embeddings={result.embedding_model}  explanation={result.explanation_source}  used_llm={result.used_llm}")
    for c in result.clauses:
        extra = f"  matched={c.matched_terms}" if c.matched_terms else ""
        print(f"  [{c.topic:15}] {c.score:.3f}  {c.citation}{extra}")
    for r in result.uncertainty_reasons:
        print(f"  UNCERTAIN: {r}")
    print("\n" + result.explanation)


if __name__ == "__main__":
    main()
