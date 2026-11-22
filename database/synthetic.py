"""
Deterministic synthetic data generator (fictional people, policies, claims).

    python -m database.synthetic            # writes data/synthetic/*.json

Beginner note:
    A seeded `random.Random(42)` makes the output identical on every run for
    the same reference date, so tests are repeatable. Dates are generated
    *relative to the reference date* (today by default) so "active" policies
    stay active whenever you run the project.

Every claim carries a `scenario` tag and the `expected_outcome` that the agent
pipeline should reach — this is our ground truth for end-to-end tests.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from database.catalog import (
    DIAGNOSES,
    EXCLUDED_DIAGNOSES,
    HOSPITALS,
    OPTIONAL_DOCUMENTS,
    PLANS,
    REQUIRED_DOCUMENTS,
    Diagnosis,
    Plan,
)
from database.models import ClaimStatus, PolicyStatus

FIRST_NAMES = (
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Ananya", "Diya", "Aadhya", "Saanvi", "Pari", "Anika", "Navya", "Myra", "Kavya", "Meera",
    "Rohan", "Karthik", "Nikhil", "Pranav", "Siddharth", "Varun", "Neha", "Pooja", "Sneha", "Lakshmi",
    "Farhan", "Imran", "Zoya", "Sana", "Gurpreet", "Harleen", "Joseph", "Maria", "Tenzin", "Priya",
)
LAST_NAMES = (
    "Sharma", "Reddy", "Iyer", "Nair", "Patel", "Gupta", "Rao", "Menon", "Joshi", "Kulkarni",
    "Banerjee", "Chatterjee", "Singh", "Verma", "Mehta", "Desai", "Pillai", "Khan", "Fernandes", "Das",
)
CITIES = (
    ("Hyderabad", "Telangana"), ("Bengaluru", "Karnataka"), ("Chennai", "Tamil Nadu"), ("Pune", "Maharashtra"),
    ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Kolkata", "West Bengal"), ("Ahmedabad", "Gujarat"),
    ("Jaipur", "Rajasthan"), ("Kochi", "Kerala"),
)

# Scenario -> (number of claims, expected AI outcome)
SCENARIOS: dict[str, tuple[int, str]] = {
    "VALID": (14, ClaimStatus.ELIGIBLE_FOR_HUMAN_REVIEW.value),
    "EXPIRED_POLICY": (2, ClaimStatus.REVIEW_REQUIRED.value),
    "UNKNOWN_POLICY": (2, ClaimStatus.POLICY_NOT_FOUND.value),
    "EXCEEDS_SUM_INSURED": (2, ClaimStatus.REVIEW_REQUIRED.value),
    "MISSING_DOCUMENTS": (3, ClaimStatus.ADDITIONAL_INFORMATION_REQUIRED.value),
    "WAITING_PERIOD": (2, ClaimStatus.REVIEW_REQUIRED.value),
    "POLICY_EXCLUSION": (2, ClaimStatus.REVIEW_REQUIRED.value),
    "DUPLICATE_INVOICE": (2, ClaimStatus.REVIEW_REQUIRED.value),
    "INVALID_DATES": (2, ClaimStatus.ADDITIONAL_INFORMATION_REQUIRED.value),
    "INCONSISTENT_AMOUNT": (2, ClaimStatus.REVIEW_REQUIRED.value),
}

N_CUSTOMERS = 55
N_EXTRA_POLICIES = 8


@dataclass
class _Ctx:
    rng: random.Random
    ref: date


def _iso(d: date | datetime | None) -> str | None:
    return d.isoformat() if d else None


def _rand_date(ctx: _Ctx, lo: date, hi: date) -> date:
    if hi < lo:
        hi = lo
    return lo + timedelta(days=ctx.rng.randint(0, (hi - lo).days))


# --------------------------------------------------------------------------- #
# Customers
# --------------------------------------------------------------------------- #
def _customers(ctx: _Ctx) -> list[dict[str, Any]]:
    rng = ctx.rng
    used: set[str] = {"Rahul Sharma"}
    out = []
    for i in range(1, N_CUSTOMERS + 1):
        if i == 1:
            name = "Rahul Sharma"  # the walkthrough example from the brief
        else:
            while True:
                name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                if name not in used:
                    used.add(name)
                    break
        # Customers 40-55 are 60+ (eligible for the senior plan).
        age = rng.randint(61, 78) if i >= 40 else rng.randint(23, 58)
        dob = date(ctx.ref.year - age, rng.randint(1, 12), rng.randint(1, 28))
        city, state = CITIES[0] if i == 1 else rng.choice(CITIES)
        first, last = name.lower().split(" ", 1)
        out.append({
            "customer_code": f"CUST-{i:04d}",
            "full_name": name,
            "date_of_birth": _iso(dob),
            "email": f"{first}.{last}.{i}@example.com",  # example.com is reserved for fiction
            "phone": f"+91-55501-{i:05d}",               # 555-01xx range: fictional numbers
            "city": city,
            "state": state,
        })
    return out


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def _policy(ctx: _Ctx, idx: int, customer: dict, plan: Plan, start: date, status: str,
            sum_insured: int | None = None, waiting: int | None = None) -> dict[str, Any]:
    si = sum_insured or ctx.rng.choice(plan.sum_insured_options)
    return {
        "policy_number": f"{plan.code}-{start.year}-{100000 + idx}",
        "customer_code": customer["customer_code"],
        "insurance_type": plan.insurance_type.value,
        "plan_name": plan.name,
        "plan_code": plan.code,
        "sum_insured": si,
        "annual_premium": round(si * plan.premium_rate, 2),
        "start_date": _iso(start),
        "end_date": _iso(start + timedelta(days=364)),
        "waiting_period_days": plan.default_waiting_days if waiting is None else waiting,
        "status": status,
        "hospitalization_covered": True,
        "exclusions": [e.title for e in plan.exclusions],
        "policy_document": plan.document_filename,
    }


def _age(customer: dict, ref: date) -> int:
    return ref.year - int(customer["date_of_birth"][:4])


def _random_plan(ctx: _Ctx, customer: dict) -> Plan:
    if _age(customer, ctx.ref) >= 60:
        return ctx.rng.choice([PLANS["SCP"], PLANS["SCP"], PLANS["CIP"]])
    return ctx.rng.choice([PLANS["IHS"], PLANS["IHS"], PLANS["FFS"], PLANS["FFS"], PLANS["PAG"], PLANS["CIP"]])


def _policies(ctx: _Ctx, customers: list[dict]) -> list[dict[str, Any]]:
    """Policy i belongs to customer i. Indices 1-17 are reserved for scenarios."""
    ref, rng = ctx.ref, ctx.rng
    active_start = lambda: _rand_date(ctx, ref - timedelta(days=330), ref - timedelta(days=100))  # noqa: E731
    out = []
    for i, cust in enumerate(customers, start=1):
        if i == 1:      # Rahul Sharma — healthy, active, ₹5L cover
            p = _policy(ctx, i, cust, PLANS["IHS"], ref - timedelta(days=300), PolicyStatus.ACTIVE, 500_000)
        elif 2 <= i <= 9:  # expired
            start = _rand_date(ctx, ref - timedelta(days=600), ref - timedelta(days=420))
            p = _policy(ctx, i, cust, rng.choice([PLANS["IHS"], PLANS["FFS"]]), start, PolicyStatus.EXPIRED)
        elif i in (10, 11):
            p = _policy(ctx, i, cust, PLANS["IHS"], active_start(), PolicyStatus.LAPSED)
        elif i == 12:
            p = _policy(ctx, i, cust, PLANS["FFS"], active_start(), PolicyStatus.CANCELLED)
        elif 13 <= i <= 15:  # brand-new policies still inside their 30-day waiting period
            p = _policy(ctx, i, cust, PLANS["IHS"], ref - timedelta(days=22), PolicyStatus.ACTIVE, waiting=30)
        elif i in (16, 17):  # smallest cover, used for "exceeds sum insured"
            p = _policy(ctx, i, cust, PLANS["IHS"], active_start(), PolicyStatus.ACTIVE, 300_000)
        else:
            p = _policy(ctx, i, cust, _random_plan(ctx, cust), active_start(), PolicyStatus.ACTIVE)
        out.append(p)

    # A few customers hold a second policy.
    for k, cust in enumerate(rng.sample(customers[19:], N_EXTRA_POLICIES), start=N_CUSTOMERS + 1):
        out.append(_policy(ctx, k, cust, _random_plan(ctx, cust), active_start(), PolicyStatus.ACTIVE))
    return out


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #
def _bill_items(rng: random.Random, total: int) -> list[dict[str, Any]]:
    heads = ["Room & boarding", "Surgeon / consultant fees", "Pharmacy & medicines",
             "Diagnostics & laboratory", "OT & consumables"]
    weights = [rng.uniform(0.18, 0.3), rng.uniform(0.18, 0.3), rng.uniform(0.1, 0.2),
               rng.uniform(0.08, 0.15), rng.uniform(0.1, 0.2)]
    scale = total / sum(weights)
    amounts = [int(w * scale) for w in weights]
    amounts[-1] += total - sum(amounts)  # make the items add up exactly
    return [{"description": h, "amount": a} for h, a in zip(heads, amounts)]


def _diagnosis_for(rng: random.Random, plan_code: str) -> Diagnosis:
    if plan_code == "PAG":
        pool = [d for d in DIAGNOSES if d.accidental]
    elif plan_code == "CIP":
        pool = [d for d in DIAGNOSES if d.critical]
    else:
        pool = [d for d in DIAGNOSES if not d.critical]
    return rng.choice(pool)


class _ClaimFactory:
    def __init__(self, ctx: _Ctx, customers: list[dict], policies: list[dict]):
        self.ctx, self.rng = ctx, ctx.rng
        self.customers = {c["customer_code"]: c for c in customers}
        self.policies = policies
        self.invoice_seq = 40000
        self.claims: list[dict[str, Any]] = []

    def _invoice(self, hospital: str, year: int) -> str:
        self.invoice_seq += self.rng.randint(3, 97)
        abbrev = "".join(w[0] for w in hospital.split(",")[0].split() if w[0].isalpha())[:3].upper()
        return f"INV-{abbrev}-{year}-{self.invoice_seq:05d}"

    def build(self, scenario: str, policy: dict | None, *, customer_code: str | None = None,
              admission: date, los: int | None = None, diagnosis: str | None = None,
              cost: int | None = None, hospital: str | None = None, invoice: str | None = None,
              claim_amount: int | None = None, documents: list[str] | None = None,
              policy_number: str | None = None, submitted: date | None = None,
              discharge: date | None = None, note: str = "") -> dict[str, Any]:
        rng = self.rng
        cust = self.customers[customer_code or policy["customer_code"]]
        if diagnosis is None:
            dx = _diagnosis_for(rng, policy["plan_code"] if policy else "IHS")
            diagnosis, cost = dx.name, cost or rng.randint(dx.min_cost, dx.max_cost) // 100 * 100
        hospital = hospital or rng.choice(HOSPITALS)
        los = los if los is not None else rng.randint(1, 6)
        discharge = discharge or admission + timedelta(days=los)
        billed = int(cost)
        submitted = submitted or min(discharge + timedelta(days=rng.randint(1, 7)), self.ctx.ref)
        submitted_dt = datetime.combine(submitted, time(rng.randint(8, 19), rng.randint(0, 59)), tzinfo=timezone.utc)
        docs = documents if documents is not None else list(REQUIRED_DOCUMENTS) + rng.sample(OPTIONAL_DOCUMENTS, 1)
        amount = claim_amount if claim_amount is not None else billed
        claim = {
            "claim_number": f"CLM-{submitted:%Y%m}-{len(self.claims) + 1:04d}",
            "customer_code": cust["customer_code"],
            "customer_name": cust["full_name"],
            "policy_number": policy_number or policy["policy_number"],
            "insurance_type": policy["insurance_type"] if policy else "Individual Health",
            "claim_amount": amount,
            "hospital_name": hospital,
            "admission_date": _iso(admission),
            "discharge_date": _iso(discharge),
            "diagnosis": diagnosis,
            "description": (f"Hospitalised at {hospital} for {diagnosis.lower()}. "
                            f"Requesting reimbursement of INR {amount:,}. {note}").strip(),
            "invoice_number": invoice or self._invoice(hospital, admission.year),
            "billed_amount": billed,
            "bill_items": _bill_items(rng, billed),
            "documents": docs,
            "submitted_at": _iso(submitted_dt),
            "scenario": scenario,
            "expected_outcome": SCENARIOS[scenario][1],
        }
        self.claims.append(claim)
        return claim


def _claims(ctx: _Ctx, customers: list[dict], policies: list[dict]) -> list[dict[str, Any]]:
    ref, rng = ctx.ref, ctx.rng
    f = _ClaimFactory(ctx, customers, policies)
    by_idx = {i: p for i, p in enumerate(policies, start=1)}

    def eligible_window(p: dict) -> tuple[date, date]:
        start = date.fromisoformat(p["start_date"]) + timedelta(days=p["waiting_period_days"] + 1)
        end = min(date.fromisoformat(p["end_date"]), ref - timedelta(days=9))
        return start, end

    active_pool = [p for i, p in by_idx.items() if i >= 18 and p["status"] == PolicyStatus.ACTIVE]

    # 1) VALID — includes Rahul Sharma's ₹85,000 appendicitis claim.
    valid: list[dict] = [f.build(
        "VALID", by_idx[1], admission=ref - timedelta(days=24), los=3,
        diagnosis="Acute appendicitis - laparoscopic appendectomy", cost=85_000,
        hospital=HOSPITALS[0], documents=["CLAIM_FORM", "HOSPITAL_BILL", "DISCHARGE_SUMMARY", "DIAGNOSTIC_REPORT"],
    )]
    for p in rng.sample(active_pool, SCENARIOS["VALID"][0] - 1):
        lo, hi = eligible_window(p)
        valid.append(f.build("VALID", p, admission=_rand_date(ctx, lo, hi - timedelta(days=6))))

    # 2) EXPIRED_POLICY — admitted after the policy end date.
    for i in (2, 3):
        end = date.fromisoformat(by_idx[i]["end_date"])
        f.build("EXPIRED_POLICY", by_idx[i], admission=min(end + timedelta(days=rng.randint(10, 40)), ref - timedelta(days=12)))

    # 3) UNKNOWN_POLICY — quotes a policy number that does not exist.
    for k, code in enumerate(rng.sample([c["customer_code"] for c in customers[20:]], 2)):
        f.build("UNKNOWN_POLICY", None, customer_code=code, policy_number=f"IHS-2025-99{9901 + k}",
                admission=_rand_date(ctx, ref - timedelta(days=120), ref - timedelta(days=15)))

    # 4) EXCEEDS_SUM_INSURED — ₹3L cover, much larger bill.
    for i in (16, 17):
        lo, hi = eligible_window(by_idx[i])
        f.build("EXCEEDS_SUM_INSURED", by_idx[i], admission=_rand_date(ctx, lo, hi - timedelta(days=8)),
                diagnosis="Coronary artery disease - angioplasty with stent", cost=rng.randint(345, 420) * 1000, los=5)

    # 5) MISSING_DOCUMENTS — bill and/or discharge summary not provided.
    for docs in (["CLAIM_FORM"], ["CLAIM_FORM", "DISCHARGE_SUMMARY"], ["CLAIM_FORM", "HOSPITAL_BILL"]):
        p = rng.choice(active_pool)
        lo, hi = eligible_window(p)
        f.build("MISSING_DOCUMENTS", p, admission=_rand_date(ctx, lo, hi - timedelta(days=6)), documents=docs)

    # 6) WAITING_PERIOD — admitted 8 days after a policy with a 30-day waiting period began.
    for i in (13, 14):
        start = date.fromisoformat(by_idx[i]["start_date"])
        f.build("WAITING_PERIOD", by_idx[i], admission=start + timedelta(days=8), los=2,
                diagnosis="Typhoid fever", cost=rng.randint(28, 46) * 1000, submitted=ref - timedelta(days=rng.randint(1, 4)))

    # 7) POLICY_EXCLUSION — cosmetic / dental treatment.
    ihs_pool = [p for p in active_pool if p["plan_code"] in ("IHS", "FFS")]
    for (dx, lo_c, hi_c), p in zip(EXCLUDED_DIAGNOSES, rng.sample(ihs_pool, 2)):
        lo, hi = eligible_window(p)
        f.build("POLICY_EXCLUSION", p, admission=_rand_date(ctx, lo, hi - timedelta(days=4)), diagnosis=dx,
                cost=rng.randint(lo_c, hi_c) // 100 * 100, los=1)

    # 8) DUPLICATE_INVOICE — re-uses an invoice number from an earlier valid claim.
    originals = sorted(valid[1:], key=lambda c: c["submitted_at"])[:2]
    other_customer = rng.choice([p for p in active_pool if p["customer_code"] not in {o["customer_code"] for o in originals}])
    for orig, p in ((originals[0], by_idx[next(i for i, x in by_idx.items() if x["policy_number"] == originals[0]["policy_number"])]),
                    (originals[1], other_customer)):
        f.build("DUPLICATE_INVOICE", p, admission=date.fromisoformat(orig["admission_date"]),
                discharge=date.fromisoformat(orig["discharge_date"]), diagnosis=orig["diagnosis"],
                cost=orig["billed_amount"], hospital=orig["hospital_name"], invoice=orig["invoice_number"],
                submitted=min(date.fromisoformat(orig["submitted_at"][:10]) + timedelta(days=rng.randint(10, 30)), ref),
                note="Resubmission of hospital invoice.")

    # 9) INVALID_DATES — discharge before admission; admission in the future.
    p1, p2 = rng.sample(active_pool, 2)
    adm = ref - timedelta(days=30)
    f.build("INVALID_DATES", p1, admission=adm, discharge=adm - timedelta(days=3), submitted=ref - timedelta(days=5))
    f.build("INVALID_DATES", p2, admission=ref + timedelta(days=12), los=2, submitted=ref - timedelta(days=2))

    # 10) INCONSISTENT_AMOUNT — claimed amount well above the hospital bill.
    for p in rng.sample(active_pool, 2):
        lo, hi = eligible_window(p)
        dx = _diagnosis_for(rng, p["plan_code"])
        billed = rng.randint(dx.min_cost, dx.max_cost) // 100 * 100
        f.build("INCONSISTENT_AMOUNT", p, admission=_rand_date(ctx, lo, hi - timedelta(days=6)), diagnosis=dx.name,
                cost=billed, claim_amount=int(billed * rng.uniform(1.3, 1.6)) // 100 * 100)

    # Order chronologically and assign final, readable claim numbers.
    claims = sorted(f.claims, key=lambda c: c["submitted_at"])
    for n, c in enumerate(claims, start=1):
        c["claim_number"] = f"CLM-{c['submitted_at'][:7].replace('-', '')}-{n:04d}"
    return claims


# --------------------------------------------------------------------------- #
def generate(reference_date: date | None = None, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    ctx = _Ctx(random.Random(seed), reference_date or date.today())
    customers = _customers(ctx)
    policies = _policies(ctx, customers)
    claims = _claims(ctx, customers, policies)
    return {"customers": customers, "policies": policies, "claims": claims}


def write_json(data: dict[str, list[dict[str, Any]]], out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or settings.synthetic_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in data.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    dataset = generate()
    for p in write_json(dataset):
        print(f"wrote {p.relative_to(settings.data_dir.parent)}")
    print({k: len(v) for k, v in dataset.items()})
