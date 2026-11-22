"""Unit tests for the deterministic synthetic data generator."""

from collections import Counter
from datetime import date

from database.catalog import PLANS, REQUIRED_DOCUMENTS
from database.synthetic import SCENARIOS, generate
from tests.conftest import REFERENCE_DATE


def test_minimum_volumes(synthetic_data):
    assert len(synthetic_data["customers"]) >= 50
    assert len(synthetic_data["policies"]) >= 50
    assert len(synthetic_data["claims"]) >= 30


def test_deterministic_for_same_reference_date(synthetic_data):
    assert generate(reference_date=REFERENCE_DATE) == synthetic_data


def test_all_ten_scenarios_present(synthetic_data):
    counts = Counter(c["scenario"] for c in synthetic_data["claims"])
    assert set(counts) == set(SCENARIOS)
    for scenario, (n, _) in SCENARIOS.items():
        assert counts[scenario] == n


def test_unique_identifiers(synthetic_data):
    for key, field in (("customers", "customer_code"), ("policies", "policy_number"), ("claims", "claim_number")):
        values = [r[field] for r in synthetic_data[key]]
        assert len(values) == len(set(values)), f"duplicate {field}"


def test_policy_mix(synthetic_data):
    statuses = Counter(p["status"] for p in synthetic_data["policies"])
    assert statuses["ACTIVE"] > 0 and statuses["EXPIRED"] > 0
    assert len({p["insurance_type"] for p in synthetic_data["policies"]}) == len(PLANS)
    assert len({p["waiting_period_days"] for p in synthetic_data["policies"]}) > 1


def test_rahul_sharma_example_claim(synthetic_data):
    rahul = [c for c in synthetic_data["claims"] if c["customer_name"] == "Rahul Sharma"]
    assert rahul and rahul[0]["claim_amount"] == 85_000 and rahul[0]["scenario"] == "VALID"


def _policies(data):
    return {p["policy_number"]: p for p in data["policies"]}


def test_scenarios_have_their_defining_property(synthetic_data):
    pol = _policies(synthetic_data)
    for c in synthetic_data["claims"]:
        adm, dis = date.fromisoformat(c["admission_date"]), date.fromisoformat(c["discharge_date"])
        p = pol.get(c["policy_number"])
        s = c["scenario"]
        if s == "VALID":
            assert p and p["status"] == "ACTIVE"
            assert date.fromisoformat(p["start_date"]) <= adm <= date.fromisoformat(p["end_date"])
            assert c["claim_amount"] == c["billed_amount"] <= p["sum_insured"]
            assert set(REQUIRED_DOCUMENTS) <= set(c["documents"])
            assert dis >= adm
        elif s == "EXPIRED_POLICY":
            assert adm > date.fromisoformat(p["end_date"])
        elif s == "UNKNOWN_POLICY":
            assert p is None
        elif s == "EXCEEDS_SUM_INSURED":
            assert c["claim_amount"] > p["sum_insured"]
        elif s == "MISSING_DOCUMENTS":
            assert not set(REQUIRED_DOCUMENTS) <= set(c["documents"])
        elif s == "WAITING_PERIOD":
            assert (adm - date.fromisoformat(p["start_date"])).days < p["waiting_period_days"]
        elif s == "INVALID_DATES":
            assert dis < adm or adm > date.fromisoformat(c["submitted_at"][:10])
        elif s == "INCONSISTENT_AMOUNT":
            assert c["claim_amount"] > c["billed_amount"] * 1.2


def test_duplicate_invoices_share_number(synthetic_data):
    invoices = Counter(c["invoice_number"] for c in synthetic_data["claims"])
    dups = [c for c in synthetic_data["claims"] if c["scenario"] == "DUPLICATE_INVOICE"]
    assert all(invoices[c["invoice_number"]] == 2 for c in dups)


def test_bill_items_sum_to_billed_amount(synthetic_data):
    for c in synthetic_data["claims"]:
        assert sum(i["amount"] for i in c["bill_items"]) == c["billed_amount"]


def test_data_is_fictional(synthetic_data):
    assert all(c["email"].endswith("@example.com") for c in synthetic_data["customers"])
