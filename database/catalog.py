"""
Fictional insurance product catalogue.

These plan definitions are the single source of truth for:
    * synthetic policy records (Phase 2),
    * the policy PDF wording generated for RAG (Phases 3 and 5),
    * deterministic business rules in the Policy Verification Agent (Phase 4).

All products, hospitals and wording are fictional and for demonstration only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from database.models import InsuranceType


@dataclass(frozen=True)
class Exclusion:
    code: str
    title: str
    keywords: tuple[str, ...]  # diagnosis keywords that trigger this exclusion (rule-based)
    wording: str


@dataclass(frozen=True)
class Plan:
    code: str
    name: str
    insurance_type: InsuranceType
    sum_insured_options: tuple[int, ...]
    default_waiting_days: int
    premium_rate: float  # annual premium as a fraction of sum insured
    exclusions: tuple[Exclusion, ...] = field(default_factory=tuple)
    covered_summary: str = ""
    min_age: int = 18
    max_age: int = 99
    copay_percent: float = 0.0  # share of every admissible claim paid by the insured

    @property
    def document_filename(self) -> str:
        return f"{self.code}_policy_wording.pdf"


# Common exclusions shared by most plans ---------------------------------------
COSMETIC = Exclusion(
    "EXC-01", "Cosmetic or aesthetic treatment",
    ("cosmetic", "aesthetic", "rhinoplasty", "liposuction", "hair transplant", "botox"),
    "Expenses for cosmetic, aesthetic or plastic surgery, unless required as treatment of an injury "
    "or illness covered under this policy, are not payable.",
)
DENTAL = Exclusion(
    "EXC-02", "Dental treatment (non-accidental)",
    ("dental implant", "root canal", "orthodontic", "teeth whitening"),
    "Dental treatment or surgery of any kind is excluded unless necessitated by an accident and "
    "requiring hospitalisation.",
)
SUBSTANCE = Exclusion(
    "EXC-03", "Substance and alcohol abuse",
    ("alcohol", "substance abuse", "drug overdose", "de-addiction"),
    "Treatment for alcoholism, drug or substance abuse, or any addictive condition and its consequences "
    "is excluded.",
)
SELF_INJURY = Exclusion(
    "EXC-04", "Intentional self-injury",
    ("self-inflicted", "self inflicted", "attempted suicide"),
    "Any intentional self-inflicted injury is excluded.",
)
INFERTILITY = Exclusion(
    "EXC-05", "Infertility and assisted reproduction",
    ("infertility", "ivf", "in vitro"),
    "Treatment related to infertility, sterility or assisted reproduction (including IVF) is excluded.",
)
EXPERIMENTAL = Exclusion(
    "EXC-06", "Unproven or experimental treatment",
    ("experimental", "unproven"),
    "Treatment that is experimental, investigational or not supported by clinical evidence is excluded.",
)
WEIGHT = Exclusion(
    "EXC-07", "Obesity and weight-control treatment",
    ("bariatric", "weight loss", "obesity surgery"),
    "Surgical or non-surgical treatment of obesity or weight control is excluded unless medically "
    "certified as life-threatening under the policy definitions.",
)
NON_ACCIDENT = Exclusion(
    "EXC-PA", "Illness not caused by an accident",
    ("fever", "dengue", "typhoid", "pneumonia", "gastroenteritis", "cataract", "appendicitis",
     "cholelithiasis", "coronary", "kidney stone"),
    "This Personal Accident policy covers hospitalisation caused solely and directly by accidental bodily "
    "injury. Hospitalisation due to illness or disease is not covered.",
)

STANDARD = (COSMETIC, DENTAL, SUBSTANCE, SELF_INJURY, INFERTILITY, EXPERIMENTAL)

PLANS: dict[str, Plan] = {
    p.code: p
    for p in [
        Plan(
            "IHS", "InsureFlow Health Secure", InsuranceType.HEALTH,
            (300_000, 500_000, 1_000_000, 1_500_000), 30, 0.028, STANDARD + (WEIGHT,),
            "In-patient hospitalisation, pre- and post-hospitalisation expenses (30/60 days), day-care "
            "procedures and ambulance charges up to the sum insured.",
        ),
        Plan(
            "FFS", "Family Floater Shield", InsuranceType.FAMILY,
            (500_000, 1_000_000, 1_500_000, 2_500_000), 30, 0.032, STANDARD + (WEIGHT,),
            "Floater cover for the insured family: in-patient hospitalisation, day-care procedures and "
            "maternity after 24 months, up to the shared sum insured.",
        ),
        Plan(
            "SCP", "Senior Care Plus", InsuranceType.SENIOR,
            (300_000, 500_000, 1_000_000), 90, 0.055, STANDARD,
            "In-patient hospitalisation for insured persons aged 60 and above, with a 10% co-payment on "
            "every admissible claim.",
            min_age=60,
            copay_percent=10.0,
        ),
        Plan(
            "PAG", "Personal Accident Guard", InsuranceType.ACCIDENT,
            (500_000, 1_000_000, 2_500_000), 0, 0.006, (SUBSTANCE, SELF_INJURY, NON_ACCIDENT),
            "Hospitalisation and medical expenses arising solely from accidental bodily injury. No initial "
            "waiting period.",
        ),
        Plan(
            "CIP", "Critical Illness Protect", InsuranceType.CRITICAL,
            (500_000, 1_000_000, 2_500_000), 90, 0.018, STANDARD,
            "Hospitalisation for listed critical illnesses (including coronary artery disease, stroke, "
            "cancer and kidney failure) after a 90-day waiting period.",
        ),
    ]
}

# Words that indicate an accidental injury. Accidents are exempt from the
# initial waiting period, and are the only thing a Personal Accident plan covers.
ACCIDENT_KEYWORDS: tuple[str, ...] = (
    "accident", "fracture", "injury", "trauma", "ligament tear", "dislocation", "burn", "road traffic",
)

ALL_EXCLUSIONS: dict[str, Exclusion] = {
    e.title: e for e in (COSMETIC, DENTAL, SUBSTANCE, SELF_INJURY, INFERTILITY, EXPERIMENTAL, WEIGHT, NON_ACCIDENT)
}


def is_accidental(diagnosis: str | None) -> bool:
    text = (diagnosis or "").lower()
    return any(k in text for k in ACCIDENT_KEYWORDS)


# Fictional hospitals -------------------------------------------------------------
HOSPITALS: tuple[str, ...] = (
    "Sunrise Multispeciality Hospital, Hyderabad",
    "Lotus Care Hospital, Bengaluru",
    "Greenfield Medical Centre, Pune",
    "Riverside General Hospital, Chennai",
    "Harmony Heart Institute, Hyderabad",
    "Silver Oak Hospital, Mumbai",
    "Meadowbrook Clinic & Hospital, Delhi",
    "Banyan Tree Hospital, Kolkata",
    "Crescent Health Hospital, Ahmedabad",
    "Northstar Orthopaedic Hospital, Jaipur",
)

# Covered diagnoses with realistic cost ranges (INR) ------------------------------
@dataclass(frozen=True)
class Diagnosis:
    name: str
    min_cost: int
    max_cost: int
    accidental: bool = False
    critical: bool = False


DIAGNOSES: tuple[Diagnosis, ...] = (
    Diagnosis("Acute appendicitis - laparoscopic appendectomy", 60_000, 140_000),
    Diagnosis("Dengue fever with thrombocytopenia", 25_000, 70_000),
    Diagnosis("Typhoid fever", 20_000, 50_000),
    Diagnosis("Community-acquired pneumonia", 40_000, 110_000),
    Diagnosis("Cholelithiasis - laparoscopic cholecystectomy", 70_000, 160_000),
    Diagnosis("Cataract - phacoemulsification with IOL", 35_000, 65_000),
    Diagnosis("Kidney stone - ureteroscopic lithotripsy", 50_000, 120_000),
    Diagnosis("Acute gastroenteritis with dehydration", 15_000, 40_000),
    Diagnosis("Fracture of right radius - open reduction internal fixation", 55_000, 130_000, accidental=True),
    Diagnosis("Road traffic accident - ligament tear, knee arthroscopy", 80_000, 180_000, accidental=True),
    Diagnosis("Coronary artery disease - angioplasty with stent", 180_000, 380_000, critical=True),
)

EXCLUDED_DIAGNOSES: tuple[tuple[str, int, int], ...] = (
    ("Cosmetic rhinoplasty (aesthetic nasal reshaping)", 90_000, 160_000),
    ("Dental implant - non-accidental tooth loss", 45_000, 90_000),
)

REQUIRED_DOCUMENTS: tuple[str, ...] = ("CLAIM_FORM", "HOSPITAL_BILL", "DISCHARGE_SUMMARY")
OPTIONAL_DOCUMENTS: tuple[str, ...] = ("DIAGNOSTIC_REPORT", "MEDICAL_REPORT")
