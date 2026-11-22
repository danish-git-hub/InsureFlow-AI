"""Customer management."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.schemas import CustomerCreate, CustomerOut
from database.connection import get_db
from database.repository import CustomerRepository, LogRepository
from services.claim_service import ServiceError

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.get("", response_model=list[CustomerOut])
def list_customers(search: str | None = Query(None, max_length=100), limit: int = Query(100, ge=1, le=500),
                   db: Session = Depends(get_db)):
    return CustomerRepository(db).list(search=search, limit=limit)


@router.get("/{customer_code}", response_model=CustomerOut)
def get_customer(customer_code: str, db: Session = Depends(get_db)):
    customer = CustomerRepository(db).get_by_code(customer_code.upper())
    if customer is None:
        raise ServiceError(f"Customer {customer_code} not found.", 404)
    return customer


@router.post("", response_model=CustomerOut, status_code=201)
def create_customer(body: CustomerCreate, db: Session = Depends(get_db)):
    """Create a synthetic customer (fictional data only)."""
    repo = CustomerRepository(db)
    customer = repo.create(customer_code=repo.next_code(), **body.model_dump())
    LogRepository(db).log("api", "CUSTOMER_CREATED", f"{customer.customer_code} created")
    return customer
