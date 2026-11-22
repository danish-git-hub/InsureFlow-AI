"""
InsureFlow AI — FastAPI backend.

Run from the project root:
    uvicorn api.main:app --reload      (or: python run.py api)

Swagger UI: http://127.0.0.1:8000/docs      ReDoc: http://127.0.0.1:8000/redoc

All errors use one JSON shape:  {"error": {"code": "...", "message": "...", "details": ...}}
Internal errors never leak stack traces or file paths to the client; they are logged server-side.
"""
import secrets
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import analytics, claims, customers, policies, reviews
from api.schemas import ErrorResponse
from app.config import get_logger, settings
from services.claim_service import ServiceError

log = get_logger("api")

TAGS = [
    {"name": "System", "description": "Health and configuration."},
    {"name": "Customers", "description": "Synthetic customer records."},
    {"name": "Policies", "description": "Policy records and the RAG policy-wording library."},
    {"name": "Claims", "description": "Submission, documents, AI processing, status and agent history."},
    {"name": "Human Review", "description": "Human decisions — AI output is advisory only."},
    {"name": "Analytics", "description": "Metrics computed from database records."},
    {"name": "Workflow", "description": "Agents and the LangGraph workflow."},
]

app = FastAPI(
    title=f"{settings.app_name} API",
    description=(f"**{settings.tagline}** Multi-agent insurance claims processing backend.\n\n"
                 "AI agents produce *recommendations*; approvals and rejections are recorded only by human "
                 "reviewers. All data is fictional."),
    version=settings.version,
    openapi_tags=TAGS,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               500: {"model": ErrorResponse}},
)

ERROR_CODES = {400: "BAD_REQUEST", 403: "FORBIDDEN", 404: "NOT_FOUND", 409: "CONFLICT", 413: "PAYLOAD_TOO_LARGE",
               422: "VALIDATION_ERROR", 500: "INTERNAL_ERROR"}


def _error(status: int, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": ERROR_CODES.get(status, "ERROR"),
                                                               "message": message, "details": details}})
@app.middleware("http")
async def verify_api_access(request: Request, call_next):
    # Keep health checks accessible to Render.
    if request.url.path == "/health":
        return await call_next(request)

    expected_token = settings.api_access_token

    if expected_token:
        supplied_token = request.headers.get("X-API-Token", "")

        if not secrets.compare_digest(supplied_token, expected_token):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Valid API access token required.",
                        "details": None,
                    }
                },
            )

    return await call_next(request)

@app.exception_handler(ServiceError)
async def service_error(_: Request, exc: ServiceError):
    return _error(exc.status_code, exc.message, exc.details)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    return _error(422, "Request validation failed.", details)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "An internal error occurred. It has been logged.")


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    ai_mode: str
    demo_mode: bool
    timestamp: datetime


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check used by the dashboard and monitoring."""
    return HealthResponse(status="ok", app=settings.app_name, version=settings.version, ai_mode=settings.mode_label,
                          demo_mode=settings.demo_mode, timestamp=datetime.now(timezone.utc))


for module in (customers, policies, claims, reviews, analytics):
    app.include_router(module.router, prefix="/api/v1")
