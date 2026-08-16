from fastapi import APIRouter

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Return service health status."""
    return HealthResponse(
        status="ok",
        service="optic-pixel",
        version="1.0.0",
    )
