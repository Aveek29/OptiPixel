import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes_download import router as download_router
from app.api.routes_enhance import router as enhance_router
from app.api.routes_health import router as health_router
from app.logging_config import setup_logging
from app.services.model_downloader import ensure_model_weights

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Optic Pixel starting up")
    threading.Thread(target=ensure_model_weights, daemon=True).start()
    yield
    logger.info("Optic Pixel shutting down")


app = FastAPI(
    title="Optic Pixel",
    version="1.0.0",
    description="Dynamic AI Image Enhancement Platform",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

_base = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_base / "static")), name="static")
templates = Jinja2Templates(directory=str(_base / "templates"))

app.include_router(health_router)
app.include_router(enhance_router)
app.include_router(download_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page."""
    return templates.TemplateResponse(request, "index.html")
