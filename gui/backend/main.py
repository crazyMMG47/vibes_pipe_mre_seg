from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_config
from .routers import subjects, slices, metrics, export

_dist = Path(__file__).parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = get_config()
    cfg.log_startup()
    if _dist.exists():
        print(f"  Serving built frontend from {_dist}\n")
    yield


app = FastAPI(title="vibes_pipe GUI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(subjects.router)
app.include_router(slices.router)
app.include_router(metrics.router)
app.include_router(export.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
