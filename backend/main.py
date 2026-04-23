# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from models.loader import load_all_models
from routers import auth, predict, resources


# ── Startup/Shutdown lifecycle ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting CHARAK API...")
    load_all_models()          # Load your .pkl files into memory once
    yield
    print("Shutting down API.")


# ── App initialization ────────────────────────────────────────────────
app = FastAPI(
    title="CHARAK API",
    description="CHARAK backend for patient flow prediction and hospital resource management",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS (allow your React frontend to call this API) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(resources.router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "api": "CHARAK v1.0"}