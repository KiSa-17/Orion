# backend/config.py
from pathlib import Path

from dotenv import load_dotenv
import os

_backend_dir = Path(__file__).resolve().parent
load_dotenv(_backend_dir / "environment.env")
load_dotenv()

class Settings:
    MODEL_DIR: str = os.getenv("MODEL_DIR", "../ml_models")
    MODELS_2_DIR: str = os.getenv(
        "MODELS_2_DIR",
        str(Path(__file__).resolve().parent.parent / "ml models 2" / "ml models"),
    )
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    @staticmethod
    def cors_origins() -> list[str]:
        raw = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
        return [o.strip() for o in raw.split(",") if o.strip()]


settings = Settings()