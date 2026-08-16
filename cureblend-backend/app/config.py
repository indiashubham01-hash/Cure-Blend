"""
CureBlend Backend — Configuration Module
==========================================
Loads environment variables for MongoDB connection, model paths,
and application-level settings using python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if it exists
load_dotenv()

# ── Base Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # cureblend-backend/
DATA_DIR = BASE_DIR / "data"

# ── MongoDB Configuration ──────────────────────────────────────
MONGODB_URL: str = os.getenv(
    "MONGODB_URL",
    "mongodb://localhost:27017"  # Fallback for local dev
)
MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "cureblend")
MONGODB_COLLECTION_HISTORY: str = os.getenv("MONGODB_COLLECTION_HISTORY", "assessment_history")

# ── Model Artifact Paths ──────────────────────────────────────
MODEL_PATH: Path = DATA_DIR / "model.joblib"
TFIDF_PATH: Path = DATA_DIR / "tfidf.joblib"
LABEL_ENCODER_PATH: Path = DATA_DIR / "label_encoder.joblib"
SCALER_PATH: Path = DATA_DIR / "scaler.joblib"
MLB_PATH: Path = DATA_DIR / "mlb.joblib"
KNOWLEDGE_BASE_PATH: Path = DATA_DIR / "knowledge_base.json"

# ── Application Settings ──────────────────────────────────────
APP_NAME: str = "CureBlend API"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = "AI-Driven Healthcare Recommendation System"
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# ── CORS Origins (supports comma-separated list or wildcard) ───
raw_cors = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,http://localhost:8080"
)
if raw_cors.strip() == "*":
    CORS_ORIGINS: list[str] = ["*"]
else:
    CORS_ORIGINS: list[str] = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]


# ── Top-K predictions to return ───────────────────────────────
TOP_K_PREDICTIONS: int = int(os.getenv("TOP_K_PREDICTIONS", "5"))
TOP_K_FEATURES: int = int(os.getenv("TOP_K_FEATURES", "10"))
