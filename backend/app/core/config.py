import os
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# Project root directory: parent of backend
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "SDG 5 Gender Stereotype AI Auditor"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Path configurations
    BASE_DIR: Path = PROJECT_ROOT
    DATASET_DIR: Path = PROJECT_ROOT / "dataset"
    BENCHMARKS_DIR: Path = PROJECT_ROOT / "data" / "benchmarks"
    STORAGE_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    
    # Database configuration (default SQLite for zero-dependency local runs)
    DB_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/sdg5_audit.db")
    
    # Model & Chunking parameters
    MODEL_MAX_SEQUENCE_LENGTH: int = 256
    CONTEXT_CHUNK_MIN_TOKENS: int = 150
    CONTEXT_CHUNK_MAX_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int = 30
    
    # PDF Parser configurable margins
    PDF_DEFAULT_HEADER_MARGIN_PCT: float = 0.05
    PDF_DEFAULT_FOOTER_MARGIN_PCT: float = 0.05
    
    model_config = ConfigDict(extra="allow", env_file=".env")


settings = Settings()

# Ensure runtime directories exist
settings.BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
