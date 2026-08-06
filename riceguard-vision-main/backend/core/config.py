import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    env_candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "backend" / ".env",
        PROJECT_ROOT / "config" / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        with env_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "RiceGuard AI")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "0.0.0.0")
    model_path: str = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "ml" / "models" / "rice_model.keras"))
    upload_folder: str = os.getenv("UPLOAD_FOLDER", str(PROJECT_ROOT / "uploads"))
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", "5242880"))
    allow_origins: str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:8080",
    )
    mongodb_uri: Optional[str] = os.getenv("MONGODB_URI")
    mongodb_db: str = os.getenv("MONGODB_DB", "riceguard_ai")
    mongodb_collection: str = os.getenv("MONGODB_COLLECTION", "predictions")


settings = Settings()
