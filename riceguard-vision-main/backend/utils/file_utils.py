import os
from pathlib import Path
from typing import List

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def ensure_upload_directory(upload_folder: str) -> Path:
    upload_path = Path(upload_folder)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def build_upload_path(upload_folder: str, filename: str) -> Path:
    upload_path = ensure_upload_directory(upload_folder)
    return upload_path / filename
