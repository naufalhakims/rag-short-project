"""
Application configuration — loads all settings from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _read_secret_file(path: str) -> str:
    """Read a local secret file if present."""
    secret_path = Path(path)
    if not secret_path.exists():
        return ""
    raw_value = secret_path.read_text(encoding="utf-8").strip()
    if "=" in raw_value:
        raw_value = raw_value.split("=", 1)[1].strip()
    return raw_value.strip("\"'")


class Settings:
    """Centralized configuration loaded from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "") or _read_secret_file("Groq-API.txt")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./chat_history.db")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "pocket_librarian")


settings = Settings()
