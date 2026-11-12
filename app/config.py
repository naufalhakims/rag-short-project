"""Application settings, loaded from environment variables and the project's .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Providers in the order "auto" mode tries them.
PROVIDER_ORDER = ("anthropic", "groq", "google", "openai")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "GROQ_MODEL=" means "use the default", not ""
        extra="ignore",
    )

    # ── LLM routing ──────────────────────────────────────────────
    # "auto" uses every provider that has credentials, in PROVIDER_ORDER.
    # Naming a provider puts it first; the others remain as fallbacks
    # unless LLM_FALLBACK is false.
    llm_provider: str = "auto"
    llm_fallback: bool = True
    llm_temperature: float = 0.2
    llm_max_output_tokens: int | None = None  # None = per-provider default
    llm_timeout_seconds: float = 30.0

    # Groq (OpenAI-compatible). Free tier: 8,000 tokens/minute per model.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "openai/gpt-oss-20b"

    # Anthropic Claude (official SDK).
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_fast_model: str = ""  # empty = use ANTHROPIC_MODEL
    anthropic_effort: str = "medium"

    # Google Gemini, through its OpenAI-compatible endpoint.
    google_api_key: str = Field(
        "", validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    google_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model: str = ""  # empty = pick from the models the key can access
    gemini_fast_model: str = ""

    # Any other OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, LM Studio…).
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    openai_fast_model: str = ""

    # ── Retrieval ────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # "none" = no reranking
    chunk_size: int = 900  # characters
    chunk_overlap: int = 120
    retrieval_candidates: int = 24  # per search leg, before fusion
    rerank_candidates: int = 12  # fused candidates the cross-encoder reads (CPU cost)
    top_k: int = 6  # most passages handed to the LLM
    min_passages: int = 3  # once something matches, always include this many
    # Cut-off on the reranker's 0..1 relevance. MS MARCO cross-encoders score long,
    # multi-topic chunks low even when they contain the answer, so this is deliberately
    # small: measured relevant passages score >= 0.017, unrelated ones <= 0.002.
    rerank_min_score: float = 0.005
    history_turns: int = 3  # previous exchanges given to the LLM

    # ── Storage ──────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path | None = None  # where local models are cached; default <data_dir>/models
    max_upload_mb: int = 25

    @model_validator(mode="after")
    def _normalise(self) -> "Settings":
        if not self.data_dir.is_absolute():
            self.data_dir = PROJECT_ROOT / self.data_dir
        if self.model_dir is not None and not self.model_dir.is_absolute():
            self.model_dir = PROJECT_ROOT / self.model_dir
        self.llm_provider = self.llm_provider.strip().lower() or "auto"
        if self.reranker_model.strip().lower() in {"none", "off", "false", "no", "0"}:
            self.reranker_model = ""
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 5
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "librarian.db"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def model_cache_dir(self) -> Path:
        return self.model_dir or self.data_dir / "models"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "app.log"


@lru_cache
def get_settings() -> Settings:
    return Settings()
