from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _backend_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return path.resolve()


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
        validate_default=True,
    )

    app_name: str = Field(
        default="Asthma Guideline RAG",
        validation_alias=AliasChoices("APP_NAME", "app_name"),
    )
    app_version: str = Field(
        default="0.1.0",
        validation_alias=AliasChoices("APP_VERSION", "app_version"),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "ollama_base_url"),
    )
    ollama_model: str = Field(
        default="llama3.2:1b",
        validation_alias=AliasChoices("OLLAMA_MODEL", "ollama_model"),
    )
    ollama_num_predict: int = Field(
        default=48,
        ge=32,
        le=2048,
        validation_alias=AliasChoices("OLLAMA_NUM_PREDICT", "ollama_num_predict"),
    )
    ollama_num_ctx: int = Field(
        default=2048,
        ge=1024,
        le=32768,
        validation_alias=AliasChoices("OLLAMA_NUM_CTX", "ollama_num_ctx"),
    )
    vector_store_path: Path = Field(
        default=Path("data/vector_store/chroma"),
        validation_alias=AliasChoices("VECTOR_STORE_PATH", "vector_store_path"),
    )
    rag_config_path: Path = Field(
        default=Path("data/rag_config.json"),
        validation_alias=AliasChoices("RAG_CONFIG_PATH", "rag_config_path"),
    )
    chunks_path: Path = Field(
        default=Path("data/chunks.jsonl"),
        validation_alias=AliasChoices("CHUNKS_PATH", "chunks_path"),
    )
    safety_rules_path: Path = Field(
        default=Path("app/core/safety_rules.json"),
        validation_alias=AliasChoices("SAFETY_RULES_PATH", "safety_rules_path"),
    )
    frontend_origins_raw: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        validation_alias=AliasChoices("FRONTEND_ORIGINS", "frontend_origins_raw"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )
    request_timeout_seconds: float = Field(
        default=600,
        gt=0,
        le=600,
        validation_alias=AliasChoices(
            "REQUEST_TIMEOUT_SECONDS", "request_timeout_seconds"
        ),
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias=AliasChoices("RETRIEVAL_TOP_K", "retrieval_top_k"),
    )
    max_question_chars: int = Field(
        default=2000,
        ge=100,
        le=10000,
        validation_alias=AliasChoices("MAX_QUESTION_CHARS", "max_question_chars"),
    )

    @field_validator(
        "vector_store_path",
        "rag_config_path",
        "chunks_path",
        "safety_rules_path",
        mode="before",
    )
    @classmethod
    def resolve_paths(cls, value: str | Path) -> Path:
        return _backend_path(value)

    @field_validator("ollama_base_url")
    @classmethod
    def normalize_ollama_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) URL")
        return value

    @field_validator("ollama_model", "app_name", "app_version", "log_level")
    @classmethod
    def non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @property
    def frontend_origins(self) -> list[str]:
        return [
            origin.strip().rstrip("/")
            for origin in self.frontend_origins_raw.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
