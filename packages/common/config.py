from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    environment: str = Field("development", alias="ENVIRONMENT")
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    llm_api_base: Optional[str] = Field(None, alias="LLM_API_BASE")
    llm_api_key: Optional[str] = Field(None, alias="LLM_API_KEY")
    asr_api_base: Optional[str] = Field(None, alias="ASR_API_BASE")
    asr_api_key: Optional[str] = Field(None, alias="ASR_API_KEY")
    tts_api_base: Optional[str] = Field(None, alias="TTS_API_BASE")
    tts_api_key: Optional[str] = Field(None, alias="TTS_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()  # type: ignore[arg-type]


settings = get_settings()
