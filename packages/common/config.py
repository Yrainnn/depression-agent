from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    environment: str = Field("development", alias="ENVIRONMENT")
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    deepseek_api_base: Optional[str] = Field(None, alias="DEEPSEEK_API_BASE")
    deepseek_api_key: Optional[str] = Field(None, alias="DEEPSEEK_API_KEY")
    alibaba_cloud_access_key_id: Optional[str] = Field(
        None, alias="ALIBABA_CLOUD_ACCESS_KEY_ID"
    )
    alibaba_cloud_access_key_secret: Optional[str] = Field(
        None, alias="ALIBABA_CLOUD_ACCESS_KEY_SECRET"
    )
    dashscope_api_key: Optional[str] = Field(None, alias="DASHSCOPE_API_KEY")

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
