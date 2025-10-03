from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    environment: str = Field("development", alias="ENVIRONMENT")
    asr_provider: str = Field("stub", alias="ASR_PROVIDER")
    redis_url: str = Field(..., alias="REDIS_URL")
    redis_prefix: str = Field("da", alias="REDIS_PREFIX")
    redis_password: Optional[str] = Field(None, alias="REDIS_PASSWORD")
    deepseek_api_base: Optional[str] = Field(None, alias="DEEPSEEK_API_BASE")
    deepseek_api_key: Optional[str] = Field(None, alias="DEEPSEEK_API_KEY")
    alibaba_cloud_access_key_id: Optional[str] = Field(
        None, alias="ALIBABA_CLOUD_ACCESS_KEY_ID"
    )
    alibaba_cloud_access_key_secret: Optional[str] = Field(
        None, alias="ALIBABA_CLOUD_ACCESS_KEY_SECRET"
    )
    dashscope_api_key: Optional[str] = Field(None, alias="DASHSCOPE_API_KEY")
    tingwu_appkey: Optional[str] = Field(None, alias="TINGWU_APPKEY")
    tingwu_ak_id: Optional[str] = Field(None, alias="TINGWU_AK_ID")
    tingwu_ak_secret: Optional[str] = Field(None, alias="TINGWU_AK_SECRET")
    tingwu_region: str = Field("cn-shanghai", alias="TINGWU_REGION")
    tingwu_base: str = Field("https://tingwu.aliyuncs.com", alias="TINGWU_BASE")
    tingwu_ws_base: str = Field(
        "wss://tingwu.aliyuncs.com/ws/v1", alias="TINGWU_WS_BASE"
    )
    tingwu_sr: int = Field(16000, alias="TINGWU_SR")
    tingwu_format: str = Field("pcm", alias="TINGWU_FORMAT")
    tingwu_lang: str = Field("cn", alias="TINGWU_LANG")

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
