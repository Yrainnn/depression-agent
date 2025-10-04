import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    environment: str = Field("development", alias="ENVIRONMENT")
    asr_provider: str = Field(
        default_factory=lambda: os.getenv("ASR_PROVIDER", "tingwu"),
        alias="ASR_PROVIDER",
    )
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
    dashscope_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY"),
        alias="DASHSCOPE_API_KEY",
    )
    tingwu_appkey: Optional[str] = Field(
        default_factory=lambda: os.getenv("TINGWU_APPKEY")
        or os.getenv("ALIBABA_TINGWU_APPKEY"),
        alias="TINGWU_APPKEY",
    )
    alibaba_tingwu_appkey: Optional[str] = Field(
        default_factory=lambda: os.getenv("ALIBABA_TINGWU_APPKEY"),
        alias="ALIBABA_TINGWU_APPKEY",
    )
    tingwu_app_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("TINGWU_APP_ID"),
        alias="TINGWU_APP_ID",
    )
    tingwu_ak_id: Optional[str] = Field(None, alias="TINGWU_AK_ID")
    tingwu_ak_secret: Optional[str] = Field(None, alias="TINGWU_AK_SECRET")
    tingwu_region: str = Field(
        default_factory=lambda: os.getenv("TINGWU_REGION", "cn-beijing"),
        alias="TINGWU_REGION",
    )
    tingwu_base: str = Field("https://tingwu.aliyuncs.com", alias="TINGWU_BASE")
    tingwu_model: str = Field(
        default_factory=lambda: os.getenv("TINGWU_MODEL", "paraformer-realtime-v2"),
        alias="TINGWU_MODEL",
    )
    tingwu_base_address: Optional[str] = Field(
        default_factory=lambda: os.getenv("TINGWU_BASE_ADDRESS"),
        alias="TINGWU_BASE_ADDRESS",
    )
    tingwu_ws_base: str = Field(
        "wss://tingwu.aliyuncs.com/ws/v1", alias="TINGWU_WS_BASE"
    )
    tingwu_sr: int = Field(
        default_factory=lambda: int(
            os.getenv("TINGWU_SR")
            or os.getenv("TINGWU_SAMPLE_RATE")
            or "16000"
        ),
        alias="TINGWU_SR",
    )
    tingwu_format: str = Field(
        default_factory=lambda: os.getenv("TINGWU_FORMAT", "pcm"),
        alias="TINGWU_FORMAT",
    )
    tingwu_lang: str = Field(
        default_factory=lambda: os.getenv("TINGWU_LANG", "cn"),
        alias="TINGWU_LANG",
    )

    # ------------------------------------------------------------------
    # Compatibility accessors (new uppercase field names exposed for callers)
    @property
    def DASHSCOPE_API_KEY(self) -> Optional[str]:
        return self.dashscope_api_key

    @property
    def TINGWU_APPKEY(self) -> Optional[str]:
        return self.tingwu_appkey or self.alibaba_tingwu_appkey

    @property
    def TINGWU_APP_ID(self) -> Optional[str]:
        return self.tingwu_app_id

    @property
    def TINGWU_BASE_ADDRESS(self) -> Optional[str]:
        return self.tingwu_base_address

    @property
    def TINGWU_MODEL(self) -> str:
        return self.tingwu_model

    @property
    def TINGWU_FORMAT(self) -> str:
        return self.tingwu_format

    @property
    def TINGWU_SR(self) -> int:
        return self.tingwu_sr

    @property
    def TINGWU_LANG(self) -> str:
        return self.tingwu_lang

    @property
    def ASR_PROVIDER(self) -> str:
        return self.asr_provider

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
