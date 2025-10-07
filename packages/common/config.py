import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    environment: str = Field("development", alias="ENVIRONMENT")
    redis_url: str = Field(..., alias="REDIS_URL")
    redis_prefix: str = Field("da", alias="REDIS_PREFIX")
    redis_password: Optional[str] = Field(None, alias="REDIS_PASSWORD")
    deepseek_api_base: Optional[str] = Field(None, alias="DEEPSEEK_API_BASE")
    deepseek_api_key: Optional[str] = Field(None, alias="DEEPSEEK_API_KEY")
    prompt_hamd17_path: Optional[str] = Field(None, alias="PROMPT_HAMD17_PATH")
    prompt_diagnosis_path: Optional[str] = Field(None, alias="PROMPT_DIAGNOSIS_PATH")
    prompt_mdd_judgment_path: Optional[str] = Field(None, alias="PROMPT_MDD_JUDGMENT_PATH")
    prompt_clarify_cn_path: Optional[str] = Field(None, alias="PROMPT_CLARIFY_CN_PATH")
    enable_ds_controller: bool = Field(True, alias="ENABLE_DS_CONTROLLER")
    alibaba_cloud_access_key_id: Optional[str] = Field(
        None, alias="ALIBABA_CLOUD_ACCESS_KEY_ID"
    )
    alibaba_cloud_access_key_secret: Optional[str] = Field(
        None, alias="ALIBABA_CLOUD_ACCESS_KEY_SECRET"
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
    tingwu_ak_id: Optional[str] = Field(None, alias="TINGWU_AK_ID")
    tingwu_ak_secret: Optional[str] = Field(None, alias="TINGWU_AK_SECRET")
    tingwu_region: str = Field(
        default_factory=lambda: os.getenv("TINGWU_REGION", "cn-beijing"),
        alias="TINGWU_REGION",
    )
    tingwu_base: str = Field("https://tingwu.aliyuncs.com", alias="TINGWU_BASE")
    tingwu_ws_base: str = Field(
        "wss://tingwu.aliyuncs.com/ws/v1", alias="TINGWU_WS_BASE"
    )
    tingwu_sample_rate: int = Field(
        default_factory=lambda: int(
            os.getenv("TINGWU_SAMPLE_RATE")
            or os.getenv("TINGWU_SR")
            or "16000"
        ),
        alias="TINGWU_SAMPLE_RATE",
    )
    tingwu_format: str = Field(
        default_factory=lambda: os.getenv("TINGWU_FORMAT", "pcm"),
        alias="TINGWU_FORMAT",
    )
    tingwu_lang: str = Field(
        default_factory=lambda: os.getenv("TINGWU_LANG", "cn"),
        alias="TINGWU_LANG",
    )
    oss_region: Optional[str] = Field(None, alias="OSS_REGION")
    oss_endpoint: Optional[str] = Field(None, alias="OSS_ENDPOINT")
    oss_bucket: Optional[str] = Field(None, alias="OSS_BUCKET")
    oss_key_prefix: str = Field("tingwu-uploads/", alias="OSS_KEY_PREFIX")

    # ------------------------------------------------------------------
    # Compatibility accessors (new uppercase field names exposed for callers)
    @property
    def ALIBABA_CLOUD_ACCESS_KEY_ID(self) -> Optional[str]:
        return self.alibaba_cloud_access_key_id

    @property
    def ALIBABA_CLOUD_ACCESS_KEY_SECRET(self) -> Optional[str]:
        return self.alibaba_cloud_access_key_secret
        
    @property
    def TINGWU_APPKEY(self) -> Optional[str]:
        return self.tingwu_appkey or self.alibaba_tingwu_appkey

    @property
    def ALIBABA_TINGWU_APPKEY(self) -> Optional[str]:
        return self.alibaba_tingwu_appkey

    @property
    def TINGWU_REGION(self) -> str:
        return self.tingwu_region

    @property
    def TINGWU_FORMAT(self) -> str:
        return self.tingwu_format

    @property
    def TINGWU_SR(self) -> int:
        return self.tingwu_sample_rate

    @property
    def TINGWU_SAMPLE_RATE(self) -> int:
        return self.tingwu_sample_rate

    @property
    def TINGWU_LANG(self) -> str:
        return self.tingwu_lang

    @property
    def OSS_REGION(self) -> Optional[str]:
        return self.oss_region

    @property
    def OSS_ENDPOINT(self) -> Optional[str]:
        return self.oss_endpoint

    @property
    def OSS_BUCKET(self) -> Optional[str]:
        return self.oss_bucket

    @property
    def OSS_KEY_PREFIX(self) -> str:
        return self.oss_key_prefix

    @property
    def ALIBABA_CLOUD_ACCESS_KEY_ID(self) -> Optional[str]:
        return self.alibaba_cloud_access_key_id

    @property
    def ALIBABA_CLOUD_ACCESS_KEY_SECRET(self) -> Optional[str]:
        return self.alibaba_cloud_access_key_secret

    @property
    def DEEPSEEK_API_BASE(self) -> Optional[str]:
        return self.deepseek_api_base

    @property
    def DEEPSEEK_API_KEY(self) -> Optional[str]:
        return self.deepseek_api_key

    @property
    def PROMPT_HAMD17_PATH(self) -> Optional[str]:
        return self.prompt_hamd17_path

    @property
    def PROMPT_DIAGNOSIS_PATH(self) -> Optional[str]:
        return self.prompt_diagnosis_path

    @property
    def PROMPT_MDD_JUDGMENT_PATH(self) -> Optional[str]:
        return self.prompt_mdd_judgment_path

    @property
    def PROMPT_CLARIFY_CN_PATH(self) -> Optional[str]:
        return self.prompt_clarify_cn_path

    @property
    def ENABLE_DS_CONTROLLER(self) -> bool:
        return self.enable_ds_controller

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
