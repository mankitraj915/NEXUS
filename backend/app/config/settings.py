from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "NEXUS"
    API_V1_STR: str = "/api/v1"

    POSTGRES_USER: str = "nexus"
    POSTGRES_PASSWORD: str = "nexus_password"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5433"
    POSTGRES_DB: str = "nexus"

    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    EMBEDDING_BACKEND: Literal["onnx", "torch"] = "onnx"
    EMBEDDING_ONNX_VARIANT: Literal["fp32", "int8"] = "fp32"
    EMBEDDING_MODEL_DIR: str = "models/all-MiniLM-L6-v2-onnx"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )


settings = Settings()
