from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings
import os
import logging



ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="allow"
    )

    # Server
    HOST: str = Field('0.0.0.0')
    PORT: int = Field(5000)
    DEV_MODE: bool = Field(False)

    # logger settings
    LOG_FORMAT: str = Field(f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - "
                            f"%(message)s")
    LOG_FOLDER: str = Field("logs")
    LOG_FILE: str = Field("logs.log")
    LOG_LEVEL: int = Field(logging.INFO)


settings = Settings()