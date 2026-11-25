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
    LOG_FORMAT: str = Field("%(asctime)s - [%(levelname)s] - [%(task_id)s] - %(name)s - "
                            "(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s")
    LOG_FOLDER: str = Field("logs")
    LOG_FILE: str = Field("logs.log")
    LOG_LEVEL: int = Field(logging.INFO)

    # upload limits and decoding
    MAX_UPLOAD_MB: float = Field(10)
    MAX_IMAGE_PIXELS: int = Field(60_000_000)
    DECODE_LONG_SIDE: int = Field(1280)

    # models
    CORNER_MODEL_PATH: str = Field("models/doccornernet_lean.ort")
    TEXT_MODEL_PATH: str = Field("models/ppocrv4_det.onnx")
    FALLBACK_MODEL_PATH: str = Field("models/docaligner_lcnet100.onnx")
    ONNX_THREADS: int = Field(0)                  # 0 = onnxruntime default

    # document presence
    DOC_MIN_CORNER_SCORE: float = Field(0.5)
    DOC_MIN_FALLBACK_SCORE: float = Field(0.5)
    FALLBACK_HEATMAP_THRESHOLD: float = Field(0.3)
    DOC_MIN_TEXT_BOXES: int = Field(1)
    DOC_TEXT_LONG_SIDE: int = Field(960)
    TEXT_PIXEL_THRESHOLD: float = Field(0.3)
    TEXT_BOX_THRESHOLD: float = Field(0.5)
    SCREENSHOT_BLOCK_STD: float = Field(0.6)      # 8x8 block std below this counts as perfectly flat
    SCREENSHOT_MIN_FLAT_SHARE: float = Field(0.5)
    SCREENSHOT_MIN_MEAN: float = Field(50)
    SCREENSHOT_MAX_MEAN: float = Field(235)

    # whole document in frame
    FRAME_EDGE_MARGIN: float = Field(0.03)
    FRAME_MIN_FALLBACK_SCORE: float = Field(0.8)

    # document size and angle
    GEOM_MIN_AREA_SHARE: float = Field(0.20)
    GEOM_MAX_SIDE_RATIO: float = Field(1.5)
    GEOM_MAX_ANGLE_DEV: float = Field(30.0)

    # image quality
    QUALITY_WORK_SIZE: int = Field(1024)
    QUALITY_DOC_INSET: float = Field(0.04)
    SHARP_WORK_SIZE: int = Field(384)             # sharpness score depends on resolution
    SHARP_PRE_SMOOTH: float = Field(0.8)
    SHARP_MIN_SCORE: float = Field(20.0)
    LIGHT_DARK_PIXEL: int = Field(15)
    LIGHT_BRIGHT_PIXEL: int = Field(250)
    LIGHT_MIN_MEAN: float = Field(60)
    LIGHT_MAX_MEAN: float = Field(240)
    LIGHT_MAX_DARK_SHARE: float = Field(0.30)
    LIGHT_MAX_BRIGHT_SHARE: float = Field(0.7)
    GLARE_MIN_VALUE: int = Field(245)
    GLARE_MAX_SATURATION: int = Field(30)
    GLARE_PAPER_PERCENTILE: float = Field(30)
    GLARE_MIN_ABOVE_PAPER: float = Field(30)
    GLARE_MIN_BLOB_SHARE: float = Field(0.002)
    LIGHT_MAX_GLARE_SHARE: float = Field(0.09)


settings = Settings()
