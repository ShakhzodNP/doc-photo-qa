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

    # upload limits and decoding
    MAX_UPLOAD_MB: float = Field(10)              # bigger files -> HTTP 413
    MAX_IMAGE_PIXELS: int = Field(60_000_000)     # bigger resolution -> HTTP 413 (decompression-bomb guard)
    DECODE_LONG_SIDE: int = Field(1280)           # images are downscaled to this long side right after decoding

    # models
    CORNER_MODEL_PATH: str = Field("models/doccornernet_lean.ort")   # relative to project root
    TEXT_MODEL_PATH: str = Field("models/ppocrv4_det.onnx")
    FALLBACK_MODEL_PATH: str = Field("models/docaligner_lcnet100.onnx")
    ONNX_THREADS: int = Field(0)                  # 0 = onnxruntime default

    # document presence
    DOC_MIN_CORNER_SCORE: float = Field(0.5)      # document-present probability below -> NO_DOCUMENT
    DOC_MIN_FALLBACK_SCORE: float = Field(0.5)    # small-document model: weakest corner peak needed to accept
    FALLBACK_HEATMAP_THRESHOLD: float = Field(0.3)  # heatmap value for a pixel to belong to a corner
    DOC_MIN_TEXT_BOXES: int = Field(1)            # fewer text regions inside the document -> BLANK_PAGE
    DOC_TEXT_LONG_SIDE: int = Field(960)          # straightened document size for text detection
    TEXT_PIXEL_THRESHOLD: float = Field(0.3)      # text probability for a pixel to count as text
    TEXT_BOX_THRESHOLD: float = Field(0.5)        # mean text probability for a region to count as a text box
    SCREENSHOT_BLOCK_STD: float = Field(0.6)      # 8x8 block with lower pixel std counts as perfectly flat
    SCREENSHOT_MIN_FLAT_SHARE: float = Field(0.5) # share of flat blocks above -> SCREENSHOT

    # image quality
    QUALITY_WORK_SIZE: int = Field(1024)          # long side for quality checks (scores depend on resolution)
    SHARP_MIN_SCORE: float = Field(60.0)          # below -> BLURRY
    LIGHT_DARK_PIXEL: int = Field(15)             # gray value counted as "almost black"
    LIGHT_BRIGHT_PIXEL: int = Field(250)          # gray value counted as "almost white"
    LIGHT_MIN_MEAN: float = Field(60)             # mean brightness below -> TOO_DARK
    LIGHT_MAX_MEAN: float = Field(225)            # mean brightness above -> OVEREXPOSED
    LIGHT_MAX_DARK_SHARE: float = Field(0.30)     # share of almost black pixels above -> TOO_DARK
    LIGHT_MAX_BRIGHT_SHARE: float = Field(0.25)   # share of almost white pixels above -> OVEREXPOSED
    GLARE_MIN_VALUE: int = Field(245)             # HSV brightness of a glare pixel
    GLARE_MAX_SATURATION: int = Field(30)         # glare is almost colorless
    GLARE_MIN_BLOB_SHARE: float = Field(0.002)    # ignore glare blobs smaller than this share of the image
    LIGHT_MAX_GLARE_SHARE: float = Field(0.01)    # total glare area above -> GLARE


settings = Settings()