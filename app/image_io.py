import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from settings import settings


ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageDecodeError(ValueError):
    pass


class ImageTooLargeError(ImageDecodeError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as probe:  # reads only the header
            if probe.format not in ALLOWED_FORMATS:
                raise ImageDecodeError(f"unsupported format: {probe.format}")
            if probe.width * probe.height > settings.MAX_IMAGE_PIXELS:
                raise ImageTooLargeError(f"resolution too high: {probe.width}x{probe.height}")
            probe.verify()

        img = Image.open(io.BytesIO(data))
        if img.format == "JPEG":
            img.draft("RGB", (settings.DECODE_LONG_SIDE, settings.DECODE_LONG_SIDE))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((settings.DECODE_LONG_SIDE, settings.DECODE_LONG_SIDE))
        rgb = np.asarray(img)
    except ImageDecodeError:
        raise
    except Exception as error:  # Pillow raises many different errors on broken files
        raise ImageDecodeError(f"cannot decode image: {error}") from error

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
